# SPDX-License-Identifier: Apache-2.0
"""Connected live e2e for Ming-Omni text streaming.

Unlike ``test_streaming_e2e_glue.py`` (which exercises each seam in isolation),
this drives the *real* thinker per-token stream callback and the *real*
``MingTextDecodeScheduler`` — running its actual queue loop on a background
thread — across a multi-token + EOS sequence. It mirrors the runtime relay hop
that turns a thinker ``OutgoingMessage(type="stream", target=decode)`` into a
``StreamItem`` on the decode stage inbox.

No GPU/model is needed: only the tokenizer is faked. Every other component is
the production one — ``decode_events``, ``build_text_usage``, the uint8 tensor
transport, the decode-stage factory, and the scheduler run loop.

Covers the gaps called out in review:
  #1 connected chain: builder output is fed into the decode scheduler, and the
     uint8 tensor contract is round-tripped end to end.
  #2 the real ``start()`` loop and the ``_emit_error`` failure path.
  #3 a multi-token sequence terminated by EOS.
  #4 no tail loss: concatenated streamed deltas equal the final decoded text,
     even across an incomplete-UTF8 boundary, and the final result drops the
     duplicate full text while keeping usage + finish_reason.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

# These components import torch (directly or transitively). Skip the whole
# module when torch is unavailable rather than erroring at collection.
torch = pytest.importorskip("torch")

from sglang_omni.models.ming_omni.bootstrap import (  # noqa: E402
    make_thinker_stream_output_builder,
)
from sglang_omni.models.ming_omni.pipeline.next_stage import (  # noqa: E402
    DECODE_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.ming_omni.stages import (  # noqa: E402
    MingTextDecodeScheduler,
    create_decode_executor,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem  # noqa: E402
from sglang_omni.proto import OmniRequest, StagePayload  # noqa: E402
from sglang_omni.scheduling.messages import IncomingMessage  # noqa: E402

_TOKENIZER_PATH = (
    "sglang_omni.models.ming_omni.components.common.load_ming_tokenizer"
)

# Content tokens (no EOS). Token 10 is an incomplete UTF-8 prefix completed by
# token 11 — so the streaming decoder must buffer it and only flush "世" once 11
# arrives. This is the exact divergence that could otherwise lose the tail.
CONTENT_TOKENS = [1, 2, 3, 10, 11]
EOS_TOKEN = 99
STREAM_SEQUENCE = CONTENT_TOKENS + [EOS_TOKEN]
EXPECTED_TEXT = "Hello world!世"


class _FakeTokenizer:
    """Byte-faithful fake tokenizer with a split multi-byte character."""

    eos_token_id = EOS_TOKEN
    _PIECES = {1: "Hello", 2: " world", 3: "!"}

    def decode(self, ids, skip_special_tokens=True):
        ids = [int(i) for i in ids]
        out: list[str] = []
        i = 0
        while i < len(ids):
            tok = ids[i]
            if tok == 10:
                if i + 1 < len(ids) and ids[i + 1] == 11:
                    out.append("世")
                    i += 2
                    continue
                out.append("�")  # incomplete multi-byte -> replacement char
                i += 1
                continue
            if tok == 11:  # 11 without its 10 prefix: still incomplete
                out.append("�")
                i += 1
                continue
            if tok == self.eos_token_id:
                i += 1
                continue
            out.append(self._PIECES.get(tok, ""))
            i += 1
        return "".join(out)


def _stream_request_payload(request_id: str = "req-live") -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs="hi",
            params={"stream": True},
            metadata={"output_modalities": ["text"]},
        ),
        data={},
    )


def _final_payload(request_id: str, *, stream: bool) -> StagePayload:
    metadata = {"output_modalities": ["text"]} if stream else {}
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs="hi", params={"stream": stream}, metadata=metadata
        ),
        data={
            "prompt": {"input_ids": [0, 0, 0, 0]},  # 4 prompt tokens
            "thinker_out": {
                "output_ids": list(CONTENT_TOKENS),  # 5 completion tokens
                "step": len(CONTENT_TOKENS),
                "is_final": True,
                "finish_reason": "stop",
                "extra_model_outputs": {},
            },
        },
    )


def _drive_thinker(builder, payload, tokens):
    """Run the real per-token callback and collect its OutgoingMessages."""
    req = SimpleNamespace(
        is_chunked=0, _ming_stream_token_ids=None, _ming_stream_emitted_text=""
    )
    req_data = SimpleNamespace(req=req, stage_payload=payload)
    msgs = []
    for tok in tokens:
        msgs.extend(builder(payload.request_id, req_data, SimpleNamespace(data=tok)))
    return msgs


def _run_decode(scheduler, stream_msgs, final_payload, *, timeout=5.0):
    """Feed stream chunks + final payload through the real scheduler loop."""
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        for chunk_id, msg in enumerate(stream_msgs):
            # Relay hop: the runtime carries the tensor payload + metadata to the
            # decode stage as a StreamItem on a stream_chunk message.
            item = StreamItem(
                chunk_id=chunk_id,
                data=msg.data,
                from_stage=THINKER_STAGE,
                metadata=msg.metadata,
            )
            scheduler.inbox.put(
                IncomingMessage(
                    request_id=msg.request_id, type="stream_chunk", data=item
                )
            )
        scheduler.inbox.put(
            IncomingMessage(
                request_id=final_payload.request_id,
                type="new_request",
                data=final_payload,
            )
        )

        streams = []
        result = None
        deadline = time.time() + timeout
        while result is None and time.time() < deadline:
            try:
                out = scheduler.outbox.get(timeout=0.1)
            except Exception:
                continue
            if out.type == "stream":
                streams.append(out)
            elif out.type == "result":
                result = out
            elif out.type == "error":
                raise AssertionError(f"decode emitted error: {out.data!r}")
        return streams, result
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)


def test_live_thinker_to_decode_streams_full_text_without_tail_loss(monkeypatch):
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(_TOKENIZER_PATH, lambda *a, **k: tokenizer)

    # Real decode-stage factory -> real MingTextDecodeScheduler wrapping the real
    # _decode closure (decode_events + build_text_usage + strip logic).
    scheduler = create_decode_executor("dummy")
    assert isinstance(scheduler, MingTextDecodeScheduler)

    builder = make_thinker_stream_output_builder(
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        target_stage=DECODE_STAGE,
        require_text_output_modality=True,
        require_request_stream=True,
    )
    stream_msgs = _drive_thinker(
        builder, _stream_request_payload(), STREAM_SEQUENCE
    )

    # #1 cross-stage contract: every relayed payload is a uint8 tensor that
    # round-trips back to the same delta the client ultimately sees.
    assert stream_msgs, "thinker produced no stream chunks"
    for msg in stream_msgs:
        assert msg.target == DECODE_STAGE
        assert isinstance(msg.data, torch.Tensor)
        assert msg.data.dtype == torch.uint8

    streams, result = _run_decode(
        scheduler, stream_msgs, _final_payload("req-live", stream=True)
    )

    assert result is not None, "decode never produced a final result"

    # #3 + #4: concatenated client-visible deltas equal the final decode, with no
    # character lost across the incomplete-UTF8 boundary.
    streamed = "".join(s.data["text"] for s in streams)
    assert streamed == EXPECTED_TEXT == tokenizer.decode(CONTENT_TOKENS)
    assert all(s.data["modality"] == "text" for s in streams)
    assert [tid for s in streams for tid in s.data["token_ids"]] == [1, 2, 3, 11]

    # #4: final result drops the duplicate full text but keeps usage + finish.
    assert "text" not in result.data.data
    assert result.data.data["finish_reason"] == "stop"
    assert result.data.data["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 5,
        "total_tokens": 9,
    }


def test_live_decode_keeps_full_text_when_not_streaming(monkeypatch):
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(_TOKENIZER_PATH, lambda *a, **k: tokenizer)
    scheduler = create_decode_executor("dummy")

    # No stream chunks were forwarded -> the final text must be preserved.
    streams, result = _run_decode(
        scheduler, [], _final_payload("req-ns", stream=False)
    )

    assert streams == []
    assert result is not None
    assert result.data.data["text"] == EXPECTED_TEXT
    assert result.data.data["usage"]["total_tokens"] == 9


def test_live_decode_scheduler_loop_emits_error_on_decode_failure():
    # #2: the real start() loop must catch a decode failure and surface it as an
    # error message rather than crashing the scheduler thread.
    def _boom(payload, *, strip_text=False):
        raise RuntimeError("decode boom")

    scheduler = MingTextDecodeScheduler(_boom)
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        scheduler.inbox.put(
            IncomingMessage(
                request_id="req-err",
                type="new_request",
                data=_stream_request_payload("req-err"),
            )
        )
        out = scheduler.outbox.get(timeout=5.0)
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)

    assert out.type == "error"
    assert isinstance(out.data, RuntimeError)
