# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Ming-Omni text streaming."""

from __future__ import annotations

from sglang_omni.models.ming_omni.pipeline.stages import (
    _build_ming_text_stream_chunk,
    _decode_ming_payload,
    _wants_text_stream,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeTokenizer:
    eos_token_id = 0

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        pieces = {
            1: "Hel",
            2: "lo",
            3: "!",
        }
        return "".join(pieces.get(int(token_id), "") for token_id in token_ids)


def _payload(*, stream: bool) -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs="hi", params={"stream": stream}),
        data={},
    )


def test_ming_text_stream_emits_incremental_deltas_without_final_duplicate() -> None:
    payload = _payload(stream=True)
    tokenizer = _FakeTokenizer()

    chunks = [
        _build_ming_text_stream_chunk(
            payload,
            token_id,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
            step=step,
        )
        for step, token_id in enumerate([1, 2, 3, tokenizer.eos_token_id], start=1)
    ]

    assert [chunk["text"] for chunk in chunks if chunk is not None] == [
        "Hel",
        "lo",
        "!",
    ]
    assert chunks[-1] is None


def test_ming_text_stream_honors_requested_text_modalities() -> None:
    payload = _payload(stream=True)
    payload.request.metadata["output_modalities"] = ["text"]

    assert _wants_text_stream(payload)


def test_ming_text_stream_ignores_non_text_modalities() -> None:
    payload = _payload(stream=True)
    payload.request.metadata["output_modalities"] = ["audio"]

    assert not _wants_text_stream(payload)


def test_ming_text_stream_flushes_only_unemitted_final_tail() -> None:
    payload = _payload(stream=True)
    payload.data = {
        "stream_state": {
            "token_ids": [1, 2],
            "text": "Hello",
            "emitted_text": "Hel",
        }
    }
    tokenizer = _FakeTokenizer()

    chunk = _build_ming_text_stream_chunk(
        payload,
        tokenizer.eos_token_id,
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        step=3,
    )

    assert chunk is not None
    assert chunk["text"] == "lo"


def test_ming_streaming_decode_result_keeps_usage_but_removes_full_text() -> None:
    payload = _payload(stream=True)
    payload.data = {
        "thinker_out": {
            "output_ids": [1, 2, 3],
            "step": 3,
            "is_final": True,
            "finish_reason": "stop",
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "extra_model_outputs": {},
        }
    }

    result_payload = _decode_ming_payload(
        payload,
        tokenizer=_FakeTokenizer(),
        eos_token_id=_FakeTokenizer.eos_token_id,
    )

    assert "text" not in result_payload.data
    assert result_payload.data["finish_reason"] == "stop"
    assert result_payload.data["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


def test_ming_non_streaming_decode_result_keeps_full_text() -> None:
    payload = _payload(stream=False)
    payload.data = {
        "thinker_out": {
            "output_ids": [1, 2, 3],
            "step": 3,
            "is_final": True,
            "finish_reason": "stop",
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "extra_model_outputs": {},
        }
    }

    result_payload = _decode_ming_payload(
        payload,
        tokenizer=_FakeTokenizer(),
        eos_token_id=_FakeTokenizer.eos_token_id,
    )

    assert result_payload.data["text"] == "Hello!"
    assert result_payload.data["usage"]["total_tokens"] == 10
