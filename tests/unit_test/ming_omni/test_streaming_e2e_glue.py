# SPDX-License-Identifier: Apache-2.0
"""Glue tests for streaming TTS: thinker emits + client merges."""

from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk
from sglang_omni.models.ming_omni.bootstrap import make_thinker_stream_output_builder
from sglang_omni.models.ming_omni.components.streaming_text import text_to_uint8_tensor
from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig
from sglang_omni.models.ming_omni.pipeline.next_stage import DECODE_STAGE, THINKER_STAGE
from sglang_omni.models.ming_omni.stages import MingTextDecodeScheduler
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


class _FakeTokenizer:
    def __init__(self) -> None:
        self.vocab = {
            5: "Hello",
            6: " world",
            7: ".",
            8: " Tail",
        }

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.vocab.get(int(i), "") for i in ids)


def _make_req():
    return SimpleNamespace(
        is_chunked=0,
        _ming_stream_token_ids=None,
        _ming_stream_emitted_text="",
    )


def _payload(
    *,
    request_id: str = "req-1",
    stream: bool = True,
    modalities: list[str] | None = None,
):
    metadata = {}
    if modalities is not None:
        metadata["output_modalities"] = modalities
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hi", params={"stream": stream}, metadata=metadata),
        data={},
    )


def _make_req_data(req, payload=None):
    return SimpleNamespace(req=req, stage_payload=payload)


def _make_req_output(token_id):
    return SimpleNamespace(data=token_id)


def test_thinker_stream_builder_emits_to_segmenter():
    builder = make_thinker_stream_output_builder(
        tokenizer=_FakeTokenizer(),
        eos_token_id=None,
    )
    req = _make_req()
    req_data = _make_req_data(req)

    msgs = builder("req-1", req_data, _make_req_output(5))
    # Thinker is not terminal, so only inter-stage stream to segmenter.
    assert len(msgs) == 1
    assert msgs[0].target == "segmenter"
    assert msgs[0].data.dtype.is_floating_point is False  # uint8 tensor
    assert bytes(msgs[0].data.tolist()).decode("utf-8") == "Hello"


def test_thinker_stream_builder_can_emit_to_decode_for_text_client():
    builder = make_thinker_stream_output_builder(
        tokenizer=_FakeTokenizer(),
        eos_token_id=None,
        target_stage=DECODE_STAGE,
        require_text_output_modality=True,
        require_request_stream=True,
    )
    req = _make_req()
    req_data = _make_req_data(req, _payload(modalities=["text"]))

    msgs = builder("req-1", req_data, _make_req_output(5))

    assert len(msgs) == 1
    assert msgs[0].target == DECODE_STAGE
    assert bytes(msgs[0].data.tolist()).decode("utf-8") == "Hello"


def test_thinker_stream_builder_honors_non_text_modalities_for_decode():
    builder = make_thinker_stream_output_builder(
        tokenizer=_FakeTokenizer(),
        eos_token_id=None,
        target_stage=DECODE_STAGE,
        require_text_output_modality=True,
        require_request_stream=True,
    )
    req = _make_req()
    req_data = _make_req_data(req, _payload(modalities=["audio"]))

    assert builder("req-1", req_data, _make_req_output(5)) == []


def test_thinker_stream_builder_honors_non_streaming_requests_for_decode():
    builder = make_thinker_stream_output_builder(
        tokenizer=_FakeTokenizer(),
        eos_token_id=None,
        target_stage=DECODE_STAGE,
        require_text_output_modality=True,
        require_request_stream=True,
    )
    req = _make_req()
    req_data = _make_req_data(req, _payload(stream=False, modalities=["text"]))

    assert builder("req-1", req_data, _make_req_output(5)) == []


def test_thinker_stream_builder_suppresses_during_chunked_prefill():
    builder = make_thinker_stream_output_builder(
        tokenizer=_FakeTokenizer(),
        eos_token_id=None,
    )
    req = _make_req()
    req.is_chunked = 1  # still consuming prompt chunks
    req_data = _make_req_data(req)

    msgs = builder("req-2", req_data, _make_req_output(5))
    assert msgs == []


def test_thinker_stream_builder_buffers_incomplete_utf8():
    # Tokenizer that produces an incomplete UTF-8 sequence on first call.
    class _IncompleteThenComplete:
        calls = 0

        def decode(self, ids, skip_special_tokens=True):
            type(self).calls += 1
            return "Hello\ufffd" if type(self).calls == 1 else "Hello\u4e16"

    builder = make_thinker_stream_output_builder(
        tokenizer=_IncompleteThenComplete(),
        eos_token_id=None,
    )
    req = _make_req()
    req_data = _make_req_data(req)
    # First token: incomplete -> no emit.
    msgs1 = builder("req-3", req_data, _make_req_output(5))
    assert msgs1 == []
    # Second token: completes UTF-8 -> emit one segmenter message with full delta.
    msgs2 = builder("req-3", req_data, _make_req_output(6))
    assert len(msgs2) == 1
    assert msgs2[0].target == "segmenter"


def test_text_config_wires_thinker_stream_to_live_decode_stage():
    config = MingOmniPipelineConfig(model_path="dummy")
    stages = {stage.name: stage for stage in config.stages}

    thinker = stages[THINKER_STAGE]
    assert thinker.factory == (
        "sglang_omni.models.ming_omni.stages."
        "create_sglang_thinker_executor_from_config"
    )
    assert thinker.factory_args["stream_text_to_stage"] == DECODE_STAGE
    assert thinker.factory_args["require_stream_text_modality"] is True
    assert thinker.factory_args["require_stream_request"] is True
    assert thinker.stream_to == [DECODE_STAGE]

    decode = stages[DECODE_STAGE]
    assert (
        decode.factory
        == "sglang_omni.models.ming_omni.stages.create_decode_executor"
    )
    assert decode.terminal is True
    assert decode.can_accept_stream_before_payload is True


def test_decode_scheduler_forwards_stream_delta_and_drops_final_duplicate():
    def _decode(payload, *, strip_text):
        payload.data = {
            "text": "Hello world.",
            "modality": "text",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
        }
        if strip_text:
            payload.data.pop("text", None)
        return payload

    scheduler = MingTextDecodeScheduler(_decode)
    scheduler._handle_message(
        IncomingMessage(
            request_id="req-1",
            type="stream_chunk",
            data=StreamItem(
                chunk_id=0,
                data=text_to_uint8_tensor("Hello"),
                from_stage=THINKER_STAGE,
                metadata={"token_id": 5, "step": 1},
            ),
        )
    )

    stream = scheduler.outbox.get_nowait()
    assert stream.type == "stream"
    assert stream.target is None
    assert stream.metadata["modality"] == "text"
    assert stream.data["text"] == "Hello"
    assert stream.data["token_ids"] == [5]

    scheduler._handle_message(
        IncomingMessage(
            request_id="req-1",
            type="new_request",
            data=_payload(request_id="req-1", modalities=["text"]),
        )
    )

    result = scheduler.outbox.get_nowait()
    assert result.type == "result"
    assert "text" not in result.data.data
    assert result.data.data["finish_reason"] == "stop"
    assert result.data.data["usage"]["total_tokens"] == 5


def test_decode_scheduler_keeps_final_text_when_no_delta_was_streamed():
    def _decode(payload, *, strip_text):
        payload.data = {"text": "Hello world.", "modality": "text"}
        if strip_text:
            payload.data.pop("text", None)
        return payload

    scheduler = MingTextDecodeScheduler(_decode)
    scheduler._handle_message(
        IncomingMessage(
            request_id="req-1",
            type="new_request",
            data=_payload(request_id="req-1", stream=False),
        )
    )

    result = scheduler.outbox.get_nowait()
    assert result.type == "result"
    assert result.data.data["text"] == "Hello world."


def test_client_result_builder_merges_decode_with_talker_stream():
    audio_bytes = (0).to_bytes(4, "little") * 8  # 8 float32 zero samples
    merged = {
        "decode": {"text": "Hello world.", "modality": "text"},
        "talker_stream": {
            "modality": "audio",
            "audio_waveform": audio_bytes,
            "audio_waveform_dtype": "float32",
            "audio_waveform_shape": [8],
            "sample_rate": 44100,
        },
    }
    chunk: GenerateChunk = Client._default_result_builder("req-x", merged)
    assert chunk.text == "Hello world."
    assert chunk.modality == "audio"
    assert chunk.audio_data is not None
    assert int(chunk.audio_data.shape[0]) == 8
    assert chunk.sample_rate == 44100
