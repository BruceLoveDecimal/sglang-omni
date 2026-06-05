# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.moss_tts.codec import split_moss_audio_segments
from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.streaming_vocoder import (
    MossStreamingVocoderScheduler,
    _decode_stream_delta,
    _MossStreamState,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeMossProcessor:
    def __init__(self, *, sample_rate: int = 24, frame_length: int = 3) -> None:
        self.model_config = SimpleNamespace(
            sampling_rate=sample_rate,
            audio_pad_code=99,
            frame_length=frame_length,
        )
        self.calls: list[torch.Tensor] = []

    def decode_audio_codes(self, segments: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []
        for segment in segments:
            segment = segment.detach().cpu().to(torch.float32)
            self.calls.append(segment.to(torch.long).clone())
            weights = torch.arange(1, segment.shape[1] + 1, dtype=torch.float32)
            frame_values = (segment * weights).sum(dim=1)
            prev_values = torch.nn.functional.pad(frame_values[:-1], (1, 0))
            frame_values = frame_values + 0.25 * prev_values
            offsets = torch.arange(
                self.model_config.frame_length,
                dtype=torch.float32,
            )
            outputs.append((frame_values[:, None] + offsets[None, :]).reshape(-1))
        return outputs


def _delayed_from_frames(frames: torch.Tensor, *, pad_code: int = 99) -> torch.Tensor:
    n_vq = int(frames.shape[1])
    delayed = torch.full(
        (int(frames.shape[0]) + n_vq - 1, n_vq),
        int(pad_code),
        dtype=torch.long,
    )
    for channel in range(n_vq):
        delayed[channel : channel + int(frames.shape[0]), channel] = frames[:, channel]
    return delayed


def _audio_tensor(payload: dict) -> torch.Tensor:
    audio = np.frombuffer(payload["audio_waveform"], dtype=np.float32)
    return torch.from_numpy(audio.copy()).reshape(payload["audio_waveform_shape"])


def _full_decode(processor: _FakeMossProcessor, delayed: torch.Tensor) -> torch.Tensor:
    segments = split_moss_audio_segments(delayed, audio_pad_code=99)
    chunks = []
    for segment in segments:
        chunks.extend(processor.decode_audio_codes([segment]))
    return torch.cat([chunk.detach().cpu().reshape(-1) for chunk in chunks], dim=0)


def test_moss_streaming_vocoder_matches_full_decode_with_segments() -> None:
    processor = _FakeMossProcessor()
    frames = torch.tensor(
        [
            [1, 2],
            [3, 4],
            [99, 99],
            [5, 6],
            [7, 8],
            [9, 10],
        ],
        dtype=torch.long,
    )
    delayed = _delayed_from_frames(frames)
    state = _MossStreamState(n_vq=2, audio_pad_code=99, sample_rate=24)
    outputs = []

    for row in delayed:
        state.delayed_rows.append(row)
        outputs.extend(
            _decode_stream_delta(
                state,
                processor=processor,
                device=torch.device("cpu"),
                stream_stride=2,
                stream_followup_stride=2,
                stream_overlap_tokens=1,
                stream_holdback_tokens=0,
                samples_per_frame=3,
                is_final=False,
            )
        )
    outputs.extend(
        _decode_stream_delta(
            state,
            processor=processor,
            device=torch.device("cpu"),
            stream_stride=2,
            stream_followup_stride=2,
            stream_overlap_tokens=1,
            stream_holdback_tokens=0,
            samples_per_frame=3,
            is_final=True,
        )
    )

    streaming_audio = torch.cat([_audio_tensor(output) for output in outputs])
    full_processor = _FakeMossProcessor()
    full_audio = _full_decode(full_processor, delayed)

    torch.testing.assert_close(streaming_audio, full_audio)


def test_moss_streaming_vocoder_does_not_full_decode_terminal_payload() -> None:
    processor = _FakeMossProcessor()
    scheduler = MossStreamingVocoderScheduler(
        processor,
        device="cpu",
        stream_stride=1,
        stream_followup_stride=1,
        stream_overlap_tokens=0,
        stream_holdback_tokens=0,
        max_batch_wait_ms=0,
    )
    terminal_delayed = torch.full((8, 2), 42, dtype=torch.long)
    state = MossTTSState(
        text="hello",
        delayed_audio_codes=terminal_delayed,
        prompt_tokens=3,
        completion_tokens=4,
    )
    payload = StagePayload(
        request_id="req",
        request=OmniRequest(inputs="hello", params={"stream": True}),
        data=state.to_dict(),
    )
    metadata = {
        "modality": "moss_delayed_audio_row",
        "stream": True,
        "n_vq": 2,
        "audio_pad_code": 99,
        "sample_rate": 24,
    }

    scheduler._on_streaming_new_request("req", payload)
    scheduler._on_chunk(
        "req",
        StreamItem(
            chunk_id=0,
            data=torch.tensor([1, 2], dtype=torch.long),
            from_stage="tts_engine",
            metadata=metadata,
        ),
    )
    scheduler._on_chunk(
        "req",
        StreamItem(
            chunk_id=1,
            data=torch.tensor([3, 4], dtype=torch.long),
            from_stage="tts_engine",
            metadata=metadata,
        ),
    )
    scheduler._on_done("req")

    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())

    assert messages[-1].type == "result"
    assert messages[-1].data.data["usage"]["completion_tokens"] == 4
    assert all(not bool((call == 42).all()) for call in processor.calls)


def test_moss_streaming_vocoder_preserves_chunks_when_done_precedes_payload() -> None:
    scheduler = MossStreamingVocoderScheduler(
        _FakeMossProcessor(),
        device="cpu",
        stream_stride=8,
        stream_followup_stride=8,
        stream_overlap_tokens=0,
        stream_holdback_tokens=0,
        max_batch_wait_ms=0,
    )

    scheduler._on_chunk("req", _row_item(0))
    scheduler._on_chunk("req", _row_item(1))
    scheduler._on_done("req")

    assert _drain_outbox(scheduler) == []
    assert "req" in scheduler._stream_states
    assert len(scheduler._stream_states["req"].delayed_rows) == 2

    scheduler._on_streaming_new_request("req", _streaming_payload("req"))
    messages = _drain_outbox(scheduler)

    assert [msg.type for msg in messages] == ["stream", "result"]
    audio = _audio_tensor(messages[0].data)
    assert audio.numel() > 0
    assert messages[-1].data.data["sample_rate"] == 24000


def _streaming_payload(request_id: str) -> StagePayload:
    state = MossTTSState(text="hi", delayed_audio_codes=None)
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hi", params={"stream": True}),
        data=state.to_dict(),
    )


def _row_item(chunk_id: int) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=torch.tensor([1, 2], dtype=torch.long),
        from_stage="tts_engine",
        metadata={
            "modality": "moss_delayed_audio_row",
            "stream": True,
            "n_vq": 2,
            "audio_pad_code": 99,
            "sample_rate": 24,
        },
    )


def _new_scheduler() -> MossStreamingVocoderScheduler:
    return MossStreamingVocoderScheduler(
        _FakeMossProcessor(),
        device="cpu",
        stream_stride=1,
        stream_followup_stride=1,
        stream_overlap_tokens=0,
        stream_holdback_tokens=0,
        max_batch_wait_ms=0,
    )


def test_is_streaming_payload_treats_malformed_params_as_non_streaming() -> None:
    scheduler = _new_scheduler()
    payload = StagePayload(
        request_id="req",
        request=OmniRequest(inputs="hi", params=["stream"]),  # type: ignore[arg-type]
        data={},
    )

    assert scheduler.is_streaming_payload(payload) is False


def test_abort_frees_streaming_state() -> None:
    scheduler = _new_scheduler()
    scheduler._on_streaming_new_request("req", _streaming_payload("req"))
    scheduler._on_chunk("req", _row_item(0))
    scheduler._on_chunk("req", _row_item(1))

    assert "req" in scheduler._stream_states
    assert "req" in scheduler._stream_payloads

    scheduler.abort("req")

    # Per-request streaming state and the latched payload are both released.
    assert "req" not in scheduler._stream_states
    assert "req" not in scheduler._stream_payloads


def _drain_outbox(scheduler: MossStreamingVocoderScheduler) -> list:
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    return messages


def test_abort_mid_stream_suppresses_outbox_and_allows_reuse() -> None:
    scheduler = _new_scheduler()
    scheduler._on_streaming_new_request("req", _streaming_payload("req"))
    scheduler._on_chunk("req", _row_item(0))
    _drain_outbox(scheduler)

    scheduler.abort("req")

    # A late chunk after abort must neither leak to the outbox nor resurrect state.
    scheduler._on_chunk("req", _row_item(1))
    assert _drain_outbox(scheduler) == []
    assert "req" not in scheduler._stream_states
    assert "req" not in scheduler._stream_payloads

    # The same id can start a fresh stream and finish normally afterwards.
    scheduler._on_streaming_new_request("req", _streaming_payload("req"))
    scheduler._on_chunk("req", _row_item(0))
    scheduler._on_done("req")
    messages = _drain_outbox(scheduler)
    assert messages, "a reused request id must stream again after abort"
    assert messages[-1].type == "result"


def test_stream_metadata_n_vq_change_raises() -> None:
    scheduler = _new_scheduler()
    scheduler._on_streaming_new_request("req", _streaming_payload("req"))
    scheduler._on_chunk("req", _row_item(0))

    bad = StreamItem(
        chunk_id=1,
        data=torch.tensor([1, 2, 3], dtype=torch.long),
        from_stage="tts_engine",
        metadata={
            "modality": "moss_delayed_audio_row",
            "stream": True,
            "n_vq": 3,
            "audio_pad_code": 99,
            "sample_rate": 24,
        },
    )
    with pytest.raises(ValueError, match="n_vq changed"):
        scheduler.on_stream_chunk("req", bad)


def test_stream_metadata_audio_pad_code_change_raises() -> None:
    scheduler = _new_scheduler()
    scheduler._on_streaming_new_request("req", _streaming_payload("req"))
    scheduler._on_chunk("req", _row_item(0))

    bad = StreamItem(
        chunk_id=1,
        data=torch.tensor([1, 2], dtype=torch.long),
        from_stage="tts_engine",
        metadata={
            "modality": "moss_delayed_audio_row",
            "stream": True,
            "n_vq": 2,
            "audio_pad_code": 7,
            "sample_rate": 24,
        },
    )
    with pytest.raises(ValueError, match="audio_pad_code changed"):
        scheduler.on_stream_chunk("req", bad)


def test_stream_chunk_without_metadata_before_latch_raises() -> None:
    scheduler = _new_scheduler()
    scheduler._on_streaming_new_request("req", _streaming_payload("req"))

    item = StreamItem(
        chunk_id=0,
        data=torch.tensor([1, 2], dtype=torch.long),
        from_stage="tts_engine",
        metadata=None,
    )
    with pytest.raises(RuntimeError, match="missing metadata"):
        scheduler.on_stream_chunk("req", item)


def test_stream_continuity_without_samples_per_frame() -> None:
    # When the processor exposes no frame size, the first decoded window must
    # latch one samples-per-frame so the stream still reconstructs the full
    # decode without gaps or overlaps.
    processor = _FakeMossProcessor()
    frames = torch.tensor(
        [[1, 2], [3, 4], [99, 99], [5, 6], [7, 8], [9, 10]],
        dtype=torch.long,
    )
    delayed = _delayed_from_frames(frames)
    state = _MossStreamState(n_vq=2, audio_pad_code=99, sample_rate=24)
    outputs = []
    for row in delayed:
        state.delayed_rows.append(row)
        outputs.extend(
            _decode_stream_delta(
                state,
                processor=processor,
                device=torch.device("cpu"),
                stream_stride=2,
                stream_followup_stride=2,
                stream_overlap_tokens=1,
                stream_holdback_tokens=0,
                samples_per_frame=None,
                is_final=False,
            )
        )
    outputs.extend(
        _decode_stream_delta(
            state,
            processor=processor,
            device=torch.device("cpu"),
            stream_stride=2,
            stream_followup_stride=2,
            stream_overlap_tokens=1,
            stream_holdback_tokens=0,
            samples_per_frame=None,
            is_final=True,
        )
    )

    streaming_audio = torch.cat([_audio_tensor(output) for output in outputs])
    full_audio = _full_decode(_FakeMossProcessor(), delayed)
    assert state.samples_per_frame is not None
    torch.testing.assert_close(streaming_audio, full_audio)
