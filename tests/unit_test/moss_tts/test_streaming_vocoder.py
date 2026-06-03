# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from sglang_omni.models.moss_tts.codec import split_moss_audio_segments
from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.streaming_vocoder import (
    MossStreamingVocoderScheduler,
    _MossStreamState,
    _decode_stream_delta,
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


def test_stream_done_after_abort_emits_no_result() -> None:
    scheduler = _new_scheduler()
    scheduler._on_streaming_new_request("req", _streaming_payload("req"))
    scheduler._on_chunk("req", _row_item(0))
    scheduler.abort("req")

    # Drain anything emitted before the abort.
    while not scheduler.outbox.empty():
        scheduler.outbox.get_nowait()

    # A late stream_done for an aborted request must not produce a result.
    scheduler._on_done("req")
    assert scheduler.outbox.empty()


def test_slot_reused_by_later_request_after_abort() -> None:
    scheduler = _new_scheduler()
    scheduler._on_streaming_new_request("req", _streaming_payload("req"))
    scheduler._on_chunk("req", _row_item(0))
    scheduler.abort("req")

    # The same id can be reused for a fresh request and complete normally.
    scheduler._on_streaming_new_request("req", _streaming_payload("req"))
    for chunk_id in range(4):
        scheduler._on_chunk("req", _row_item(chunk_id))
    scheduler._on_done("req")

    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())

    assert messages, "reused request emitted nothing"
    assert messages[-1].type == "result"
    # State cleared again after the successful done.
    assert "req" not in scheduler._stream_states
