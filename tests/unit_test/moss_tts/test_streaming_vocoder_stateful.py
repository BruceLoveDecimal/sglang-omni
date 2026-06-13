# SPDX-License-Identifier: Apache-2.0
"""Stateful (KV-cache) streaming vocoder path tests.

These inject a fake StatefulMossCodec so they run without the real checkpoint;
the numerical parity of the *real* codec is covered by
``test_codec_streaming_parity.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sglang_omni.models.moss_tts.payload_types import MossTTSState  # noqa: E402
from sglang_omni.models.moss_tts.streaming_vocoder import (  # noqa: E402
    MossStreamingVocoderScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem  # noqa: E402
from sglang_omni.proto import OmniRequest, StagePayload  # noqa: E402


class _FakeMossProcessor:
    def __init__(self) -> None:
        self.model_config = SimpleNamespace(sampling_rate=24000, audio_pad_code=99)


class _FakeStatefulCodec:
    """Records decode calls; 2 samples/frame, value = sum of frame codes."""

    def __init__(self) -> None:
        self.decode_frames: list[torch.Tensor] = []
        self.states_created = 0
        self.states_used: list[int] = []

    def new_state(self):
        self.states_created += 1
        return {"id": self.states_created}

    def decode_stream(self, frames, state, *, last_chunk=False):
        self.decode_frames.append(frames.clone())
        self.states_used.append(state["id"])
        vals = frames.sum(dim=1).to(torch.float32)
        return vals.repeat_interleave(2)


def _delayed_from_frames(frames: torch.Tensor, *, pad_code: int = 99) -> torch.Tensor:
    n_vq = int(frames.shape[1])
    delayed = torch.full(
        (int(frames.shape[0]) + n_vq - 1, n_vq), int(pad_code), dtype=torch.long
    )
    for c in range(n_vq):
        delayed[c : c + int(frames.shape[0]), c] = frames[:, c]
    return delayed


def _row_item(chunk_id: int, row: torch.Tensor) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=row,
        from_stage="tts_engine",
        metadata={
            "modality": "moss_delayed_audio_row",
            "stream": True,
            "n_vq": 2,
            "audio_pad_code": 99,
            "sample_rate": 24000,
        },
    )


def _streaming_payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hi", params={"stream": True}),
        data=MossTTSState(text="hi", delayed_audio_codes=None).to_dict(),
    )


def _audio(payload: dict) -> torch.Tensor:
    a = np.frombuffer(payload["audio_waveform"], dtype=np.float32)
    return torch.from_numpy(a.copy()).reshape(payload["audio_waveform_shape"])


def _stateful_scheduler() -> tuple[MossStreamingVocoderScheduler, _FakeStatefulCodec]:
    sched = MossStreamingVocoderScheduler(
        _FakeMossProcessor(),
        device="cpu",
        stream_codec_mode="stateful",
        max_batch_wait_ms=0,
    )
    codec = _FakeStatefulCodec()
    sched._stateful_codec = codec  # bypass real-checkpoint build
    return sched, codec


def _drain(sched) -> list:
    out = []
    while not sched.outbox.empty():
        out.append(sched.outbox.get_nowait())
    return out


def test_stateful_decodes_frames_and_resets_per_segment() -> None:
    sched, codec = _stateful_scheduler()
    # Two audio segments split by a pad-only separator frame.
    frames = torch.tensor(
        [[1, 2], [3, 4], [99, 99], [5, 6], [7, 8]], dtype=torch.long
    )
    delayed = _delayed_from_frames(frames)

    sched._on_streaming_new_request("req", _streaming_payload("req"))
    for i, row in enumerate(delayed):
        sched._on_chunk("req", _row_item(i, row))
    sched._on_done("req")

    messages = _drain(sched)
    assert messages[-1].type == "result"

    decoded = torch.cat([f for f in codec.decode_frames], dim=0)
    # All five real frames reach the codec; the pad separator does not.
    audio_frames = frames[(frames != 99).any(dim=1)]
    assert decoded.shape[0] == audio_frames.shape[0]

    # Two segments → two independent codec states.
    assert codec.states_created == 2
    assert set(codec.states_used) == {1, 2}


def test_stateful_emits_audio_before_terminal() -> None:
    sched, _ = _stateful_scheduler()
    frames = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long)
    delayed = _delayed_from_frames(frames)

    sched._on_streaming_new_request("req", _streaming_payload("req"))
    for i, row in enumerate(delayed):
        sched._on_chunk("req", _row_item(i, row))
    msgs_before_done = _drain(sched)
    sched._on_done("req")
    msgs_after = _drain(sched)

    stream_msgs = [m for m in (msgs_before_done + msgs_after) if m.type == "stream"]
    assert stream_msgs, "expected at least one audio chunk"
    assert _audio(stream_msgs[0].data).numel() > 0
    assert (msgs_after[-1].type) == "result"


def test_stateful_abort_clears_state() -> None:
    sched, _ = _stateful_scheduler()
    sched._on_streaming_new_request("req", _streaming_payload("req"))
    sched._on_chunk("req", _row_item(0, torch.tensor([1, 2], dtype=torch.long)))
    assert "req" in sched._stream_states
    sched.abort("req")
    assert "req" not in sched._stream_states
    assert "req" not in sched._stream_payloads
