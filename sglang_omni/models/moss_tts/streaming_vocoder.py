# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for MOSS-TTS Delay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.moss_tts.codec import split_moss_audio_segments
from sglang_omni.models.moss_tts.payload_types import (
    MossTTSState,
    resolve_moss_audio_pad_code,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload


@dataclass
class _MossSegmentDecodeState:
    emitted_frames: int = 0


@dataclass
class _MossStreamState:
    delayed_rows: list[torch.Tensor] = field(default_factory=list)
    segment_states: list[_MossSegmentDecodeState] = field(default_factory=list)
    next_decode_rows: int = 0
    has_emitted: bool = False
    n_vq: int | None = None
    audio_pad_code: int | None = None
    sample_rate: int = 24000
    # Latched once from the first decoded window so every chunk trims on the
    # same samples-per-frame and the stream stays gap-free.
    samples_per_frame: int | None = None


def _as_audio_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value).detach().reshape(-1).to(torch.float32).cpu()


def _resolve_sample_rate(processor: Any, fallback: int = 24000) -> int:
    for obj in (
        getattr(processor, "model_config", None),
        getattr(getattr(processor, "audio_tokenizer", None), "config", None),
        getattr(processor, "config", None),
    ):
        value = getattr(obj, "sampling_rate", None)
        if value:
            return int(value)
    return int(fallback or 24000)


def _resolve_samples_per_frame(processor: Any, sample_rate: int) -> int | None:
    for obj in (
        getattr(processor, "model_config", None),
        getattr(getattr(processor, "audio_tokenizer", None), "config", None),
        getattr(processor, "config", None),
    ):
        for attr in (
            "downsample_rate",
            "samples_per_frame",
            "frame_length",
            "hop_length",
        ):
            value = getattr(obj, attr, None)
            if value:
                value_i = int(value)
                if value_i > 0:
                    return value_i
        frame_rate = getattr(obj, "frame_rate", None)
        if frame_rate:
            frame_rate_f = float(frame_rate)
            if frame_rate_f > 0:
                return max(int(round(float(sample_rate) / frame_rate_f)), 1)
    return None


def _build_usage(state: MossTTSState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage: dict[str, Any] = {
        "prompt_tokens": int(state.prompt_tokens),
        "completion_tokens": int(state.completion_tokens),
        "total_tokens": int(state.prompt_tokens + state.completion_tokens),
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage


def _decode_moss_audio_segment(
    processor: Any,
    segment: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    segment = segment.to(device=device, dtype=torch.long)
    with torch.no_grad():
        decoded = processor.decode_audio_codes([segment])
    if not decoded:
        return None
    return _as_audio_tensor(decoded[0])


def _decode_stream_delta(
    state: _MossStreamState,
    *,
    processor: Any,
    device: torch.device,
    stream_stride: int,
    stream_followup_stride: int,
    stream_overlap_tokens: int,
    stream_holdback_tokens: int,
    samples_per_frame: int | None,
    is_final: bool,
) -> list[dict[str, Any]]:
    n_vq = state.n_vq
    audio_pad_code = state.audio_pad_code
    if n_vq is None or audio_pad_code is None:
        raise RuntimeError("MOSS stream metadata is missing n_vq or audio_pad_code")
    delayed_count = len(state.delayed_rows)
    if delayed_count < n_vq:
        return []

    next_decode_rows = state.next_decode_rows or max(n_vq, stream_stride)
    if not is_final and delayed_count < next_decode_rows:
        state.next_decode_rows = next_decode_rows
        return []

    raw_total = delayed_count - n_vq + 1
    emit_until_raw = raw_total
    if not is_final and stream_holdback_tokens > 0:
        emit_until_raw = max(0, raw_total - stream_holdback_tokens)
    if emit_until_raw <= 0:
        state.next_decode_rows = delayed_count + stream_followup_stride
        return []

    rows_end = emit_until_raw + n_vq - 1
    delayed_rows = torch.stack(state.delayed_rows[:rows_end], dim=0).to(torch.long)
    segments = split_moss_audio_segments(
        delayed_rows,
        audio_pad_code=int(audio_pad_code),
        assistant_start_length=0,
    )
    while len(state.segment_states) < len(segments):
        state.segment_states.append(_MossSegmentDecodeState())

    # A config-provided rate wins; otherwise reuse whatever the first window
    # already latched so the trim boundary never drifts between chunks.
    stable_spf = (
        samples_per_frame if samples_per_frame is not None else state.samples_per_frame
    )

    chunks: list[dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        segment_state = state.segment_states[idx]
        total_frames = int(segment.shape[0])
        emitted_frames = int(segment_state.emitted_frames)
        if total_frames <= emitted_frames:
            continue

        window_start = max(0, emitted_frames - int(stream_overlap_tokens))
        window = segment[window_start:total_frames].contiguous()
        audio = _decode_moss_audio_segment(processor, window, device=device)
        if audio is None or audio.numel() == 0:
            continue

        if stable_spf is None:
            decoded_frames = total_frames - window_start
            stable_spf = max(int(audio.shape[-1]) // max(decoded_frames, 1), 1)
            state.samples_per_frame = stable_spf
        spf = stable_spf
        trim_frames = emitted_frames - window_start
        trim_samples = min(int(trim_frames * spf), int(audio.shape[-1]))
        segment_is_closed = is_final or idx < len(segments) - 1
        if not segment_is_closed:
            new_frames = total_frames - emitted_frames
            emit_samples = int(new_frames * spf)
            delta = audio[trim_samples : trim_samples + emit_samples].contiguous()
        else:
            delta = audio[trim_samples:].contiguous()
        if delta.numel() == 0:
            continue

        segment_state.emitted_frames = total_frames
        state.has_emitted = True
        chunks.append(
            audio_waveform_payload(
                delta,
                sample_rate=state.sample_rate,
                modality="audio",
                source_hint="MOSS-TTS streaming",
            )
        )

    state.next_decode_rows = delayed_count + stream_followup_stride
    return chunks


class MossStreamingVocoderScheduler(StreamingSimpleScheduler):
    """Decode MOSS delayed audio rows incrementally.

    Streaming uses the repository-known ``processor.decode_audio_codes`` API
    with retained overlap windows. Non-streaming requests continue to use the
    full batched vocoder path.
    """

    def __init__(
        self,
        processor: Any,
        *,
        device: str | torch.device = "cuda:0",
        stream_stride: int = 8,
        stream_followup_stride: int = 8,
        stream_overlap_tokens: int = 2,
        stream_holdback_tokens: int = 1,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
    ) -> None:
        if stream_stride <= 0 or stream_followup_stride <= 0:
            raise ValueError("stream_stride and stream_followup_stride must be > 0")
        if stream_overlap_tokens < 0 or stream_holdback_tokens < 0:
            raise ValueError(
                "stream_overlap_tokens and stream_holdback_tokens must be >= 0"
            )
        self._processor = processor
        self._device = torch.device(device)
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        self._stream_overlap_tokens = int(stream_overlap_tokens)
        self._stream_holdback_tokens = int(stream_holdback_tokens)
        self._sample_rate = _resolve_sample_rate(processor)
        self._samples_per_frame = _resolve_samples_per_frame(
            processor, self._sample_rate
        )
        self._stream_states: dict[str, _MossStreamState] = {}

        super().__init__(
            self._vocode_payload,
            batch_compute_fn=self._vocode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        params = payload.request.params
        return isinstance(params, dict) and bool(params.get("stream", False))

    def validate_non_streaming_payload(self, payload: StagePayload) -> None:
        self._prepare_vocoder_item(payload)

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        # Always start from a clean state so a reused id never inherits rows left
        # behind by an aborted attempt.
        stream_state = _MossStreamState()
        stream_state.sample_rate = self._resolve_payload_sample_rate(payload)
        self._stream_states[request_id] = stream_state

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        if self._is_aborted(request_id):
            # A late chunk for an aborted request must not resurrect state.
            return []
        stream_state = self._stream_states.setdefault(request_id, _MossStreamState())
        self._latch_stream_metadata(request_id, stream_state, item.metadata)
        row = item.data
        if not isinstance(row, torch.Tensor):
            row = torch.as_tensor(row, dtype=torch.long)
        row = row.to(dtype=torch.long)
        if row.ndim != 1:
            raise ValueError(
                f"MOSS-TTS stream row for {request_id!r} must be 1-D, "
                f"got {tuple(row.shape)}"
            )
        n_vq = stream_state.n_vq
        if n_vq is None:
            raise RuntimeError(f"MOSS-TTS stream metadata missing n_vq for {request_id}")
        if int(row.shape[0]) != int(n_vq):
            raise ValueError(
                f"MOSS-TTS stream row for {request_id!r} has {int(row.shape[0])} "
                f"codebooks, expected {int(n_vq)}"
            )
        stream_state.delayed_rows.append(row.detach().cpu())
        chunks = self._decode_stream_state(stream_state, is_final=False)
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=chunk,
                metadata={"modality": "audio"},
            )
            for chunk in chunks
        ]

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        stream_state = self._stream_states.setdefault(request_id, _MossStreamState())
        chunks = self._decode_stream_state(stream_state, is_final=True)
        messages = [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=chunk,
                metadata={"modality": "audio"},
            )
            for chunk in chunks
        ]
        payload = self._stream_payloads[request_id]
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=self._store_streaming_result(payload, stream_state),
            )
        )
        return messages

    def clear_stream_state(self, request_id: str) -> None:
        self._stream_states.pop(request_id, None)

    def _decode_stream_state(
        self, state: _MossStreamState, *, is_final: bool
    ) -> list[dict[str, Any]]:
        return _decode_stream_delta(
            state,
            processor=self._processor,
            device=self._device,
            stream_stride=self._stream_stride,
            stream_followup_stride=self._stream_followup_stride,
            stream_overlap_tokens=self._stream_overlap_tokens,
            stream_holdback_tokens=self._stream_holdback_tokens,
            samples_per_frame=self._samples_per_frame,
            is_final=is_final,
        )

    def _latch_stream_metadata(
        self,
        request_id: str,
        state: _MossStreamState,
        metadata: dict[str, Any] | None,
    ) -> None:
        if not isinstance(metadata, dict):
            if state.n_vq is None or state.audio_pad_code is None:
                raise RuntimeError(
                    f"MOSS-TTS stream chunk for {request_id!r} is missing metadata"
                )
            return
        if metadata.get("modality") not in (None, "moss_delayed_audio_row"):
            raise ValueError(
                f"MOSS-TTS stream chunk modality must be moss_delayed_audio_row, "
                f"got {metadata.get('modality')!r}"
            )
        if metadata.get("stream") is not True:
            raise RuntimeError(
                f"MOSS-TTS stream chunk for {request_id!r} must include "
                "metadata['stream'] == True"
            )
        if "n_vq" in metadata:
            n_vq = int(metadata["n_vq"])
            if n_vq <= 0:
                raise ValueError(f"MOSS-TTS n_vq must be > 0, got {n_vq}")
            if state.n_vq is not None and int(state.n_vq) != n_vq:
                raise ValueError(
                    f"MOSS-TTS stream n_vq changed for {request_id!r}: "
                    f"{state.n_vq} -> {n_vq}"
                )
            state.n_vq = n_vq
        if "audio_pad_code" in metadata:
            audio_pad_code = int(metadata["audio_pad_code"])
            if (
                state.audio_pad_code is not None
                and state.audio_pad_code != audio_pad_code
            ):
                raise ValueError(
                    f"MOSS-TTS stream audio_pad_code changed for {request_id!r}: "
                    f"{state.audio_pad_code} -> {audio_pad_code}"
                )
            state.audio_pad_code = audio_pad_code
        if "sample_rate" in metadata:
            state.sample_rate = int(metadata["sample_rate"] or state.sample_rate)
        if state.n_vq is None or state.audio_pad_code is None:
            raise RuntimeError(
                f"MOSS-TTS stream metadata for {request_id!r} must include "
                "n_vq and audio_pad_code"
            )

    def _resolve_payload_sample_rate(self, payload: StagePayload) -> int:
        state = MossTTSState.from_dict(payload.data)
        return int(state.sample_rate or self._sample_rate)

    def _prepare_vocoder_item(
        self,
        payload: StagePayload,
    ) -> tuple[MossTTSState, torch.Tensor]:
        state = MossTTSState.from_dict(payload.data)
        if state.delayed_audio_codes is None:
            raise RuntimeError("MOSS-TTS vocoder requires delayed_audio_codes")
        delayed_codes = torch.as_tensor(state.delayed_audio_codes, dtype=torch.long)
        if delayed_codes.numel() == 0:
            raise RuntimeError("MOSS-TTS generated no delayed audio codes")
        return state, delayed_codes

    def _decode_audio(
        self,
        state: MossTTSState,
        delayed_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        delayed_codes = delayed_codes.to(device=self._device, dtype=torch.long)
        audio_pad_code = resolve_moss_audio_pad_code(
            getattr(self._processor, "model_config", None)
        )
        segments = split_moss_audio_segments(
            delayed_codes,
            audio_pad_code=audio_pad_code,
            assistant_start_length=int(state.assistant_start_length),
        )
        decoded = []
        for segment in segments:
            decoded.extend(self._processor.decode_audio_codes([segment]))
        if not decoded:
            raise RuntimeError("MOSS-TTS vocoder decoded no audio segments")
        waveforms = [_as_audio_tensor(wav) for wav in decoded]
        return torch.cat(waveforms, dim=0), self._resolve_payload_or_processor_rate(state)

    def _resolve_payload_or_processor_rate(self, state: MossTTSState) -> int:
        return int(self._sample_rate or state.sample_rate or 24000)

    def _store_vocoder_result(
        self,
        payload: StagePayload,
        state: MossTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        audio_payload = audio_waveform_payload(
            wav,
            sample_rate=sample_rate,
            modality="audio",
            source_hint="MOSS-TTS",
        )
        state.delayed_audio_codes = None
        state.sample_rate = int(sample_rate)
        data = state.to_dict()
        data.update(audio_payload)
        usage = _build_usage(state)
        if usage is not None:
            data["usage"] = usage
        payload.data = data
        return payload

    def _store_streaming_result(
        self,
        payload: StagePayload,
        stream_state: _MossStreamState,
    ) -> StagePayload:
        # The audio (including the final tail) already left as stream chunks, so
        # the terminal result carries no waveform — only usage/metadata. This
        # keeps a long stream bounded in host memory and matches the terminal
        # SSE event, whose ``audio`` is always null.
        state = MossTTSState.from_dict(payload.data)
        state.delayed_audio_codes = None
        state.sample_rate = int(stream_state.sample_rate or state.sample_rate)
        data = state.to_dict()
        data["modality"] = "audio"
        data["sample_rate"] = state.sample_rate
        usage = _build_usage(state)
        if usage is not None:
            data["usage"] = usage
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=data,
        )

    def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        return self._vocode_payloads([payload])[0]

    def _vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        results: list[StagePayload] = []
        for payload in payloads:
            state, delayed_codes = self._prepare_vocoder_item(payload)
            wav, sample_rate = self._decode_audio(state, delayed_codes)
            results.append(self._store_vocoder_result(payload, state, wav, sample_rate))
        return results


__all__ = [
    "MossStreamingVocoderScheduler",
    "_MossStreamState",
    "_decode_stream_delta",
]
