# SPDX-License-Identifier: Apache-2.0
"""Native cache-aware PCM streaming scheduler for Nemotron 3.5 ASR."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler

from .model_runner import (
    Nemotron3_5ASRDecodeState,
    Nemotron3_5ASRModelRunner,
    Nemotron3_5ASRStreamingBatchResult,
)
from .request_builders import (
    build_nemotron3_5_asr_result,
    normalize_nemotron_language,
    validate_nemotron_greedy_params,
)

PCM16_BYTES_PER_SAMPLE = 2
PCM16_AMPLITUDE_SCALE = 32768.0


@dataclass(frozen=True, slots=True)
class Nemotron3_5ASRStreamingChunkSpec:
    sample_rate: int
    first_samples: int
    subsequent_samples: int
    first_frames: int
    subsequent_frames: int
    hop_length: int
    n_fft: int
    streaming_latency_ms: int


@dataclass(frozen=True, slots=True)
class Nemotron3_5ASRAudioWindow:
    waveform: np.ndarray
    model_chunk_index: int
    is_first: bool
    ready_wait_s: float


@dataclass(slots=True)
class Nemotron3_5ASRStreamMetrics:
    request_started_s: float = field(default_factory=time.perf_counter)
    input_done_s: float | None = None
    first_text_s: float | None = None
    finalized_s: float | None = None
    model_compute_s: float = 0.0
    packet_count: int = 0
    model_chunk_count: int = 0
    cache_reuse_count: int = 0
    max_queue_depth: int = 0
    batch_sizes: list[int] = field(default_factory=list)
    chunk_latency_ms: list[float] = field(default_factory=list)
    chunk_ready_wait_ms: list[float] = field(default_factory=list)


@dataclass(slots=True)
class Nemotron3_5ASRStreamState:
    request_id: str
    payload: StagePayload
    language: str
    spec: Nemotron3_5ASRStreamingChunkSpec
    decode: Nemotron3_5ASRDecodeState
    max_new_tokens: int | None = None
    pcm_bytes: bytearray = field(default_factory=bytearray)
    total_samples: int = 0
    covered_audio_end: int = 0
    model_chunk_index: int = 0
    next_mel_frame: int = 0
    is_input_done: bool = False
    ready_since_s: float | None = None
    raw_text: str = ""
    clean_text: str = ""
    detected_language: str | None = None
    metrics: Nemotron3_5ASRStreamMetrics = field(
        default_factory=Nemotron3_5ASRStreamMetrics
    )

    def append_pcm16(
        self, tensor: torch.Tensor, metadata: Mapping[str, object]
    ) -> None:
        if tensor.device.type != "cpu":
            raise ValueError("Nemotron streaming chunks must be CPU PCM16 tensors")
        if tensor.dtype not in {torch.int16, torch.uint8}:
            raise TypeError(
                "Nemotron streaming chunks must use PCM16 samples (torch.int16) "
                f"or raw little-endian bytes (torch.uint8), got {tensor.dtype}"
            )
        if tensor.ndim not in {1, 2}:
            raise ValueError(
                "Nemotron streaming PCM16 tensors must be one-dimensional or mono"
            )
        if tensor.ndim == 2 and 1 not in tensor.shape:
            raise ValueError("Nemotron streaming accepts mono PCM16 only")
        sample_rate = metadata.get("sample_rate", self.spec.sample_rate)
        if isinstance(sample_rate, bool) or sample_rate != self.spec.sample_rate:
            raise ValueError(
                f"Nemotron streaming requires sample_rate={self.spec.sample_rate}"
            )
        modality = metadata.get("modality")
        if modality not in {None, "audio", "pcm16"}:
            raise ValueError(
                f"Nemotron streaming chunk modality must be audio or pcm16, got {modality!r}"
            )
        if self.is_input_done:
            raise RuntimeError(f"Nemotron stream {self.request_id!r} is already done")

        pcm_samples = tensor.detach().contiguous().reshape(-1)
        if pcm_samples.numel() == 0:
            raise ValueError("Nemotron streaming PCM16 chunks must not be empty")
        if pcm_samples.dtype == torch.int16:
            packet_bytes = pcm_samples.numpy().astype("<i2", copy=False).tobytes()
        else:
            packet_bytes = pcm_samples.numpy().tobytes()
        self.pcm_bytes.extend(packet_bytes)
        self.total_samples = len(self.pcm_bytes) // PCM16_BYTES_PER_SAMPLE
        now = time.perf_counter()
        self.metrics.packet_count += 1
        self.mark_ready(now)

    @property
    def has_reached_decode_limit(self) -> bool:
        return (
            self.max_new_tokens is not None
            and self.decode.decoder_steps >= self.max_new_tokens
        )

    def mark_done(self) -> None:
        if self.is_input_done:
            raise RuntimeError(f"Nemotron stream {self.request_id!r} is already done")
        if self.total_samples == 0:
            raise ValueError("Nemotron streaming input contains no PCM16 samples")
        if len(self.pcm_bytes) % PCM16_BYTES_PER_SAMPLE:
            raise ValueError(
                "Nemotron streaming input ends with an incomplete PCM16 sample"
            )
        self.is_input_done = True
        self.metrics.input_done_s = time.perf_counter()
        self.mark_ready(self.metrics.input_done_s)

    def next_window_bounds(self) -> tuple[int, int]:
        if self.model_chunk_index == 0:
            return 0, self.spec.first_samples
        start = self.next_mel_frame * self.spec.hop_length - self.spec.n_fft // 2
        return start, start + self.spec.subsequent_samples

    def has_ready_window(self, *, finalizing: bool = False) -> bool:
        if self.model_chunk_index == 0:
            return self.total_samples >= self.spec.first_samples or (
                finalizing and self.total_samples > 0
            )
        _, end = self.next_window_bounds()
        if self.total_samples >= end:
            return True
        return finalizing and self.total_samples > self.covered_audio_end

    def mark_ready(self, now: float) -> None:
        if self.ready_since_s is not None or not self.has_ready_window(
            finalizing=self.is_input_done
        ):
            return
        self.ready_since_s = now

    def pop_ready_window(
        self, *, finalizing: bool = False
    ) -> Nemotron3_5ASRAudioWindow:
        assert self.has_ready_window(
            finalizing=finalizing
        ), f"Nemotron stream {self.request_id!r} has no ready window"
        now = time.perf_counter()
        start, end = self.next_window_bounds()
        window_samples = end - start
        source_start = max(start, 0)
        source_end = min(end, self.total_samples)
        complete_bytes = memoryview(self.pcm_bytes)[
            : self.total_samples * PCM16_BYTES_PER_SAMPLE
        ]
        pcm_samples = np.frombuffer(complete_bytes, dtype="<i2")
        window_pcm = pcm_samples[source_start : max(source_start, source_end)]
        left_padding = max(-start, 0)
        right_padding = window_samples - left_padding - int(window_pcm.shape[0])
        waveform = np.pad(
            window_pcm.astype(np.float32) / PCM16_AMPLITUDE_SCALE,
            (left_padding, right_padding),
        ).astype(np.float32, copy=False)
        is_first = self.model_chunk_index == 0
        ready_wait_s = (
            max(now - self.ready_since_s, 0.0)
            if self.ready_since_s is not None
            else 0.0
        )
        window = Nemotron3_5ASRAudioWindow(
            waveform=waveform,
            model_chunk_index=self.model_chunk_index,
            is_first=is_first,
            ready_wait_s=ready_wait_s,
        )

        self.covered_audio_end = max(
            self.covered_audio_end, min(end, self.total_samples)
        )
        self.model_chunk_index += 1
        if is_first:
            self.next_mel_frame = self.spec.first_frames
        else:
            self.next_mel_frame += self.spec.subsequent_frames
        self.ready_since_s = None
        self.mark_ready(now)
        return window


class Nemotron3_5ASRStreamingScheduler(StreamingSimpleScheduler):
    """Serialize request-owned RNNT state with abort cleanup through state_lock."""

    supports_external_input_stream = True
    can_batch_stream_chunks = True
    stream_chunk_batch_distinct_requests = True

    def __init__(
        self,
        runner: Nemotron3_5ASRModelRunner,
        compute_fn: Callable[[StagePayload], StagePayload],
        *,
        batch_compute_fn: Callable[
            [Sequence[StagePayload]], list[StagePayload | BaseException]
        ],
        prompt_dictionary: Mapping[str, int],
        max_batch_size: int,
        max_batch_wait_ms: float,
        max_pending_messages: int,
    ) -> None:
        self.runner = runner
        self.chunk_spec = Nemotron3_5ASRStreamingChunkSpec(
            **runner.streaming_chunk_spec
        )
        self.prompt_dictionary = dict(prompt_dictionary)
        self.stream_states: dict[str, Nemotron3_5ASRStreamState] = {}
        self.is_closed = False
        self.aggregate_metrics: dict[str, int | float] = {
            "input_packets": 0,
            "model_chunks": 0,
            "model_batches": 0,
            "cache_reuses": 0,
            "audio_samples": 0,
            "model_compute_s": 0.0,
            "max_batch_size": 0,
            "completed_streams": 0,
            "aborted_streams": 0,
        }
        self.stream_chunk_batch_max = max_batch_size
        super().__init__(
            compute_fn,
            batch_compute_fn=batch_compute_fn,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            max_pending_messages=max_pending_messages,
        )

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return payload.external_input_stream

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        if request_id in self.stream_states:
            raise ValueError(f"Nemotron stream {request_id!r} already exists")
        params = payload.request.params or {}
        max_new_tokens = validate_nemotron_greedy_params(params)
        language = normalize_nemotron_language(
            params.get("language"), self.prompt_dictionary
        )
        self.stream_states[request_id] = Nemotron3_5ASRStreamState(
            request_id=request_id,
            payload=payload,
            language=language,
            spec=self.chunk_spec,
            decode=self.runner.new_streaming_decode_state(),
            max_new_tokens=max_new_tokens,
        )

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        raise RuntimeError(
            "Nemotron streaming chunks must use the cross-request batch path"
        )

    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        failed: list[str] = []
        with self.state_lock:
            touched: set[str] = set()
            for request_id, item in items:
                if self.is_aborted(request_id):
                    continue
                try:
                    state = self.stream_states[request_id]
                    metadata = item.metadata or {}
                    if not isinstance(metadata, dict):
                        raise TypeError(
                            "Nemotron streaming chunk metadata must be a dict"
                        )
                    if not isinstance(item.data, torch.Tensor):
                        raise TypeError(
                            "Nemotron streaming chunks must carry torch.Tensor"
                        )
                    samples_before = state.total_samples
                    state.append_pcm16(item.data, metadata)
                except Exception as exc:
                    self.emit_error(request_id, exc)
                    self.abort_state(request_id)
                    self.aggregate_metrics["aborted_streams"] += 1
                    failed.append(request_id)
                    continue
                state.metrics.max_queue_depth = max(
                    state.metrics.max_queue_depth, self.inbox.qsize()
                )
                self.aggregate_metrics["input_packets"] += 1
                self.aggregate_metrics["audio_samples"] += (
                    state.total_samples - samples_before
                )
                touched.add(request_id)
            ready = [
                (request_id, state, state.pop_ready_window())
                for request_id, state in self.stream_states.items()
                if request_id in touched
                and not self.is_aborted(request_id)
                and not state.has_reached_decode_limit
                and state.has_ready_window()
            ]
            failed.extend(self.run_ready_windows(ready))
        for request_id in dict.fromkeys(failed):
            self.cleanup_aborted_request(request_id)

    def run_ready_windows(
        self,
        ready: Sequence[
            tuple[str, Nemotron3_5ASRStreamState, Nemotron3_5ASRAudioWindow]
        ],
    ) -> list[str]:
        groups: dict[
            int,
            list[tuple[str, Nemotron3_5ASRStreamState, Nemotron3_5ASRAudioWindow]],
        ] = defaultdict(list)
        for item in ready:
            groups[item[2].model_chunk_index].append(item)

        failed: list[str] = []
        for group in groups.values():
            for offset in range(0, len(group), self.max_batch_size):
                batch = group[offset : offset + self.max_batch_size]
                try:
                    prepared_chunks = [
                        self.runner.prepare_streaming_chunk(
                            window.waveform,
                            language=state.language,
                            is_first=window.is_first,
                        )
                        for _, state, window in batch
                    ]
                    batch_result = self.runner.run_streaming_batch(
                        [state.decode for _, state, _ in batch],
                        prepared_chunks,
                        requested_languages=[state.language for _, state, _ in batch],
                        max_new_tokens=[state.max_new_tokens for _, state, _ in batch],
                    )
                    self.record_batch(batch, batch_result)
                    for index, (request_id, state, _) in enumerate(batch):
                        message = self.partial_message(state, batch_result, index)
                        if message is None or self.is_aborted(request_id):
                            continue
                        self.outbox.put(message)
                except Exception as exc:
                    for request_id, _, _ in batch:
                        self.emit_error(request_id, exc)
                        self.abort_state(request_id)
                        self.aggregate_metrics["aborted_streams"] += 1
                        failed.append(request_id)
        return failed

    def record_batch(
        self,
        batch: Sequence[
            tuple[str, Nemotron3_5ASRStreamState, Nemotron3_5ASRAudioWindow]
        ],
        batch_result: Nemotron3_5ASRStreamingBatchResult,
    ) -> None:
        batch_size = len(batch)
        self.aggregate_metrics["model_batches"] += 1
        self.aggregate_metrics["model_chunks"] += batch_size
        self.aggregate_metrics["model_compute_s"] += batch_result.elapsed_s
        self.aggregate_metrics["max_batch_size"] = max(
            self.aggregate_metrics["max_batch_size"], batch_size
        )
        per_request_compute_s = batch_result.elapsed_s / batch_size
        for _, state, window in batch:
            state.metrics.model_compute_s += per_request_compute_s
            state.metrics.model_chunk_count += 1
            state.metrics.batch_sizes.append(batch_size)
            state.metrics.chunk_latency_ms.append(batch_result.elapsed_s * 1000.0)
            state.metrics.chunk_ready_wait_ms.append(window.ready_wait_s * 1000.0)
            if window.model_chunk_index == 0:
                continue
            state.metrics.cache_reuse_count += 1
            self.aggregate_metrics["cache_reuses"] += 1

    def partial_message(
        self,
        state: Nemotron3_5ASRStreamState,
        batch_result: Nemotron3_5ASRStreamingBatchResult,
        index: int,
    ) -> OutgoingMessage | None:
        previous_text = state.clean_text
        state.raw_text = batch_result.raw_texts[index]
        state.clean_text = batch_result.clean_texts[index]
        state.detected_language = batch_result.languages[index]
        if not state.clean_text or state.clean_text == previous_text:
            return None
        if previous_text and not state.clean_text.startswith(previous_text):
            raise RuntimeError(
                "Nemotron streaming transcript changed a previously emitted prefix"
            )
        text_delta = state.clean_text[len(previous_text) :]
        now = time.perf_counter()
        state.metrics.first_text_s = (
            now if state.metrics.first_text_s is None else state.metrics.first_text_s
        )
        return OutgoingMessage(
            request_id=state.request_id,
            type="stream",
            data={
                "text": text_delta,
                "full_text": state.clean_text,
                "raw_text": state.raw_text,
                "language": state.detected_language,
                "token_ids": list(state.decode.tokens),
                "modality": "text",
                "metrics": self.metrics_snapshot(state, now=now),
            },
            metadata={"modality": "text"},
        )

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        state = self.stream_states[request_id]
        state.mark_done()
        messages: list[OutgoingMessage] = []
        while not state.has_reached_decode_limit and state.has_ready_window(
            finalizing=True
        ):
            window = state.pop_ready_window(finalizing=True)
            prepared_chunk = self.runner.prepare_streaming_chunk(
                window.waveform,
                language=state.language,
                is_first=window.is_first,
            )
            batch_result = self.runner.run_streaming_batch(
                [state.decode],
                [prepared_chunk],
                requested_languages=[state.language],
                max_new_tokens=[state.max_new_tokens],
            )
            self.record_batch([(request_id, state, window)], batch_result)
            partial = self.partial_message(state, batch_result, 0)
            if partial is None:
                continue
            messages.append(partial)

        state.metrics.finalized_s = time.perf_counter()
        self.aggregate_metrics["completed_streams"] += 1
        metrics = self.metrics_snapshot(state, now=state.metrics.finalized_s)
        final_payload = build_nemotron3_5_asr_result(
            state.payload,
            raw_text=state.raw_text,
            requested_language=state.language,
            duration_s=state.total_samples / state.spec.sample_rate,
            asr_latency_s=max(
                state.metrics.finalized_s - state.metrics.request_started_s, 0.0
            ),
            model_latency_s=state.metrics.model_compute_s,
            extra_data={
                "token_ids": list(state.decode.tokens),
                "durations": list(state.decode.durations),
                "encoder_frames": state.decode.encoder_frames,
                "decoder_steps": state.decode.decoder_steps,
                "streaming_latency_ms": state.spec.streaming_latency_ms,
                "metrics": metrics,
            },
        )
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=final_payload,
            )
        )
        return messages

    def metrics_snapshot(
        self, state: Nemotron3_5ASRStreamState, *, now: float
    ) -> dict[str, float | int | list[float] | list[int] | None]:
        audio_s = state.total_samples / state.spec.sample_rate
        compute_s = state.metrics.model_compute_s
        elapsed_s = max(now - state.metrics.request_started_s, 0.0)
        ttft_s = (
            state.metrics.first_text_s - state.metrics.request_started_s
            if state.metrics.first_text_s is not None
            else None
        )
        if (
            state.metrics.finalized_s is not None
            and state.metrics.input_done_s is not None
        ):
            speech_end_to_final_s = (
                state.metrics.finalized_s - state.metrics.input_done_s
            )
        else:
            speech_end_to_final_s = None
        latencies = state.metrics.chunk_latency_ms
        if latencies:
            p50, p99 = np.percentile(
                latencies, [50, 99], method="inverted_cdf"
            ).tolist()
        else:
            p50, p99 = None, None
        return {
            "ttft_s": ttft_s,
            "speech_end_to_final_s": speech_end_to_final_s,
            "elapsed_s": elapsed_s,
            "model_compute_s": compute_s,
            "rtf": compute_s / audio_s if audio_s > 0 else None,
            "rtfx": audio_s / compute_s if compute_s > 0 else None,
            "throughput_audio_s_per_s": audio_s / elapsed_s if elapsed_s > 0 else None,
            "chunk_latency_ms": list(latencies),
            "chunk_latency_p50_ms": p50,
            "chunk_latency_p99_ms": p99,
            "chunk_ready_wait_ms": list(state.metrics.chunk_ready_wait_ms),
            "input_packets": state.metrics.packet_count,
            "model_chunks": state.metrics.model_chunk_count,
            "cache_reuses": state.metrics.cache_reuse_count,
            "batch_sizes": list(state.metrics.batch_sizes),
            "max_queue_depth": state.metrics.max_queue_depth,
        }

    def clear_stream_state(self, request_id: str) -> None:
        self.stream_states.pop(request_id, None)

    def stats(self) -> dict[str, int | float]:
        with self.state_lock:
            return {
                **self.aggregate_metrics,
                "active_streams": len(self.stream_states),
                "inbox_depth": self.inbox.qsize(),
            }

    def start(self) -> None:
        try:
            super().start()
        finally:
            with self.state_lock:
                self.stream_states.clear()
            self.close_runner()

    def stop(self) -> None:
        was_running = self.running
        super().stop()
        if was_running:
            return
        with self.state_lock:
            self.stream_states.clear()
        self.close_runner()

    def close_runner(self) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        self.runner.close()


__all__ = [
    "Nemotron3_5ASRAudioWindow",
    "Nemotron3_5ASRStreamMetrics",
    "Nemotron3_5ASRStreamState",
    "Nemotron3_5ASRStreamingChunkSpec",
    "Nemotron3_5ASRStreamingScheduler",
]
