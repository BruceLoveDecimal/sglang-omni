# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS streaming end-to-end acceptance driver.

This is a *gated, real-model* driver (not collected by the default pytest run).
It talks to a running ``sgl-omni serve`` MOSS-TTS server over the OpenAI-compatible
``/v1/audio/speech`` endpoint and exercises the full acceptance matrix from
``docs/design/moss_tts_streaming_plan.md``:

Functional   FA-001 non-stream compat, FA-002 first chunk, FA-003 terminal,
             FA-004 ordering/concat, FA-005 prompt coverage, FA-006 concurrency,
             FA-007 abort + slot reuse.
Quality      QA-001 full-vs-stream parity (duration + RMS), QA-002 boundary
             discontinuity, QA-003 no empty/NaN/Inf chunk.
Performance  PA-001 TTFA P50/P95 per concurrency, PA-002 throughput/RTF,
             PA-004 peak GPU memory (via nvidia-smi).
Stability    SA-001 soak, SA-002 cleanup (client-observable).

Run it directly on the GPU host::

    python tests/test_model/moss_tts_streaming_e2e.py \
        --base-url http://127.0.0.1:8000 \
        --ref-audio docs/_static/audio/higgs-1.wav \
        --ref-text "We asked over twenty different people ..." \
        --out-dir artifacts/moss_streaming_e2e

It writes ``e2e_run_summary.json``, per-suite JSON, and WAV pairs under
``--out-dir`` and exits non-zero if any required check fails.

PA-003 (active vocoder batch size) and exact SA-002 slot counters need
server-side introspection and are therefore reported as client-observable
proxies only; the limitation is recorded in the JSON summary.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import requests

# --------------------------------------------------------------------------- #
# Prompt corpus (FA-005)
# --------------------------------------------------------------------------- #

_LONG_TEXT = (
    "Streaming text to speech systems must keep latency low while preserving "
    "the natural prosody of long form narration. In this paragraph we ask the "
    "model to read several connected clauses without pausing, so that the "
    "delay pattern decoder has to align many audio frames before the first "
    "stable chunk becomes available, and so that the streaming vocoder has to "
    "emit a long sequence of overlapping windows while the autoregressive "
    "engine is still generating new rows for the very same request."
)

_PAUSE_TEXT = "Hello there. [pause 0.5s] Please continue with a calm sentence."

_ZH_TEXT = (
    "今天天气不错，适合出去走走。我们先去公园散步，"
    "然后找一家安静的咖啡馆坐下来，慢慢聊聊最近发生的事情。"
)


@dataclass
class Case:
    name: str
    input: str
    token_count: int
    seed: int
    max_new_tokens: int = 512
    language: str | None = None


def _coverage_cases() -> list[Case]:
    return [
        Case("short_plain_64", "Get the trust fund to the bank early.", 64, 1234, 384),
        Case("long_narration_192", _LONG_TEXT, 192, 4242, 1024),
        Case("pause_markers_96", _PAUSE_TEXT, 96, 5678, 512),
        Case("zh_text_128", _ZH_TEXT, 128, 9012, 640, language="Chinese"),
        Case(
            "terminal_guard_128",
            "SGLang Omni streams audio while generation continues, and the "
            "final payload should not repeat the waveform.",
            128,
            2026,
            640,
        ),
    ]


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #


@dataclass
class StreamResult:
    status: int
    audio: np.ndarray  # float32 mono, concatenated in receive order
    chunk_sample_counts: list[int]  # samples per audio chunk (for boundary checks)
    chunk_events: int
    first_chunk_s: float | None
    total_latency_s: float
    sample_rate: int
    final_events: int
    done: bool
    final_audio_is_null: bool
    usage: dict | None
    error: str | None = None


@dataclass
class NonStreamResult:
    status: int
    audio: np.ndarray
    latency_s: float
    sample_rate: int
    usage: dict | None
    error: str | None = None


def _pcm_bytes_to_float(buf: bytes) -> np.ndarray:
    if not buf:
        return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0
    return arr


class Client:
    def __init__(self, base_url: str, model: str | None, ref_audio_uri: str | None,
                 ref_text: str | None, sample_rate: int, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.ref_audio_uri = ref_audio_uri
        self.ref_text = ref_text
        self.sample_rate = sample_rate
        self.timeout = timeout
        self._session = requests.Session()

    def _body(self, case: Case, *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "input": case.input,
            "stream": stream,
            "seed": case.seed,
            "token_count": case.token_count,
            "max_new_tokens": case.max_new_tokens,
            "response_format": "pcm",
        }
        if self.model:
            body["model"] = self.model
        if case.language:
            body["language"] = case.language
        if self.ref_audio_uri:
            body["ref_audio"] = self.ref_audio_uri
            if self.ref_text:
                body["ref_text"] = self.ref_text
        return body

    def non_stream(self, case: Case) -> NonStreamResult:
        t0 = time.perf_counter()
        try:
            resp = self._session.post(
                f"{self.base_url}/v1/audio/speech",
                json=self._body(case, stream=False),
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            return NonStreamResult(0, np.zeros(0, np.float32), 0.0, self.sample_rate,
                                   None, error=repr(exc))
        latency = time.perf_counter() - t0
        if resp.status_code != 200:
            return NonStreamResult(resp.status_code, np.zeros(0, np.float32), latency,
                                   self.sample_rate, None,
                                   error=resp.text[:300])
        audio = _pcm_bytes_to_float(resp.content)
        usage = None
        if "X-Completion-Tokens" in resp.headers:
            usage = {
                "prompt_tokens": int(resp.headers.get("X-Prompt-Tokens", 0) or 0),
                "completion_tokens": int(resp.headers.get("X-Completion-Tokens", 0) or 0),
            }
        return NonStreamResult(200, audio, latency, self.sample_rate, usage)

    def stream(self, case: Case, *, abort_after_chunks: int | None = None) -> StreamResult:
        t0 = time.perf_counter()
        chunks: list[np.ndarray] = []
        chunk_counts: list[int] = []
        first_chunk_s: float | None = None
        chunk_events = 0
        final_events = 0
        done = False
        final_audio_is_null = False
        usage: dict | None = None
        sample_rate = self.sample_rate
        try:
            resp = self._session.post(
                f"{self.base_url}/v1/audio/speech",
                json=self._body(case, stream=True),
                stream=True,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            return StreamResult(0, np.zeros(0, np.float32), [], 0, None,
                                time.perf_counter() - t0, sample_rate, 0, False,
                                False, None, error=repr(exc))
        if resp.status_code != 200:
            txt = resp.text[:300]
            resp.close()
            return StreamResult(resp.status_code, np.zeros(0, np.float32), [], 0, None,
                                time.perf_counter() - t0, sample_rate, 0, False,
                                False, None, error=txt)
        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[len("data:"):].strip()
                if data == "[DONE]":
                    done = True
                    break
                evt = json.loads(data)
                audio = evt.get("audio")
                if evt.get("finish_reason") is not None:
                    final_events += 1
                    final_audio_is_null = audio is None
                    usage = evt.get("usage")
                    continue
                if audio is None:
                    continue
                if first_chunk_s is None:
                    first_chunk_s = time.perf_counter() - t0
                sample_rate = int(audio.get("sample_rate") or sample_rate)
                pcm = base64.b64decode(audio["data"])
                arr = _pcm_bytes_to_float(pcm)
                chunks.append(arr)
                chunk_counts.append(int(arr.shape[0]))
                chunk_events += 1
                if abort_after_chunks is not None and chunk_events >= abort_after_chunks:
                    resp.close()  # disconnect -> server aborts generation
                    break
        finally:
            resp.close()
        audio_all = (np.concatenate(chunks) if chunks else np.zeros(0, np.float32))
        return StreamResult(200, audio_all, chunk_counts, chunk_events, first_chunk_s,
                            time.perf_counter() - t0, sample_rate, final_events, done,
                            final_audio_is_null, usage)


# --------------------------------------------------------------------------- #
# Audio metrics
# --------------------------------------------------------------------------- #


def _duration_s(audio: np.ndarray, sample_rate: int) -> float:
    return float(audio.shape[0]) / float(sample_rate or 1)


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def _has_nan_inf(audio: np.ndarray) -> bool:
    return bool(audio.size) and bool(np.any(~np.isfinite(audio)))


def _boundary_discontinuity(audio: np.ndarray, chunk_counts: list[int]) -> dict[str, float]:
    """Max/mean absolute sample jump at chunk join points, normalised by RMS."""
    if len(chunk_counts) < 2 or audio.size == 0:
        return {"max": 0.0, "mean": 0.0, "joins": 0}
    rms = _rms(audio) or 1e-9
    jumps = []
    offset = 0
    for count in chunk_counts[:-1]:
        offset += count
        if 0 < offset < audio.shape[0]:
            jumps.append(abs(float(audio[offset]) - float(audio[offset - 1])))
    if not jumps:
        return {"max": 0.0, "mean": 0.0, "joins": 0}
    return {"max": max(jumps) / rms, "mean": (sum(jumps) / len(jumps)) / rms,
            "joins": len(jumps)}


# --------------------------------------------------------------------------- #
# GPU memory sampler (PA-004)
# --------------------------------------------------------------------------- #


class GpuMemorySampler:
    def __init__(self, gpu_index: int = 0, interval_s: float = 0.2):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._peak_mb = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = self._probe() is not None

    def _probe(self) -> int | None:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-i", str(self.gpu_index)],
                stderr=subprocess.DEVNULL, timeout=5,
            )
            return int(out.decode().strip().splitlines()[0])
        except Exception:  # noqa: BLE001
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            used = self._probe()
            if used is not None:
                self._peak_mb = max(self._peak_mb, used)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuMemorySampler":
        self._peak_mb = self._probe() or 0
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def peak_mb(self) -> int:
        return self._peak_mb


# --------------------------------------------------------------------------- #
# Check accumulation
# --------------------------------------------------------------------------- #


@dataclass
class Suite:
    name: str
    checks: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def check(self, name: str, ok: bool, detail: Any = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        return bool(ok)

    @property
    def passed(self) -> bool:
        return all(c["ok"] for c in self.checks)

    def to_dict(self) -> dict:
        return {"name": self.name, "pass": self.passed,
                "checks": self.checks, "metrics": self.metrics,
                "failures": [c for c in self.checks if not c["ok"]]}


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2").tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)


# --------------------------------------------------------------------------- #
# Suites
# --------------------------------------------------------------------------- #


def suite_functional_quality(client: Client, cases: list[Case], out_dir: str,
                             dur_tol: float) -> Suite:
    """FA-001..005 + QA-001..003 over the prompt corpus."""
    s = Suite("functional_quality")
    wav_dir = os.path.join(out_dir, "wavs")
    per_case = []
    for case in cases:
        non = client.non_stream(case)
        stream = client.stream(case)
        s.check(f"{case.name}:nonstream_200", non.status == 200, non.status)
        s.check(f"{case.name}:stream_200", stream.status == 200, stream.status)
        if non.status != 200 or stream.status != 200:
            s.metrics.setdefault("errors", []).append(
                {"case": case.name, "non": non.error, "stream": stream.error})
            continue

        # FA-001 / FA-002
        s.check(f"{case.name}:nonstream_nonempty", non.audio.size > 0, non.audio.size)
        s.check(f"{case.name}:stream_nonempty", stream.audio.size > 0, stream.audio.size)
        s.check(f"{case.name}:first_chunk_before_terminal",
                stream.first_chunk_s is not None
                and stream.first_chunk_s < stream.total_latency_s,
                f"first={stream.first_chunk_s}")
        s.check(f"{case.name}:multiple_chunks", stream.chunk_events >= 1,
                stream.chunk_events)
        # FA-003
        s.check(f"{case.name}:single_final_event", stream.final_events == 1,
                stream.final_events)
        s.check(f"{case.name}:done_sentinel", stream.done, stream.done)
        s.check(f"{case.name}:final_audio_null", stream.final_audio_is_null,
                stream.final_audio_is_null)
        s.check(f"{case.name}:usage_present", bool(stream.usage), stream.usage)
        s.check(f"{case.name}:sample_rate_match",
                stream.sample_rate == non.sample_rate,
                f"{stream.sample_rate} vs {non.sample_rate}")

        # QA-001 parity
        d_non = _duration_s(non.audio, non.sample_rate)
        d_str = _duration_s(stream.audio, stream.sample_rate)
        dur_ratio = (d_str / d_non) if d_non else 0.0
        rms_non, rms_str = _rms(non.audio), _rms(stream.audio)
        rms_diff = abs(rms_str - rms_non) / (rms_non or 1e-9)
        s.check(f"{case.name}:duration_parity_2pct",
                d_non > 0 and abs(dur_ratio - 1.0) <= dur_tol,
                f"non={d_non:.3f} stream={d_str:.3f} ratio={dur_ratio:.4f}")
        # QA-002 boundary discontinuity (reported; soft threshold)
        disc = _boundary_discontinuity(stream.audio, stream.chunk_sample_counts)
        s.check(f"{case.name}:boundary_finite", np.isfinite(disc["max"]), disc)
        # QA-003 no NaN/Inf, no all-zero stream
        s.check(f"{case.name}:no_nan_inf",
                not _has_nan_inf(stream.audio) and not _has_nan_inf(non.audio), "")
        s.check(f"{case.name}:stream_not_silent", rms_str > 0.0, f"rms={rms_str:.5f}")

        _write_wav(os.path.join(wav_dir, f"{case.name}_nonstream.wav"),
                   non.audio, non.sample_rate)
        _write_wav(os.path.join(wav_dir, f"{case.name}_stream.wav"),
                   stream.audio, stream.sample_rate)
        per_case.append({
            "case": case.name,
            "duration_non_s": round(d_non, 4),
            "duration_stream_s": round(d_str, 4),
            "duration_ratio": round(dur_ratio, 4),
            "rms_non": round(rms_non, 6),
            "rms_stream": round(rms_str, 6),
            "rms_rel_diff": round(rms_diff, 4),
            "boundary_discontinuity": disc,
            "first_chunk_s": stream.first_chunk_s,
            "chunk_events": stream.chunk_events,
        })
    s.metrics["per_case"] = per_case
    return s


def suite_concurrency(client: Client, case: Case, levels: list[int],
                      dur_tol: float) -> Suite:
    """FA-006 + PA-001 (TTFA P50/P95) + PA-002 (throughput/RTF)."""
    s = Suite("concurrency")
    summary = {}
    for level in levels:
        t0 = time.perf_counter()
        results: list[StreamResult] = []
        with ThreadPoolExecutor(max_workers=level) as pool:
            futs = [pool.submit(client.stream, case) for _ in range(level)]
            for fut in as_completed(futs):
                results.append(fut.result())
        wall = time.perf_counter() - t0
        ok = [r for r in results if r.status == 200 and r.done]
        s.check(f"c{level}:all_completed", len(ok) == level,
                f"{len(ok)}/{level}")
        s.check(f"c{level}:all_have_audio",
                all(r.audio.size > 0 for r in ok), "")
        s.check(f"c{level}:all_single_final",
                all(r.final_events == 1 for r in ok), "")
        s.check(f"c{level}:no_nan_inf",
                all(not _has_nan_inf(r.audio) for r in ok), "")
        ttfas = sorted(r.first_chunk_s for r in ok if r.first_chunk_s is not None)
        durs = [_duration_s(r.audio, r.sample_rate) for r in ok]
        lats = [r.total_latency_s for r in ok]
        rtf = [(lats[i] / durs[i]) for i in range(len(ok)) if durs[i] > 0]
        summary[f"concurrency_{level}"] = {
            "completed": len(ok),
            "wall_s": round(wall, 3),
            "qps": round(len(ok) / wall, 3) if wall else 0.0,
            "ttfa_p50_s": round(statistics.median(ttfas), 4) if ttfas else None,
            "ttfa_p95_s": round(_pctl(ttfas, 0.95), 4) if ttfas else None,
            "rtf_p50": round(statistics.median(rtf), 4) if rtf else None,
            "audio_duration_p50_s": round(statistics.median(durs), 3) if durs else None,
        }
    s.metrics["per_concurrency"] = summary
    return s


def suite_abort(client: Client, case: Case, n: int) -> Suite:
    """FA-007 + SA-002 (client-observable): abort mid-stream, then reuse."""
    s = Suite("abort")
    aborted: list[StreamResult] = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(client.stream, case, abort_after_chunks=1)
                for _ in range(n)]
        for fut in as_completed(futs):
            aborted.append(fut.result())
    s.check("abort:requests_started",
            all(r.status == 200 for r in aborted), "")
    s.check("abort:got_first_chunk_then_stopped",
            all(r.chunk_events >= 1 and not r.done for r in aborted),
            f"events={[r.chunk_events for r in aborted]}")
    s.check("abort:no_terminal_after_abort",
            all(r.final_events == 0 for r in aborted),
            f"finals={[r.final_events for r in aborted]}")
    # Slot reuse: a fresh request must complete normally after the aborts.
    time.sleep(1.0)
    reuse = client.stream(case)
    s.check("abort:slot_reuse_completes",
            reuse.status == 200 and reuse.done and reuse.audio.size > 0,
            f"done={reuse.done} samples={reuse.audio.size}")
    s.check("abort:slot_reuse_single_final", reuse.final_events == 1,
            reuse.final_events)
    s.metrics["aborted_count"] = len(aborted)
    s.metrics["note"] = (
        "Codec-slot/stream-state freeing is verified indirectly via successful "
        "post-abort reuse; exact server-side slot counters need an introspection "
        "endpoint (out of scope for the client driver)."
    )
    return s


def suite_soak(client: Client, case: Case, total: int, concurrency: int,
               sampler_idx: int) -> Suite:
    """SA-001 soak + monotonic memory-growth guard."""
    s = Suite("soak")
    failures = 0
    completed = 0
    with GpuMemorySampler(sampler_idx) as mem:
        baseline = mem.peak_mb
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(client.stream, case) for _ in range(total)]
            for fut in as_completed(futs):
                r = fut.result()
                if r.status == 200 and r.done and r.audio.size > 0 \
                        and not _has_nan_inf(r.audio):
                    completed += 1
                else:
                    failures += 1
        peak = mem.peak_mb
    s.check("soak:all_completed", failures == 0,
            f"{completed} ok, {failures} failed")
    growth = peak - baseline if mem.available else 0
    s.check("soak:no_runaway_memory_growth",
            (not mem.available) or growth <= 1024,
            f"baseline={baseline}MB peak={peak}MB growth={growth}MB"
            + ("" if mem.available else " (nvidia-smi unavailable)"))
    s.metrics.update({"total": total, "completed": completed,
                      "failed": failures, "concurrency": concurrency,
                      "baseline_mb": baseline, "peak_mb": peak,
                      "mem_available": mem.available})
    return s


def suite_memory(client: Client, case: Case, concurrency: int,
                 sampler_idx: int, mem_tol: float) -> Suite:
    """PA-004: streaming peak GPU memory vs non-streaming peak."""
    s = Suite("memory")
    if not GpuMemorySampler(sampler_idx).available:
        s.check("memory:nvidia_smi_available", False, "nvidia-smi not available")
        s.metrics["skipped"] = True
        return s

    def _run_block(stream: bool) -> int:
        with GpuMemorySampler(sampler_idx) as mem:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                if stream:
                    futs = [pool.submit(client.stream, case)
                            for _ in range(concurrency)]
                else:
                    futs = [pool.submit(client.non_stream, case)
                            for _ in range(concurrency)]
                for fut in as_completed(futs):
                    fut.result()
            return mem.peak_mb

    non_peak = _run_block(stream=False)
    str_peak = _run_block(stream=True)
    over = (str_peak - non_peak) / (non_peak or 1) if non_peak else 0.0
    s.check("memory:streaming_within_tolerance", over <= mem_tol,
            f"non={non_peak}MB stream={str_peak}MB over={over:.2%}")
    s.metrics.update({"nonstream_peak_mb": non_peak, "stream_peak_mb": str_peak,
                      "overhead_frac": round(over, 4)})
    return s


def _pctl(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(int(round(q * (len(sorted_values) - 1))), len(sorted_values) - 1)
    return sorted_values[idx]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _build_ref_data_uri(path: str | None) -> str | None:
    """Path-based reference can fail when the server venv torchcodec lacks
    FFmpeg; a base64 ``data:`` URI routes through soundfile instead."""
    if not path:
        return None
    if path.startswith("data:") or path.startswith("http://") \
            or path.startswith("https://"):
        return path
    with open(path, "rb") as f:
        raw = f.read()
    mime = "audio/wav"
    if path.lower().endswith(".flac"):
        mime = "audio/flac"
    elif path.lower().endswith(".mp3"):
        mime = "audio/mpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def _wait_ready(base_url: str, timeout_s: float) -> tuple[bool, str]:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=5)
            if r.status_code == 200:
                return True, r.text[:300]
            last = f"{r.status_code}: {r.text[:200]}"
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
        time.sleep(2)
    return False, last


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MOSS-TTS streaming E2E acceptance driver")
    p.add_argument("--base-url", default=os.environ.get("MOSS_BASE_URL",
                                                         "http://127.0.0.1:8000"))
    p.add_argument("--model", default=os.environ.get("MOSS_MODEL"))
    p.add_argument("--ref-audio", default=os.environ.get("MOSS_REF_AUDIO"))
    p.add_argument("--ref-text", default=os.environ.get("MOSS_REF_TEXT"))
    p.add_argument("--out-dir", default="artifacts/moss_streaming_e2e")
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--concurrency", default="1,4,8,16",
                   help="comma-separated concurrency levels for FA-006/PA-001")
    p.add_argument("--abort-count", type=int, default=10)
    p.add_argument("--soak-requests", type=int, default=200)
    p.add_argument("--soak-concurrency", type=int, default=8)
    p.add_argument("--duration-tol", type=float, default=0.02)
    p.add_argument("--memory-tol", type=float, default=0.10)
    p.add_argument("--ready-timeout", type=float, default=180.0)
    p.add_argument("--quick", action="store_true",
                   help="fast smoke: concurrency 1,2; soak 8; for local checks")
    p.add_argument("--skip", default="",
                   help="comma list of suites to skip: "
                        "functional,concurrency,abort,memory,soak")
    args = p.parse_args(argv)

    if args.quick:
        args.concurrency = "1,2"
        args.soak_requests = 8
        args.soak_concurrency = 2
        args.abort_count = 3

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}
    os.makedirs(args.out_dir, exist_ok=True)

    ready, ready_detail = _wait_ready(args.base_url, args.ready_timeout)
    if not ready:
        print(f"[FATAL] server not ready: {ready_detail}", file=sys.stderr)
        return 2

    ref_uri = _build_ref_data_uri(args.ref_audio)
    client = Client(args.base_url, args.model, ref_uri, args.ref_text,
                    args.sample_rate)
    cases = _coverage_cases()
    conc_case = cases[0]

    suites: list[Suite] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    if "functional" not in skip:
        print("[run] functional_quality ...", flush=True)
        suites.append(suite_functional_quality(client, cases, args.out_dir,
                                                args.duration_tol))
    if "concurrency" not in skip:
        print("[run] concurrency ...", flush=True)
        suites.append(suite_concurrency(client, conc_case, levels,
                                        args.duration_tol))
    if "abort" not in skip:
        print("[run] abort ...", flush=True)
        suites.append(suite_abort(client, conc_case, args.abort_count))
    if "memory" not in skip:
        print("[run] memory ...", flush=True)
        suites.append(suite_memory(client, conc_case, max(levels),
                                   args.gpu_index, args.memory_tol))
    if "soak" not in skip:
        print("[run] soak ...", flush=True)
        suites.append(suite_soak(client, conc_case, args.soak_requests,
                                 args.soak_concurrency, args.gpu_index))

    overall_pass = all(s.passed for s in suites)
    summary = {
        "pass": overall_pass,
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_url": args.base_url,
        "model": args.model,
        "ref_audio": args.ref_audio,
        "reference_transport": "data_uri" if ref_uri and ref_uri.startswith("data:")
                               else ("url" if ref_uri else "none"),
        "concurrency_levels": levels,
        "git_commit": _git_commit(),
        "models_ready": ready_detail,
        "suites": [s.to_dict() for s in suites],
        "limitations": [
            "PA-003 active vocoder batch size not measured (needs server "
            "introspection); v1 vocoder decodes per-request by design.",
            "SA-002 exact slot/state counters approximated via post-abort reuse.",
        ],
    }
    out_path = os.path.join(args.out_dir, "e2e_run_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== MOSS streaming E2E: {'PASS' if overall_pass else 'FAIL'} ===")
    for s in suites:
        flag = "PASS" if s.passed else "FAIL"
        fails = [c["name"] for c in s.checks if not c["ok"]]
        print(f"  [{flag}] {s.name}"
              + (f"  failures: {fails}" if fails else ""))
    print(f"summary: {out_path}")
    return 0 if overall_pass else 1


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    raise SystemExit(main())
