# SPDX-License-Identifier: Apache-2.0
"""Lightweight gated MOSS-TTS streaming e2e smoke driver.

Run it directly against a running MOSS-TTS server:

    python tests/test_model/moss_tts_streaming_e2e.py \
        --base-url http://127.0.0.1:8000 \
        --ref-audio docs/_static/audio/higgs-1.wav \
        --ref-text "We asked over twenty different people ..." \
        --out-dir artifacts/moss_streaming_e2e
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import requests


@dataclass
class Case:
    name: str
    input: str
    token_count: int
    seed: int
    max_new_tokens: int


@dataclass
class Checks:
    checks: list[dict[str, str | bool]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: Any = "") -> None:
        self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    @property
    def passed(self) -> bool:
        return all(bool(item["ok"]) for item in self.checks)

    @property
    def failures(self) -> list[dict[str, str | bool]]:
        return [item for item in self.checks if not item["ok"]]


def _smoke_case() -> Case:
    return Case(
        name="short_plain_64",
        input="Get the trust fund to the bank early.",
        token_count=64,
        seed=1234,
        max_new_tokens=384,
    )


def _pcm_bytes_to_float(buf: bytes) -> np.ndarray:
    if not buf:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0


def _duration_s(audio: np.ndarray, sample_rate: int) -> float:
    return float(audio.shape[0]) / float(sample_rate or 1)


def _has_nan_inf(audio: np.ndarray) -> bool:
    return bool(audio.size) and bool(np.any(~np.isfinite(audio)))


def _body(
    case: Case,
    *,
    stream: bool,
    model: str | None,
    ref_audio: str | None,
    ref_text: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "input": case.input,
        "stream": stream,
        "seed": case.seed,
        "token_count": case.token_count,
        "max_new_tokens": case.max_new_tokens,
        "response_format": "pcm",
    }
    if model:
        body["model"] = model
    if ref_audio:
        body["ref_audio"] = ref_audio
        if ref_text:
            body["ref_text"] = ref_text
    return body


def _usage_from_headers(
    headers: requests.structures.CaseInsensitiveDict,
) -> dict | None:
    if "X-Completion-Tokens" not in headers:
        return None
    return {
        "prompt_tokens": int(headers.get("X-Prompt-Tokens", 0) or 0),
        "completion_tokens": int(headers.get("X-Completion-Tokens", 0) or 0),
    }


def _non_stream(
    session: requests.Session,
    *,
    base_url: str,
    model: str | None,
    ref_audio: str | None,
    ref_text: str | None,
    case: Case,
    sample_rate: int,
    timeout: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        resp = session.post(
            f"{base_url}/v1/audio/speech",
            json=_body(
                case,
                stream=False,
                model=model,
                ref_audio=ref_audio,
                ref_text=ref_text,
            ),
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": 0,
            "audio": np.zeros(0, np.float32),
            "latency_s": 0.0,
            "sample_rate": sample_rate,
            "usage": None,
            "error": repr(exc),
        }

    latency_s = time.perf_counter() - t0
    if resp.status_code != 200:
        return {
            "status": resp.status_code,
            "audio": np.zeros(0, np.float32),
            "latency_s": latency_s,
            "sample_rate": sample_rate,
            "usage": None,
            "error": resp.text[:300],
        }

    return {
        "status": 200,
        "audio": _pcm_bytes_to_float(resp.content),
        "latency_s": latency_s,
        "sample_rate": sample_rate,
        "usage": _usage_from_headers(resp.headers),
        "error": None,
    }


def _empty_stream_result(
    *,
    status: int,
    started_at: float,
    sample_rate: int,
    error: str | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "audio": np.zeros(0, np.float32),
        "chunk_events": 0,
        "first_chunk_s": None,
        "total_latency_s": time.perf_counter() - started_at,
        "sample_rate": sample_rate,
        "final_events": 0,
        "done": False,
        "final_audio_is_null": False,
        "usage": None,
        "error": error,
    }


def _stream(
    session: requests.Session,
    *,
    base_url: str,
    model: str | None,
    ref_audio: str | None,
    ref_text: str | None,
    case: Case,
    sample_rate: int,
    timeout: float,
    abort_after_chunks: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        resp = session.post(
            f"{base_url}/v1/audio/speech",
            json=_body(
                case,
                stream=True,
                model=model,
                ref_audio=ref_audio,
                ref_text=ref_text,
            ),
            stream=True,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return _empty_stream_result(
            status=0,
            started_at=t0,
            sample_rate=sample_rate,
            error=repr(exc),
        )

    if resp.status_code != 200:
        text = resp.text[:300]
        resp.close()
        return _empty_stream_result(
            status=resp.status_code,
            started_at=t0,
            sample_rate=sample_rate,
            error=text,
        )

    chunks: list[np.ndarray] = []
    first_chunk_s: float | None = None
    final_events = 0
    done = False
    final_audio_is_null = False
    usage = None

    try:
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[len("data:") :].strip()
            if data == "[DONE]":
                done = True
                break

            event = json.loads(data)
            audio = event.get("audio")
            if event.get("finish_reason") is not None:
                final_events += 1
                final_audio_is_null = audio is None
                usage = event.get("usage")
                continue
            if audio is None:
                continue

            if first_chunk_s is None:
                first_chunk_s = time.perf_counter() - t0
            sample_rate = int(audio.get("sample_rate") or sample_rate)
            chunks.append(_pcm_bytes_to_float(base64.b64decode(audio["data"])))

            if abort_after_chunks is not None and len(chunks) >= abort_after_chunks:
                resp.close()
                break
    finally:
        resp.close()

    return {
        "status": 200,
        "audio": np.concatenate(chunks) if chunks else np.zeros(0, np.float32),
        "chunk_events": len(chunks),
        "first_chunk_s": first_chunk_s,
        "total_latency_s": time.perf_counter() - t0,
        "sample_rate": sample_rate,
        "final_events": final_events,
        "done": done,
        "final_audio_is_null": final_audio_is_null,
        "usage": usage,
        "error": None,
    }


def _build_ref_data_uri(path: str | None) -> str | None:
    if not path:
        return None
    if (
        path.startswith("data:")
        or path.startswith("http://")
        or path.startswith("https://")
    ):
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
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return True, response.text[:300]
            last = f"{response.status_code}: {response.text[:200]}"
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
        time.sleep(2)
    return False, last


def _run_smoke(
    session: requests.Session,
    *,
    base_url: str,
    model: str | None,
    ref_audio: str | None,
    ref_text: str | None,
    sample_rate: int,
    timeout: float,
    duration_tol: float,
    skip_abort: bool,
) -> Checks:
    case = _smoke_case()
    checks = Checks()
    common = {
        "session": session,
        "base_url": base_url,
        "model": model,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "case": case,
        "sample_rate": sample_rate,
        "timeout": timeout,
    }
    non = _non_stream(**common)
    stream = _stream(**common)

    checks.add("nonstream_200", non["status"] == 200, non["status"])
    checks.add("stream_200", stream["status"] == 200, stream["status"])
    checks.metrics["errors"] = {"nonstream": non["error"], "stream": stream["error"]}
    if non["status"] != 200 or stream["status"] != 200:
        return checks

    non_duration = _duration_s(non["audio"], non["sample_rate"])
    stream_duration = _duration_s(stream["audio"], stream["sample_rate"])
    duration_ratio = stream_duration / non_duration if non_duration else 0.0

    checks.add("nonstream_nonempty", non["audio"].size > 0, non["audio"].size)
    checks.add("stream_nonempty", stream["audio"].size > 0, stream["audio"].size)
    checks.add(
        "first_chunk_before_terminal",
        stream["first_chunk_s"] is not None
        and stream["first_chunk_s"] < stream["total_latency_s"],
        stream["first_chunk_s"],
    )
    checks.add("stream_has_chunk", stream["chunk_events"] >= 1, stream["chunk_events"])
    checks.add(
        "single_final_event", stream["final_events"] == 1, stream["final_events"]
    )
    checks.add("done_sentinel", stream["done"], stream["done"])
    checks.add("final_audio_null", stream["final_audio_is_null"])
    checks.add("usage_present", bool(stream["usage"]), stream["usage"])
    checks.add(
        "sample_rate_match",
        stream["sample_rate"] == non["sample_rate"],
        f"{stream['sample_rate']} vs {non['sample_rate']}",
    )
    checks.add(
        "duration_parity",
        non_duration > 0 and abs(duration_ratio - 1.0) <= duration_tol,
        f"non={non_duration:.3f} stream={stream_duration:.3f} ratio={duration_ratio:.4f}",
    )
    checks.add(
        "no_nan_inf",
        not _has_nan_inf(non["audio"]) and not _has_nan_inf(stream["audio"]),
    )

    if not skip_abort:
        aborted = _stream(**common, abort_after_chunks=1)
        checks.add("abort_request_started", aborted["status"] == 200, aborted["status"])
        checks.add(
            "abort_after_first_chunk",
            aborted["chunk_events"] >= 1 and not aborted["done"],
            f"chunks={aborted['chunk_events']} done={aborted['done']}",
        )
        checks.add(
            "abort_no_final_event",
            aborted["final_events"] == 0,
            aborted["final_events"],
        )
        time.sleep(1.0)
        reuse = _stream(**common)
        checks.add(
            "post_abort_reuse_completes",
            reuse["status"] == 200 and reuse["done"] and reuse["audio"].size > 0,
            f"status={reuse['status']} done={reuse['done']} samples={reuse['audio'].size}",
        )

    checks.metrics.update(
        {
            "case": case.name,
            "nonstream_latency_s": round(non["latency_s"], 4),
            "stream_latency_s": round(stream["total_latency_s"], 4),
            "first_chunk_s": (
                round(stream["first_chunk_s"], 4)
                if stream["first_chunk_s"] is not None
                else None
            ),
            "chunk_events": stream["chunk_events"],
            "nonstream_duration_s": round(non_duration, 4),
            "stream_duration_s": round(stream_duration, 4),
            "duration_ratio": round(duration_ratio, 4),
            "nonstream_usage": non["usage"],
            "stream_usage": stream["usage"],
        }
    )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MOSS-TTS streaming e2e smoke driver")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MOSS_BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--model", default=os.environ.get("MOSS_MODEL"))
    parser.add_argument("--ref-audio", default=os.environ.get("MOSS_REF_AUDIO"))
    parser.add_argument("--ref-text", default=os.environ.get("MOSS_REF_TEXT"))
    parser.add_argument("--out-dir", default="artifacts/moss_streaming_e2e")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--duration-tol", type=float, default=0.05)
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--skip-abort", action="store_true")
    args = parser.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    os.makedirs(args.out_dir, exist_ok=True)
    ready, ready_detail = _wait_ready(base_url, args.ready_timeout)
    if not ready:
        print(f"[FATAL] server not ready: {ready_detail}", file=sys.stderr)
        return 2

    ref_audio = _build_ref_data_uri(args.ref_audio)
    checks = _run_smoke(
        requests.Session(),
        base_url=base_url,
        model=args.model,
        ref_audio=ref_audio,
        ref_text=args.ref_text,
        sample_rate=args.sample_rate,
        timeout=args.timeout,
        duration_tol=args.duration_tol,
        skip_abort=args.skip_abort,
    )

    summary = {
        "pass": checks.passed,
        "base_url": base_url,
        "model": args.model,
        "ref_audio": args.ref_audio,
        "reference_transport": (
            "data_uri"
            if ref_audio and ref_audio.startswith("data:")
            else ("url" if ref_audio else "none")
        ),
        "models_ready": ready_detail,
        "checks": checks.checks,
        "metrics": checks.metrics,
        "failures": checks.failures,
    }
    out_path = os.path.join(args.out_dir, "e2e_run_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== MOSS streaming e2e smoke: {'PASS' if checks.passed else 'FAIL'} ===")
    if checks.failures:
        print(f"failures: {[item['name'] for item in checks.failures]}")
    print(f"summary: {out_path}")
    return 0 if checks.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
