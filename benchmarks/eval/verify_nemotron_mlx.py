# SPDX-License-Identifier: Apache-2.0
"""Real-weight Torch/MLX token parity followed by real HTTP/SSE serving checks.

Run with the Apple environment from the repository root:
  python -m benchmarks.eval.verify_nemotron_mlx --model-path /path/to/model \
      --output /tmp/nemotron-validation.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import torch


def verify_components(model_path, audio_paths):
    import mlx.core as mx

    from sglang_omni.models.nemotron3_5_asr.mlx.runner import Nemotron3_5ASRMLXRunner
    from sglang_omni.models.nemotron3_5_asr.model_runner import (
        Nemotron3_5ASRModelRunner,
    )
    from sglang_omni.serve.transcription_adapters.nemotron3_5_asr import (
        Nemotron3_5ASRTranscriptionAdapter,
    )
    from sglang_omni.utils.audio import load_audio

    torch.set_num_threads(4)
    reference = Nemotron3_5ASRModelRunner(model_path, device="cpu")
    native = Nemotron3_5ASRMLXRunner(model_path)
    adapter = Nemotron3_5ASRTranscriptionAdapter()
    results = []
    for lookahead in (0, 3, 6, 13):
        native.processor.set_num_lookahead_tokens(lookahead)
        for path in audio_paths:
            audio = load_audio(
                str(path), source_name="Nemotron validation", target_sample_rate=16000
            )
            for language in ("en-US", "auto"):
                inputs = dict(
                    native.processor(
                        audio,
                        sampling_rate=16000,
                        language=language,
                        return_tensors="pt",
                    )
                )
                started = time.perf_counter()
                expected = reference._generate_sequences(inputs, max_new_tokens=None)[
                    0
                ].tolist()
                torch_s = time.perf_counter() - started
                started = time.perf_counter()
                actual = native._generate_sequences(inputs, max_new_tokens=None)[0]
                mlx_s = time.perf_counter() - started
                assert (
                    actual == expected
                ), f"Token mismatch: {path.name} {language} lookahead={lookahead}"
                raw = native.processor.decode(actual, skip_special_tokens=False)
                text = adapter.postprocess_text(raw)
                assert text, f"Empty speech transcript: {path}"
                row = dict(
                    audio=path.name,
                    language=language,
                    lookahead=lookahead,
                    tokens=actual,
                    text=text,
                    raw_text=raw,
                    torch_s=torch_s,
                    mlx_s=mlx_s,
                )
                results.append(row)
                print(
                    json.dumps({k: v for k, v in row.items() if k != "tokens"}),
                    flush=True,
                )
    peak = mx.get_peak_memory()
    reference.close()
    native.close()
    del reference, native
    gc.collect()
    return results, peak


def verify_http(model_path, audio_paths, component_results, output):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, SGLANG_USE_MLX="1", TOKENIZERS_PARALLELISM="false")
    env["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/ffmpeg@7/lib" + (
        ":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else ""
    )
    server_log = output.with_suffix(".server.log")
    rows = []
    output.with_suffix(".http.jsonl").write_text("")
    started = time.perf_counter()
    with server_log.open("w") as log:
        proc = subprocess.Popen(
            [
                str(Path(sys.executable).parent / "sgl-omni"),
                "serve",
                "--model-path",
                model_path,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            with httpx.Client(base_url=base, timeout=180) as client:
                deadline = time.monotonic() + 180
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Server exited; see {server_log}")
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Server readiness timed out; see {server_log}"
                        )
                    time.sleep(0.5)
                startup_s = time.perf_counter() - started

                def request(audio, **params):
                    data = dict(
                        model=model_path, language="en-US", response_format="json"
                    )
                    data.update(params)
                    tick = time.perf_counter()
                    response = client.post(
                        "/v1/audio/transcriptions",
                        data=data,
                        files={"file": ("audio.wav", audio, "audio/wav")},
                    )
                    row = dict(
                        params=data,
                        status=response.status_code,
                        elapsed_s=time.perf_counter() - tick,
                        body=response.text,
                    )
                    rows.append(row)
                    with output.with_suffix(".http.jsonl").open("a") as records:
                        records.write(json.dumps(row) + "\n")
                    return response

                expected = {
                    (row["audio"], row["language"]): row["text"]
                    for row in component_results
                    if row["lookahead"] == 3
                }
                for path in audio_paths:
                    audio = path.read_bytes()
                    for language in ("en-US", "auto"):
                        for fmt in ("json", "text", "verbose_json"):
                            response = request(
                                audio, language=language, response_format=fmt
                            )
                            response.raise_for_status()
                            text = (
                                response.text
                                if fmt == "text"
                                else response.json()["text"]
                            )
                            assert text.strip() == expected[path.name, language]
                            if fmt == "verbose_json":
                                assert response.json()["language"] == "en-US"
                    response = request(audio, stream="true")
                    response.raise_for_status()
                    events = [
                        json.loads(line[6:])
                        for line in response.text.splitlines()
                        if line.startswith("data: ") and line != "data: [DONE]"
                    ]
                    text = "".join(
                        e.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        for e in events
                    )
                    # The endpoint may use transcription delta events instead of chat deltas.
                    if not text:
                        text = "".join(
                            e.get("delta", "")
                            for e in events
                            if e.get("type") == "transcript.text.delta"
                        )
                    done = [
                        e for e in events if e.get("type") == "transcript.text.done"
                    ]
                    assert (
                        len(done) == 1
                        and done[0]["text"] == expected[path.name, "en-US"]
                    ), events
                    assert "data: [DONE]" in response.text
                    if text:
                        assert text.strip() == expected[path.name, "en-US"], events

                # Queued peers and repeat requests must not inherit predictor state.
                def concurrent_request(path):
                    response = request(path.read_bytes())
                    response.raise_for_status()
                    assert response.json()["text"] == expected[path.name, "en-US"]

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(concurrent_request, audio_paths * 2))

                for params in (
                    {"language": "xx-XX"},
                    {"temperature": "0.5"},
                    {"prompt": "context"},
                ):
                    assert (
                        request(audio_paths[0].read_bytes(), **params).status_code
                        == 400
                    )
                assert request(b"not audio").status_code == 400
                empty = io.BytesIO()
                sf.write(empty, np.zeros(0, dtype=np.float32), 16000, format="WAV")
                assert request(empty.getvalue()).status_code == 400
                too_long = io.BytesIO()
                sf.write(
                    too_long,
                    np.zeros(61 * 16000, dtype=np.float32),
                    16000,
                    format="WAV",
                )
                assert request(too_long.getvalue()).status_code == 400
                # Recovery after bad requests.
                concurrent_request(audio_paths[0])
                return dict(
                    startup_s=startup_s, requests=rows, server_log=str(server_log)
                )
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    audio_paths = [
        root / "tests/data/query_to_cars.wav",
        root / "tests/data/query_to_draw.wav",
    ]
    report = dict(
        hardware=platform.platform(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "mlx", "transformers", "sglang")
        },
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        model_path=args.model_path,
    )
    import mlx.core as mx

    report["metal_device"] = mx.device_info()
    report["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "sglang_omni/models/nemotron3_5_asr").rglob("*.py"))
    }
    report["audio_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in audio_paths
    }
    manifest = Path(args.model_path) / "conversion.json"
    if manifest.exists():
        report["conversion"] = json.loads(manifest.read_text())
    try:
        report["components"], report["mlx_peak_bytes"] = verify_components(
            args.model_path, audio_paths
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["http"] = verify_http(
            args.model_path, audio_paths, report["components"], args.output
        )
        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["error"] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validation passed: {args.output}")


if __name__ == "__main__":
    main()
