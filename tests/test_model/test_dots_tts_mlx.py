# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-checkpoint Apple HTTP tests.

DOTS_TTS_MLX_MODEL_PATH=/path/to/dots.tts-mf pytest -q -s \
    tests/test_model/test_dots_tts_mlx.py

Set DOTS_TTS_MLX_OUTPUT_DIR to retain WAV/PCM outputs, timings and server logs.
"""

from __future__ import annotations

import io
import json
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import requests
import soundfile as sf
import torch

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(
        not os.environ.get("DOTS_TTS_MLX_MODEL_PATH")
        or not torch.backends.mps.is_available(),
        reason="set DOTS_TTS_MLX_MODEL_PATH on an Apple Silicon host",
    ),
]
ROOT = Path(__file__).resolve().parents[2]
TEXT = "Hello, this is a test of speech synthesis on Apple Silicon."


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    model = str(Path(os.environ["DOTS_TTS_MLX_MODEL_PATH"]).resolve())
    output = Path(
        os.environ.get("DOTS_TTS_MLX_OUTPUT_DIR") or tmp_path_factory.mktemp("dots_mlx")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        model,
        "--config",
        str(ROOT / "examples/configs/dots_tts_mlx.yaml"),
        "--allowed-local-media-path",
        str(ROOT / "docs/_static/audio"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    proc = start_server_from_cmd(
        command,
        output / "server.log",
        port,
        timeout=600,
        env={"SGLANG_USE_MLX": "1", "PYTHONPATH": str(ROOT)},
    )
    metrics = []
    try:
        yield f"http://127.0.0.1:{port}", model, output, metrics
    finally:
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        stop_server(proc)


def _body(model, **overrides):
    return {
        "model": model,
        "input": TEXT,
        "references": [
            {
                "audio_path": str(ROOT / "docs/_static/audio/male-voice.wav"),
                "text": "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time.",
            }
        ],
        "seed": 42,
        "stage_params": {"latent_engine": {"max_generate_length": 128}},
        **overrides,
    }


def _wav(server, name):
    url, model, output, metrics = server
    start = time.perf_counter()
    with requests.Session() as client:
        client.trust_env = False
        response = client.post(url + "/v1/audio/speech", json=_body(model), timeout=300)
        response.raise_for_status()
    elapsed = time.perf_counter() - start
    waveform, rate = sf.read(io.BytesIO(response.content), dtype="float32")
    assert rate == 48000 and waveform.ndim == 1
    assert waveform.size > rate // 2 and np.isfinite(waveform).all()
    assert float(np.sqrt(np.mean(waveform**2))) > 1e-4
    (output / f"{name}.wav").write_bytes(response.content)
    metrics.append(
        {"name": name, "seconds": elapsed, "audio_seconds": len(waveform) / rate}
    )
    return waveform


def test_nonstreaming_seed_reproducibility(server):
    first = _wav(server, "nonstreaming")
    second = _wav(server, "repeat")
    np.testing.assert_array_equal(first, second)


def test_streaming_pcm(server):
    url, model, output, metrics = server
    chunks = []
    first_byte = None
    start = time.perf_counter()
    with requests.Session() as client:
        client.trust_env = False
        with client.post(
            url + "/v1/audio/speech",
            json=_body(model, stream=True, response_format="pcm"),
            stream=True,
            timeout=300,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    if first_byte is None:
                        first_byte = time.perf_counter() - start
                    chunks.append(chunk)
    pcm = b"".join(chunks)
    assert len(pcm) % 2 == 0
    waveform = np.frombuffer(pcm, dtype="<i2")
    assert len(chunks) > 1 and waveform.size > 24000
    assert np.max(np.abs(waveform.astype(np.int32))) > 100
    (output / "streaming.pcm").write_bytes(pcm)
    sf.write(output / "streaming.wav", waveform, 48000, subtype="PCM_16")
    metrics.append(
        {
            "name": "streaming",
            "seconds": time.perf_counter() - start,
            "first_byte_seconds": first_byte,
            "chunks": len(chunks),
            "audio_seconds": len(waveform) / 48000,
        }
    )


def test_concurrent_clients_queue_without_state_leakage(server):
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(
            pool.map(lambda name: _wav(server, name), ["queued_a", "queued_b"])
        )
    np.testing.assert_array_equal(first, second)


def test_disconnect_then_recovery(server):
    url, model, _, _ = server
    with requests.Session() as client:
        client.trust_env = False
        with client.post(
            url + "/v1/audio/speech",
            json=_body(model, stream=True, response_format="pcm"),
            stream=True,
            timeout=300,
        ) as response:
            response.raise_for_status()
            assert next(response.iter_content(chunk_size=4096))
    _wav(server, "after_disconnect")


def test_missing_reference_is_rejected(server):
    url, model, _, _ = server
    with requests.Session() as client:
        client.trust_env = False
        response = client.post(
            url + "/v1/audio/speech",
            json=_body(model, references=[]),
            timeout=30,
        )
    assert response.status_code == 400
