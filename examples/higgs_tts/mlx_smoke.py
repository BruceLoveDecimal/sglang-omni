# SPDX-License-Identifier: Apache-2.0
"""Exercise a running Higgs MLX server and save inspectable audio/results.

Start the server with SGLANG_USE_MLX=1 before running this script. The server's
allowed-local-media-path must include the reference audio directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18083")
    parser.add_argument("--output-dir", type=Path, default=Path("higgs-mlx-smoke"))
    parser.add_argument(
        "--reference-audio",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "docs/_static/audio/male-voice.wav",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    url = args.base_url.rstrip("/") + "/v1/audio/speech"
    defaults = dict(
        voice="default",
        input="Hello, this is a local speech test.",
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        seed=42,
        max_new_tokens=256,
    )
    results = []

    def generate(name, **overrides):
        payload = {**defaults, **overrides}
        stream = payload.get("stream", False)
        started = time.perf_counter()
        first_byte = None
        chunks = []
        with requests.post(
            url, json=payload, stream=True, timeout=(10, 300)
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    if first_byte is None:
                        first_byte = time.perf_counter() - started
                    chunks.append(chunk)
            content = b"".join(chunks)
            if stream:
                assert response.headers["content-type"].startswith(
                    "audio/pcm"
                ), response.headers
                assert (
                    response.headers.get("X-Sample-Rate") == "24000"
                ), response.headers
                assert len(content) % 2 == 0
                waveform = (
                    np.frombuffer(content, dtype="<i2").astype(np.float32) / 32768
                )
                sample_rate = 24000
                assert len(chunks) > 1, "Expected incremental PCM chunks"
            else:
                waveform, sample_rate = sf.read(io.BytesIO(content), dtype="float32")
        assert sample_rate == 24000 and waveform.ndim == 1
        assert waveform.size >= 4800 and np.isfinite(waveform).all()
        rms = float(np.sqrt(np.mean(waveform**2)))
        assert rms > 1e-5, "Output is silent"
        sf.write(args.output_dir / f"{name}.wav", waveform, sample_rate)
        result = dict(
            name=name,
            duration_s=waveform.size / sample_rate,
            rms=rms,
            first_byte_s=first_byte,
            total_s=time.perf_counter() - started,
            chunks=len(chunks),
            bytes=len(content),
        )
        print(json.dumps(result), flush=True)
        return waveform, result

    for name, overrides in [
        ("english", {}),
        ("chinese", {"input": "你好，这是一次本地语音合成测试。"}),
        ("stream", {"stream": True, "response_format": "pcm"}),
        (
            "reference",
            {
                "references": [
                    {
                        "audio_path": str(args.reference_audio.resolve()),
                        "text": "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time.",
                    }
                ]
            },
        ),
    ]:
        _, result = generate(name, **overrides)
        results.append(result)

    # Both requests use the same seed; queueing must not share RNG or KV state.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(generate, "queued-1")
        second = pool.submit(generate, "queued-2")
        wav1, res1 = first.result()
        wav2, res2 = second.result()
    np.testing.assert_array_equal(wav1, wav2)
    results.extend([res1, res2])

    # Disconnect after the first PCM chunk, then ensure another request succeeds.
    with requests.post(
        url,
        json={
            **defaults,
            "input": "This is a longer sentence. " * 20,
            "stream": True,
            "response_format": "pcm",
        },
        stream=True,
        timeout=(10, 300),
    ) as response:
        response.raise_for_status()
        assert next(response.iter_content(chunk_size=None))
    _, result = generate("after-cancel")
    results.append(result)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        "PASS: WAV, PCM streaming, voice reference, queued requests, and cancellation recovery",
        flush=True,
    )


if __name__ == "__main__":
    main()
