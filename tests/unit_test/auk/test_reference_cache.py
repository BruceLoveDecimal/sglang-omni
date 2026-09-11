# SPDX-License-Identifier: Apache-2.0
"""Content-keyed reference audio and VAE posterior caches (no GPU, no weights)."""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path

import httpx
import numpy as np
import pybase64
import pytest
import torch

from sglang_omni.models.auk.constants import QWEN_AUDIO_SAMPLE_RATE, SAMPLE_RATE
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.models.auk.reference_cache import (
    AuKReferenceEncoder,
    AuKReferenceIdentity,
    AuKReferenceLoader,
    AuKReferenceWaveform,
)
from sglang_omni.models.auk.vae import BigVGANFlowVAE

# --------------------------------------------------------------------------- #
# Reference audio loader
# --------------------------------------------------------------------------- #


def wav_bytes(seconds: float, *, seed: int, sample_rate: int = 8000) -> bytes:
    rng = np.random.default_rng(seed)
    pcm = (rng.uniform(-0.5, 0.5, int(seconds * sample_rate)) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def data_uri(payload: bytes) -> str:
    return "data:audio/wav;base64," + pybase64.b64encode(payload).decode()


@pytest.fixture
def loader():
    loader = AuKReferenceLoader(SAMPLE_RATE)
    yield loader
    loader.close()


def counts(stats: dict[str, int] | None, *names: str) -> tuple[int, ...]:
    assert stats is not None
    return tuple(stats[name] for name in names)


def test_loader_resamples_for_both_consumers(loader):
    audio = loader.load(wav_bytes(0.5, seed=1))
    assert audio.vae_audio.dtype == audio.qwen_audio.dtype == np.float32
    assert audio.vae_audio.shape == (SAMPLE_RATE // 2,)
    assert audio.qwen_audio.shape == (QWEN_AUDIO_SAMPLE_RATE // 2,)


def test_loader_keys_inline_references_by_content(loader):
    same = wav_bytes(0.5, seed=1)
    other = wav_bytes(0.5, seed=2)

    first = loader.load(same)
    assert counts(loader.stats(), "hits", "misses") == (0, 1)
    # A data URI carrying the same bytes shares the entry.
    second = loader.load(data_uri(same))
    assert counts(loader.stats(), "hits", "misses") == (1, 1)
    np.testing.assert_array_equal(first.vae_audio, second.vae_audio)
    np.testing.assert_array_equal(first.qwen_audio, second.qwen_audio)

    loader.load(other)
    assert counts(loader.stats(), "hits", "misses", "entries") == (1, 2, 2)


def test_loader_hands_out_private_copies(loader):
    payload = wav_bytes(0.5, seed=1)
    first = loader.load(payload)
    first.vae_audio[:] = 0
    first.qwen_audio[:] = 0
    second = loader.load(payload)
    assert counts(loader.stats(), "hits") == (1,)
    assert np.abs(second.vae_audio).max() > 0
    assert np.abs(second.qwen_audio).max() > 0


def test_loader_revalidates_local_files_by_content(loader, tmp_path):
    path = tmp_path / "ref.wav"
    path.write_bytes(wav_bytes(0.5, seed=1))
    loader.load(str(path))
    loader.load(f"file://{path}")
    assert counts(loader.stats(), "hits", "misses") == (1, 1)

    # Same path and size, different samples: the entry must not be reused.
    path.write_bytes(wav_bytes(0.5, seed=2))
    os.utime(path, ns=(1, 1))
    loader.load(str(path))
    assert counts(loader.stats(), "hits", "misses") == (1, 2)


def test_loader_bypasses_cache_for_unreadable_paths(loader, tmp_path):
    with pytest.raises((OSError, RuntimeError)):
        loader.load(str(tmp_path / "missing.wav"))
    assert counts(loader.stats(), "uncacheable", "failed", "entries") == (1, 1, 0)


def test_loader_never_trusts_the_url_string(loader, monkeypatch):
    served = [wav_bytes(0.5, seed=1), wav_bytes(0.5, seed=2), wav_bytes(0.5, seed=2)]

    def fake_get(url, **kwargs):
        return httpx.Response(
            200, content=served.pop(0), request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    url = "https://example.invalid/speaker.wav"
    loader.load(url)
    loader.load(url)
    assert counts(loader.stats(), "hits", "misses") == (0, 2)
    loader.load(url)
    assert counts(loader.stats(), "hits", "misses") == (1, 2)


def test_loader_respects_the_byte_budget():
    loader = AuKReferenceLoader(SAMPLE_RATE, max_bytes=1024)
    try:
        loader.load(wav_bytes(0.5, seed=1))
        assert counts(loader.stats(), "misses", "entries", "bytes") == (1, 0, 0)
    finally:
        loader.close()


def test_loader_can_run_without_a_cache():
    loader = AuKReferenceLoader(SAMPLE_RATE, cache=False)
    audio = loader.load(wav_bytes(0.5, seed=1))
    assert audio.vae_audio.shape == (SAMPLE_RATE // 2,)
    assert loader.stats() is None


def test_loader_rejects_unknown_source_types(loader):
    with pytest.raises(ValueError, match="Unsupported AuK reference audio input"):
        loader.load(123)


# --------------------------------------------------------------------------- #
# VAE posterior cache
# --------------------------------------------------------------------------- #


class CountingEncoder(torch.nn.Conv1d):
    def __init__(self, hop_size: int):
        torch.manual_seed(0)
        super().__init__(1, 8, 1, stride=hop_size)
        self.calls = 0

    def forward(self, sample):
        self.calls += 1
        return super().forward(sample)


class PosteriorVAE(torch.nn.Module):
    """A one-layer stand-in that keeps the real posterior sampling code."""

    encode_posterior = BigVGANFlowVAE.encode_posterior
    sample_and_normalize = BigVGANFlowVAE.sample_and_normalize
    encoding_and_normalization = BigVGANFlowVAE.encoding_and_normalization

    def __init__(self):
        super().__init__()
        self.hop_size = 480
        self.audio_encoder = CountingEncoder(self.hop_size)
        self.register_buffer("global_mean", torch.full((4,), 0.5))
        self.register_buffer("global_log_std", torch.full((4,), 4.0))


@pytest.fixture
def vae():
    return PosteriorVAE()


@pytest.fixture
def audio():
    return np.random.default_rng(3).standard_normal(24001).astype(np.float32)


def test_cached_posterior_reproduces_the_uncached_seeded_latent(vae, audio):
    device = torch.device("cpu")
    uncached = AuKReferenceEncoder(vae, device)
    cached = AuKReferenceEncoder(vae, device, AuKReferenceIdentity("stub", "cfg"))

    expected, length = uncached.encode(audio, seed=7)
    for _ in range(2):
        latent, cached_length = cached.encode(audio, seed=7)
        assert torch.equal(latent, expected)
        assert cached_length == length == 50
    assert counts(cached.stats(), "hits", "misses", "entries") == (1, 1, 1)
    # Three encode calls, two of them through the cache: one posterior encode.
    assert vae.audio_encoder.calls == 2


def test_cache_hits_still_draw_fresh_noise_for_unseeded_requests(vae, audio):
    cached = AuKReferenceEncoder(
        vae, torch.device("cpu"), AuKReferenceIdentity("stub", "cfg")
    )
    first, _ = cached.encode(audio, seed=None)
    second, _ = cached.encode(audio, seed=None)
    assert counts(cached.stats(), "hits", "misses") == (1, 1)
    assert not torch.equal(first, second)
    # Different seeds on a hit also differ; the same seed matches.
    third, _ = cached.encode(audio, seed=1)
    fourth, _ = cached.encode(audio, seed=1)
    fifth, _ = cached.encode(audio, seed=2)
    assert torch.equal(third, fourth)
    assert not torch.equal(third, fifth)


def test_posterior_cache_is_keyed_by_waveform_content(vae, audio):
    cached = AuKReferenceEncoder(
        vae, torch.device("cpu"), AuKReferenceIdentity("stub", "cfg")
    )
    cached.encode(audio, seed=1)
    cached.encode(audio.copy(), seed=1)
    cached.encode(np.concatenate([audio, audio]), seed=1)
    assert counts(cached.stats(), "hits", "misses", "entries") == (1, 2, 2)
    assert AuKReferenceWaveform.from_audio(audio).content_key.startswith("waveform:")


def test_posterior_cache_stores_cpu_copies(vae, audio):
    cached = AuKReferenceEncoder(
        vae, torch.device("cpu"), AuKReferenceIdentity("stub", "cfg")
    )
    cached.encode(audio, seed=1)
    stats = cached.stats()
    assert stats is not None
    # (2 * latent_dim) x frames float32: 8 x 51 x 4 bytes.
    assert stats["bytes"] == 8 * 51 * 4


def test_identity_tracks_checkpoint_weights_and_vae_config(tmp_path: Path):
    checkpoint = tmp_path / "AuK"
    checkpoint.mkdir()
    weights = checkpoint / "vae.safetensors"
    weights.write_bytes(b"v1")
    config = AuKRuntimeConfig(
        model_path=str(checkpoint), vae={"model_init_kwargs": {"latent_dim": 64}}
    )

    base = AuKReferenceIdentity.of(str(checkpoint), config)
    assert base == AuKReferenceIdentity.of(str(checkpoint), config)

    weights.write_bytes(b"v2+")
    reweighted = AuKReferenceIdentity.of(str(checkpoint), config)
    assert reweighted.checkpoint_revision != base.checkpoint_revision
    assert reweighted.vae_config_hash == base.vae_config_hash

    config.vae["model_init_kwargs"]["latent_dim"] = 32
    assert AuKReferenceIdentity.of(str(checkpoint), config).vae_config_hash != (
        base.vae_config_hash
    )
    config.vae["target_sample_rate"] = 48000
    assert AuKReferenceIdentity.of(str(checkpoint), config).vae_config_hash not in {
        base.vae_config_hash,
        reweighted.vae_config_hash,
    }
