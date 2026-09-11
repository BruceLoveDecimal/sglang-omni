# SPDX-License-Identifier: Apache-2.0
"""Content-keyed caches for AuK reference audio.

Voice-cloning traffic tends to reuse a small set of speaker references, so two
request-independent artifacts are shared across requests through the common
``ReferenceEncodeService`` (content-keyed LRU, byte budget, same-key
single-flight, hit/miss statistics):

* ``AuKReferenceLoader`` (preprocessing stage) keeps the decoded waveforms at
  the VAE and Qwen sample rates, keyed by the bytes or file content that was
  actually consumed. URL strings are never used as keys: remote content is
  downloaded and hashed, local files are hashed and revalidated by stat.
* ``AuKReferenceEncoder`` (conditioning stage) keeps the VAE posterior
  statistics, keyed by the waveform the encoder consumed plus the checkpoint,
  VAE configuration and sample rate.

The posterior (``mean`` and ``log_std``) is cached rather than the sampled
latent because every request draws its own noise:
``latent = mean + noise * exp(log_std)``. Sampling after the cache lookup keeps
unseeded requests random and seeded requests reproducible.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import numpy as np
import torch

from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.flow_matching import request_generator
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.models.auk.vae import BigVGANFlowVAE
from sglang_omni.models.auk.weight_loader import resolve_vae_file
from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key
from sglang_omni.scheduling.reference_encoder import (
    KeyedReferenceEncodeHook,
    ReferenceEncodeService,
    TensorReferenceEncodeHook,
)
from sglang_omni.utils.audio import audio_fingerprint, decode_audio_data_uri, load_audio

DEFAULT_AUDIO_CACHE_MAX_ITEMS = 256
# Two float32 waveforms at 24 kHz and 16 kHz cost ~160 KiB per second, so this
# holds a few dozen 10-second references.
DEFAULT_AUDIO_CACHE_MAX_BYTES = 128 * 1024 * 1024
DEFAULT_POSTERIOR_CACHE_MAX_ITEMS = 256
# A posterior is 2 * latent_dim float32 values per 20 ms frame, ~25 KiB per
# second of reference audio.
DEFAULT_POSTERIOR_CACHE_MAX_BYTES = 64 * 1024 * 1024

_REMOTE_TIMEOUT_S = 5


# --------------------------------------------------------------------------- #
# Preprocessing: download, decode and resample once per content
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AuKReferenceSource:
    """A reference after URI resolution: the raw bytes or a local file path."""

    payload: bytes | str
    desc: str


@dataclass(frozen=True)
class AuKReferenceAudio:
    """One reference decoded for both consumers."""

    vae_audio: np.ndarray
    """float32 mono waveform at the VAE sample rate."""
    qwen_audio: np.ndarray
    """float32 mono waveform at the Qwen2.5-Omni audio sample rate."""


def resolve_reference_source(source: object) -> AuKReferenceSource:
    """Turn a request reference into bytes or a local path without decoding it."""
    if isinstance(source, memoryview):
        source = source.tobytes()
    if isinstance(source, bytearray):
        source = bytes(source)
    if isinstance(source, bytes):
        return AuKReferenceSource(source, "bytes")
    if not isinstance(source, str):
        raise ValueError(
            f"Unsupported AuK reference audio input: {type(source).__name__}"
        )
    decoded = decode_audio_data_uri(source)
    if decoded is not None:
        return AuKReferenceSource(decoded, "data-URI")
    if source.startswith(("http://", "https://")):
        import httpx

        response = httpx.get(source, timeout=_REMOTE_TIMEOUT_S, follow_redirects=True)
        response.raise_for_status()
        return AuKReferenceSource(response.content, repr(source))
    if source.startswith("file://"):
        return AuKReferenceSource(unquote(urlparse(source).path), repr(source))
    return AuKReferenceSource(source, repr(source))


def decode_reference_audio(payload: bytes | str, sample_rate: int) -> AuKReferenceAudio:
    """Decode and resample one resolved reference for the VAE and Qwen."""
    import librosa

    vae_audio = load_audio(payload, source_name="AuK", target_sample_rate=sample_rate)
    qwen_audio, _ = librosa.load(
        io.BytesIO(payload) if isinstance(payload, bytes) else payload,
        sr=C.QWEN_AUDIO_SAMPLE_RATE,
        mono=True,
    )
    return AuKReferenceAudio(
        vae_audio=np.ascontiguousarray(vae_audio, dtype=np.float32).reshape(-1),
        qwen_audio=np.ascontiguousarray(qwen_audio, dtype=np.float32).reshape(-1),
    )


class _ReferenceAudioHook(
    KeyedReferenceEncodeHook[
        AuKReferenceSource, AuKReferenceAudio, dict[str, torch.Tensor]
    ]
):
    model_id = "auk"
    model_revision = ""
    encoder_id = "auk_reference_audio"
    artifact_kind = "resampled_waveforms_v1"

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = int(sample_rate)
        self.encoder_config_hash = (
            f"vae_sr{self._sample_rate}:qwen_sr{C.QWEN_AUDIO_SAMPLE_RATE}"
        )

    def input_key(self, item: AuKReferenceSource) -> str | None:
        if isinstance(item.payload, bytes):
            return f"bytes:{hash_bytes(item.payload)}"
        # Full-content hash, memoized and revalidated by (size, mtime, ctime).
        # Unreadable paths bypass the cache so the decoder reports the error.
        return reference_path_cache_key(item.payload, trust_stat=False)

    def encode_one(self, item: AuKReferenceSource) -> AuKReferenceAudio:
        return decode_reference_audio(item.payload, self._sample_rate)

    # Tensors rather than arrays so the cache's byte budget counts them.
    def store_artifact(self, artifact: AuKReferenceAudio) -> dict[str, torch.Tensor]:
        return {
            "vae_audio": torch.from_numpy(artifact.vae_audio).clone(),
            "qwen_audio": torch.from_numpy(artifact.qwen_audio).clone(),
        }

    def load_artifact(self, stored: dict[str, torch.Tensor]) -> AuKReferenceAudio:
        return AuKReferenceAudio(
            vae_audio=stored["vae_audio"].numpy().copy(),
            qwen_audio=stored["qwen_audio"].numpy().copy(),
        )


class AuKReferenceLoader:
    """Resolve a reference source, then decode it at most once per content."""

    def __init__(
        self,
        sample_rate: int,
        *,
        cache: bool = True,
        max_items: int | None = DEFAULT_AUDIO_CACHE_MAX_ITEMS,
        max_bytes: int | None = DEFAULT_AUDIO_CACHE_MAX_BYTES,
    ) -> None:
        self._sample_rate = int(sample_rate)
        self._service: (
            ReferenceEncodeService[
                AuKReferenceSource, AuKReferenceAudio, dict[str, torch.Tensor]
            ]
            | None
        ) = None
        if cache:
            self._service = ReferenceEncodeService(
                _ReferenceAudioHook(self._sample_rate),
                max_items=max_items,
                max_bytes=max_bytes,
                log_prefix="AuK reference audio cache",
            )

    def load(self, source: object) -> AuKReferenceAudio:
        item = resolve_reference_source(source)
        if self._service is None:
            return decode_reference_audio(item.payload, self._sample_rate)
        return self._service.get_or_encode(item, desc=item.desc)

    def stats(self) -> dict[str, int] | None:
        return None if self._service is None else self._service.stats()

    def close(self) -> None:
        if self._service is not None:
            self._service.close()


# --------------------------------------------------------------------------- #
# Conditioning: VAE posterior once per waveform, fresh noise per request
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AuKReferenceWaveform:
    """The exact float32 waveform handed to the VAE encoder, with its identity."""

    audio: np.ndarray
    content_key: str

    @classmethod
    def from_audio(cls, audio: np.ndarray) -> AuKReferenceWaveform:
        waveform = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        return cls(waveform, f"waveform:{audio_fingerprint(waveform)}")


@dataclass(frozen=True)
class AuKReferenceIdentity:
    """Everything a cached posterior depends on besides the waveform content."""

    checkpoint_revision: str
    vae_config_hash: str

    @classmethod
    def of(cls, checkpoint: str, config: AuKRuntimeConfig) -> AuKReferenceIdentity:
        revision = str(checkpoint)
        vae_file = resolve_vae_file(checkpoint)
        if vae_file is not None:
            stat = vae_file.stat()
            revision = f"{revision}:{stat.st_size}:{stat.st_mtime_ns}"
        spec = {
            "vae": config.vae_init_kwargs,
            "sample_rate": config.sample_rate,
            "downsample_rate": config.downsample_rate,
        }
        digest = hash_bytes(json.dumps(spec, sort_keys=True, default=str).encode())
        return cls(checkpoint_revision=revision, vae_config_hash=digest)


class _PosteriorHook(TensorReferenceEncodeHook[AuKReferenceWaveform]):
    model_id = "auk"
    encoder_id = "auk_bigvgan_flow_vae_encoder"
    # Bump when the stored statistics or their layout change.
    artifact_kind = "vae_posterior_v1"
    storage_dtype = torch.float32
    output_dtype = torch.float32

    def __init__(
        self, encoder: AuKReferenceEncoder, identity: AuKReferenceIdentity
    ) -> None:
        self._encoder = encoder
        self.model_revision = identity.checkpoint_revision
        self.encoder_config_hash = identity.vae_config_hash

    def input_key(self, item: AuKReferenceWaveform) -> str:
        return item.content_key

    def encode_one(self, item: AuKReferenceWaveform) -> torch.Tensor:
        return self._encoder.posterior(item)


class AuKReferenceEncoder:
    """Encode reference waveforms to normalized latents through the VAE.

    With an identity the deterministic posterior is cached across requests;
    the noise draw and normalization always run per request.
    """

    def __init__(
        self,
        vae: BigVGANFlowVAE,
        device: torch.device,
        identity: AuKReferenceIdentity | None = None,
        *,
        max_items: int | None = DEFAULT_POSTERIOR_CACHE_MAX_ITEMS,
        max_bytes: int | None = DEFAULT_POSTERIOR_CACHE_MAX_BYTES,
    ) -> None:
        self._vae = vae
        self._device = device
        self._service: (
            ReferenceEncodeService[AuKReferenceWaveform, torch.Tensor, torch.Tensor]
            | None
        ) = None
        if identity is not None:
            self._service = ReferenceEncodeService(
                _PosteriorHook(self, identity),
                max_items=max_items,
                max_bytes=max_bytes,
                log_prefix="AuK reference posterior cache",
            )

    def posterior(self, item: AuKReferenceWaveform) -> torch.Tensor:
        """Posterior statistics ``(2 * latent_dim, frames)`` on the device."""
        waveform = torch.from_numpy(item.audio).reshape(1, 1, -1).to(self._device)
        return self._vae.encode_posterior(waveform)[0]

    def encode(self, audio: np.ndarray, seed: int | None) -> tuple[torch.Tensor, int]:
        """Normalized latent ``(frames, latent_dim)`` and its valid frame count."""
        item = AuKReferenceWaveform.from_audio(audio)
        if self._service is None:
            stats = self.posterior(item)
        else:
            stats = self._service.get_or_encode(item, desc=item.content_key)
        stats = stats.to(self._device).unsqueeze(0)
        hop = self._vae.hop_size
        lengths = torch.tensor([item.audio.shape[-1] // hop * hop], device=self._device)
        latent, lengths = self._vae.sample_and_normalize(
            stats, lengths, generator=request_generator(seed, self._device)
        )
        return latent[0], int(lengths[0])

    def stats(self) -> dict[str, int] | None:
        return None if self._service is None else self._service.stats()

    def close(self) -> None:
        if self._service is not None:
            self._service.close()
