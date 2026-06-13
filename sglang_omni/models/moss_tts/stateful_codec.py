# SPDX-License-Identifier: Apache-2.0
"""Stateful streaming codec wrapper for MOSS-TTS.

Builds the vendored :class:`MossAudioTokenizerModel` (which carries the streaming
KV-cache decode) from an already-loaded MOSS processor, transplanting the remote
codec's weights with name-remapping. See
``docs/design/moss_tts_stateful_codec_streaming_rfc.en.md`` §5.3/§5.6 and the
``_vendored/moss_audio_tokenizer/README.md`` for provenance.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.models.moss_tts._vendored.moss_audio_tokenizer import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerModel,
    StreamingCache,
)

logger = logging.getLogger(__name__)

# Upstream/remote module names → vendored module names (RFC §5.6). The vendored
# ``_Attention`` also carries a load-state-dict pre-hook for the in_proj/out_proj
# legacy names; this list covers the remaining transformer/quantizer submodules.
_SUFFIX_REMAP: list[tuple[str, str]] = [
    (".self_attn.in_projs.0.", ".attn.in_proj."),
    (".self_attn.out_projs.0.", ".attn.out_proj."),
    (".self_attn.", ".attn."),
    (".linear1.", ".ff1."),
    (".linear2.", ".ff2."),
    (".layer_scale_1.", ".ls1."),
    (".layer_scale_2.", ".ls2."),
    (".input_proj.", ".in_proj."),
    (".output_proj.", ".out_proj."),
]

_MIN_LOAD_FRACTION = 0.95


def _remap(name: str) -> str:
    for src, dst in _SUFFIX_REMAP:
        if src in name:
            return name.replace(src, dst)
    return name


def _config_from_remote(remote_cfg: Any) -> MossAudioTokenizerConfig:
    """Reconstruct a vendored config from the remote audio-tokenizer config."""
    kwargs: dict[str, Any] = {}
    for attr in (
        "version",
        "sampling_rate",
        "downsample_rate",
        "causal_transformer_context_duration",
        "encoder_kwargs",
        "decoder_kwargs",
        "quantizer_type",
        "quantizer_kwargs",
    ):
        value = getattr(remote_cfg, attr, None)
        if value is not None:
            kwargs[attr] = value
    return MossAudioTokenizerConfig(**kwargs)


def build_stateful_codec_from_processor(
    processor: Any,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> MossAudioTokenizerModel:
    """Build the vendored streaming codec and load weights from the processor."""
    remote = getattr(processor, "audio_tokenizer", None)
    if remote is None:
        raise RuntimeError(
            "MOSS-TTS stateful codec requires processor.audio_tokenizer to be loaded"
        )
    remote_cfg = getattr(remote, "config", None)
    if remote_cfg is None:
        raise RuntimeError("MOSS-TTS remote audio_tokenizer has no config")

    cfg = _config_from_remote(remote_cfg)
    model = MossAudioTokenizerModel(cfg)

    remote_state = remote.state_dict()
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    targets = {**params, **buffers}

    loaded = 0
    skipped: list[str] = []
    own = model.state_dict()
    new_state: dict[str, torch.Tensor] = {}
    for name, tensor in remote_state.items():
        tgt = name if name in targets else _remap(name)
        if tgt in own and own[tgt].shape == tensor.shape:
            new_state[tgt] = tensor
            loaded += 1
        else:
            skipped.append(name)

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    frac = loaded / max(len(own), 1)
    logger.info(
        "MOSS stateful codec weights: loaded=%d/%d (%.1f%%) skipped=%d missing=%d "
        "(first skipped: %s)",
        loaded,
        len(own),
        100.0 * frac,
        len(skipped),
        len(missing),
        skipped[:3] if skipped else "none",
    )
    if frac < _MIN_LOAD_FRACTION:
        raise RuntimeError(
            f"MOSS stateful codec only loaded {frac:.1%} of weights "
            f"({loaded}/{len(own)}); the _SUFFIX_REMAP in stateful_codec.py likely "
            f"needs updating for this checkpoint. First skipped: {skipped[:5]}"
        )

    model.to(device=device, dtype=dtype)
    model.eval()
    return model


class StatefulMossCodec:
    """Thin per-stream adapter over the vendored streaming codec.

    Accepts de-delayed audio-code frames as ``[T, n_vq]`` (the layout produced by
    the de-delay path) and decodes incrementally, threading a per-stream
    :class:`StreamingCache`.
    """

    def __init__(self, model: MossAudioTokenizerModel, *, n_vq: int) -> None:
        self._model = model
        self._n_vq = int(n_vq)
        self._device = next(model.parameters()).device

    @property
    def downsample_rate(self) -> int:
        return int(self._model.downsample_rate)

    def new_state(self) -> StreamingCache:
        return StreamingCache()

    @torch.no_grad()
    def decode_stream(
        self,
        frames: torch.Tensor,
        state: StreamingCache,
        *,
        last_chunk: bool = False,
    ) -> torch.Tensor:
        """Decode ``frames`` (``[T, n_vq]``) → 1-D waveform delta for those frames."""
        if frames.ndim != 2:
            raise ValueError(f"frames must be [T, n_vq], got {tuple(frames.shape)}")
        codes = frames.to(device=self._device, dtype=torch.long).transpose(0, 1)  # (n_vq, T)
        out, _ = self._model.decode_stream(
            codes, state=state, last_chunk=last_chunk, num_quantizers=self._n_vq
        )
        if out.audio is None:
            return torch.zeros(0, dtype=torch.float32)
        return out.audio.reshape(-1).to(torch.float32).cpu()


__all__ = ["build_stateful_codec_from_processor", "StatefulMossCodec"]
