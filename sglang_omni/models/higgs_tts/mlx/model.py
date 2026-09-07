# SPDX-License-Identifier: Apache-2.0
"""Higgs audio embeddings/head around mlx-lm's Qwen3 backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

from sglang_omni.models.higgs_tts.weight_loader import DiscreteWeightMapper


@dataclass
class ModelConfig:
    text_config: dict[str, Any]
    audio_encoder_config: dict[str, Any]
    audio_token_id: int = -100
    model_type: str = "higgs_multimodal_qwen3"

    @classmethod
    def from_dict(cls, config):
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class HiggsRotaryEmbedding(nn.Module):
    """Match Higgs' BF16-truncated training-time cosine/sine tables."""

    def __init__(self, dims, base):
        super().__init__()
        self.dims = dims
        self.base = base
        self.traditional = False
        self.scale = 1.0

    def __call__(self, x, offset=0):
        positions = mx.arange(x.shape[-2], dtype=mx.float32) + offset
        frequencies = self.base ** (
            -mx.arange(0, self.dims, 2, dtype=mx.float32) / self.dims
        )
        angles = positions[:, None] * frequencies[None, :]
        cos = mx.cos(angles).astype(mx.bfloat16).astype(x.dtype)
        sin = mx.sin(angles).astype(mx.bfloat16).astype(x.dtype)
        first, second = mx.split(x, 2, axis=-1)
        return mx.concatenate(
            [first * cos - second * sin, second * cos + first * sin], axis=-1
        )


class HiggsMlxModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        text = dict(config.text_config)
        rope = text.get("rope_parameters") or {}
        text["rope_theta"] = text.get("rope_theta") or rope.get("rope_theta", 1e6)
        if rope.get("rope_type", "default") != "default":
            text["rope_scaling"] = rope
        self.args = ModelArgs.from_dict(text)
        if self.args.rope_scaling:
            raise ValueError("Higgs MLX currently requires default Qwen3 RoPE")
        enc = config.audio_encoder_config
        if enc.get("encoder_type", "discrete") != "discrete":
            raise ValueError("Higgs MLX requires a discrete TTS checkpoint")
        self.num_codebooks = int(enc["num_codebooks"])
        self.codebook_size = int(enc["vocab_size"])
        if self.num_codebooks != 8 or self.codebook_size != 1026:
            raise ValueError("Higgs Audio v3 requires 8 codebooks of size 1026")
        if int(enc.get("out_dim", self.args.hidden_size)) != self.args.hidden_size:
            raise ValueError("Audio embedding dimension must match the Qwen3 backbone")
        self.tie_modality = bool(enc.get("tie_word_embeddings", True))
        self.model = Qwen3Model(self.args)
        for layer in self.model.layers:
            layer.self_attn.rope = HiggsRotaryEmbedding(
                self.args.head_dim, self.args.rope_theta
            )
        self.audio_embedding = nn.Embedding(
            self.num_codebooks * self.codebook_size, self.args.hidden_size
        )
        if not self.tie_modality:
            self.audio_head = nn.Linear(
                self.args.hidden_size,
                self.num_codebooks * self.codebook_size,
                bias=False,
            )

    @property
    def layers(self):
        return self.model.layers

    def embed_codes(self, codes):
        offsets = mx.arange(self.num_codebooks) * self.codebook_size
        return self.audio_embedding(codes + offsets).sum(axis=-2)

    def prompt_embeddings(self, token_ids, reference_codes):
        if not token_ids:
            raise ValueError("Higgs MLX requires a nonempty prompt")
        positions = [
            i
            for i, token in enumerate(token_ids)
            if token == self.config.audio_token_id
        ]
        references = reference_codes or []
        if len(positions) != len(references):
            raise ValueError("Higgs reference codes must match audio placeholders")
        if any(
            t != self.config.audio_token_id and not 0 <= t < self.args.vocab_size
            for t in token_ids
        ):
            raise ValueError("Higgs prompt contains an invalid text token")
        ids = mx.array(
            [[0 if t == self.config.audio_token_id else t for t in token_ids]]
        )
        embeds = self.model.embed_tokens(ids)
        if references:
            if any(
                len(row) != self.num_codebooks
                or any(not 0 <= c < self.codebook_size for c in row)
                for row in references
            ):
                raise ValueError("Invalid Higgs reference codebook shape or token")
            embeds[:, mx.array(positions), :] = self.embed_codes(mx.array([references]))
        return embeds

    def __call__(self, inputs=None, cache=None, input_embeddings=None):
        hidden = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        hidden = hidden[:, -1:, :]
        logits = (
            self.audio_embedding.as_linear(hidden)
            if self.tie_modality
            else self.audio_head(hidden)
        )
        return logits.reshape(hidden.shape[0], self.num_codebooks, self.codebook_size)

    def sanitize(self, weights):
        mapper = DiscreteWeightMapper(
            text_prefix_map={
                "tied.embedding.text_embedding.": "model.embed_tokens.",
                "body.layers.": "model.layers.",
                "body.norm.": "model.norm.",
            },
            embedding_dest="audio_embedding.",
            head_dest="audio_head.",
            tie_modality=self.tie_modality,
        )
        result = {}
        for name, value in weights.items():
            # The text head is unused in the TTS graph, even when untied.
            if name.startswith("tied.head.text_head."):
                continue
            if self.tie_modality and name.startswith("tied.head.modality_heads.0."):
                continue
            mapped = mapper.map(name)
            if mapped is not None:
                result[mapped] = value
        return result
