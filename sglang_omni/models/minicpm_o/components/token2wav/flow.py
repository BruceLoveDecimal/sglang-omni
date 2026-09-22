# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications: retain MiniCPM-o chunked inference only; local imports and typing.
"""Chunked flow matching for MiniCPM-o with per-row attention caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.components.token2wav.conformer import (
    EncoderCache,
    UpsampleConformerEncoderV2,
)
from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT

# Silence codec token appended as lookahead so a reference cache does not
# depend on the request's generated tokens.
PROMPT_LOOKAHEAD_TOKEN = 4218


@dataclass(kw_only=True)
class FlowCache:
    """Attention and convolution state of one or more rows, left-padded in time.

    Rows with fewer cached tokens than the widest row are padded on the left so
    the newest positions line up; token_lens records each row's valid width.
    """

    encoder: EncoderCache
    estimator_att: torch.Tensor  # (steps, depth, 2, batch, heads, frames, 2 * head_dim)
    estimator_conv: torch.Tensor  # (steps, depth, 2, batch, channels, 2)
    token_lens: list[int]
    prompt_tokens: list[int]
    frame_offsets: list[int]  # frames consumed so far; indexes the fixed noise


@dataclass(kw_only=True)
class FlowCacheRow:
    """One row of a batched cache, kept as a reference to avoid copies."""

    batch: FlowCache
    index: int


def pad_cat(
    tensors: list[torch.Tensor], *, batch_dim: int, time_dim: int
) -> torch.Tensor:
    """Concatenate single-row tensors along batch_dim, left-padding time_dim."""
    width = max(tensor.shape[time_dim] for tensor in tensors)
    padded = []
    for tensor in tensors:
        missing = width - tensor.shape[time_dim]
        if missing:
            pad_shape = list(tensor.shape)
            pad_shape[time_dim] = missing
            tensor = torch.cat([tensor.new_zeros(pad_shape), tensor], dim=time_dim)
        padded.append(tensor)
    return torch.cat(padded, dim=batch_dim)


def flow_cache_row(cache: FlowCache, index: int, up_rate: int) -> FlowCache:
    """View one row of a batched cache without its left padding."""
    tokens = cache.token_lens[index]
    frames = tokens * up_rate
    return FlowCache(
        encoder=EncoderCache(
            token_att=cache.encoder.token_att[:, index : index + 1, :, -tokens:],
            frame_att=cache.encoder.frame_att[:, index : index + 1, :, -frames:],
            lookahead_conv=cache.encoder.lookahead_conv[index : index + 1],
            upsample_conv=cache.encoder.upsample_conv[index : index + 1],
        ),
        estimator_att=cache.estimator_att[:, :, :, index : index + 1, :, -frames:],
        estimator_conv=cache.estimator_conv[:, :, :, index : index + 1],
        token_lens=[tokens],
        prompt_tokens=[cache.prompt_tokens[index]],
        frame_offsets=[cache.frame_offsets[index]],
    )


def stack_flow_caches(rows: list[FlowCacheRow], up_rate: int) -> FlowCache:
    """Batch row references; rows that already form one batch are reused as is."""
    batch = rows[0].batch
    if all(row.batch is batch for row in rows) and [row.index for row in rows] == list(
        range(len(batch.token_lens))
    ):
        return batch
    caches = [flow_cache_row(row.batch, row.index, up_rate) for row in rows]
    return FlowCache(
        encoder=EncoderCache(
            token_att=pad_cat(
                [row.encoder.token_att for row in caches], batch_dim=1, time_dim=3
            ),
            frame_att=pad_cat(
                [row.encoder.frame_att for row in caches], batch_dim=1, time_dim=3
            ),
            lookahead_conv=torch.cat(
                [row.encoder.lookahead_conv for row in caches], dim=0
            ),
            upsample_conv=torch.cat(
                [row.encoder.upsample_conv for row in caches], dim=0
            ),
        ),
        estimator_att=pad_cat(
            [row.estimator_att for row in caches], batch_dim=3, time_dim=5
        ),
        estimator_conv=torch.cat([row.estimator_conv for row in caches], dim=3),
        token_lens=[row.token_lens[0] for row in caches],
        prompt_tokens=[row.prompt_tokens[0] for row in caches],
        frame_offsets=[row.frame_offsets[0] for row in caches],
    )


def trim_flow_cache(cache: FlowCache, *, tail_tokens: int, up_rate: int) -> FlowCache:
    """Keep each reference plus the most recent tail_tokens.

    Rows must share one width and one reference length so the same time slices
    apply to every row; callers split mixed batches into rows first.
    """
    tokens, prompt = cache.token_lens[0], cache.prompt_tokens[0]
    assert set(cache.token_lens) == {tokens} and set(cache.prompt_tokens) == {prompt}
    if tokens <= prompt + tail_tokens:
        return cache
    prompt_frames, tail_frames = prompt * up_rate, tail_tokens * up_rate

    def keep(tensor: torch.Tensor, head: int, tail: int, dim: int) -> torch.Tensor:
        return torch.cat(
            [
                tensor.narrow(dim, 0, head),
                tensor.narrow(dim, tensor.shape[dim] - tail, tail),
            ],
            dim=dim,
        )

    return FlowCache(
        encoder=EncoderCache(
            token_att=keep(cache.encoder.token_att, prompt, tail_tokens, 3),
            frame_att=keep(cache.encoder.frame_att, prompt_frames, tail_frames, 3),
            lookahead_conv=cache.encoder.lookahead_conv,
            upsample_conv=cache.encoder.upsample_conv,
        ),
        estimator_att=keep(cache.estimator_att, prompt_frames, tail_frames, 5),
        estimator_conv=cache.estimator_conv,
        token_lens=[prompt + tail_tokens] * len(cache.token_lens),
        prompt_tokens=cache.prompt_tokens,
        frame_offsets=cache.frame_offsets,
    )


class CausalConditionalCFM(torch.nn.Module):

    def __init__(self, estimator: DiT, inference_cfg_rate: float = 0.7) -> None:
        super().__init__()
        self.estimator = estimator
        self.inference_cfg_rate = inference_cfg_rate
        self.out_channels = estimator.out_channels
        self.register_buffer(
            "rand_noise",
            torch.randn([1, self.out_channels, 50 * 600]),
            persistent=False,
        )

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        *,
        noise_offsets: list[int],
        n_timesteps: int,
        att_cache: torch.Tensor | None,
        conv_cache: torch.Tensor | None,
        key_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Denoise one chunk with classifier-free guidance on the doubled batch.

        Caches carry one entry per Euler step, stacked on the leading axis, with
        the guided and unguided halves on the next axis after the block depth.
        """
        if n_timesteps <= 0:
            raise ValueError("n_timesteps must be positive")
        assert self.inference_cfg_rate > 0, "inference_cfg_rate better > 0"
        batch_size, frames = mu.size(0), mu.size(2)
        if max(noise_offsets) + frames > self.rand_noise.size(2):
            raise ValueError(
                "Combined reference and generated audio exceed 600 seconds"
            )
        x = torch.stack(
            [
                self.rand_noise[0, :, offset : offset + frames]
                for offset in noise_offsets
            ]
        )
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t = t_span[0].expand(batch_size)
        dt = t_span[1] - t_span[0]
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        if key_mask is None:
            mask_in = None
        else:
            new_frames = key_mask.new_ones(batch_size, frames)
            mask_in = torch.cat([key_mask, new_frames], dim=1).repeat(2, 1)
        new_att: torch.Tensor | None = None
        new_conv: torch.Tensor | None = None
        for step in range(1, len(t_span)):
            step_att = None if att_cache is None else att_cache[step - 1].flatten(1, 2)
            step_conv = (
                None if conv_cache is None else conv_cache[step - 1].flatten(1, 2)
            )
            dphi_dt, step_new_conv, step_new_att = self.estimator(
                torch.cat([x, x], dim=0),
                mu_in,
                torch.cat([t, t], dim=0),
                spks_in,
                cond_in,
                conv_cache=step_conv,
                att_cache=step_att,
                key_mask=mask_in,
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [batch_size, batch_size], dim=0)
            dphi_dt = (
                1.0 + self.inference_cfg_rate
            ) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_span[step]
            # Write each step straight into the stacked cache instead of holding
            # every step's copy alive until a final stack.
            step_new_att = step_new_att.unflatten(1, (2, batch_size))
            step_new_conv = step_new_conv.unflatten(1, (2, batch_size))
            if new_att is None or new_conv is None:
                new_att = step_new_att.new_empty((n_timesteps, *step_new_att.shape))
                new_conv = step_new_conv.new_empty((n_timesteps, *step_new_conv.shape))
            new_att[step - 1] = step_new_att
            new_conv[step - 1] = step_new_conv
            del step_new_att, step_new_conv
        assert new_att is not None and new_conv is not None
        return x, new_att, new_conv


class CausalMaskedDiffWithXvec(torch.nn.Module):

    def __init__(
        self,
        encoder: UpsampleConformerEncoderV2,
        decoder: CausalConditionalCFM,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: Literal["mel"] = "mel",
        vocab_size: int = 6561,
    ) -> None:
        super().__init__()
        if output_type != "mel":
            raise ValueError("MiniCPM-o flow output must be mel")
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.pre_lookahead_len = int(encoder.pre_lookahead_layer.pre_lookahead_len)
        self.up_rate = int(encoder.up_layer.stride)
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_dim, output_size)
        self.decoder = decoder

    def speaker_projection(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.spk_embed_affine_layer(F.normalize(embedding, dim=1))

    @torch.inference_mode()
    def prompt_cache(
        self,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int,
    ) -> FlowCache:
        """Run one reference through both stages and keep its state as a single row."""
        assert prompt_token.shape[0] == 1, prompt_token.shape
        prompt_tokens = prompt_token.shape[1]
        assert prompt_feat.shape[1] == prompt_tokens * self.up_rate, (
            prompt_feat.shape,
            prompt_tokens,
        )
        lookahead = torch.full(
            (1, self.pre_lookahead_len),
            PROMPT_LOOKAHEAD_TOKEN,
            dtype=prompt_token.dtype,
            device=prompt_token.device,
        )
        token = self.input_embedding(torch.cat([prompt_token, lookahead], dim=1))
        h, encoder_cache = self.encoder(
            token, last_chunk=False, cache=None, token_key_mask=None
        )
        h = self.encoder_proj(h)
        _, estimator_att, estimator_conv = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            spks=self.speaker_projection(embedding),
            cond=prompt_feat.transpose(1, 2).contiguous(),
            noise_offsets=[0],
            n_timesteps=n_timesteps,
            att_cache=None,
            conv_cache=None,
            key_mask=None,
        )
        return FlowCache(
            encoder=encoder_cache,
            estimator_att=estimator_att,
            estimator_conv=estimator_conv,
            token_lens=[prompt_tokens],
            prompt_tokens=[prompt_tokens],
            frame_offsets=[prompt_tokens * self.up_rate],
        )

    @torch.inference_mode()
    def decode_chunk(
        self,
        token: torch.Tensor,
        embedding: torch.Tensor,
        cache: FlowCache,
        *,
        last_chunk: bool,
        n_timesteps: int,
    ) -> tuple[torch.Tensor, FlowCache]:
        """Generate mel for one chunk per row on top of the batched cache.

        Args:
            token: (batch, tokens); non-final chunks include pre_lookahead_len
                trailing lookahead tokens that produce no frames.
        Returns:
            (batch, mel bins, frames) mel and the cache extended by this chunk.
        """
        assert token.shape[0] == len(cache.token_lens), (token.shape, cache.token_lens)
        width = max(cache.token_lens)
        if min(cache.token_lens) == width:
            key_mask = None
            frame_mask = None
        else:
            positions = torch.arange(width, device=token.device)
            lens = torch.tensor(cache.token_lens, device=token.device)
            key_mask = positions.unsqueeze(0) >= (width - lens).unsqueeze(1)
            frame_mask = key_mask.repeat_interleave(self.up_rate, dim=1)
        h, encoder_cache = self.encoder(
            self.input_embedding(token),
            last_chunk=last_chunk,
            cache=cache.encoder,
            token_key_mask=key_mask,
        )
        h = self.encoder_proj(h)
        frames = h.shape[1]
        feat, estimator_att, estimator_conv = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            spks=self.speaker_projection(embedding),
            cond=torch.zeros_like(h).transpose(1, 2).contiguous(),
            noise_offsets=cache.frame_offsets,
            n_timesteps=n_timesteps,
            att_cache=cache.estimator_att,
            conv_cache=cache.estimator_conv,
            key_mask=frame_mask,
        )
        new_tokens = frames // self.up_rate
        new_cache = FlowCache(
            encoder=encoder_cache,
            estimator_att=estimator_att,
            estimator_conv=estimator_conv,
            token_lens=[length + new_tokens for length in cache.token_lens],
            prompt_tokens=cache.prompt_tokens,
            frame_offsets=[offset + frames for offset in cache.frame_offsets],
        )
        return feat, new_cache
