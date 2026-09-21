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
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Flow for MiniCPM-o."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.components.token2wav.conformer import (
    UpsampleConformerEncoderV2,
    make_pad_mask,
)
from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT


@dataclass
class FlowChunkCache:
    """Encoder and estimator context carried from one chunk to the next."""

    conformer_cnn: torch.Tensor
    conformer_att: torch.Tensor
    estimator_cnn: torch.Tensor
    estimator_att: torch.Tensor

    def truncate(self, prompt_frames: int, keep_frames: int) -> None:
        """Bound the attention caches to the prompt plus the most recent frames.

        The estimator cache lists newest frames first, so its kept head is the
        latest generated context while its tail is the end of the prompt.
        """
        limit = prompt_frames + keep_frames
        if self.estimator_att.shape[4] > limit:
            self.estimator_att = torch.cat(
                [
                    self.estimator_att[:, :, :, :, :prompt_frames],
                    self.estimator_att[:, :, :, :, -keep_frames:],
                ],
                dim=4,
            )
        if self.conformer_att.shape[3] > limit:
            self.conformer_att = torch.cat(
                [
                    self.conformer_att[:, :, :, :prompt_frames],
                    self.conformer_att[:, :, :, -keep_frames:],
                ],
                dim=3,
            )


def cosine_time_span(n_timesteps: int, reference: torch.Tensor) -> torch.Tensor:
    if n_timesteps <= 0:
        raise ValueError("n_timesteps must be positive")
    t_span = torch.linspace(
        0, 1, n_timesteps + 1, device=reference.device, dtype=reference.dtype
    )
    return 1 - torch.cos(t_span * 0.5 * torch.pi)


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

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        t = t_span[0].expand(batch_size)
        dt = t_span[1] - t_span[0]
        assert self.inference_cfg_rate > 0, "inference_cfg_rate better > 0"
        mask_in = torch.cat([mask, mask], dim=0)
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        for step in range(1, len(t_span)):
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            dphi_dt = self.estimator.forward(
                x_in, mask_in, mu_in, t_in, spks_in, cond_in
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = (
                1.0 + self.inference_cfg_rate
            ) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_span[step]
        return x

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if mu.size(2) > self.rand_noise.size(2):
            raise ValueError(
                "Combined reference and generated audio exceed 600 seconds"
            )
        z = (
            self.rand_noise[:, :, : mu.size(2)].expand(mu.size(0), -1, -1).clone()
            * temperature
        )
        t_span = cosine_time_span(n_timesteps, mu)
        return self.solve_euler(z, t_span, mu, mask, spks, cond)

    def solve_euler_chunk(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve one chunk against per-timestep caches of the preceding chunks.

        Args:
            cnn_cache: (n_timesteps, depth, 2, channels, kernel - 1) or None.
            att_cache: (n_timesteps, depth, 2, heads, cached_frames, 2 * head_dim)
                or None for the first chunk.
        Returns:
            The chunk's mel and the caches extended by this chunk's frames.
        """
        assert self.inference_cfg_rate > 0, "inference_cfg_rate better > 0"
        n_timesteps = len(t_span) - 1
        t = t_span[0].unsqueeze(0)
        dt = t_span[1] - t_span[0]
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        empty_cnn, empty_att = self.estimator.empty_chunk_caches(2, mu.device)
        new_cnn_cache = empty_cnn.new_empty((n_timesteps, *empty_cnn.shape))
        cached_frames = 0 if att_cache is None else att_cache.shape[4]
        new_att_cache = empty_att.new_empty(
            (
                n_timesteps,
                *empty_att.shape[:3],
                cached_frames + x.shape[2],
                empty_att.shape[4],
            )
        )
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator.forward_chunk(
                x.repeat(2, 1, 1),
                mu_in,
                t.repeat(2),
                spks_in,
                cond_in,
                empty_cnn if cnn_cache is None else cnn_cache[step - 1],
                empty_att if att_cache is None else att_cache[step - 1],
                new_cnn_cache[step - 1],
                new_att_cache[step - 1],
            )
            dphi_dt, cfg_dphi_dt = dphi_dt.chunk(2, dim=0)
            dphi_dt = (
                1.0 + self.inference_cfg_rate
            ) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x, new_cnn_cache, new_att_cache

    @torch.inference_mode()
    def forward_chunk(
        self,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        n_timesteps: int,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate one chunk of mel; the noise offset follows the cached length."""
        assert mu.shape[0] == 1, f"chunked flow runs one stream, got {mu.shape[0]}"
        offset = 0 if att_cache is None else att_cache.shape[4]
        if offset + mu.size(2) > self.rand_noise.size(2):
            raise ValueError("Cached and generated audio exceed 600 seconds")
        z = self.rand_noise[:, :, offset : offset + mu.size(2)]
        t_span = cosine_time_span(n_timesteps, mu)
        return self.solve_euler_chunk(z, t_span, mu, spks, cond, cnn_cache, att_cache)


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

    @torch.inference_mode()
    def inference(
        self,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        assert token.shape[0] == prompt_token.shape[0], (
            f"flow batch size mismatch: token={token.shape[0]} "
            f"prompt_token={prompt_token.shape[0]}"
        )
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        token_len = prompt_token_len + token_len
        token = torch.concat([prompt_token, token], dim=1)
        token_mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * token_mask
        h, _ = self.encoder.forward(token, token_len)
        frame_mask = (~make_pad_mask(token_len * self.up_rate, h.shape[1])).to(h)
        h = self.encoder_proj(h) * frame_mask.unsqueeze(-1)
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - prompt_feat.shape[1]
        conds = torch.zeros_like(h)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2).contiguous()
        feat = self.decoder.forward(
            mu=h.transpose(1, 2).contiguous(),
            mask=frame_mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat

    @torch.inference_mode()
    def setup_chunk_cache(
        self,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int,
    ) -> FlowChunkCache:
        """Prime a stream's caches with the reference prompt.

        Args:
            prompt_token: (1, tokens + pre_lookahead_len) prompt codec tokens
                followed by lookahead tokens.
            prompt_feat: (1, tokens * up_rate, mel) prompt mel.
        """
        prompt_tokens = prompt_token.shape[1] - self.pre_lookahead_len
        assert prompt_tokens * self.up_rate == prompt_feat.shape[1], (
            f"prompt mel frames {prompt_feat.shape[1]} do not match "
            f"{prompt_tokens} prompt tokens"
        )
        embedding = self.spk_embed_affine_layer(F.normalize(embedding, dim=1))
        h, conformer_cnn, conformer_att = self.encoder.forward_chunk(
            self.input_embedding(prompt_token), False, None, None
        )
        h = self.encoder_proj(h)
        _, estimator_cnn, estimator_att = self.decoder.forward_chunk(
            mu=h.transpose(1, 2).contiguous(),
            spks=embedding,
            cond=prompt_feat.transpose(1, 2).contiguous(),
            n_timesteps=n_timesteps,
            cnn_cache=None,
            att_cache=None,
        )
        return FlowChunkCache(
            conformer_cnn, conformer_att, estimator_cnn, estimator_att
        )

    @torch.inference_mode()
    def inference_chunk(
        self,
        token: torch.Tensor,
        embedding: torch.Tensor,
        cache: FlowChunkCache,
        last_chunk: bool,
        n_timesteps: int,
    ) -> tuple[torch.Tensor, FlowChunkCache]:
        """Generate mel for one token chunk and return the advanced caches.

        A non-final chunk ends with pre_lookahead_len lookahead tokens that
        produce no frames; the last chunk is zero padded instead.
        """
        embedding = self.spk_embed_affine_layer(F.normalize(embedding, dim=1))
        h, conformer_cnn, conformer_att = self.encoder.forward_chunk(
            self.input_embedding(token),
            last_chunk,
            cache.conformer_cnn,
            cache.conformer_att,
        )
        h = self.encoder_proj(h)
        feat, estimator_cnn, estimator_att = self.decoder.forward_chunk(
            mu=h.transpose(1, 2).contiguous(),
            spks=embedding,
            cond=torch.zeros_like(h).transpose(1, 2).contiguous(),
            n_timesteps=n_timesteps,
            cnn_cache=cache.estimator_cnn,
            att_cache=cache.estimator_att,
        )
        return feat, FlowChunkCache(
            conformer_cnn, conformer_att, estimator_cnn, estimator_att
        )
