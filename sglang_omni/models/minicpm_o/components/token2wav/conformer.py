# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2024 Alibaba Inc (Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Conformer for MiniCPM-o."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from sglang_omni.models.minicpm_o.components.token2wav.conformer_layers import (
    ConformerEncoderLayer,
    EspnetRelPositionalEncoding,
    LinearNoSubsampling,
    PositionwiseFeedForward,
    RelPositionMultiHeadedAttention,
)


@dataclass(kw_only=True)
class EncoderCache:
    """Per-chunk state of the encoder, batch-first after the layer axis."""

    token_att: torch.Tensor  # (num_blocks, batch, heads, tokens, 2 * d_k)
    frame_att: torch.Tensor  # (num_up_blocks, batch, heads, frames, 2 * d_k)
    lookahead_conv: torch.Tensor  # (batch, channels, 2)
    upsample_conv: torch.Tensor  # (batch, channels, 2 * stride)


class Upsample1D(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: int,
        stride: int = 2,
        scale_factor: float | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv = nn.Conv1d(
            self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0
        )
        self.scale_factor = (
            float(self.stride) if scale_factor is None else float(scale_factor)
        )

    def forward(
        self, inputs: torch.Tensor, cache: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="nearest")
        if cache is None:
            cache = outputs.new_zeros(
                outputs.shape[0], outputs.shape[1], self.stride * 2
            )
        outputs = torch.cat([cache, outputs], dim=2)
        new_cache = outputs[..., -self.stride * 2 :]
        return self.conv(outputs), new_cache


class PreLookaheadLayer(nn.Module):

    def __init__(self, channels: int, pre_lookahead_len: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=0)

    def forward(
        self, inputs: torch.Tensor, cache: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume pre_lookahead_len trailing tokens as lookahead context."""
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.leaky_relu(self.conv1(outputs))
        if cache is None:
            cache = outputs.new_zeros(outputs.shape[0], outputs.shape[1], 2)
        new_cache = outputs[..., -2:]
        outputs = torch.cat([cache, outputs], dim=2)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = outputs + inputs[:, : -self.pre_lookahead_len]
        return outputs, new_cache


class UpsampleConformerEncoderV2(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        input_layer: Literal["linear"] = "linear",
        pre_lookahead_len: int = 3,
        num_blocks: int = 6,
        num_up_blocks: int = 4,
        up_stride: int = 2,
        up_scale_factor: float = 2,
        attention_heads: int = 4,
        pos_enc_layer_type: Literal["rel_pos_espnet"] = "rel_pos_espnet",
        selfattention_layer_type: Literal["rel_selfattn"] = "rel_selfattn",
        key_bias: bool = True,
        linear_units: int = 2048,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        activation_type: Literal["swish"] = "swish",
    ) -> None:
        super().__init__()
        if (
            input_layer,
            pos_enc_layer_type,
            selfattention_layer_type,
            activation_type,
        ) != ("linear", "rel_pos_espnet", "rel_selfattn", "swish"):
            raise ValueError("Unsupported MiniCPM-o flow encoder configuration")
        self.output_dim = output_size
        self.embed = LinearNoSubsampling(
            input_size,
            output_size,
            dropout_rate,
            EspnetRelPositionalEncoding(output_size, positional_dropout_rate),
        )
        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-05)
        activation = nn.SiLU()
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        positionwise_layer_args = (output_size, linear_units, dropout_rate, activation)
        self.pre_lookahead_layer = PreLookaheadLayer(
            channels=output_size, pre_lookahead_len=pre_lookahead_len
        )
        self.encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(num_blocks)
            ]
        )
        self.up_layer = Upsample1D(
            channels=output_size,
            out_channels=output_size,
            stride=up_stride,
            scale_factor=up_scale_factor,
        )
        self.up_embed = LinearNoSubsampling(
            input_size,
            output_size,
            dropout_rate,
            EspnetRelPositionalEncoding(output_size, positional_dropout_rate),
        )
        self.up_encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(num_up_blocks)
            ]
        )

    def forward(
        self,
        xs: torch.Tensor,
        *,
        last_chunk: bool,
        cache: EncoderCache | None,
        token_key_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, EncoderCache]:
        """Encode one chunk of token embeddings on top of the cached history.

        Args:
            xs: (batch, tokens, input_size); non-final chunks carry
                pre_lookahead_len extra trailing tokens that are consumed here.
            token_key_mask: (batch, cached tokens) validity of left-padded cache
                positions, or None when every row's cache is full.
        """
        batch, tokens = xs.shape[0], xs.shape[1]
        lookahead = self.pre_lookahead_layer.pre_lookahead_len
        out_tokens = tokens if last_chunk else tokens - lookahead
        cached_tokens = 0 if cache is None else cache.token_att.shape[3]
        xs, pos_emb = self.embed(xs, cached_tokens + out_tokens)
        if last_chunk:
            xs = F.pad(xs, (0, 0, 0, lookahead))
        xs, lookahead_conv = self.pre_lookahead_layer(
            xs, None if cache is None else cache.lookahead_conv
        )
        if token_key_mask is None:
            token_mask = None
            frame_mask = None
        else:
            new_tokens = token_key_mask.new_ones(batch, out_tokens)
            token_mask = torch.cat([token_key_mask, new_tokens], dim=1).unsqueeze(1)
            stride = self.up_layer.stride
            frame_mask = torch.cat(
                [
                    token_key_mask.repeat_interleave(stride, dim=1),
                    token_key_mask.new_ones(batch, out_tokens * stride),
                ],
                dim=1,
            ).unsqueeze(1)
        token_att = []
        for idx, layer in enumerate(self.encoders):
            xs, layer_cache = layer(
                xs, token_mask, pos_emb, None if cache is None else cache.token_att[idx]
            )
            token_att.append(layer_cache)
        xs = xs.transpose(1, 2).contiguous()
        xs, upsample_conv = self.up_layer(
            xs, None if cache is None else cache.upsample_conv
        )
        xs = xs.transpose(1, 2).contiguous()
        cached_frames = 0 if cache is None else cache.frame_att.shape[3]
        xs, pos_emb = self.up_embed(xs, cached_frames + xs.shape[1])
        frame_att = []
        for idx, layer in enumerate(self.up_encoders):
            xs, layer_cache = layer(
                xs, frame_mask, pos_emb, None if cache is None else cache.frame_att[idx]
            )
            frame_att.append(layer_cache)
        if self.normalize_before:
            xs = self.after_norm(xs)
        new_cache = EncoderCache(
            token_att=torch.stack(token_att),
            frame_att=torch.stack(frame_att),
            lookahead_conv=lookahead_conv,
            upsample_conv=upsample_conv,
        )
        return xs, new_cache
