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
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return (outputs, input_lengths * self.stride)

    def forward_chunk(
        self, inputs: torch.Tensor, cache: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Upsample one chunk; the cache holds the last 2 * stride upsampled frames."""
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.pad(
            outputs, (0, self.pre_lookahead_len), mode="constant", value=0.0
        )
        outputs = F.leaky_relu(self.conv1(outputs))
        outputs = F.pad(outputs, (2, 0), mode="constant", value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = outputs + inputs
        return outputs

    def forward_chunk(
        self, inputs: torch.Tensor, cache: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume the chunk's trailing lookahead frames instead of zero padding.

        The output is pre_lookahead_len frames shorter than the input; the
        cache holds the two conv2 context frames for the next chunk.
        """
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.leaky_relu(self.conv1(outputs))
        if cache is None:
            cache = outputs.new_zeros(outputs.shape[0], outputs.shape[1], 2)
        new_cache = outputs[..., -2:]
        outputs = torch.cat([cache, outputs], dim=2)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        return outputs + inputs[:, : -self.pre_lookahead_len], new_cache


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
        self, xs: torch.Tensor, xs_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        xs, pos_emb, masks = self.embed(xs, masks)
        xs = xs * masks.transpose(1, 2).to(xs)
        xs = self.pre_lookahead_layer(xs)
        xs = xs * masks.transpose(1, 2).to(xs)
        for layer in self.encoders:
            xs, _ = layer(xs, masks, pos_emb)
        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        xs = xs * masks.transpose(1, 2).to(xs)
        for layer in self.up_encoders:
            xs, _ = layer(xs, masks, pos_emb)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return (xs, masks)

    def forward_chunk(
        self,
        xs: torch.Tensor,
        last_chunk: bool,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one token chunk against the caches of the preceding chunks.

        The chunk carries pre_lookahead_len trailing lookahead tokens unless it
        is the last one, which is zero padded instead. The attention cache
        stacks both conformer stacks along dim 0: the token-rate stack's keys
        are repeated twice along time so they share the frame-rate layout.

        Args:
            xs: (batch, tokens, channels) embedded tokens.
            cnn_cache: (batch, channels, 2 + 2 * stride) conv context.
            att_cache: (blocks + up_blocks, batch, heads, 2 * cached_tokens, 2 * d_k).
        Returns:
            Encoded frames and the updated caches in the same layouts.
        """
        num_blocks = len(self.encoders)
        if att_cache is None:
            token_offset = 0
            att_cache1: list[torch.Tensor | None] | torch.Tensor = [None] * num_blocks
            att_cache2: list[torch.Tensor | None] | torch.Tensor = [None] * len(
                self.up_encoders
            )
        else:
            token_offset = att_cache.shape[3] // 2
            att_cache1 = att_cache[:num_blocks, :, :, :token_offset]
            att_cache2 = att_cache[num_blocks:]
        if cnn_cache is None:
            cnn_cache1 = cnn_cache2 = None
        else:
            cnn_cache1, cnn_cache2 = cnn_cache[:, :, :2], cnn_cache[:, :, 2:]
        xs, _, _ = self.embed(xs, None)
        if last_chunk:
            xs = F.pad(xs, (0, 0, 0, self.pre_lookahead_layer.pre_lookahead_len))
        xs, new_cnn_cache1 = self.pre_lookahead_layer.forward_chunk(xs, cnn_cache1)
        pos_emb = self.embed.pos_enc.position_encoding(token_offset + xs.shape[1])
        new_att_cache1 = []
        for idx, layer in enumerate(self.encoders):
            xs, layer_cache = layer(xs, None, pos_emb, att_cache1[idx])
            new_att_cache1.append(layer_cache)
        xs = xs.transpose(1, 2).contiguous()
        xs, new_cnn_cache2 = self.up_layer.forward_chunk(xs, cnn_cache2)
        xs = xs.transpose(1, 2).contiguous()
        xs, _, _ = self.up_embed(xs, None)
        pos_emb = self.up_embed.pos_enc.position_encoding(
            token_offset * self.up_layer.stride + xs.shape[1]
        )
        new_att_cache2 = []
        for idx, layer in enumerate(self.up_encoders):
            xs, layer_cache = layer(xs, None, pos_emb, att_cache2[idx])
            new_att_cache2.append(layer_cache)
        if self.normalize_before:
            xs = self.after_norm(xs)
        new_att_cache = torch.cat(
            [
                torch.stack(new_att_cache1).repeat(1, 1, 1, 2, 1),
                torch.stack(new_att_cache2),
            ],
            dim=0,
        )
        new_cnn_cache = torch.cat([new_cnn_cache1, new_cnn_cache2], dim=2)
        return xs, new_cnn_cache, new_att_cache


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask
