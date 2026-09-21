# SPDX-License-Identifier: Apache-2.0
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Dit for MiniCPM-o."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, repeat


class MLP(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] | None = None,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.to_q = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(self.inner_dim, dim)

    def to_heads(self, ts: torch.Tensor) -> torch.Tensor:
        b, t, c = ts.shape
        ts = ts.reshape(b, t, self.num_heads, c // self.num_heads)
        ts = ts.transpose(1, 2)
        return ts

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = self.to_heads(q)
        k = self.to_heads(k)
        v = self.to_heads(v)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_mask = attn_mask.unsqueeze(1)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(b, t, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_chunk(
        self, x: torch.Tensor, att_cache: torch.Tensor, attn_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend over this chunk's keys followed by the cached ones.

        Args:
            x: (batch, frames, dim) chunk.
            att_cache: (batch, heads, cached_frames, 2 * head_dim) keys and values.
            attn_mask: (batch, frames, frames + cached_frames) bool, or None.
        Returns:
            The chunk output and the cache with this chunk's keys prepended.
        """
        b, t, c = x.shape
        q = self.to_heads(self.to_q(x))
        k = self.to_heads(self.to_k(x))
        v = self.to_heads(self.to_v(x))
        q = self.q_norm(q)
        k = self.k_norm(k)
        k_cache, v_cache = att_cache.chunk(2, dim=3)
        k = torch.cat([k, k_cache], dim=2)
        v = torch.cat([v, v_cache], dim=2)
        new_att_cache = torch.cat([k, v], dim=3)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(b, t, -1)
        return self.proj(x), new_att_cache


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.scale = 1000

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t * self.scale, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Transpose(torch.nn.Module):

    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class CausalConv1d(torch.nn.Conv1d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size)
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.causal_padding)
        x = super(CausalConv1d, self).forward(x)
        return x

    def forward_chunk(
        self, x: torch.Tensor, cnn_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convolve with the previous chunk's kernel_size - 1 trailing frames."""
        x = torch.cat([cnn_cache, x], dim=2)
        new_cnn_cache = x[..., -self.causal_padding[0] :]
        return super(CausalConv1d, self).forward(x), new_cnn_cache


class CausalConvBlock(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.block = torch.nn.Sequential(
            Transpose(1, 2),
            CausalConv1d(in_channels, out_channels, kernel_size),
            Transpose(1, 2),
            nn.LayerNorm(out_channels),
            nn.Mish(),
            Transpose(1, 2),
            CausalConv1d(out_channels, out_channels, kernel_size),
            Transpose(1, 2),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            x = x * mask
        x = self.block(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_chunk(
        self,
        x: torch.Tensor,
        cnn_cache: torch.Tensor,
        frame_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the block on one chunk; the cache stacks both convs' contexts on dim 1.

        Padded frames are zeroed before each conv so real frames see the same
        context as an unpadded chunk would.
        """
        cnn_cache1, cnn_cache2 = cnn_cache.split(
            (self.in_channels, self.out_channels), dim=1
        )
        if frame_mask is not None:
            x = x * frame_mask
        x = self.block[0](x)
        x, new_cnn_cache1 = self.block[1].forward_chunk(x, cnn_cache1)
        x = self.block[2:5](x)
        if frame_mask is not None:
            x = x * frame_mask
        x = self.block[5](x)
        x, new_cnn_cache2 = self.block[6].forward_chunk(x, cnn_cache2)
        x = self.block[7](x)
        return x, torch.cat((new_cnn_cache1, new_cnn_cache2), dim=1)


class DiTBlock(nn.Module):

    def __init__(
        self, hidden_size: int, num_heads: int, head_dim: int, mlp_ratio: float = 4.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            qkv_bias=True,
            qk_norm=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = MLP(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.conv = CausalConvBlock(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=3
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_conv,
            scale_conv,
            gate_conv,
        ) = self.adaLN_modulation(c).chunk(9, dim=-1)
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), attn_mask
        )
        x = x + gate_conv * self.conv(modulate(self.norm3(x), shift_conv, scale_conv))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward_chunk(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_cache: torch.Tensor,
        attn_mask: torch.Tensor | None,
        frame_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_conv,
            scale_conv,
            gate_conv,
        ) = self.adaLN_modulation(c).chunk(9, dim=-1)
        x_att, new_att_cache = self.attn.forward_chunk(
            modulate(self.norm1(x), shift_msa, scale_msa), att_cache, attn_mask
        )
        x = x + gate_msa * x_att
        x_conv, new_cnn_cache = self.conv.forward_chunk(
            modulate(self.norm3(x), shift_conv, scale_conv), cnn_cache, frame_mask
        )
        x = x + gate_conv * x_conv
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, new_cnn_cache, new_att_cache


class FinalLayer(nn.Module):

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ChunkStepRunner(Protocol):
    """Replays a captured DiT chunk step when the shapes fit a captured graph."""

    def run(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_cache: torch.Tensor,
        new_cnn_cache: torch.Tensor,
        new_att_cache: torch.Tensor,
    ) -> torch.Tensor | None: ...


class DiT(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float = 4.0,
        depth: int = 28,
        num_heads: int = 8,
        head_dim: int = 64,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.in_proj = nn.Linear(in_channels, hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, head_dim, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.chunk_graphs: ChunkStepRunner | None = None

    def initialize_weights(self) -> None:

        def initialize_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(initialize_linear)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = self.t_embedder(t).unsqueeze(1)
        x = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]
        x = x.transpose(1, 2)
        attn_mask = mask.bool()
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, t, attn_mask)
        x = self.final_layer(x, t)
        x = x.transpose(1, 2)
        return x

    def pack_chunk_inputs(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate the estimator inputs on channels and embed the timestep."""
        t_emb = self.t_embedder(t).unsqueeze(1)
        spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
        return pack([x, mu, spks, cond], "b * t")[0], t_emb

    def empty_chunk_caches(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero conv context and an empty attention cache for a stream's first chunk."""
        conv = self.blocks[0].conv
        dtype = self.in_proj.weight.dtype
        cnn_cache = torch.zeros(
            len(self.blocks),
            batch_size,
            conv.in_channels + conv.out_channels,
            conv.kernel_size - 1,
            device=device,
            dtype=dtype,
        )
        att_cache = torch.zeros(
            len(self.blocks),
            batch_size,
            self.num_heads,
            0,
            2 * self.head_dim,
            device=device,
            dtype=dtype,
        )
        return cnn_cache, att_cache

    def blocks_forward_chunk(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        attn_mask: torch.Tensor | None,
        frame_mask: torch.Tensor | None,
        cnn_cache: torch.Tensor,
        att_cache: torch.Tensor,
        new_cnn_cache: torch.Tensor,
        new_att_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Run the packed chunk through the blocks, writing caches into the outputs.

        Args:
            x: (batch, in_channels, frames) packed inputs.
            attn_mask: (batch, frames, frames + cached_frames) bool, or None.
            frame_mask: (batch, frames, 1) marking real frames, or None.
            cnn_cache: (depth, batch, 2 * hidden, kernel - 1).
            att_cache: (depth, batch, heads, cached_frames, 2 * head_dim).
            new_att_cache: (depth, batch, heads, frames + cached_frames, 2 * head_dim).
        """
        x = self.in_proj(x.transpose(1, 2))
        for idx, block in enumerate(self.blocks):
            x, new_cnn_cache[idx], new_att_cache[idx] = block.forward_chunk(
                x, t_emb, cnn_cache[idx], att_cache[idx], attn_mask, frame_mask
            )
        x = self.final_layer(x, t_emb)
        return x.transpose(1, 2)

    def forward_chunk(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_cache: torch.Tensor,
        new_cnn_cache: torch.Tensor,
        new_att_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate one chunk's velocity, writing the updated caches into the outputs.

        A captured graph is replayed when one fits the chunk and cache sizes.
        """
        packed, t_emb = self.pack_chunk_inputs(x, mu, t, spks, cond)
        if self.chunk_graphs is None:
            velocity = None
        else:
            velocity = self.chunk_graphs.run(
                packed, t_emb, cnn_cache, att_cache, new_cnn_cache, new_att_cache
            )
        if velocity is None:
            velocity = self.blocks_forward_chunk(
                packed,
                t_emb,
                None,
                None,
                cnn_cache,
                att_cache,
                new_cnn_cache,
                new_att_cache,
            )
        return velocity
