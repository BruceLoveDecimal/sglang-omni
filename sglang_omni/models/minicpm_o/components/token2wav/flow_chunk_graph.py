# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs for the MiniCPM-o DiT chunk step.

One graph is captured per (query_frames, cache_frames) key. A chunk shorter than
the key is padded: a stream's first chunk pads on the left, so the causal convs
see the same zero context as an unpadded start, while later chunks pad on the
right and keep their real conv cache. Padded frames are masked out of attention
and zeroed before each conv, so real frames match the eager result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT

CFG_BATCH = 2


@dataclass(frozen=True, order=True)
class ChunkGraphKey:
    query_frames: int
    cache_frames: int


@dataclass
class CapturedChunkGraph:
    graph: torch.cuda.CUDAGraph | None
    x: torch.Tensor
    t_emb: torch.Tensor
    attn_mask: torch.Tensor
    frame_mask: torch.Tensor
    cnn_cache: torch.Tensor
    att_cache: torch.Tensor
    velocity: torch.Tensor
    new_cnn_cache: torch.Tensor
    new_att_cache: torch.Tensor


class DiTChunkGraphs:
    """Captures the DiT chunk step once per key and replays it for fitting chunks."""

    def __init__(
        self,
        estimator: DiT,
        *,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
        keys: Sequence[ChunkGraphKey],
    ) -> None:
        if not keys:
            raise ValueError("chunk graph keys must not be empty")
        self.estimator = estimator
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.keys = sorted(set(keys))
        self.graphs: dict[ChunkGraphKey, CapturedChunkGraph] = {}

    def autocast(self) -> torch.autocast:
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        )

    def static_tensors(self, key: ChunkGraphKey) -> CapturedChunkGraph:
        estimator = self.estimator
        dtype = estimator.in_proj.weight.dtype
        depth = len(estimator.blocks)
        conv = estimator.blocks[0].conv
        query, cache = key.query_frames, key.cache_frames
        zeros = lambda *shape: torch.zeros(*shape, device=self.device, dtype=dtype)
        return CapturedChunkGraph(
            graph=None,
            x=zeros(CFG_BATCH, estimator.in_channels, query),
            t_emb=zeros(CFG_BATCH, 1, estimator.hidden_size),
            attn_mask=torch.ones(
                CFG_BATCH, query, query + cache, device=self.device, dtype=torch.bool
            ),
            frame_mask=torch.ones(CFG_BATCH, query, 1, device=self.device, dtype=dtype),
            cnn_cache=zeros(
                depth,
                CFG_BATCH,
                conv.in_channels + conv.out_channels,
                conv.kernel_size - 1,
            ),
            att_cache=zeros(
                depth, CFG_BATCH, estimator.num_heads, cache, 2 * estimator.head_dim
            ),
            velocity=zeros(CFG_BATCH, estimator.out_channels, query),
            new_cnn_cache=zeros(
                depth,
                CFG_BATCH,
                conv.in_channels + conv.out_channels,
                conv.kernel_size - 1,
            ),
            new_att_cache=zeros(
                depth,
                CFG_BATCH,
                estimator.num_heads,
                query + cache,
                2 * estimator.head_dim,
            ),
        )

    def step(self, captured: CapturedChunkGraph) -> None:
        captured.velocity.copy_(
            self.estimator.blocks_forward_chunk(
                captured.x,
                captured.t_emb,
                captured.attn_mask,
                captured.frame_mask,
                captured.cnn_cache,
                captured.att_cache,
                captured.new_cnn_cache,
                captured.new_att_cache,
            )
        )

    @torch.inference_mode()
    def capture(self) -> None:
        # Capture on a side stream so unrelated default-stream work is not recorded.
        current_stream = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current_stream)
        graphs: dict[ChunkGraphKey, CapturedChunkGraph] = {}
        with torch.cuda.device(self.device), torch.cuda.stream(stream), self.autocast():
            pool = torch.cuda.graph_pool_handle()
            for key in self.keys:
                captured = self.static_tensors(key)
                self.step(captured)
                captured.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    cuda_graph=captured.graph,
                    pool=pool,
                    stream=stream,
                    capture_error_mode="thread_local",
                ):
                    self.step(captured)
                graphs[key] = captured
        current_stream.wait_stream(stream)
        torch.cuda.empty_cache()
        self.graphs = graphs

    def select(self, query_frames: int, cache_frames: int) -> ChunkGraphKey | None:
        for key in self.keys:
            if key.query_frames >= query_frames and key.cache_frames >= cache_frames:
                return key
        return None

    @staticmethod
    def load(
        captured: CapturedChunkGraph,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_cache: torch.Tensor,
    ) -> tuple[int, int]:
        """Pad the chunk into the static inputs; returns its frame range."""
        query, cached = x.shape[2], att_cache.shape[3]
        query_frames = captured.x.shape[2]
        assert (
            x.dtype == captured.x.dtype and att_cache.dtype == captured.att_cache.dtype
        ), f"chunk graph captured {captured.x.dtype}, got {x.dtype}/{att_cache.dtype}"
        # The first chunk of a stream carries no cache and pads on the left.
        start = query_frames - query if cached == 0 else 0
        stop = start + query
        captured.x.zero_()
        captured.x[:, :, start:stop].copy_(x)
        captured.t_emb.copy_(t_emb)
        captured.cnn_cache.copy_(cnn_cache)
        captured.att_cache.zero_()
        captured.att_cache[:, :, :, :cached].copy_(att_cache)
        captured.frame_mask.zero_()
        captured.frame_mask[:, start:stop] = 1
        # Keys are ordered [this chunk, cache]; padded query rows keep every
        # key so their (discarded) softmax rows stay finite.
        captured.attn_mask.fill_(True)
        captured.attn_mask[:, start:stop, :] = False
        captured.attn_mask[:, start:stop, start:stop] = True
        captured.attn_mask[:, start:stop, query_frames : query_frames + cached] = True
        return start, stop

    @staticmethod
    def store(
        captured: CapturedChunkGraph,
        start: int,
        stop: int,
        new_cnn_cache: torch.Tensor,
        new_att_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Copy the real frames' caches out of the static outputs; returns velocity.

        A right-padded chunk's conv cache covers padded frames; that chunk is a
        stream's last, so the cache is never read.
        """
        query = stop - start
        cached = new_att_cache.shape[3] - query
        query_frames = captured.x.shape[2]
        new_cnn_cache.copy_(captured.new_cnn_cache)
        new_att_cache[:, :, :, :query].copy_(
            captured.new_att_cache[:, :, :, start:stop]
        )
        new_att_cache[:, :, :, query:].copy_(
            captured.new_att_cache[:, :, :, query_frames : query_frames + cached]
        )
        return captured.velocity[:, :, start:stop].clone()

    def run(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_cache: torch.Tensor,
        new_cnn_cache: torch.Tensor,
        new_att_cache: torch.Tensor,
    ) -> torch.Tensor | None:
        """Replay the smallest fitting graph, or return None when none fits."""
        key = self.select(x.shape[2], att_cache.shape[3])
        if key is None or x.shape[0] != CFG_BATCH:
            return None
        captured = self.graphs[key]
        start, stop = self.load(captured, x, t_emb, cnn_cache, att_cache)
        captured.graph.replay()
        return self.store(captured, start, stop, new_cnn_cache, new_att_cache)
