# SPDX-License-Identifier: Apache-2.0
"""Chunked MiniCPM-o flow: chunk scheduling, cache bounds and graph padding."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.minicpm_o.components.code2wav import (
    SAMPLES_PER_CODEC_TOKEN,
    MiniCPMOCode2Wav,
)
from sglang_omni.models.minicpm_o.components.token2wav.conformer import (
    UpsampleConformerEncoderV2,
)
from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT
from sglang_omni.models.minicpm_o.components.token2wav.flow import (
    CausalConditionalCFM,
    CausalMaskedDiffWithXvec,
    FlowChunkCache,
)
from sglang_omni.models.minicpm_o.components.token2wav.flow_chunk_graph import (
    ChunkGraphKey,
    DiTChunkGraphs,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MEL = 8


def tiny_flow() -> CausalMaskedDiffWithXvec:
    torch.manual_seed(0)
    encoder = UpsampleConformerEncoderV2(
        input_size=16,
        output_size=16,
        pre_lookahead_len=3,
        num_blocks=1,
        num_up_blocks=1,
        attention_heads=2,
        linear_units=32,
    )
    estimator = DiT(
        in_channels=4 * MEL,
        out_channels=MEL,
        depth=2,
        num_heads=2,
        head_dim=4,
        hidden_size=16,
    )
    for parameter in estimator.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    flow = CausalMaskedDiffWithXvec(
        encoder,
        CausalConditionalCFM(estimator),
        input_size=16,
        output_size=MEL,
        spk_embed_dim=6,
        vocab_size=4300,
    )
    return flow.eval()


def tiny_prompt(
    prompt_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.arange(prompt_tokens, dtype=torch.int32).unsqueeze(0),
        torch.tensor([prompt_tokens], dtype=torch.int32),
        torch.randn(1, 6),
        torch.randn(1, prompt_tokens * 2, MEL),
    )


def chunked_model(flow: CausalMaskedDiffWithXvec) -> MiniCPMOCode2Wav:
    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    model.token2wav = SimpleNamespace(
        device=torch.device("cpu"), dtype=torch.float32, n_timesteps=2, flow=flow
    )
    model.chunked_flow = True
    return model


@pytest.mark.parametrize("num_tokens", [5, 25, 28, 60])
def test_chunked_mel_has_up_rate_frames_per_token(num_tokens: int) -> None:
    flow = tiny_flow()
    model = chunked_model(flow)
    tokens = list(range(1, num_tokens + 1))
    mel = model.chunked_mel(tokens, tiny_prompt(7))
    assert mel.shape == (1, MEL, num_tokens * flow.up_rate)
    assert torch.isfinite(mel).all()


def test_chunked_vocode_slices_waveforms_to_token_lengths() -> None:
    class FakeHiFT:
        def __call__(self, speech_feat: torch.Tensor) -> tuple[torch.Tensor, None]:
            samples = speech_feat.shape[-1] * (SAMPLES_PER_CODEC_TOKEN // 2)
            return speech_feat.new_ones(speech_feat.shape[0], 1, samples), None

    model = chunked_model(tiny_flow())
    model.token2wav.hift = FakeHiFT()
    model.speaker_prompt = lambda prompt_wav: tiny_prompt(7)
    waveforms = model.vocode([[1, 2], [3, 4, 5]], b"ref")
    assert [wave.shape for wave in waveforms] == [
        (2 * SAMPLES_PER_CODEC_TOKEN,),
        (3 * SAMPLES_PER_CODEC_TOKEN,),
    ]


def test_chunk_cache_truncate_keeps_prompt_and_recent_frames() -> None:
    estimator_att = torch.arange(30.0).view(1, 1, 1, 1, 30, 1)
    conformer_att = torch.arange(30.0).view(1, 1, 1, 30, 1)
    cache = FlowChunkCache(torch.zeros(1), conformer_att, torch.zeros(1), estimator_att)
    cache.truncate(prompt_frames=10, keep_frames=5)
    kept = list(range(10)) + list(range(25, 30))
    assert cache.estimator_att.flatten().tolist() == kept
    assert cache.conformer_att.flatten().tolist() == kept
    cache.truncate(prompt_frames=10, keep_frames=5)
    assert cache.estimator_att.shape[4] == 15


@pytest.mark.parametrize("cached_frames", [0, 7])
def test_padded_chunk_step_matches_unpadded(cached_frames: int) -> None:
    """The graph's padding and masks must not change the real frames' result."""
    flow = tiny_flow()
    estimator = flow.decoder.estimator
    query = 5
    padded_query = 12
    x = torch.randn(2, MEL, query)
    mu = torch.randn(2, MEL, query)
    t = torch.tensor([0.3, 0.3])
    spks = torch.randn(2, MEL)
    cond = torch.randn(2, MEL, query)
    packed, t_emb = estimator.pack_chunk_inputs(x, mu, t, spks, cond)
    cnn_cache, att_cache = estimator.empty_chunk_caches(2, torch.device("cpu"))
    # A stream's first chunk (no attention cache) also starts from zero conv context.
    if cached_frames:
        cnn_cache = torch.randn_like(cnn_cache)
    att_cache = torch.randn(*att_cache.shape[:3], cached_frames, att_cache.shape[4])

    expected_cnn = torch.empty_like(cnn_cache)
    expected_att = torch.empty(
        *att_cache.shape[:3], cached_frames + query, att_cache.shape[4]
    )
    expected = estimator.blocks_forward_chunk(
        packed, t_emb, None, None, cnn_cache, att_cache, expected_cnn, expected_att
    )

    graphs = DiTChunkGraphs(
        estimator,
        device=torch.device("cpu"),
        autocast_dtype=None,
        keys=[ChunkGraphKey(padded_query, 16)],
    )
    captured = graphs.static_tensors(graphs.keys[0])
    start, stop = graphs.load(captured, packed, t_emb, cnn_cache, att_cache)
    graphs.step(captured)
    actual_cnn = torch.empty_like(cnn_cache)
    actual_att = torch.empty_like(expected_att)
    actual = graphs.store(captured, start, stop, actual_cnn, actual_att)

    assert (start, stop) == (
        (padded_query - query, padded_query) if cached_frames == 0 else (0, query)
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(actual_att, expected_att, atol=1e-5, rtol=1e-4)
    # A right-padded chunk is always a stream's last, so only the left-padded
    # first chunk must hand a correct conv context to the next chunk.
    if cached_frames == 0:
        torch.testing.assert_close(actual_cnn, expected_cnn, atol=1e-5, rtol=1e-4)


def test_chunk_graph_selects_smallest_fitting_key() -> None:
    graphs = DiTChunkGraphs(
        tiny_flow().decoder.estimator,
        device=torch.device("cpu"),
        autocast_dtype=None,
        keys=[ChunkGraphKey(50, 1024), ChunkGraphKey(64, 0), ChunkGraphKey(128, 0)],
    )
    assert graphs.select(50, 300) == ChunkGraphKey(50, 1024)
    assert graphs.select(20, 1000) == ChunkGraphKey(50, 1024)
    assert graphs.select(60, 0) == ChunkGraphKey(64, 0)
    assert graphs.select(100, 0) == ChunkGraphKey(128, 0)
    assert graphs.select(50, 2000) is None
    assert graphs.select(200, 0) is None


def checkpoint_dir() -> Path | None:
    env = os.environ.get("MINICPMO_CHECKPOINT")
    for path in [Path(env)] if env else []:
        if (path / "assets" / "token2wav").is_dir():
            return path
    return None


@pytest.mark.accelerator
def test_chunked_vocoder_graph_matches_eager_with_checkpoint() -> None:
    checkpoint = checkpoint_dir()
    if checkpoint is None or not torch.cuda.is_available():
        pytest.skip("Set MINICPMO_CHECKPOINT and provide CUDA for vocoder validation")
    model = MiniCPMOCode2Wav(str(checkpoint), device="cuda:0", chunked_flow=True)
    tokens = [1498, 1734, 3732, 3726, 3645] * 12
    prompt = model.speaker_prompt(None)
    with torch.inference_mode():
        graph_mel = model.chunked_mel(tokens, prompt)
        estimator = model.token2wav.flow.decoder.estimator
        graphs, estimator.chunk_graphs = estimator.chunk_graphs, None
        eager_mel = model.chunked_mel(tokens, prompt)
        estimator.chunk_graphs = graphs
    assert graph_mel.shape == (1, 80, len(tokens) * 2)
    torch.testing.assert_close(graph_mel, eager_mel, atol=1e-3, rtol=1e-3)
    waveform = model.vocode([tokens], None)[0]
    assert waveform.shape == (len(tokens) * SAMPLES_PER_CODEC_TOKEN,)
    assert np.isfinite(waveform).all()
