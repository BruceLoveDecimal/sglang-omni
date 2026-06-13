# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the vendored MOSS codec streaming decode.

These use a *tiny* random-init model (no checkpoint download) and assert that
``decode_stream`` chunked equals ``batch_decode`` single-pass, which is the core
correctness claim of the stateful-streaming RFC (§6).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sglang_omni.models.moss_tts._vendored.moss_audio_tokenizer import (  # noqa: E402
    MossAudioTokenizerConfig,
    MossAudioTokenizerModel,
    StreamingCache,
)
from sglang_omni.models.moss_tts._vendored.moss_audio_tokenizer.audio_tokenizer import (  # noqa: E402
    _apply_rope,
)


def _tiny_block(in_dim, out_dim, d_model, heads, layers):
    return {
        "module_type": "Transformer",
        "input_dimension": in_dim,
        "output_dimension": out_dim,
        "d_model": d_model,
        "num_heads": heads,
        "num_layers": layers,
        "dim_feedforward": d_model * 2,
        "causal": True,
        "norm": "layer_norm",
        "positional_embedding": "rope",
        "max_period": 10000,
        "gating": "none",
        "layer_scale": 0.01,
        "conv_layout": True,
    }


def _tiny_model(seed: int = 0) -> MossAudioTokenizerModel:
    torch.manual_seed(seed)
    cfg = MossAudioTokenizerConfig(
        sampling_rate=16,
        downsample_rate=16,  # 2 * 2 * 4
        encoder_kwargs=[
            {"module_type": "PatchedPretransform", "patch_size": 16},
            _tiny_block(16, 16, 32, 2, 1),
        ],
        decoder_kwargs=[
            _tiny_block(16, 32, 32, 2, 2),
            {"module_type": "PatchedPretransform", "patch_size": 2},
            _tiny_block(16, 32, 32, 2, 2),
            {"module_type": "PatchedPretransform", "patch_size": 2},
            _tiny_block(16, 4, 16, 2, 1),
            {"module_type": "PatchedPretransform", "patch_size": 4},
        ],
        quantizer_kwargs={
            "input_dim": 16,
            "rvq_dim": 8,
            "output_dim": 16,
            "num_quantizers": 2,
            "codebook_size": 16,
            "codebook_dim": 4,
            "quantizer_type": "rlfq",
        },
    )
    model = MossAudioTokenizerModel(cfg)
    model.eval()
    return model


def _rand_codes(model: MossAudioTokenizerModel, t: int, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    nq = model.config.num_quantizers
    cb = model.config.codebook_size
    return torch.randint(0, cb, (nq, t), dtype=torch.long)


def _stream(model, codes, chunk_sizes):
    state = StreamingCache()
    outs = []
    pos = 0
    for cs in chunk_sizes:
        chunk = codes[:, pos : pos + cs]
        pos += cs
        out, state = model.decode_stream(chunk, state=state, last_chunk=(pos >= codes.shape[-1]))
        outs.append(out.audio.reshape(-1))
    return torch.cat(outs, dim=0)


@pytest.mark.parametrize("chunk_sizes", [[10], [3, 3, 4], [1] * 10, [7, 1, 2], [5, 5]])
def test_decode_stream_matches_batch_decode(chunk_sizes):
    model = _tiny_model()
    t = sum(chunk_sizes)
    codes = _rand_codes(model, t)

    with torch.no_grad():
        full = model.batch_decode([codes]).audio.reshape(-1)
        streamed = _stream(model, codes, chunk_sizes)

    assert full.shape == streamed.shape
    assert torch.allclose(full, streamed, atol=1e-4, rtol=1e-4), (
        f"max abs diff = {(full - streamed).abs().max().item():.3e}"
    )


def test_single_frame_equals_one_shot():
    model = _tiny_model()
    codes = _rand_codes(model, 12)
    with torch.no_grad():
        one_shot = _stream(model, codes, [12])
        per_frame = _stream(model, codes, [1] * 12)
    assert torch.allclose(one_shot, per_frame, atol=1e-4, rtol=1e-4)


def test_rope_offset_continuation():
    # RoPE on a window starting at `offset` equals the corresponding slice of RoPE
    # applied to the full sequence from position 0.
    torch.manual_seed(3)
    B, H, T, D = 1, 2, 9, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    qf, kf = _apply_rope(q, k, offset=0)
    split = 4
    q1, k1 = _apply_rope(q[:, :, :split], k[:, :, :split], offset=0)
    q2, k2 = _apply_rope(q[:, :, split:], k[:, :, split:], offset=split)
    assert torch.allclose(qf[:, :, :split], q1, atol=1e-6)
    assert torch.allclose(qf[:, :, split:], q2, atol=1e-6)
    assert torch.allclose(kf[:, :, split:], k2, atol=1e-6)


def test_state_reset():
    model = _tiny_model()
    codes = _rand_codes(model, 6)
    with torch.no_grad():
        a = _stream(model, codes, [6])
        b = _stream(model, codes, [2, 2, 2])
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)
