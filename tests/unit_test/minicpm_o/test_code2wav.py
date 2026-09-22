# SPDX-License-Identifier: Apache-2.0
"""Public MiniCPM-o vocoder contracts: import, checkpoint decode, speaker ref."""

from __future__ import annotations

import base64
import math
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.components.code2wav import (
    MEL_OVERLAP_FRAMES,
    SAMPLES_PER_CODEC_TOKEN,
    SAMPLES_PER_MEL_FRAME,
    SOURCE_OVERLAP_SAMPLES,
    HiFTGraphs,
    MiniCPMOCode2Wav,
    plan_chunks,
)
from sglang_omni.models.minicpm_o.components.token2wav.conformer import (
    UpsampleConformerEncoderV2,
)
from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT, TimestepEmbedder
from sglang_omni.models.minicpm_o.components.token2wav.flow import (
    CausalConditionalCFM,
    CausalMaskedDiffWithXvec,
    FlowCacheRow,
    flow_cache_row,
    stack_flow_caches,
    trim_flow_cache,
)
from sglang_omni.models.minicpm_o.components.token2wav.hift import (
    ConvRNNF0Predictor,
    HiFTGenerator,
)
from sglang_omni.models.minicpm_o.config import MiniCPMOSpeechPipelineConfig
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.routing import (
    code2wav_reference_audio,
    project_talker_to_code2wav,
)
from sglang_omni.models.minicpm_o.stages import vocode_code2wav_payloads
from sglang_omni.proto import OmniRequest, StagePayload

REPO_ROOT = Path(__file__).resolve().parents[3]
MEL_BINS = 6
SPEAKER_DIM = 4
PROMPT_TOKENS = {b"a": 3, b"b": 5, b"c": 4}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("frequency_size", [255, 256])
def test_timestep_embedding_matches_reference(
    dtype: torch.dtype, frequency_size: int
) -> None:
    model = TimestepEmbedder(16, frequency_size).to(dtype).eval()
    t = torch.linspace(0, 1, 11, dtype=dtype)
    half = frequency_size // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half) / half).to(t)
    angles = (t * 1000)[:, None] * frequencies[None]
    embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
    if frequency_size % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    torch.testing.assert_close(model(t), model.mlp(embedding), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_timestep_embedding_autocast_preserves_frequencies(dtype: torch.dtype) -> None:
    model = TimestepEmbedder(16).to(device="cuda", dtype=dtype).eval()
    t = torch.linspace(0, 1, 11, device="cuda", dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000) * torch.arange(128) / 128).to(t)
    angles = (t * 1000)[:, None] * frequencies[None]
    embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        torch.testing.assert_close(model(t), model.mlp(embedding), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_timestep_embedding_cuda_graph_replays_new_inputs() -> None:
    model = TimestepEmbedder(16).cuda().eval()
    t = torch.zeros(2, device="cuda")
    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model(t)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(t)
        t.fill_(0.25)
        expected = model(t)
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_native_vocoder_import_does_not_require_legacy_packages() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "stepaudio2", "s3tokenizer", "minicpmo", "hyperpyyaml"
        }:
            raise ImportError(f"Legacy dependency requested: {fullname}")

sys.meta_path.insert(0, BlockLegacy())
from sglang_omni.models.minicpm_o.components.token2wav.vocoder import Token2Wav
""",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def _checkpoint_dir() -> Path | None:
    env = os.environ.get("MINICPMO_CHECKPOINT")
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    candidates = [Path(env)] if env else []
    candidates += [REPO_ROOT / "MiniCPM-o-4_6", REPO_ROOT / "MiniCPM-o-4_5"]
    for hub in (
        hf_home / "hub" / "models--openbmb--MiniCPM-o-4_5" / "snapshots",
        hf_home / "models--openbmb--MiniCPM-o-4_5" / "snapshots",
    ):
        if hub.is_dir():
            candidates.extend(sorted(hub.iterdir(), reverse=True))
    for path in candidates:
        if path is not None and (path / "assets" / "token2wav").is_dir():
            return path
    return None


@pytest.mark.accelerator
def test_native_vocoder_with_checkpoint() -> None:
    checkpoint = _checkpoint_dir()
    if checkpoint is None or not torch.cuda.is_available():
        pytest.skip("Set MINICPMO_CHECKPOINT and provide CUDA for vocoder validation")
    model = MiniCPMOCode2Wav(str(checkpoint), chunk_tokens=25, device="cuda:0")
    tokens = [1498, 1734, 3732, 3726, 3645]
    output = model(codec_tokens=torch.tensor(tokens))
    waveform = output["waveform"]
    assert output["sample_rate"] == 24000
    assert waveform.dtype == np.float32
    assert waveform.shape == (len(tokens) * SAMPLES_PER_CODEC_TOKEN,)
    assert np.isfinite(waveform).all()
    assert np.max(np.abs(waveform)) > 1e-5
    assert np.max(np.abs(waveform)) <= 0.99


@pytest.mark.accelerator
def test_native_vocoder_batch_matches_single_request_shapes() -> None:
    checkpoint = _checkpoint_dir()
    if checkpoint is None or not torch.cuda.is_available():
        pytest.skip("Set MINICPMO_CHECKPOINT and provide CUDA for vocoder validation")
    model = MiniCPMOCode2Wav(str(checkpoint), chunk_tokens=25, device="cuda:0")
    tokens_a = [1498, 1734, 3732, 3726, 3645] * 8
    tokens_b = tokens_a + [3645, 3726] * 5
    batched = model.vocode([tokens_a, tokens_b], None)
    single_a = model.vocode([tokens_a], None)[0]
    single_b = model.vocode([tokens_b], None)[0]
    assert (
        batched[0].shape == single_a.shape == (len(tokens_a) * SAMPLES_PER_CODEC_TOKEN,)
    )
    assert (
        batched[1].shape == single_b.shape == (len(tokens_b) * SAMPLES_PER_CODEC_TOKEN,)
    )
    assert all(np.isfinite(wave).all() for wave in (*batched, single_a, single_b))


def _data_uri(audio: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")


def _payload(
    *,
    request_id: str = "test",
    tokens: list[int] | None = None,
    params: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=None, params=params or {}, metadata=metadata or {}),
        data=MiniCPMOPipelineState(
            engine_outputs={"talker": {"codec_tokens": torch.tensor(tokens or [1, 2])}}
        ).to_dict(),
    )


def test_chat_api_forwards_reference_to_vocoder() -> None:
    from sglang_omni.client.client import build_params
    from sglang_omni.serve.openai_api import (
        ChatCompletionRequest,
        build_chat_generate_request,
    )

    reference = _data_uri(b"reference")
    request = ChatCompletionRequest(
        model="minicpm-o",
        messages=[{"role": "user", "content": "Hello"}],
        modalities=["text", "audio"],
        audio={"format": "wav", "ref_audio": reference},
    )
    generate_request = build_chat_generate_request(request)
    payload = _payload(
        params=build_params(generate_request), metadata=generate_request.metadata
    )
    assert code2wav_reference_audio(project_talker_to_code2wav(payload)) == b"reference"


def test_invalid_reference_does_not_silently_use_default() -> None:
    payload = _payload(params={"ref_audio": "/tmp/ref.wav"})
    with pytest.raises(ValueError, match="inline audio"):
        code2wav_reference_audio(payload)


def test_speech_pipeline_enables_code2wav_batching_by_default() -> None:
    config = MiniCPMOSpeechPipelineConfig(model_path="unused")
    code2wav = next(stage for stage in config.stages if stage.name == "code2wav")
    assert code2wav.factory.max_batch_size == 8
    assert code2wav.factory.max_batch_wait_ms == 0.0
    assert code2wav.factory.batch_wait_when_idle is False
    assert code2wav.factory.chunk_tokens == 25


def test_plan_chunks_gives_lookahead_to_all_but_the_last_chunk() -> None:
    assert plan_chunks(60, 25, 3) == [(0, 25, False), (25, 50, False), (50, 60, True)]
    assert plan_chunks(28, 25, 3) == [(0, 28, True)]
    assert plan_chunks(29, 25, 3) == [(0, 25, False), (25, 29, True)]
    assert plan_chunks(1, 25, 3) == [(0, 1, True)]


def _small_flow(seed: int = 0) -> CausalMaskedDiffWithXvec:
    torch.manual_seed(seed)
    encoder = UpsampleConformerEncoderV2(
        input_size=8,
        output_size=8,
        attention_heads=2,
        linear_units=16,
        num_blocks=2,
        num_up_blocks=1,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
    )
    estimator = DiT(
        in_channels=4 * MEL_BINS,
        out_channels=MEL_BINS,
        depth=2,
        num_heads=2,
        head_dim=4,
        hidden_size=8,
    )
    flow = CausalMaskedDiffWithXvec(
        encoder,
        CausalConditionalCFM(estimator),
        input_size=8,
        output_size=MEL_BINS,
        spk_embed_dim=SPEAKER_DIM,
    )
    # The checkpoint-style zero init would make the estimator ignore its caches.
    for parameter in flow.parameters():
        parameter.data.normal_(0, 0.2)
    return flow.eval()


def _fake_speaker_prompt(
    reference: bytes,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    length = PROMPT_TOKENS[reference]
    generator = torch.Generator().manual_seed(length)
    tokens = torch.randint(0, 100, (1, length), generator=generator, dtype=torch.int32)
    embedding = torch.randn(1, SPEAKER_DIM, generator=generator)
    mel = torch.randn(1, 2 * length, MEL_BINS, generator=generator)
    return tokens, torch.tensor([length], dtype=torch.int32), embedding, mel


class _OverlapSensitiveHiFT:
    """Deterministic stand-in whose output depends on neighbours and the source cache."""

    def __call__(
        self, speech_feat: torch.Tensor, cache_source: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = speech_feat.new_ones(1, 1, 3)
        hidden = F.conv1d(speech_feat[:, :1], kernel, padding=1) + 1
        source = hidden.repeat_interleave(SAMPLES_PER_MEL_FRAME, dim=-1)
        if cache_source is not None:
            source[:, :, : cache_source.shape[2]] = cache_source
        wav = F.conv1d(source, kernel, padding=1).squeeze(1)
        return wav, source


def _chunked_model(chunk_tokens: int = 4) -> MiniCPMOCode2Wav:
    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    model.token2wav = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        n_timesteps=2,
        flow=_small_flow(),
        hift=_OverlapSensitiveHiFT(),
    )
    model.chunk_tokens = chunk_tokens
    model.hift_graphs = None
    model.default_prompt_wav = None
    model.prompt_cache = OrderedDict()
    model.prompt_cache_capacity = 4
    model.flow_cache = OrderedDict()
    model.flow_cache_capacity = 4
    model.speech_window = torch.hamming_window(
        2 * SOURCE_OVERLAP_SAMPLES, periodic=False
    )
    model.speaker_prompt = _fake_speaker_prompt
    return model


def _sequences(lengths: list[int]) -> list[list[int]]:
    generator = torch.Generator().manual_seed(7)
    return [
        torch.randint(0, 100, (length,), generator=generator).tolist()
        for length in lengths
    ]


def test_vocode_batch_matches_single_rows_across_references_and_lengths() -> None:
    model = _chunked_model()
    sequences = _sequences([3, 9, 12, 5, 8])
    references = [b"a", b"b", b"a", b"c", b"b"]
    batched = model.vocode(sequences, references)
    for tokens, reference, waveform in zip(sequences, references, batched, strict=True):
        assert waveform.shape == (len(tokens) * SAMPLES_PER_CODEC_TOKEN,)
        single = model.vocode([tokens], reference)[0]
        np.testing.assert_allclose(waveform, single, rtol=1e-4, atol=1e-4)


def test_vocode_reuses_reference_flow_state_across_rows_and_calls() -> None:
    model = _chunked_model()
    flow = model.token2wav.flow
    model.token2wav.flow.prompt_cache = MagicMock(wraps=flow.prompt_cache)
    model.vocode(_sequences([4, 6, 5]), [b"a", b"b", b"a"])
    model.vocode(_sequences([7]), b"b")
    assert model.token2wav.flow.prompt_cache.call_count == 2


def test_vocode_overlap_windows_carry_the_previous_chunk_tail() -> None:
    model = _chunked_model(chunk_tokens=4)
    tokens = _sequences([13])[0]
    hift = model.token2wav.hift
    calls: list[tuple[int, int]] = []

    def recording_hift(
        speech_feat: torch.Tensor, cache_source: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(
            (
                speech_feat.shape[-1],
                0 if cache_source is None else cache_source.shape[-1],
            )
        )
        return hift(speech_feat, cache_source)

    model.token2wav.hift = recording_hift
    waveform = model.vocode([tokens], b"a")[0]
    assert waveform.shape == (13 * SAMPLES_PER_CODEC_TOKEN,)
    assert calls == [
        (8, 0),
        (8 + MEL_OVERLAP_FRAMES, SOURCE_OVERLAP_SAMPLES),
        (10 + MEL_OVERLAP_FRAMES, SOURCE_OVERLAP_SAMPLES),
    ]


def test_decode_chunk_masks_shorter_cached_histories() -> None:
    flow = _small_flow()
    rows = []
    embeddings = []
    for reference in (b"a", b"b"):
        tokens, _, embedding, mel = _fake_speaker_prompt(reference)
        rows.append(flow.prompt_cache(tokens, mel, embedding, 2))
        embeddings.append(embedding)
    chunk = torch.randint(0, 100, (2, 7), dtype=torch.int32)
    batched_mel, batched_cache = flow.decode_chunk(
        chunk,
        torch.cat(embeddings),
        stack_flow_caches(
            [FlowCacheRow(batch=row, index=0) for row in rows], flow.up_rate
        ),
        last_chunk=False,
        n_timesteps=2,
    )
    assert batched_mel.shape == (2, MEL_BINS, 8)
    assert batched_cache.token_lens == [7, 9]
    for idx in range(2):
        row_cache = flow_cache_row(batched_cache, idx, flow.up_rate)
        single_mel, single_cache = flow.decode_chunk(
            chunk[idx : idx + 1],
            embeddings[idx],
            rows[idx],
            last_chunk=False,
            n_timesteps=2,
        )
        torch.testing.assert_close(
            batched_mel[idx : idx + 1], single_mel, atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            row_cache.estimator_att, single_cache.estimator_att, atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            row_cache.encoder.token_att,
            single_cache.encoder.token_att,
            atol=1e-4,
            rtol=1e-4,
        )


def test_trim_flow_cache_keeps_reference_and_recent_tail() -> None:
    flow = _small_flow()
    tokens, _, embedding, mel = _fake_speaker_prompt(b"b")
    cache = flow.prompt_cache(tokens, mel, embedding, 2)
    _, cache = flow.decode_chunk(
        torch.randint(0, 100, (1, 6), dtype=torch.int32),
        embedding,
        cache,
        last_chunk=True,
        n_timesteps=2,
    )
    assert cache.token_lens == [11]
    trimmed = trim_flow_cache(cache, tail_tokens=2, up_rate=flow.up_rate)
    assert trimmed.token_lens == [7]
    assert trimmed.encoder.token_att.shape[3] == 7
    assert trimmed.encoder.frame_att.shape[3] == 14
    assert trimmed.estimator_att.shape[5] == 14
    torch.testing.assert_close(
        trimmed.estimator_att[..., :10, :], cache.estimator_att[..., :10, :]
    )
    torch.testing.assert_close(
        trimmed.estimator_att[..., 10:, :], cache.estimator_att[..., -4:, :]
    )
    assert trim_flow_cache(trimmed, tail_tokens=2, up_rate=flow.up_rate) is trimmed


def test_stack_flow_caches_reuses_an_intact_batch() -> None:
    flow = _small_flow()
    tokens, _, embedding, mel = _fake_speaker_prompt(b"a")
    batch = stack_flow_caches(
        [FlowCacheRow(batch=flow.prompt_cache(tokens, mel, embedding, 2), index=0)] * 2,
        flow.up_rate,
    )
    rows = [FlowCacheRow(batch=batch, index=idx) for idx in range(2)]
    assert stack_flow_caches(rows, flow.up_rate) is batch
    assert stack_flow_caches(rows[::-1], flow.up_rate) is not batch
    assert stack_flow_caches(rows[:1], flow.up_rate).token_lens == [3]


def test_hift_istft_matches_torch() -> None:
    hift = HiFTGenerator(in_channels=MEL_BINS, base_channels=16).eval()
    torch.manual_seed(0)
    n_fft = hift.istft_params["n_fft"]
    magnitude = torch.rand(2, n_fft // 2 + 1, 40) * 3
    phase = torch.rand(2, n_fft // 2 + 1, 40) * 6
    expected = torch.istft(
        torch.complex(magnitude * phase.cos(), magnitude * phase.sin()),
        n_fft,
        hift.istft_params["hop_len"],
        n_fft,
        window=hift.stft_window,
    )
    torch.testing.assert_close(hift.istft(magnitude, phase), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hift_graphs_replay_steady_windows_and_reuse_shapes() -> None:
    torch.manual_seed(0)
    hift = (
        HiFTGenerator(
            in_channels=MEL_BINS,
            base_channels=16,
            f0_predictor=ConvRNNF0Predictor(in_channels=MEL_BINS),
        )
        .cuda()
        .eval()
    )
    graphs = HiFTGraphs(hift)
    mel = torch.randn(2, MEL_BINS, 8 + MEL_OVERLAP_FRAMES, device="cuda")
    cache = torch.randn(2, 1, SOURCE_OVERLAP_SAMPLES, device="cuda")
    with torch.inference_mode():
        speech, source = graphs(mel, cache)
        again, _ = graphs(mel * 0.5, cache)
        first_chunk, _ = graphs(mel[:, :, MEL_OVERLAP_FRAMES:], None)
        eager, eager_source = hift(mel, cache)
    assert speech.shape == eager.shape == (2, 16 * SAMPLES_PER_MEL_FRAME)
    assert source.shape == eager_source.shape
    assert first_chunk.shape == (2, 8 * SAMPLES_PER_MEL_FRAME)
    torch.testing.assert_close(source[:, :, :SOURCE_OVERLAP_SAMPLES], cache)
    assert not torch.equal(speech, again)
    assert torch.isfinite(speech).all() and torch.isfinite(again).all()
    assert len(graphs.graphs) == 2


def test_vocode_rejects_mismatched_reference_count() -> None:
    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    with pytest.raises(ValueError, match="does not match"):
        model.vocode([[1], [2]], [b"a"])


def test_prompt_cache_reuses_references_across_calls() -> None:
    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    model.default_prompt_wav = None
    model.prompt_cache = OrderedDict()
    model.prompt_cache_capacity = 4
    model.token2wav = SimpleNamespace(prepare_prompt=MagicMock())
    model.token2wav.prepare_prompt.return_value = (
        torch.zeros(1, 2, dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.zeros(1, 4),
        torch.zeros(1, 4, 80),
    )
    for reference in (b"a", b"b", b"a", b"c", b"b"):
        model.speaker_prompt(reference)
    assert model.token2wav.prepare_prompt.call_count == 3


def test_vocode_rejects_empty_sequences() -> None:
    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    assert model.vocode([], b"ref") == []
    with pytest.raises(ValueError, match="non-empty"):
        model.vocode([[1], []], b"ref")


def _fake_code2wav_model() -> MagicMock:
    fake = MagicMock()
    fake.sample_rate = 24000
    fake.resolve_prompt_wav.side_effect = lambda reference: (
        b"default" if reference is None else reference
    )
    fake.vocode.side_effect = lambda sequences, references: [
        np.full(
            len(tokens) * SAMPLES_PER_CODEC_TOKEN,
            float(len(tokens)),
            dtype=np.float32,
        )
        for tokens in sequences
    ]
    return fake


def test_vocode_payloads_vocodes_one_reference_per_row() -> None:
    fake = _fake_code2wav_model()
    output = vocode_code2wav_payloads(fake, [_payload(tokens=[7, 8, 9])])[0]
    fake.vocode.assert_called_once_with([[7, 8, 9]], [b"default"])
    assert output.data["sample_rate"] == 24000
    assert output.data["audio_waveform_shape"] == [3 * SAMPLES_PER_CODEC_TOKEN]


def test_vocode_payloads_keeps_mixed_references_in_one_call() -> None:
    fake = _fake_code2wav_model()
    outputs = vocode_code2wav_payloads(
        fake,
        [
            _payload(
                request_id="a", tokens=[1, 2], params={"ref_audio": _data_uri(b"spk-a")}
            ),
            _payload(
                request_id="b", tokens=[3], params={"ref_audio": _data_uri(b"spk-b")}
            ),
            _payload(
                request_id="c",
                tokens=[4, 5, 6],
                params={"ref_audio": _data_uri(b"spk-a")},
            ),
        ],
    )
    fake.vocode.assert_called_once_with(
        [[1, 2], [3], [4, 5, 6]], [b"spk-a", b"spk-b", b"spk-a"]
    )
    assert [out.data["audio_waveform_shape"][0] for out in outputs] == [
        2 * SAMPLES_PER_CODEC_TOKEN,
        SAMPLES_PER_CODEC_TOKEN,
        3 * SAMPLES_PER_CODEC_TOKEN,
    ]


def test_vocode_payloads_resolves_default_reference_per_row() -> None:
    fake = _fake_code2wav_model()
    vocode_code2wav_payloads(
        fake,
        [_payload(request_id="a", tokens=[1]), _payload(request_id="b", tokens=[2, 3])],
    )
    fake.vocode.assert_called_once_with([[1], [2, 3]], [b"default", b"default"])
