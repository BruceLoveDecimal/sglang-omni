# SPDX-License-Identifier: Apache-2.0
"""Metal component and native runner regressions; no checkpoint download required."""

import json
import queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache

from sglang_omni.models.higgs_tts.mlx.model import HiggsMlxModel, ModelConfig
from sglang_omni.models.higgs_tts.mlx.runner import (
    SamplingState,
    make_higgs_mlx_runner_class,
)
from sglang_omni.models.higgs_tts.mlx.scheduler_runner import HiggsMlxSchedulerRunner
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.request_builders import build_sglang_higgs_request
from sglang_omni.models.higgs_tts.sampler import HiggsSamplerState, step


def tiny_config(tied=True):
    return ModelConfig(
        text_config=dict(
            model_type="qwen3",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            rms_norm_eps=1e-6,
            max_position_embeddings=4096,
            tie_word_embeddings=True,
            rope_theta=None,
        ),
        audio_encoder_config=dict(
            encoder_type="discrete",
            num_codebooks=8,
            vocab_size=1026,
            out_dim=32,
            tie_word_embeddings=tied,
        ),
    )


@pytest.mark.parametrize("tied", [True, False])
def test_embedding_head_and_cached_decode(tied):
    mx.random.seed(3)
    model = HiggsMlxModel(tiny_config(tied))
    codes = mx.array([[list(range(8))]])
    actual = model.embed_codes(codes)
    weight = torch.tensor(np.array(model.audio_embedding.weight))
    expected = sum(weight[i * 1026 + i] for i in range(8))
    np.testing.assert_allclose(np.array(actual[0, 0]), expected.numpy(), atol=1e-6)
    prompt = model.prompt_embeddings([1, -100, 2], [list(range(8))])
    np.testing.assert_allclose(np.array(prompt[:, 1:2]), np.array(actual))
    cache = [KVCache() for _ in model.layers]
    model(cache=cache, input_embeddings=prompt)
    actual_logits = model(cache=cache, input_embeddings=actual)
    expected_logits = model(input_embeddings=mx.concatenate([prompt, actual], axis=1))
    np.testing.assert_allclose(
        np.array(actual_logits), np.array(expected_logits), rtol=1e-4, atol=2e-5
    )
    hidden = model.model(
        None, input_embeddings=mx.concatenate([prompt, actual], axis=1)
    )[:, -1, :]
    head = model.audio_embedding.weight if tied else model.audio_head.weight
    expected_head = torch.tensor(np.array(hidden)) @ torch.tensor(np.array(head)).T
    np.testing.assert_allclose(
        np.array(expected_logits).reshape(1, -1),
        expected_head.numpy(),
        rtol=1e-4,
        atol=2e-5,
    )


@pytest.mark.parametrize(
    "ids,refs",
    [
        ([1, -100], None),
        ([1], [[0] * 8]),
        ([-2], None),
        ([64], None),
        ([-100], [[0] * 7]),
        ([-100], [[1026] * 8]),
    ],
)
def test_bad_prompt_rejected(ids, refs):
    with pytest.raises(ValueError):
        HiggsMlxModel(tiny_config()).prompt_embeddings(ids, refs)


@pytest.mark.parametrize("tied", [True, False])
def test_official_weight_mapping_is_strict(tmp_path, tied):
    model = HiggsMlxModel(tiny_config(tied))
    original = dict(tree_flatten(model.parameters()))
    checkpoint = {}
    prefixes = {
        "model.embed_tokens.": "tied.embedding.text_embedding.",
        "model.layers.": "body.layers.",
        "model.norm.": "body.norm.",
        "audio_embedding.": "tied.embedding.modality_embeddings.0.embedding.",
        "audio_head.": "tied.head.modality_heads.0.",
    }
    for name, value in original.items():
        for dest, source in prefixes.items():
            if name.startswith(dest):
                checkpoint[source + name[len(dest) :]] = value
                break
    checkpoint["tied.embedding.modality_embeddings.0.model.codec.weight"] = mx.zeros(
        (2,)
    )
    checkpoint["tied.head.text_head.weight"] = mx.zeros((64, 32))
    if tied:
        checkpoint["tied.head.modality_heads.0.weight"] = original[
            "audio_embedding.weight"
        ]
    mapped = model.sanitize(checkpoint)
    assert set(mapped) == set(original)
    model.load_weights(list(mapped.items()), strict=True)
    # Unknown tensors must not disappear and hide an incompatible checkpoint.
    mapped["unexpected.weight"] = mx.zeros((1,))
    with pytest.raises(ValueError):
        model.load_weights(list(mapped.items()), strict=True)


@pytest.mark.parametrize("eoc_step", [0, 7, 8, 12])
def test_greedy_delay_and_eoc_match_torch(eoc_step):
    native = SamplingState(temperature=0)
    reference = HiggsSamplerState(num_codebooks=8)
    for index in range(24):
        logits = np.full((8, 1026), -10, dtype=np.float32)
        logits[:, index] = 10
        if index == eoc_step:
            logits[0, 1025] = 20
        expected = step(torch.from_numpy(logits), reference, temperature=0).numpy()
        actual = np.array(native.sample(mx.array(logits)))
        native.commit(actual)
        np.testing.assert_array_equal(actual, expected)
        assert native.generation_done == reference.generation_done
        assert native.eoc_countdown == reference.eoc_countdown
        if native.generation_done:
            break


@pytest.mark.parametrize("top_k,top_p", [(1, 1.0), (5, 1.0), (-1, 0.01), (1026, 0.9)])
def test_sampling_is_seeded_and_respects_filters(top_k, top_p):
    logits = mx.broadcast_to(mx.arange(1026, dtype=mx.float32) / 10, (8, 1026))

    def generate():
        state = SamplingState(
            key=mx.random.key(123), top_k=top_k, top_p=top_p, delay_count=8
        )
        return np.stack([np.array(state.sample(logits)) for _ in range(12)])

    result = generate()
    np.testing.assert_array_equal(result, generate())
    if top_k > 0:
        assert result.min() >= 1026 - top_k
    if top_k == 1 or top_p == 0.01:
        assert np.all(result == 1025)


@pytest.fixture
def native_runner(tmp_path):
    config = tiny_config()
    model = HiggsMlxModel(config)
    (tmp_path / "config.json").write_text(json.dumps(config.__dict__))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    runner = make_higgs_mlx_runner_class()(
        model_path=str(tmp_path), disable_radix_cache=True, pool_size=256
    )
    yield runner
    runner.clear()


def request_data(rid="first", refs=None):
    return build_sglang_higgs_request(
        HiggsTtsState(
            prompt_token_ids=[1, 2] if refs is None else [1, -100, 2],
            reference_codes_delayed=refs,
            temperature=0,
            max_new_tokens=32,
            seed=42,
        ),
        request_id=rid,
    )


def prefill(runner, data):
    ids = data.req.origin_input_ids
    runner.register_request(data.req.rid, data)
    pending = runner.prefill_start(
        data.req.rid, ids, ids, [], list(range(len(ids))), 0, req=data.req
    )
    mx.eval(pending.lazy_token)
    return runner.prefill_finalize(pending)


def test_native_runner_prefill_decode_release_reuse(native_runner):
    runner = native_runner
    data = request_data(refs=[list(range(8))])
    first = prefill(runner, data)
    codes, done = runner.take_frame("first")
    assert codes[0] == first
    np.testing.assert_array_equal(codes[1:], [1024] * 7)
    pending = runner.decode_batch_start(["first"])
    mx.eval(pending.lazy_tokens)
    token = runner.decode_batch_finalize(pending)[0]
    codes, _ = runner.take_frame("first")
    assert codes[0] == token
    assert runner._req_caches["first"][0].offset == 4
    with pytest.raises(ValueError, match="one active"):
        runner.register_request("second", request_data("second"))
    runner.remove_request("first")
    assert not runner.has_request("first")
    assert not runner._higgs_requests and not runner._higgs_frames
    assert prefill(runner, request_data(refs=[list(range(8))])) == first
    runner.clear()
    assert not runner._higgs_requests and not runner._higgs_pending


def test_scheduler_reuses_streaming_buffers(native_runner):
    runner = native_runner
    data = request_data()
    data.stream_metadata = {
        "modality": "audio_codes",
        "stream": True,
        "num_codebooks": 8,
        "codebook_size": 1026,
    }
    prefill(runner, data)
    bridge = object.__new__(HiggsMlxSchedulerRunner)
    bridge.tp_worker = SimpleNamespace(_mlx_runner=runner)
    bridge._outbox = queue.Queue()
    bridge._vocoder_target = "vocoder"
    bridge._collect_frames([SimpleNamespace(request_id="first", data=data)])
    assert data.output_code_count == 1
    bridge.on_request_finished("first", data)
    chunk = bridge._outbox.get_nowait()
    assert chunk.target == "vocoder"
    np.testing.assert_array_equal(
        chunk.data.numpy(), data.output_code_buffer[0].numpy()
    )
    assert bridge._outbox.empty()


def test_higgs_rope_matches_bf16_training_tables():
    from sglang_omni.models.higgs_tts.mlx.model import HiggsRotaryEmbedding

    x = np.random.default_rng(5).normal(size=(1, 2, 3, 8)).astype(np.float32)
    actual = HiggsRotaryEmbedding(8, 1e6)(mx.array(x), offset=17)
    positions = torch.arange(17, 20, dtype=torch.float32)
    inv = 1e6 ** (-torch.arange(0, 8, 2, dtype=torch.float32) / 8)
    angles = positions[:, None] * inv[None, :]
    cos = angles.cos().bfloat16().float()
    sin = angles.sin().bfloat16().float()
    first, second = torch.from_numpy(x).chunk(2, dim=-1)
    expected = torch.cat(
        [first * cos - second * sin, second * cos + first * sin], dim=-1
    )
    np.testing.assert_allclose(np.array(actual), expected.numpy(), rtol=1e-6, atol=2e-6)


@pytest.fixture
def mlx_builder(monkeypatch):
    from sglang.srt.utils import tensor_bridge

    from sglang_omni.models.higgs_tts.engine_builder import HiggsTtsEngineBuilder
    from sglang_omni.platforms import current_platform

    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: True)
    monkeypatch.setattr(current_platform, "is_mps", lambda: True)
    return HiggsTtsEngineBuilder(
        max_new_tokens=2048,
        max_running_requests=64,
        cuda_graph_max_bs=64,
        enable_async_decode=True,
        async_decode_min_batch_size=2,
    )


@pytest.mark.parametrize(
    "override",
    [
        {"max_running_requests": 2},
        {"disable_radix_cache": False},
        {"chunked_prefill_size": 128},
        {"mlx_enable_sampling": True},
        {"tp_size": 2},
        {"enable_torch_compile": True},
        {"disable_cuda_graph": False},
    ],
)
def test_backend_rejects_unsupported_engine_options(mlx_builder, override):
    args = mlx_builder.generation_defaults(dtype="bfloat16")
    args.update(mlx_enable_sampling=False, tp_size=1)
    mlx_builder.validate_before_infrastructure(SimpleNamespace(**args))
    args.update(override)
    with pytest.raises(ValueError, match="Higgs MLX"):
        mlx_builder.validate_before_infrastructure(SimpleNamespace(**args))
    assert mlx_builder.extra_scheduler_kwargs() == {"enable_async_decode": False}


@pytest.mark.parametrize(
    "fields",
    [
        {"max_new_tokens": 0},
        {"prompt_token_ids": []},
        {"temperature": -1},
        {"temperature": float("nan")},
        {"return_logprob": True},
        {"return_omni_rollout": True},
        {"num_codebooks": 4},
        {"max_new_tokens": 2048, "prompt_token_ids": [1] * 2049},
    ],
)
def test_backend_rejects_unsupported_requests(mlx_builder, fields):
    from sglang_omni.proto import OmniRequest, StagePayload

    state = HiggsTtsState(prompt_token_ids=[1, 2], temperature=0, max_new_tokens=32)
    for key, value in fields.items():
        setattr(state, key, value)
    payload = StagePayload(
        request_id="invalid", request=OmniRequest(inputs={}), data=state.to_dict()
    )
    build, _ = mlx_builder.make_adapters(None)
    with pytest.raises(ValueError, match="Higgs"):
        build(payload)


def test_apple_pipeline_preserves_shared_stages(mlx_builder):
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

    config = HiggsTtsPipelineConfig(model_path="unused")
    assert [s.name for s in config.stages] == [
        "preprocessing",
        "audio_encoder",
        "tts_engine",
        "vocoder",
    ]
    assert config.stage_named("tts_engine").stream_to == ["vocoder"]
    assert config.stage_named("audio_encoder").factory.dtype == "float32"
    assert config.stage_named("vocoder").factory.dtype == "float32"
    assert (
        config.stage_factory_kwargs("vocoder")["decode_cuda_graph_frame_counts"] == ()
    )


def test_native_logits_match_torch_qwen3_with_reference_audio():
    from transformers import Qwen3Model

    from sglang_omni.models.higgs_tts.hf_config import _build_text_config

    mx.random.seed(17)
    native = HiggsMlxModel(tiny_config())
    reference = Qwen3Model(_build_text_config(tiny_config().text_config)).eval()
    reference.load_state_dict(
        {
            name: torch.tensor(np.array(value))
            for name, value in tree_flatten(native.model.parameters())
        },
        strict=True,
    )
    rotary_forward = reference.rotary_emb.forward
    reference.rotary_emb.forward = lambda *args, **kwargs: tuple(
        value.bfloat16().float() for value in rotary_forward(*args, **kwargs)
    )
    prompt = native.prompt_embeddings([1, -100, 2, 3], [list(range(8))])
    with torch.no_grad():
        hidden = reference(
            inputs_embeds=torch.tensor(np.array(prompt))
        ).last_hidden_state[:, -1, :]
        expected = hidden @ torch.tensor(np.array(native.audio_embedding.weight)).T
    actual = native(input_embeddings=prompt)
    np.testing.assert_allclose(
        np.array(actual).reshape(1, -1), expected.numpy(), rtol=2e-4, atol=3e-5
    )
