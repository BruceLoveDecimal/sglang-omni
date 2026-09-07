# SPDX-License-Identifier: Apache-2.0
"""Backend policy, cache lifecycle and numerical checks for dots.tts MLX."""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.dots_tts.engine_builder import DotsTTSEngineBuilder
from sglang_omni.models.dots_tts.mlx_runner import DotsTTSMlxModelRunner


@pytest.fixture
def mlx_builder(monkeypatch):
    monkeypatch.setattr(DotsTTSEngineBuilder, "_uses_mlx", staticmethod(lambda: True))
    return DotsTTSEngineBuilder()


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_running_requests": 2},
        {"tp_size": 2},
        {"disable_cuda_graph": False},
        {"disable_radix_cache": False},
        {"disable_overlap_schedule": False},
        {"enable_torch_compile": True},
        {"chunked_prefill_size": 256},
        {"quantization": "awq"},
        {"mlx_enable_sampling": True},
        {"max_total_tokens": 1024},
    ],
)
def test_mlx_rejects_unqualified_execution(mlx_builder, overrides):
    with pytest.raises(ValueError, match="dots.tts MLX"):
        mlx_builder.adjust_overrides(overrides)


def test_mlx_keeps_single_request_and_unsplit_prefill(mlx_builder):
    overrides = {}
    mlx_builder.adjust_overrides(overrides)
    assert overrides["max_running_requests"] == 1
    assert overrides["chunked_prefill_size"] == -1
    assert mlx_builder.max_running_requests == 1


@pytest.mark.parametrize("finished", [False, True])
def test_mlx_releases_backbone_and_acoustic_state(finished):
    released = []
    flow_state = object()
    data = SimpleNamespace(flow_state=flow_state, pending_feedback_queue=[1])
    runner = object.__new__(DotsTTSMlxModelRunner)
    runner.model = SimpleNamespace(
        caches={"r": [object()]},
        flow=SimpleNamespace(release_request=released.append),
    )
    runner._request_data = {"r": data}
    if finished:
        runner.on_request_finished("r", data)
    else:
        runner.reset_request("r")
    runner.reset_request("r")  # cleanup is idempotent
    assert runner.model.caches == {}
    assert runner._request_data == {}
    assert released == [flow_state]
    assert data.pending_feedback_queue == []
    assert data.flow_state is None


def test_mlx_refuses_multi_request_batch_before_forward():
    runner = object.__new__(DotsTTSMlxModelRunner)
    with pytest.raises(RuntimeError, match="max_running_requests=1"):
        runner._build_forward_batch(
            SimpleNamespace(batch_data=object(), requests=[object(), object()])
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_flow_rng_is_reproducible_and_restores_global_state():
    from sglang_omni.models.dots_tts.flow_head import DotsTTSFlowHead

    head = object.__new__(DotsTTSFlowHead)
    initial = torch.Generator(device="mps").manual_seed(42).get_state()
    state = SimpleNamespace(fm_sequence=torch.zeros(1, device="mps"), rng_state=initial)
    global_state = torch.mps.get_rng_state().clone()
    with head._request_rng(state):
        first = torch.randn(32, device="mps").cpu()
    assert torch.equal(global_state, torch.mps.get_rng_state())
    with head._request_rng(state):
        second = torch.randn(32, device="mps").cpu()
    assert not torch.equal(first, second)
    state.rng_state = initial
    with head._request_rng(state):
        replay = torch.randn(32, device="mps").cpu()
    torch.testing.assert_close(first, replay, rtol=0, atol=0)
    assert torch.equal(global_state, torch.mps.get_rng_state())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MLX")
def test_qwen2_mlx_prefill_and_feedback_match_torch():
    pytest.importorskip("mlx.core")
    from mlx_lm.models.qwen2 import ModelArgs, Qwen2Model
    from sglang.srt.utils.tensor_bridge import torch_to_mlx
    from transformers import Qwen2Config
    from transformers import Qwen2Model as TorchQwen2Model

    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=1000000.0,
    )
    torch.manual_seed(42)
    reference = TorchQwen2Model(config).eval()
    model = Qwen2Model(ModelArgs.from_dict(config.to_dict()))
    model.load_weights(
        [(key, torch_to_mlx(value)) for key, value in reference.state_dict().items()]
    )
    from sglang_omni.models.dots_tts.mlx_runner import DotsMlxModel

    wrapper = DotsMlxModel.__new__(DotsMlxModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.backbone = model
    wrapper.caches = {}
    embeddings = torch.randn(1, 8, 32)
    feedback = torch.randn(1, 1, 32)
    with torch.inference_mode():
        first = reference(inputs_embeds=embeddings, use_cache=True)
        second = reference(
            inputs_embeds=feedback,
            past_key_values=first.past_key_values,
            use_cache=True,
        )
    mlx_first = wrapper.forward_hidden("r", embeddings, prefill=True)
    mlx_second = wrapper.forward_hidden("r", feedback, prefill=False)
    torch.testing.assert_close(
        mlx_first.cpu(), first.last_hidden_state[0], rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        mlx_second.cpu(), second.last_hidden_state[0], rtol=1e-4, atol=1e-5
    )
    assert all(layer.offset == 9 for layer in wrapper.caches["r"])
    wrapper.forward_hidden("r", embeddings, prefill=True)
    assert all(layer.offset == 8 for layer in wrapper.caches["r"])
    with pytest.raises(RuntimeError, match="no prefill cache"):
        wrapper.forward_hidden("absent", feedback, prefill=False)


def test_mlx_forward_returns_hidden_states_without_token_sampling():
    runner = object.__new__(DotsTTSMlxModelRunner)
    hidden = torch.randn(3, 8)
    calls = []

    def forward(rid, embeddings, *, prefill):
        calls.append((rid, embeddings, prefill))
        return hidden

    runner.model = SimpleNamespace(forward_hidden=forward)
    embeddings = torch.zeros(3, 8)
    result = runner.custom_prefill_forward(
        SimpleNamespace(input_embeds=embeddings),
        None,
        [SimpleNamespace(request_id="r")],
    )
    assert calls == [("r", embeddings, True)]
    assert result.logits_output.hidden_states is hidden
    assert result.logits_output.next_token_logits is None
    assert result.next_token_ids.tolist() == [0]


def test_mlx_retraction_drops_cache_and_preserves_acoustic_replay_state():
    from sglang_omni.models.dots_tts.request_builders import DotsFlowResume

    rng = torch.tensor([42], dtype=torch.uint8)
    data = SimpleNamespace(
        req=SimpleNamespace(rid="r"),
        flow_state=object(),
        pending_feedback_queue=[1],
        decoded_latent_patches=[torch.ones(1, 4, 128)],
    )
    runner = object.__new__(DotsTTSMlxModelRunner)
    runner.model = SimpleNamespace(
        caches={"r": [object()]},
        flow=SimpleNamespace(suspend_request=lambda state: rng),
    )
    runner._suspend_request_data(data)
    assert runner.model.caches == {}
    assert isinstance(data.flow_state, DotsFlowResume)
    assert data.flow_state.rng_state is rng
    assert data.pending_feedback_queue == []
    assert len(data.decoded_latent_patches) == 1
