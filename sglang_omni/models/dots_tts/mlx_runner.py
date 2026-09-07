# SPDX-License-Identifier: Apache-2.0
"""MLX Qwen2 execution with the shared dots.tts Torch/MPS acoustic tail.

The scheduler owns admission and token bookkeeping. This runner owns only the
single request's native MLX attention cache; acoustic feedback, EOS, retraction
replay and streaming continue through DotsTTSModelRunner.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open

from sglang_omni.models.dots_tts.flow_head import DotsTTSFlowHead
from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner


class DotsMlxModel(torch.nn.Module):
    graph_feedback_buffer = None

    def __init__(self, checkpoint: Path, *, dtype: str = "bfloat16") -> None:
        super().__init__()
        import mlx.core as mx
        from mlx_lm.models.qwen2 import ModelArgs, Qwen2Model

        config = json.loads((checkpoint / "config.json").read_text())
        llm_config = json.loads((checkpoint / "llm_config.json").read_text())
        self.backbone = Qwen2Model(ModelArgs.from_dict(llm_config))
        self.flow = DotsTTSFlowHead(
            config,
            llm_hidden_size=llm_config["hidden_size"],
            latent_stats_path=str(checkpoint / "latent_stats.pt"),
            optimize=False,
        )
        backbone_weights = []
        flow_weights = {}
        with safe_open(checkpoint / "model.safetensors", framework="pt") as weights:
            for name in weights.keys():
                value = weights.get_tensor(name)
                if name.startswith("llm.model."):
                    from sglang.srt.utils.tensor_bridge import torch_to_mlx

                    backbone_weights.append(
                        (name.removeprefix("llm.model."), torch_to_mlx(value))
                    )
                elif name == "llm.lm_head.weight":
                    # dots.tts consumes hidden states, never vocabulary logits.
                    continue
                else:
                    flow_weights[name] = value
        self.backbone.load_weights(backbone_weights, strict=True)
        mlx_dtype = {
            "float32": mx.float32,
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
        }[dtype]
        self.backbone.set_dtype(mlx_dtype)
        self.backbone.eval()
        mx.eval(self.backbone.parameters())
        self.flow.load_state_dict(flow_weights, strict=True)
        # The upstream acoustic solver performs explicit operations outside
        # autocast. Float32 also preserves VAE/speaker precision on MPS.
        self.flow.to(device="mps", dtype=torch.float32).eval()
        self.caches: dict[str, list[Any]] = {}

    def get_input_embeddings(self):
        from sglang.srt.utils.tensor_bridge import mlx_to_torch, torch_to_mlx

        def embed(input_ids):
            values = self.backbone.embed_tokens(torch_to_mlx(input_ids))
            return mlx_to_torch(values, device="mps").float()

        return embed

    def forward_hidden(self, rid: str, embeddings: torch.Tensor, *, prefill: bool):
        from mlx_lm.models.cache import KVCache
        from sglang.srt.utils.tensor_bridge import mlx_to_torch, torch_to_mlx

        if prefill:
            self.caches[rid] = [KVCache() for _ in self.backbone.layers]
        elif rid not in self.caches:
            raise RuntimeError("dots.tts MLX decode has no prefill cache")
        values = torch_to_mlx(embeddings.reshape(1, -1, embeddings.shape[-1]))
        values = values.astype(self.backbone.embed_tokens.weight.dtype)
        hidden = self.backbone(None, cache=self.caches[rid], input_embeddings=values)
        return mlx_to_torch(hidden, device="mps").float().reshape(-1, hidden.shape[-1])


class DotsMlxWorkerRunner:
    """Model and bounded cache capacity supplied to SGLang's MLX worker stub."""

    def __init__(self, *, model_path: str, pool_size: int = 2048, **kwargs):
        self.pool_size = int(pool_size)
        self.model = DotsMlxModel(
            Path(model_path), dtype=kwargs.get("dtype", "bfloat16")
        )

    def has_request(self, rid: str) -> bool:
        return rid in self.model.caches

    def remove_request(self, rid: str) -> None:
        self.model.caches.pop(rid, None)

    def store_auxiliary_state_for_request(self, rid: str) -> None:
        # No auxiliary state or radix reuse on this single-request backend.
        pass


class DotsTTSMlxModelRunner(DotsTTSModelRunner):
    def lookahead_eligible(self, batch: Any) -> bool:
        return False

    def _build_forward_batch(self, scheduler_output: Any):
        batch = scheduler_output.batch_data
        if batch is None:
            return None
        if len(scheduler_output.requests) != 1:
            raise RuntimeError("dots.tts MLX requires max_running_requests=1")
        # SGLang's MLX bookkeeping stub deliberately has no Torch attention
        # backend. The shared dots hooks need only ids and feedback embeddings.
        forward = SimpleNamespace(
            input_ids=batch.input_ids.to("mps"), input_embeds=None
        )
        return forward, batch, bool(batch.forward_mode.is_extend())

    def _forward(self, forward_batch, requests, *, prefill):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers.utils import GenerationBatchResult

        hidden = self.model.forward_hidden(
            requests[0].request_id, forward_batch.input_embeds, prefill=prefill
        )
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(
                next_token_logits=None, hidden_states=hidden
            ),
            next_token_ids=torch.zeros(1, dtype=torch.long),
            can_run_cuda_graph=False,
        )

    def _launch_flow_batch(self, result, requests, hidden, *, append_hidden):
        launch = super()._launch_flow_batch(
            result, requests, hidden, append_hidden=append_hidden
        )
        # MLX scheduler bookkeeping lives on CPU, independently of the MPS
        # acoustic tensors. Only control ids cross this boundary.
        result.next_token_ids = result.next_token_ids.cpu()
        return launch

    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        return self._forward(forward_batch, requests, prefill=True)

    def custom_decode_forward(self, forward_batch, schedule_batch, requests):
        return self._forward(forward_batch, requests, prefill=False)

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        self.model.caches.pop(request_id, None)
        super().on_request_finished(request_id, req_data)

    def _suspend_request_data(self, req_data: Any) -> None:
        self.model.caches.pop(req_data.req.rid, None)
        super()._suspend_request_data(req_data)

    def reset_request(self, request_id: str) -> None:
        self.model.caches.pop(request_id, None)
        super().reset_request(request_id)
