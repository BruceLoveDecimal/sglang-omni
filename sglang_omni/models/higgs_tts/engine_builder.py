# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS SGLang engine builder."""

from __future__ import annotations

import importlib
import logging
import math
from typing import Any

from sglang_omni.models.higgs_tts import request_builders
from sglang_omni.models.higgs_tts import utils as higgs_utils
from sglang_omni.models.higgs_tts.vocoder_scheduler import (
    DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES,
    DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_HIGGS_STREAM_STRIDE,
)
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    build_default_prefill_cuda_graph_bs,
)
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)


class HiggsTtsEngineBuilder(TtsEngineBuilder):
    model_name = "Higgs TTS"
    model_arch_override = "HiggsMultimodalQwen3ForConditionalGeneration"
    context_length = 4096
    supports_breakable_prefill_cuda_graph = True

    def __init__(
        self,
        *,
        max_new_tokens: int | None,
        max_running_requests: int,
        cuda_graph_max_bs: int,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        stream_stride: int = DEFAULT_HIGGS_STREAM_STRIDE,
        stream_followup_stride: int = DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
        initial_chunk_frames: int = DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES,
        prefill_coalesce_requests: int = 0,
        prefill_coalesce_wait_ms: float = 60.0,
        total_gpu_memory_fraction: float | None = None,
    ) -> None:
        if total_gpu_memory_fraction is not None and not (
            0.0 < total_gpu_memory_fraction < 1.0
        ):
            raise ValueError(
                "Higgs tts_engine total_gpu_memory_fraction must be in (0, 1): "
                "it drives sglang mem_fraction_static, which requires < 1"
            )
        self.max_new_tokens = max_new_tokens
        self.max_running_requests = max_running_requests
        self.cuda_graph_max_bs = cuda_graph_max_bs
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.stream_stride = stream_stride
        self.stream_followup_stride = stream_followup_stride
        self.initial_chunk_frames = initial_chunk_frames
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.total_gpu_memory_fraction = total_gpu_memory_fraction
        self.model: Any | None = None

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        from sglang.srt.utils.tensor_bridge import use_mlx

        from sglang_omni.platforms import current_platform

        if use_mlx():
            if not current_platform.is_mps():
                raise ValueError("Higgs MLX requires Apple Silicon")
            return {
                "max_running_requests": 1,
                "disable_cuda_graph": True,
                "disable_overlap_schedule": True,
                "disable_radix_cache": True,
                "enable_torch_compile": False,
                "max_total_tokens": self.context_length,
                "max_prefill_tokens": self.context_length,
                "chunked_prefill_size": -1,
                "mem_fraction_static": self.total_gpu_memory_fraction or 0.8,
                "dtype": dtype,
            }
        if current_platform.is_mps():
            raise ValueError(
                "Higgs Audio v3 on Apple Silicon requires SGLANG_USE_MLX=1"
            )
        del dtype
        # note (luojiaxuan): Radix cache is namespaced per ref-audio via
        # Req.extra_key (set in build_sglang_higgs_request); shared -100
        # placeholder prefixes from different ref audios can't cross-contaminate
        # the KV tree.
        return {
            "max_running_requests": self.max_running_requests,
            "cuda_graph_max_bs": self.cuda_graph_max_bs,
            "disable_cuda_graph": False,
            "mem_fraction_static": (
                self.total_gpu_memory_fraction
                if self.total_gpu_memory_fraction is not None
                else 0.85
            ),
            "chunked_prefill_size": 8192,
            # Qualified capture budget; longer prefills run eager.
            "cuda_graph_backend_prefill": CudaGraphBackend.BREAKABLE,
            "cuda_graph_bs_prefill": build_default_prefill_cuda_graph_bs(512),
            "dtype": "bfloat16",
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        # Note: (Jiaxin Deng) an explicit mem_fraction_static override (e.g.
        # --tts_engine.engine.mem_fraction_static) wins, but never silently.
        expected = self.total_gpu_memory_fraction
        if expected is None:
            return
        actual = overrides.get("mem_fraction_static")
        if actual is not None and abs(actual - expected) <= 1e-9:
            return
        logger.warning(
            "Higgs tts_engine mem_fraction_static=%s overrides the "
            "placement-validated total_gpu_memory_fraction=%s",
            actual,
            expected,
        )

    def validate_before_infrastructure(self, server_args: Any) -> None:
        from sglang.srt.utils.tensor_bridge import use_mlx

        if not use_mlx():
            return
        if server_args.max_running_requests != 1:
            raise ValueError("Higgs MLX requires max_running_requests=1")
        if (
            not server_args.disable_radix_cache
            or server_args.chunked_prefill_size != -1
        ):
            raise ValueError(
                "Higgs MLX requires disabled radix cache and chunked prefill"
            )
        if server_args.mlx_enable_sampling:
            raise ValueError(
                "Higgs MLX owns codebook sampling; disable mlx_enable_sampling"
            )
        if server_args.tp_size != 1:
            raise ValueError("Higgs MLX requires tensor parallel size 1")
        if server_args.enable_torch_compile or not server_args.disable_cuda_graph:
            raise ValueError(
                "Higgs MLX requires disabled Torch compilation and CUDA graphs"
            )

    def customize_server_args(self, server_args: Any) -> None:
        override_server_args(
            server_args,
            "sglang_omni.higgs_tts.disable_overlap_schedule",
            disable_overlap_schedule=True,
        )

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, device, gpu_id, server_args
        from sglang.srt.utils.tensor_bridge import use_mlx

        if use_mlx():
            self.model = model_worker._mlx_runner
        else:
            self.model = model_worker.model_runner.model
            higgs_utils.truncate_rope_to_bf16(self.model)

    def get_model_buffer_bs(self, model: Any) -> int | None:
        from sglang.srt.utils.tensor_bridge import use_mlx

        return 1 if use_mlx() else model.sampler_pool_max_running_requests

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang.srt.utils.tensor_bridge import use_mlx

        if use_mlx():
            from sglang_omni.models.higgs_tts.mlx.scheduler_runner import (
                HiggsMlxSchedulerRunner,
            )

            return HiggsMlxSchedulerRunner(model_worker, output_proc)
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.higgs_tts.model_runner"
        )

        return model_runner_mod.HiggsTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        adapters = request_builders.make_higgs_scheduler_adapters(
            max_new_tokens_cap=self.max_new_tokens,
            stream_stride=self.stream_stride,
            stream_followup_stride=self.stream_followup_stride,
            initial_chunk_frames=self.initial_chunk_frames,
        )

        from sglang.srt.utils.tensor_bridge import use_mlx

        if not use_mlx():
            return adapters
        build_request, adapt_result = adapters

        def build_mlx_request(payload):
            data = build_request(payload)
            if not math.isfinite(data.temperature) or data.temperature < 0:
                raise ValueError(
                    "Higgs MLX temperature must be finite and non-negative"
                )
            data.req.sampling_params.verify(vocab_size=151936)
            if not len(data.input_ids) or data.max_new_tokens < 1:
                raise ValueError(
                    "Higgs MLX requires a nonempty prompt and a positive output budget"
                )
            if data.return_logprob or data.return_omni_rollout:
                raise ValueError("Higgs MLX does not support rollout/logprob capture")
            if data.num_codebooks != 8 or data.codebook_size != 1026:
                raise ValueError("Higgs Audio v3 requires 8 codebooks of size 1026")
            if len(data.input_ids) + data.max_new_tokens > self.context_length:
                raise ValueError(
                    f"Higgs MLX prompt plus output budget exceeds {self.context_length} tokens"
                )
            return data

        return build_mlx_request, adapt_result

    def make_abort_callback(self) -> Any | None:
        assert self.model is not None
        from sglang.srt.utils.tensor_bridge import use_mlx

        return self.model.remove_request if use_mlx() else self.model.reset_request

    def make_request_finished_callback(self) -> Any | None:
        assert self.model is not None
        from sglang.srt.utils.tensor_bridge import use_mlx

        return self.model.remove_request if use_mlx() else self.model.reset_request

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        from sglang.srt.utils.tensor_bridge import use_mlx

        if use_mlx():
            return {"enable_async_decode": False}
        return {
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            "prefill_coalesce_requests": self.prefill_coalesce_requests,
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
        }

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        model_runner.set_stream_outbox(scheduler.outbox)
