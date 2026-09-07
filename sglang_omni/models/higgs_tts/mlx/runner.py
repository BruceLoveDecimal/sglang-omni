# SPDX-License-Identifier: Apache-2.0
"""Embedding-driven Higgs generation using SGLang's MLX cache lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from sglang_omni.models.higgs_tts.utils import BOC_ID, EOC_ID


@dataclass
class SamplingState:
    num_codebooks: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    key: Any = None
    delay_count: int = 0
    eoc_countdown: int | None = None
    generation_done: bool = False
    last_codes: Any = None

    def sample(self, logits):
        """Sample independently, then apply Higgs' delayed-codebook warm-up."""
        if self.generation_done:
            raise RuntimeError("Cannot sample a finished Higgs request")
        logits = logits.astype(mx.float32)
        if self.temperature <= 1e-5:
            codes = mx.argmax(logits, axis=-1)
        else:
            logits = logits / self.temperature
            if self.top_k > 0:
                k = min(self.top_k, logits.shape[-1])
                threshold = mx.sort(logits, axis=-1)[:, -k, None]
                logits = mx.where(logits < threshold, -mx.inf, logits)
            if self.top_p < 1.0:
                order = mx.argsort(-logits, axis=-1)
                sorted_logits = mx.take_along_axis(logits, order, axis=-1)
                probs = mx.softmax(sorted_logits, axis=-1)
                remove = mx.concatenate(
                    [
                        mx.zeros((logits.shape[0], 1), dtype=mx.bool_),
                        mx.cumsum(probs, axis=-1)[:, :-1] > self.top_p,
                    ],
                    axis=-1,
                )
                sorted_logits = mx.where(remove, -mx.inf, sorted_logits)
                logits = mx.take_along_axis(
                    sorted_logits, mx.argsort(order, axis=-1), axis=-1
                )
            self.key, sample_key = mx.random.split(self.key)
            codes = mx.random.categorical(logits, key=sample_key)
        if self.delay_count < self.num_codebooks:
            codes = mx.where(
                mx.arange(self.num_codebooks) > self.delay_count, BOC_ID, codes
            )
        return codes

    def commit(self, codes):
        """Match the shared Torch sampler's warm-up and EOC countdown."""
        if self.delay_count < self.num_codebooks:
            self.delay_count += 1
        elif self.eoc_countdown is not None:
            self.eoc_countdown -= 1
            self.generation_done = self.eoc_countdown <= 0
        elif int(codes[0]) == EOC_ID:
            if self.num_codebooks <= 2:
                self.generation_done = True
            else:
                self.eoc_countdown = self.num_codebooks - 2
        self.last_codes = mx.array(codes)


class HiggsMlxModelRunner:
    """One active request; the native worker owns admission and KV recycling."""

    def __init__(self, *args, **kwargs):
        self._higgs_requests = {}
        self._higgs_pending = {}
        self._higgs_frames = {}
        super().__init__(*args, **kwargs)

    def _load_model(self):
        from mlx_lm.utils import load_model
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            ensure_remote_code_allowed,
            resolve_model_directory,
        )

        from .model import HiggsMlxModel, ModelConfig

        if self._quantization is not None:
            raise ValueError(
                "Higgs MLX currently loads official unquantized checkpoints; omit quantization"
            )
        path = resolve_model_directory(self.model_path, revision=self.revision)
        ensure_remote_code_allowed(path, self.trust_remote_code)
        self.model, config = load_model(
            path, get_model_classes=lambda config: (HiggsMlxModel, ModelConfig)
        )
        if config.get("quantization"):
            raise ValueError("Quantized Higgs checkpoints are not supported yet")

    def register_request(self, request_id, data):
        if request_id in self._higgs_requests:
            return
        if self._higgs_requests:
            raise ValueError("Higgs MLX supports one active request")
        if data.return_logprob or data.return_omni_rollout:
            raise ValueError("Higgs MLX does not support rollout/logprob capture")
        seed = data.req.sampling_params.sampling_seed
        if seed is None:
            # Request-local keys avoid changing another Metal stage's RNG.
            key = mx.random.key(int(np.random.default_rng().integers(0, 2**32)))
        else:
            key = mx.random.key(int(seed) & 0xFFFFFFFF)
        self._higgs_requests[request_id] = (
            data,
            SamplingState(
                num_codebooks=data.num_codebooks,
                temperature=data.temperature,
                top_p=data.top_p,
                top_k=data.top_k,
                key=key,
            ),
        )

    def prefill_start(
        self,
        req_id,
        new_token_ids,
        full_token_ids,
        prefix_slot_ids,
        new_slot_ids,
        req_pool_idx,
        req=None,
        needs_logits=True,
        logit_edit_row=None,
        logprob_spec=None,
    ):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingPrefill

        if prefix_slot_ids or not self.disable_radix_cache:
            raise ValueError("Higgs MLX requires disable_radix_cache=True")
        if not needs_logits or new_token_ids != full_token_ids:
            raise ValueError("Higgs MLX does not support chunked prefill")
        if logit_edit_row is not None or logprob_spec is not None:
            raise ValueError(
                "Higgs MLX owns codebook sampling; disable MLX text sampling"
            )
        data, state = self._higgs_requests[req_id]
        embeddings = self.model.prompt_embeddings(
            new_token_ids, data.reference_codes_delayed
        )
        cache = self._acquire_cache()
        try:
            codes = state.sample(
                self.model(cache=cache, input_embeddings=embeddings)[0]
            )
        except Exception:
            self._release_cache(cache)
            raise
        self._higgs_pending[req_id] = codes
        return MlxPendingPrefill(
            lazy_token=codes[:1],
            cache=cache,
            req_id=req_id,
            full_token_ids=list(full_token_ids),
            req_pool_idx=req_pool_idx,
            synced_offset=0,
            lazy_logprobs=None,
        )

    def _commit_frame(self, req_id):
        codes = np.array(self._higgs_pending.pop(req_id), copy=True)
        self._higgs_requests[req_id][1].commit(codes)
        self._higgs_frames[req_id] = codes

    def prefill_finalize(self, pending):
        token = super().prefill_finalize(pending)
        self._commit_frame(pending.req_id)
        return token

    def decode_batch_start(
        self, req_ids, edit_rows=None, logprob_spec=None, logits_hook=None
    ):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingDecode

        if len(req_ids) != 1:
            raise ValueError("Higgs MLX supports one active request")
        if edit_rows is not None or logprob_spec is not None or logits_hook is not None:
            raise ValueError(
                "Higgs MLX owns codebook sampling; disable MLX text sampling"
            )
        rid = req_ids[0]
        state = self._higgs_requests[rid][1]
        cache = self._req_caches[rid]
        embeddings = self.model.embed_codes(state.last_codes[None, None, :])
        codes = state.sample(self.model(cache=cache, input_embeddings=embeddings)[0])
        self._higgs_pending[rid] = codes
        return MlxPendingDecode(
            lazy_tokens=codes[:1],
            req_ids=req_ids,
            caches=[cache],
            lazy_logprobs=None,
            logprob_spec=None,
            edit_rows=None,
        )

    def decode_batch_start_chained(self, prev):
        raise ValueError("Higgs MLX requires synchronous decode")

    def decode_batch_finalize(self, pending):
        tokens = super().decode_batch_finalize(pending)
        self._commit_frame(pending.req_ids[0])
        return tokens

    def take_frame(self, req_id):
        codes = self._higgs_frames.pop(req_id)
        return codes, self._higgs_requests[req_id][1].generation_done

    def remove_request(self, req_id):
        self._higgs_requests.pop(req_id, None)
        self._higgs_pending.pop(req_id, None)
        self._higgs_frames.pop(req_id, None)
        super().remove_request(req_id)

    def clear(self):
        self._higgs_requests.clear()
        self._higgs_pending.clear()
        self._higgs_frames.clear()
        super().clear()


def make_higgs_mlx_runner_class():
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class HiggsMlxRunner(HiggsMlxModelRunner, MlxModelRunner):
        pass

    return HiggsMlxRunner
