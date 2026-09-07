# SPDX-License-Identifier: Apache-2.0
"""Higgs code collection and streaming over the shared MLX worker bridge."""

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.mlx_model_worker import MlxSchedulerModelRunner
from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner


class HiggsMlxSchedulerRunner(MlxSchedulerModelRunner, HiggsTTSModelRunner):
    """Reuse Higgs output buffers/cadence without its CUDA execution hooks."""

    before_prefill = ModelRunner.before_prefill
    before_decode = ModelRunner.before_decode
    next_input_token_ids = ModelRunner.next_input_token_ids

    def lookahead_eligible(self, batch):
        return False

    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        for request in requests:
            self.tp_worker._mlx_runner.register_request(
                request.request_id, request.data
            )
        return super().custom_prefill_forward(forward_batch, schedule_batch, requests)

    def _collect_frames(self, requests):
        for request in requests:
            codes, done = self.tp_worker._mlx_runner.take_frame(request.request_id)
            data = request.data
            row = self._append_output_code(data, torch.from_numpy(codes).long())
            data.generation_done = done
            self._queue_or_emit_code_chunk(
                request, row, force=self._is_final_code_step(data)
            )
            self._mark_sampler_finished(data.req, done)

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        self._collect_frames(requests)

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        self._collect_frames(requests)
