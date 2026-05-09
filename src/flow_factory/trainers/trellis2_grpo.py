# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/trainers/trellis2_grpo.py
"""
Trellis2 GRPO Trainer with cross-GPU upstream stage sharing.

Uses ``Trellis2TrainerMixin`` for multi-stage rollout logic and inherits
``optimize()`` from ``GRPOTrainer``.
"""
from functools import partial
from typing import List

import torch
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ..samples import BaseSample
from ..utils.logger_utils import setup_logger
from ..utils.trajectory_collector import compute_trajectory_indices
from .grpo import GRPOTrainer
from .registry import register_trainer
from .trellis2_mixin import Trellis2TrainerMixin, _WindowOOMSkipped

logger = setup_logger(__name__)


@register_trainer("trellis2_grpo")
class Trellis2GRPOTrainer(Trellis2TrainerMixin, GRPOTrainer):
    """GRPO trainer with cross-GPU upstream stage sharing for Trellis2.

    For multi-stage Trellis2 training, upstream stages (dense, shape) are
    deterministic given the same conditioning -- so they only need to run
    once per unique prompt.  This trainer:

    1. Merges ``_batches_to_merge`` consecutive dataloader batches into one
       rollout window (topology-dependent: ``K // bs`` for group_contiguous,
       ``1`` for distributed samplers).
    2. Runs upstream stages via a cross-GPU owner-broadcast protocol:
       each unique prompt is computed by exactly one owner rank, then
       broadcast to all ranks holding copies.
    3. Calls ``inference()`` for the remaining (training + downstream)
       stages with pre-filled samples (stage-skip kicks in).
    4. Inherits ``optimize()`` from ``GRPOTrainer`` unchanged.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_trellis2()

    def sample(self) -> List[BaseSample]:
        """Generate rollouts with cross-GPU upstream stage sharing."""
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []
        data_iter = iter(self.dataloader)
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.scheduler.train_timesteps,
            num_inference_steps=self.training_args.num_inference_steps,
        )

        btm = self._batches_to_merge
        if self.training_args.num_batches_per_epoch % btm != 0:
            raise ValueError(
                f"num_batches_per_epoch ({self.training_args.num_batches_per_epoch}) "
                f"must be divisible by batches_to_merge ({btm})."
            )
        num_windows = self.training_args.num_batches_per_epoch // btm

        skipped_windows = 0
        with torch.no_grad(), self.autocast():
            for window_idx in tqdm(
                range(num_windows),
                desc=f"Epoch {self.epoch} Sampling",
                disable=not self.show_progress_bar,
            ):
                window_batches = [next(data_iter) for _ in range(btm)]
                merged_batch = self._merge_batches(window_batches)
                try:
                    sample_batch = self._rollout_group(
                        merged_batch,
                        trajectory_indices,
                        compute_log_prob=True,
                    )
                except _WindowOOMSkipped:
                    logger.warning("Window %d skipped due to OOM", window_idx)
                    skipped_windows += 1
                    self.accelerator.wait_for_everyone()
                    continue
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)
                self.accelerator.wait_for_everyone()

        if skipped_windows > 0:
            self.log_data({"train/skipped_windows": skipped_windows}, step=self.step)
        return samples
