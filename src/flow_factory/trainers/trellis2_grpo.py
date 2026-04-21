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
Trellis2 GRPO Trainer with upstream stage sharing.

Extends GRPOTrainer to accumulate K // per_device_batch_size consecutive
dataloader batches (same prompt, guaranteed by GroupContiguousSampler)
before calling one of the shared-upstream inference methods.
"""
from typing import List, Dict, Any, Union
from functools import partial

import torch
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .grpo import GRPOTrainer
from .registry import register_trainer
from ..hparams import GRPOTrainingArguments
from ..samples import BaseSample
from ..utils.base import filter_kwargs
from ..utils.logger_utils import setup_logger
from ..utils.trajectory_collector import compute_trajectory_indices

logger = setup_logger(__name__)


@register_trainer('trellis2_grpo')
class Trellis2GRPOTrainer(GRPOTrainer):
    """GRPO trainer with upstream stage sharing for Trellis2.

    For multi-stage Trellis2 training, upstream stages (dense, shape) are
    deterministic given the same conditioning — so all K samples in a
    group can share one upstream run.  This trainer:

    1. Accumulates ``group_size // per_device_batch_size`` consecutive
       dataloader batches into one merged batch of K samples.
    2. Dispatches to the appropriate shared-upstream inference method
       based on ``target_flow_model``.
    3. Inherits ``optimize()`` from ``GRPOTrainer`` unchanged — it
       continues to use ``per_device_batch_size`` for gradient batching.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.config.data_args.sampler_type != 'group_contiguous':
            raise ValueError(
                "Trellis2GRPOTrainer requires data.sampler_type='group_contiguous' "
                "so consecutive batches belong to the same prompt group."
            )
        train_stage = self.adapter.pipeline._target_flow_model.split('_')[0]
        if train_stage == 'shape':
            self._inference_fn = self.adapter.inference_with_shared_dense
        elif train_stage == 'tex':
            self._inference_fn = self.adapter.inference_with_shared_dense_shape
        else:
            self._inference_fn = self.adapter.inference

        extra = self.model_args.extra_kwargs
        self._render_kwargs = {
            'decode_output': extra.get('decode_output', False),
            'render_num_frames': extra.get('render_num_frames', 24),
            'render_resolution': extra.get('render_resolution', 512),
            'envmap_path': extra.get('envmap_path', None),
        }


    def sample(self) -> List[BaseSample]:
        """Generate rollouts with upstream stage sharing.

        Accumulates consecutive batches until one full group (K samples)
        is assembled, then calls the shared-upstream inference method once.
        """
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []
        data_iter = iter(self.dataloader)
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.scheduler.train_timesteps,
            num_inference_steps=self.training_args.num_inference_steps,
        )

        K = self.training_args.group_size
        bs = self.training_args.per_device_batch_size
        if K % bs != 0:
            raise ValueError(
                f"group_size ({K}) must be divisible by "
                f"per_device_batch_size ({bs}) for batch accumulation."
            )
        batches_per_group = K // bs
        if self.training_args.num_batches_per_epoch % batches_per_group != 0:
            raise ValueError(
                "num_batches_per_epoch "
                f"({self.training_args.num_batches_per_epoch}) must be divisible by "
                f"batches_per_group ({batches_per_group}) to avoid dropping merged groups."
            )
        num_groups = self.training_args.num_batches_per_epoch // batches_per_group

        with torch.no_grad(), self.autocast():
            for group_idx in tqdm(
                range(num_groups),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                group_batches = [next(data_iter) for _ in range(batches_per_group)]
                merged_batch = self._merge_batches(group_batches)

                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': True,
                    'trajectory_indices': trajectory_indices,
                    **self._render_kwargs,
                    **merged_batch,
                }
                sample_kwargs = filter_kwargs(self._inference_fn, **sample_kwargs)
                sample_batch = self._inference_fn(**sample_kwargs)

                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)
                self.accelerator.wait_for_everyone()

        return samples

    def _extra_eval_inference_kwargs(self) -> dict:
        return {**self._render_kwargs, 'stages': ['dense', 'shape', 'tex']}

    @staticmethod
    def _merge_batches(batches: list) -> dict:
        """Merge consecutive dataloader batches (same prompt group) into one.

        List-typed values are concatenated; tensor values are cat-ed along
        dim 0; scalar/other values are kept from the first batch (they are
        identical across same-prompt repetitions).
        """
        merged = {}
        for key in batches[0]:
            values = [b[key] for b in batches]
            if isinstance(values[0], list):
                merged[key] = [item for sublist in values for item in sublist]
            elif isinstance(values[0], torch.Tensor):
                merged[key] = torch.cat(values, dim=0)
            else:
                merged[key] = values[0]
        return merged
