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

# src/flow_factory/trainers/trellis2_nft.py
"""
Trellis2 DiffusionNFT Trainer with cross-GPU upstream stage sharing.

Uses ``Trellis2TrainerMixin`` for multi-stage rollout logic and inherits
``optimize()`` from ``DiffusionNFTTrainer``.  Overrides the three sparse-layout
hooks so that the base-class optimize loop handles Trellis2's
``(N_total, C)`` sparse latents correctly.
"""
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, List

import torch
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ..samples import BaseSample
from ..utils.logger_utils import setup_logger
from .nft import DiffusionNFTTrainer
from .registry import register_trainer
from .trellis2_mixin import Trellis2TrainerMixin, _WindowOOMSkipped

logger = setup_logger(__name__)


@register_trainer("trellis2_nft")
class Trellis2NFTTrainer(Trellis2TrainerMixin, DiffusionNFTTrainer):
    """DiffusionNFT trainer with cross-GPU upstream stage sharing for Trellis2.

    Inherits ``optimize()`` / ``prepare_feedback()`` / ``start()`` from
    ``DiffusionNFTTrainer``.  Overrides ``sample()`` to use the mixin's
    multi-stage rollout with ``compute_log_prob=False``, and provides the
    three sparse-layout hooks so that the base-class optimize loop
    handles ``(N_total, C)`` latent tensors correctly.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_trellis2()

    # Sampling (override)

    def sample(self) -> List[BaseSample]:
        """Generate rollouts with cross-GPU upstream stage sharing.

        Same window-merge + upstream-broadcast protocol as the GRPO
        variant, but passes ``compute_log_prob=False`` and
        ``trajectory_indices=[-1]`` (NFT only needs final latents).
        """
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []
        data_iter = iter(self.dataloader)

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
                        trajectory_indices=[-1],
                        compute_log_prob=False,
                    )
                except _WindowOOMSkipped:
                    logger.warning("Window %d skipped due to OOM", window_idx)
                    skipped_windows += 1
                    self.accelerator.wait_for_everyone()
                    continue
                self._maybe_offload_samples_to_cpu(sample_batch)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)
                self.accelerator.wait_for_everyone()

        if skipped_windows > 0:
            self.log_data({"train/skipped_windows": skipped_windows}, step=self.step)

        # FlowEdit phase
        if self._flowedit_cfg is not None:
            edited = self._sample_flowedit(samples)
            if edited:
                samples.extend(edited)

        return samples

    def _sample_flowedit(
        self,
        original_samples: List[BaseSample],
    ) -> List[BaseSample]:
        """Generate FlowEdit-edited copies of original samples.

        Mirrors the structure of the ODE rollout loop in ``sample()``:
        iterate groups, call ``_flowedit_group`` (peer of ``_rollout_group``),
        offload, buffer, sync.

        Implements all-or-nothing OOM recovery: if any group fails, all
        successfully edited samples are discarded to maintain a uniform
        ``advantage_processor.group_size`` across the epoch.
        """
        K = self.training_args.group_size
        self.advantage_processor.group_size = K * 2
        self.reward_buffer.group_size = K * 2
        edited_all: List[BaseSample] = []
        skipped_groups = 0

        self.adapter.rollout()
        use_ref = self._flowedit_cfg.get("flowedit_use_ref", True)
        fe_ctx = self.adapter.use_ref_parameters() if use_ref else nullcontext()
        with torch.no_grad(), self.autocast(), fe_ctx:
            for group_idx in tqdm(
                range(0, len(original_samples), K),
                desc=f"Epoch {self.epoch} FlowEdit",
                disable=not self.show_progress_bar,
            ):
                # Pass samples directly: _inference_flowedit already moves every
                # needed field (sparse_coords, latents, conditioning) to device
                # explicitly, so calling s.to(device) here would incorrectly move
                # sample.video (CPU from render_mesh) to GPU, causing a
                # device mismatch with FlowEdit samples whose video stays on CPU.
                group = original_samples[group_idx : group_idx + K]
                try:
                    edited_group = self._flowedit_group(group, self._flowedit_cfg)
                except _WindowOOMSkipped:
                    logger.warning("FlowEdit group %d skipped due to OOM", group_idx // K)
                    skipped_groups += 1
                    self.accelerator.wait_for_everyone()
                    continue
                self._maybe_offload_samples_to_cpu(edited_group)
                edited_all.extend(edited_group)
                self.accelerator.wait_for_everyone()

        if skipped_groups > 0:
            # All-or-nothing: partial FlowEdit would cause group_size mismatch
            # in advantage_processor and reward_buffer (expects uniform K*2 per group).
            logger.warning(
                "%d FlowEdit groups failed; discarding all %d edited samples "
                "to maintain uniform group_size=%d",
                skipped_groups,
                len(edited_all),
                self.training_args.group_size,
            )
            self.advantage_processor.group_size = self.training_args.group_size
            self.reward_buffer.group_size = self.training_args.group_size
            self.reward_buffer.clear()
            self.reward_buffer.add_samples(original_samples)
            self.log_data({"train/flowedit_skipped_groups": skipped_groups}, step=self.step)
            return []
        self.reward_buffer.add_samples(edited_all)
        return edited_all

    # Sparse-layout hooks (override)

    def _get_optimize_batch_size(self, batch: Dict[str, Any]) -> int:
        """Infer B from sparse_coords (shape[0] of all_latents is N_total)."""
        return int(batch["sparse_coords"][:, 0].max().item()) + 1

    def _broadcast_sigma(
        self,
        sigma_B: torch.Tensor,
        ref_tensor: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Expand ``(B,)`` sigma to ``(N_total, 1)`` via batch index."""
        batch_idx = batch["sparse_coords"][:, 0].long()  # (N_total,)
        return sigma_B.to(ref_tensor.device)[batch_idx].unsqueeze(-1)  # (N_total, 1)

    def _reduce_elementwise_loss(
        self,
        per_element_loss: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Scatter-mean per-point ``(N_total,)`` loss to per-sample ``(B,)``."""
        batch_idx = batch["sparse_coords"][:, 0].long()  # (N_total,)
        B = int(batch_idx.max().item()) + 1
        per_sample = torch.zeros(
            B, device=per_element_loss.device, dtype=per_element_loss.dtype
        )  # (B,)
        counts = torch.zeros(
            B, device=per_element_loss.device, dtype=per_element_loss.dtype
        )  # (B,)
        per_sample.scatter_add_(0, batch_idx, per_element_loss)  # (B,)
        counts.scatter_add_(0, batch_idx, torch.ones_like(per_element_loss))  # (B,)
        return per_sample / counts.clamp(min=1)  # (B,)
