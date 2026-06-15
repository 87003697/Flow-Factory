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

# src/flow_factory/trainers/trellis2_opd.py
"""Trellis2 OPD self-distillation trainer.

Distills the pretrained base model (teacher, LoRA disabled) into the LoRA
student along the student's own rollout trajectories using a pathwise
mean-matching loss with cross-view conditioning:

  PASS 1 (no_grad, base via use_ref_parameters):
      mu_T_j = base.forward(x_j, t_j, image_cond=c_tgt)
  PASS 2 (gradient, LoRA enabled):
      mu_S_j = (base+LoRA).forward(x_j, t_j, image_cond=c_ref)
      loss   = 0.5 * ||mu_S_j - mu_T_j||^2 / denom_j

c_tgt = student rollout multi-view render -> random frame -> re-encoded.

Reference:
[1] On-Policy Distillation of Diffusion Models — https://github.com/ali-vilab/DiffusionOPD
"""
from __future__ import annotations

import math
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import tqdm as tqdm_
from torchvision.transforms.functional import to_pil_image

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ..hparams import Trellis2OPDTrainingArguments
from ..hparams.training_args.opd import resolve_distill_step_band
from ..samples import BaseSample
from ..utils.base import create_generator, filter_kwargs
from ..utils.logger_utils import setup_logger
from ..utils.trajectory_collector import compute_trajectory_indices
from .abc import BaseTrainer
from .registry import register_trainer
from .trellis2_mixin import Trellis2TrainerMixin, _WindowOOMSkipped

logger = setup_logger(__name__)
import o_voxel


@torch.no_grad()
def compute_voxel_visibility(
    coords_3d: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    voxel_size: float,
    render_resolution: int = 512,
) -> torch.Tensor:
    """Voxel visibility via o_voxel cube rasterization (ray-cube intersection).

    Args:
        coords_3d: (N, 3) world-space positions in [-0.5, 0.5] AABB.
        extrinsics: (4, 4) camera extrinsic matrix.
        intrinsics: (3, 3) normalized camera intrinsic matrix.
        voxel_size: Side length of each voxel cube.
        render_resolution: Rasterization resolution for ray-cube test.

    Returns:
        Boolean mask (N,) where True = visible from camera.
    """

    device = coords_3d.device
    N = coords_3d.shape[0]

    attrs = torch.zeros(N, 1, device=device)
    renderer = o_voxel.rasterize.VoxelRenderer({"resolution": render_resolution})
    ret = renderer.render(coords_3d, attrs, voxel_size, extrinsics, intrinsics)

    voxel_id = ret["voxel_id"]
    visible_ids = voxel_id.unique()
    visible_ids = visible_ids[visible_ids >= 0]

    visible = torch.zeros(N, dtype=torch.bool, device=device)
    visible[visible_ids.long()] = True
    return visible





@register_trainer("trellis2_opd")
class Trellis2OPDTrainer(Trellis2TrainerMixin, BaseTrainer):
    """OPD self-distillation: pretrained base (teacher) -> LoRA (student)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.training_args: Trellis2OPDTrainingArguments
        self._init_trellis2()

        # Enable visibility mask pre-computation during rollout
        if self.training_args.use_visibility_mask:
            self._render_kwargs["compute_visibility_masks"] = True
            self._render_kwargs["visibility_mask_mode"] = self.training_args.visibility_mask_mode
            self._render_kwargs["visibility_mask_target_res"] = 16

        # OPD needs rendered views for c_tgt, so rollout must run all stages
        if self._training_stage == "dense":
            self._inference_stages = ["dense", "shape", "tex"]

        scheduler = self.adapter.scheduler
        self._is_sde = scheduler.dynamics_type != "ODE"
        self._student_noise_level = float(scheduler.noise_level) if self._is_sde else 0.0
        self._mu_store_device = (
            "cpu" if self.training_args.offload_samples_to_cpu else self.accelerator.device
        )

        logger.info(
            f"Trellis2OPDTrainer initialized: stage={self._training_stage}, "
            f"dynamics={scheduler.dynamics_type!r} (is_sde={self._is_sde}, "
            f"noise_level={self._student_noise_level}), "
            f"frame_strategy={self.training_args.teacher_frame_strategy!r}."
        )

    # =============================== Lifecycle ===============================

    def start(self) -> None:
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)

            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    "checkpoints",
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            if self.eval_args.eval_freq > 0 and self.epoch % self.eval_args.eval_freq == 0:
                self.evaluate()

            samples = self.sample()
            self.prepare_feedback(samples)
            self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    def sample(self) -> List[BaseSample]:
        """Generate rollouts with cross-GPU upstream stage sharing."""
        self.adapter.rollout()
        samples: List[BaseSample] = []
        data_iter = iter(self.dataloader)

        train_step_indices = self._select_train_step_indices(
            self.training_args.num_inference_steps, self.training_args.timestep_range
        )
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=train_step_indices,
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
                        compute_log_prob=False,
                    )
                except _WindowOOMSkipped:
                    logger.warning("Window %d skipped due to OOM", window_idx)
                    skipped_windows += 1
                    self.accelerator.wait_for_everyone()
                    continue
                samples.extend(sample_batch)
                self.accelerator.wait_for_everyone()

        if skipped_windows > 0:
            self.log_data({"train/skipped_windows": skipped_windows}, step=self.step)
        return samples

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        if not samples:
            logger.warning("All sample windows skipped (OOM), skipping feedback.")
            return

    # =============================== Optimization ===============================

    def optimize(self, samples: List[BaseSample]) -> None:
        if not samples:
            logger.warning("Trellis2OPD optimize() received no samples; skipping epoch.")
            return

        self.adapter.train()
        train_timesteps = self._select_train_step_indices(
            self.training_args.num_inference_steps, self.training_args.timestep_range
        )

        self._compute_c_tgt(samples)
        if self.training_args.use_visibility_mask:
            self._build_visibility_masks(samples)
        self._precompute_mu_T(samples, train_timesteps)

        if self.training_args.disable_grad_checkpoint_for_distill:
            self._set_transformer_checkpoint(False)
        self._distill(samples, train_timesteps)
        if self.training_args.disable_grad_checkpoint_for_distill:
            self._set_transformer_checkpoint(True)

    # =============================== c_tgt encoding ===============================

    @torch.no_grad()
    def _compute_c_tgt(self, samples: List[BaseSample]) -> None:
        """Extract a random frame from each sample's video and encode as c_tgt.

        Must run AFTER sample() — inference overwrites extra_kwargs.
        """
        batch_pil: List[List] = []
        for sample in samples:
            if sample.video is None:
                batch_pil.append([])
                continue
            num_frames = sample.video.shape[0]
            gen = create_generator(self.training_args.seed, self.epoch, sample.unique_id)
            frame_idx = torch.randint(0, num_frames, (1,), generator=gen).item()
            sample.extra_kwargs["c_tgt_frame_idx"] = frame_idx
            frame_pil = to_pil_image(sample.video[frame_idx].clamp(0, 1))
            batch_pil.append([frame_pil])

        # preprocess_func needs rembg_model + image_encoder on GPU; they were
        # offloaded to CPU after dataset preprocessing (_init_dataloader).
        # Only load these two — sparse_structure_flow_model is also a preprocessing
        # module but is the trainable transformer; offloading it would break forward.
        _preprocess_only = ["rembg_model", "image_encoder"]
        device = self.accelerator.device
        self.adapter.on_load_components(components=_preprocess_only, device=device)
        result = self.adapter.preprocess_func(images=batch_pil)
        self.adapter.off_load_components(components=_preprocess_only)
        cond_512_list = result.get("image_cond_512", [])

        for i, sample in enumerate(samples):
            if i < len(cond_512_list):
                sample.extra_kwargs["image_cond_tgt"] = cond_512_list[i]

    # =============================== Visibility mask ===============================

    @torch.no_grad()
    def _build_visibility_masks(
        self,
        samples: List[BaseSample],
    ) -> None:
        """Index pre-computed visibility masks by c_tgt frame index.

        During rollout, render_latents stores (num_frames, D, D, D) masks on
        each sample. Here we simply index the correct frame for the selected
        c_tgt view — zero compute at this point.
        """
        for sample in samples:
            frame_idx = sample.extra_kwargs.get("c_tgt_frame_idx")
            masks = sample.extra_kwargs.get("visibility_masks")
            if frame_idx is None or masks is None:
                continue
            sample.extra_kwargs["visibility_mask"] = masks[frame_idx].unsqueeze(0).to(
                self.accelerator.device
            )  # (1, D, D, D)

    # =============================== PASS 1: teacher targets ===============================

    @torch.no_grad()
    def _precompute_mu_T(
        self,
        samples: List[BaseSample],
        train_timesteps: torch.Tensor,
    ) -> None:
        """PASS 1: cache teacher per-step mean mu_T on each sample."""
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = math.ceil(len(samples) / per_device_batch_size)

        with self.adapter.use_ref_parameters():
            with self.autocast():
                for batch_idx in tqdm(
                    range(num_batches),
                    total=num_batches,
                    desc=f"Epoch {self.epoch} Teacher targets",
                    disable=not self.show_progress_bar,
                ):
                    start = batch_idx * per_device_batch_size
                    micro_batch_samples = [
                        sample.to(device)
                        for sample in samples[start : start + per_device_batch_size]
                    ]
                    batch = BaseSample.stack(micro_batch_samples)

                    cond_override = {"image_cond": batch["image_cond_tgt"]}

                    mu_teacher_steps = [
                        self._forward_step(batch, timestep_index, cond_override=cond_override)[0]
                        .detach()
                        for timestep_index in train_timesteps
                    ]
                    mu_teacher_stacked = torch.stack(mu_teacher_steps, dim=1)

                    # Reference KL anchor: ref model under c_ref (no cond_override)
                    mu_ref_stacked = None
                    if self.training_args.ref_kl_beta > 0:
                        mu_ref_steps = [
                            self._forward_step(batch, timestep_index)[0]
                            .detach()
                            for timestep_index in train_timesteps
                        ]
                        mu_ref_stacked = torch.stack(mu_ref_steps, dim=1)

                    for i, sample in enumerate(micro_batch_samples):
                        sample.extra_kwargs["mu_teacher"] = (
                            mu_teacher_stacked[i].to(self._mu_store_device).clone()
                        )
                        if mu_ref_stacked is not None:
                            sample.extra_kwargs["mu_ref"] = (
                                mu_ref_stacked[i].to(self._mu_store_device).clone()
                            )
        torch.clear_autocast_cache()

    # =============================== PASS 2: student distillation ===============================

    def _distill(
        self,
        samples: List[BaseSample],
        train_timesteps: torch.Tensor,
    ) -> None:
        """PASS 2: student-only gradient loop matching mu_S to cached mu_T."""
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = math.ceil(len(samples) / per_device_batch_size)

        for inner_epoch in range(self.training_args.num_inner_epochs):
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            self.adapter.train()
            kl_sum = torch.zeros(1, device=device)
            kl_count = torch.zeros(1, device=device)
            ref_kl_sum = torch.zeros(1, device=device)
            mask_cov_sum = torch.zeros(1, device=device)
            mask_cov_count = torch.zeros(1, device=device)
            grad_norm = None

            # Load mu_ref if ref_kl_beta > 0
            use_ref_kl = self.training_args.ref_kl_beta > 0
            use_vis_mask = self.training_args.use_visibility_mask

            with self.autocast():
                for batch_idx in tqdm(
                    range(num_batches),
                    total=num_batches,
                    desc=f"Epoch {self.epoch} Distill",
                    position=0,
                    disable=not self.show_progress_bar,
                ):
                    start = batch_idx * per_device_batch_size
                    batch_samples = [
                        s.to(device)
                        for s in shuffled_samples[start : start + per_device_batch_size]
                    ]
                    batch = BaseSample.stack(batch_samples)

                    # extra_kwargs tensors are not moved by BaseSample.to()
                    mu_teacher_all = batch["mu_teacher"]
                    if not isinstance(mu_teacher_all, torch.Tensor):
                        raise RuntimeError(
                            "Expected cached teacher means `mu_teacher` (a tensor) on every "
                            f"sample, got {type(mu_teacher_all).__name__}. "
                            "PASS 1 (_precompute_mu_T) must run before PASS 2."
                        )
                    mu_teacher_all = mu_teacher_all.to(device)

                    mu_ref_all = None
                    if use_ref_kl:
                        mu_ref_all = batch["mu_ref"].to(device)

                    vis_mask = None
                    if use_vis_mask:
                        masks = [s.extra_kwargs.get("visibility_mask") for s in batch_samples]
                        if all(m is not None for m in masks):
                            vis_mask = torch.stack(masks).to(device)  # (B, 1, D, H, W)

                    for idx, timestep_index in enumerate(
                        tqdm(
                            train_timesteps,
                            desc=f"Epoch {self.epoch} Timestep",
                            position=1,
                            leave=False,
                            disable=not self.show_progress_bar,
                        )
                    ):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
                            mu_S, std_dev_t, dt = self._forward_step(batch, timestep_index)

                            mu_T = mu_teacher_all[:, idx]
                            diff_sq = (mu_S.float() - mu_T.float()).pow(2)
                            if vis_mask is not None:
                                diff_sq = diff_sq * vis_mask
                                C = mu_S.shape[1]
                                mask_count = vis_mask.flatten(1).sum(dim=1) * C
                                per_sample_mse = diff_sq.flatten(1).sum(dim=1) / mask_count.clamp(min=1.0)
                            else:
                                per_sample_mse = diff_sq.flatten(1).mean(dim=1)

                            denom = self.adapter.scheduler.get_kl_divergence_denominator(
                                std_dev_t, dt
                            )
                            if isinstance(denom, torch.Tensor):
                                denom = denom.reshape(per_sample_mse.shape[0], -1).mean(dim=1)

                            per_sample_kl = 0.5 * (per_sample_mse / denom)
                            distill_loss = per_sample_kl.mean()

                            # Reference KL regularization
                            ref_kl_loss = torch.zeros(1, device=device)
                            if use_ref_kl:
                                mu_ref = mu_ref_all[:, idx]
                                ref_mse = (
                                    (mu_S.float() - mu_ref.float()).pow(2).flatten(1).mean(dim=1)
                                )
                                ref_kl_loss = self.training_args.ref_kl_beta * ref_mse.mean()

                            loss = distill_loss + ref_kl_loss

                            self.accelerator.backward(loss)

                            with torch.no_grad():
                                kl_sum += per_sample_kl.detach().sum()
                                kl_count += per_sample_kl.shape[0]
                                if use_ref_kl:
                                    ref_kl_sum += ref_kl_loss.detach()
                                if vis_mask is not None:
                                    total_voxels = vis_mask[0].numel()
                                    coverage = vis_mask.flatten(1).sum(dim=1) / total_voxels
                                    mask_cov_sum += coverage.sum()
                                    mask_cov_count += coverage.shape[0]

                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                self._log_distill_metrics(
                                    kl_sum, kl_count, grad_norm, ref_kl_sum,
                                    mask_cov_sum, mask_cov_count,
                                )
                                self.step += 1
                                kl_sum.zero_()
                                kl_count.zero_()
                                ref_kl_sum.zero_()
                                mask_cov_sum.zero_()
                                mask_cov_count.zero_()

    # =============================== Helpers ===============================

    def _forward_step(
        self,
        batch: Dict[str, Any],
        timestep_index: Union[int, torch.Tensor],
        cond_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward the adapter at stored trajectory step ``timestep_index``."""
        latents_index_map = batch["latent_index_map"]
        num_timesteps = batch["timesteps"].shape[1]

        t = batch["timesteps"][:, timestep_index]
        t_next = (
            batch["timesteps"][:, timestep_index + 1]
            if timestep_index + 1 < num_timesteps
            else torch.zeros_like(t)
        )
        latents = batch["all_latents"][:, latents_index_map[timestep_index]]
        next_latents = batch["all_latents"][:, latents_index_map[timestep_index + 1]]

        forward_inputs = {
            **self.training_args,
            **batch,
            "t": t,
            "t_next": t_next,
            "latents": latents,
            "next_latents": next_latents,
            "compute_log_prob": False,
            "noise_level": self._student_noise_level,
        }
        if cond_override:
            forward_inputs.update(cond_override)
        forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)
        output = self.adapter.forward(**forward_inputs)

        if output.next_latents_mean is None:
            raise RuntimeError(
                "Trellis2OPD requires `next_latents_mean` from adapter.forward, got None."
            )
        return output.next_latents_mean, output.std_dev_t, output.dt

    @staticmethod
    def _select_train_step_indices(
        num_inference_steps: int,
        timestep_range: Union[float, Tuple[float, float]],
    ) -> torch.Tensor:
        lo, hi = resolve_distill_step_band(num_inference_steps, timestep_range)
        return torch.arange(lo, hi, dtype=torch.long)

    def _log_distill_metrics(
        self,
        kl_sum: torch.Tensor,
        kl_count: torch.Tensor,
        grad_norm: Optional[torch.Tensor],
        ref_kl_sum: Optional[torch.Tensor] = None,
        mask_cov_sum: Optional[torch.Tensor] = None,
        mask_cov_count: Optional[torch.Tensor] = None,
    ) -> None:
        packed = torch.stack([kl_sum.squeeze(), kl_count.squeeze()])
        packed = cast(torch.Tensor, self.accelerator.reduce(packed, reduction="sum"))
        g_sum, g_count = packed[0], packed[1]

        metrics: Dict[str, Any] = {}
        if g_count > 0:
            metrics["kl_div"] = g_sum / g_count
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm
        if ref_kl_sum is not None and ref_kl_sum.item() > 0:
            metrics["ref_kl"] = ref_kl_sum.squeeze()
        if mask_cov_count is not None and mask_cov_count.item() > 0:
            metrics["mask_coverage"] = (mask_cov_sum / mask_cov_count).squeeze()

        self.log_data({f"train/{k}": v for k, v in metrics.items()}, step=self.step)
