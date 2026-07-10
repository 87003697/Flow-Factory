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

import base64
import math
import os
from functools import partial
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import requests
import torch
import tqdm as tqdm_
from PIL import Image
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
from trellis2.modules.sparse import SparseTensor as _SparseTensor


@torch.no_grad()
def compute_voxel_visibility(
    coords_3d: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    voxel_size: float,
    render_resolution: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Voxel visibility via o_voxel cube rasterization (ray-cube intersection).

    Args:
        coords_3d: (N, 3) world-space positions in [-0.5, 0.5] AABB.
        extrinsics: (4, 4) camera extrinsic matrix.
        intrinsics: (3, 3) normalized camera intrinsic matrix.
        voxel_size: Side length of each voxel cube.
        render_resolution: Rasterization resolution for ray-cube test.

    Returns:
        (visible, voxel_id): Boolean mask (N,) and per-pixel hit index (H, W).
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
    return visible, voxel_id


class TargetImageBuffer:
    """Buffer for target images extracted from rollout videos.

    When flowedit_fn is provided, each raw render frame is also sent through
    FlowEdit to produce an edited variant. The buffer stores (raw, edited) pairs.
    """

    def __init__(self, seed: int, flowedit_fn: Optional[Callable[[Image.Image, Image.Image], Image.Image]] = None):
        self._seed = seed
        self._flowedit_fn = flowedit_fn
        self._raw_images: List[Optional[Image.Image]] = []
        self._edited_images: List[Optional[Image.Image]] = []

    def clear(self):
        self._raw_images.clear()
        self._edited_images.clear()

    def add_samples(self, samples: List[BaseSample], epoch: int):
        for s in samples:
            if s.video is None:
                self._raw_images.append(None)
                self._edited_images.append(None)
                continue
            gen = create_generator(self._seed, epoch, s.unique_id)
            idx = torch.randint(0, s.video.shape[0], (1,), generator=gen).item()
            s.extra_kwargs["c_tgt_frame_idx"] = idx
            raw_pil = to_pil_image(s.video[idx].clamp(0, 1))
            self._raw_images.append(raw_pil)

            if self._flowedit_fn is not None:
                if s.condition_images:
                    ci = s.condition_images[0]
                    cond_pil = ci if isinstance(ci, Image.Image) else to_pil_image(ci.clamp(0, 1))
                else:
                    cond_pil = raw_pil
                try:
                    edited_pil = self._flowedit_fn(raw_pil, cond_pil)
                except Exception as e:
                    logger.warning("FlowEdit call failed (uid=%s): %s; falling back to raw", s.unique_id, e)
                    edited_pil = raw_pil
                self._edited_images.append(edited_pil)
            else:
                self._edited_images.append(raw_pil)

    def get_raw_images(self) -> List[Optional[Image.Image]]:
        return self._raw_images

    def get_edited_images(self) -> List[Optional[Image.Image]]:
        return self._edited_images


def _call_flowedit_server(
    source: Image.Image,
    condition: Image.Image,
    *,
    server: str,
    prompt: str,
    cfg_tgt: float,
    cfg_src: float,
    n_max: int,
    steps: int,
    timeout: float,
) -> Image.Image:
    """Call a FlowEdit vllm-omni server to edit source guided by condition."""

    def _img_to_b64(img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_img_to_b64(source)}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_img_to_b64(condition)}"}},
            ],
        }],
        "extra_body": {
            "num_inference_steps": steps,
            "guidance_scale": 1,
            "true_cfg_scale": cfg_tgt,
            "true_cfg_scale_src": cfg_src,
            "n_max": n_max,
        },
    }
    resp = requests.post(f"{server}/v1/chat/completions", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    b64_url = data["choices"][0]["message"]["content"][0]["image_url"]["url"]
    b64_str = b64_url.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")



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
            if self._training_stage == "dense":
                self._render_kwargs["visibility_mask_target_res"] = 16
            else:
                self._render_kwargs["visibility_mask_target_res"] = self._infer_ss_resolution()

        # OPD needs rendered views for c_tgt, so rollout must run all stages
        if self._training_stage == "dense":
            self._inference_stages = ["dense", "shape", "tex"]

        # FlowEdit contrastive distillation setup
        flowedit_fn: Optional[Callable[[Image.Image, Image.Image], Image.Image]] = None
        self._use_contrastive = self.training_args.flowedit is not None
        if self._use_contrastive:
            fe_cfg = self.training_args.flowedit
            rank = int(os.environ.get("LOCAL_RANK", 0))
            server_url = fe_cfg.server_url_template.format(rank=rank)
            flowedit_fn = partial(
                _call_flowedit_server,
                server=server_url,
                prompt=fe_cfg.prompt,
                cfg_tgt=fe_cfg.cfg_tgt,
                cfg_src=fe_cfg.cfg_src,
                n_max=fe_cfg.n_max,
                steps=fe_cfg.steps,
                timeout=fe_cfg.timeout,
            )
            logger.info("Contrastive FlowEdit enabled: server=%s", server_url)

        self._tgt_buffer = TargetImageBuffer(self.training_args.seed, flowedit_fn=flowedit_fn)

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
        self._tgt_buffer.clear()
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
                self._tgt_buffer.add_samples(sample_batch, self.epoch)
                self.accelerator.wait_for_everyone()

        if skipped_windows > 0:
            self.log_data({"train/skipped_windows": skipped_windows}, step=self.step)
        return samples

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        if not samples:
            logger.warning("All sample windows skipped (OOM), skipping feedback.")
            return
        if self._use_contrastive:
            self._encode_c_tgt(samples, self._tgt_buffer.get_edited_images(), key="image_cond_tgt")
            self._encode_c_tgt(samples, self._tgt_buffer.get_raw_images(), key="image_cond_neg")
        else:
            self._encode_c_tgt(samples, self._tgt_buffer.get_edited_images(), key="image_cond_tgt")
        self._build_visibility(samples)
        self._log_train_samples(samples)
        self._tgt_buffer.clear()

    # =============================== Optimization ===============================

    def optimize(self, samples: List[BaseSample]) -> None:
        if not samples:
            logger.warning("Trellis2OPD optimize() received no samples; skipping epoch.")
            return

        self.adapter.train()
        train_timesteps = self._select_train_step_indices(
            self.training_args.num_inference_steps, self.training_args.timestep_range
        )

        self._precompute_mu_T(samples, train_timesteps)

        self._distill(samples, train_timesteps)

    # =============================== c_tgt encoding ===============================

    @torch.no_grad()
    def _encode_c_tgt(
        self,
        samples: List[BaseSample],
        images: List[Optional[Image.Image]],
        key: str = "image_cond_tgt",
    ) -> None:
        """Encode pre-extracted target frames as conditioning tensor.

        Frame extraction is done by TargetImageBuffer during sample(); this
        method only handles the encoding pass (rembg + DINOv2).
        """
        batch_pil: List[List] = []
        for img in images:
            batch_pil.append([img] if img is not None else [])

        _preprocess_only = ["rembg_model", "image_encoder"]
        device = self.accelerator.device
        self.adapter.on_load_components(components=_preprocess_only, device=device)
        result = self.adapter.preprocess_func(images=batch_pil)
        self.adapter.off_load_components(components=_preprocess_only)
        cond_512_list = result.get("image_cond_512", [])

        for i, sample in enumerate(samples):
            if i < len(cond_512_list):
                sample.extra_kwargs[key] = cond_512_list[i]

    # =============================== Visibility mask ===============================

    @torch.no_grad()
    def _build_visibility_masks_dense(self, samples):
        """Index pre-computed 16³ visibility grid by frame_idx."""
        for sample in samples:
            frame_idx = sample.extra_kwargs.get("c_tgt_frame_idx")  # int
            masks = sample.extra_kwargs.get("visibility_masks")  # (num_frames, 16, 16, 16)
            if frame_idx is None or masks is None:
                continue
            # Select the frame's visibility grid and add channel dim
            sample.extra_kwargs["visibility_mask"] = masks[frame_idx].unsqueeze(0).to(
                self.accelerator.device
            )  # (1, 16, 16, 16) — broadcasts with (B, C, 16, 16, 16) latent

    @torch.no_grad()
    def _build_visibility_masks_sparse(self, samples):
        """Index ss_res³ grid by sparse_coords → per-voxel visibility scalar."""
        for sample in samples:
            frame_idx = sample.extra_kwargs.get("c_tgt_frame_idx")  # int
            masks = sample.extra_kwargs.get("visibility_masks")  # (num_frames, ss_res, ss_res, ss_res)
            if frame_idx is None or masks is None:
                continue
            frame_mask = masks[frame_idx]  # (ss_res, ss_res, ss_res)
            ss_res = frame_mask.shape[0]
            # sparse_coords[:, 0] = batch_idx, [:, 1:4] = (x, y, z) in [0, ss_res)
            coords = sample.sparse_coords[:, 1:4].long().cpu()  # (N_b, 3)
            assert coords.max() < ss_res and coords.min() >= 0, (
                f"sparse_coords range [{coords.min()}, {coords.max()}] outside [0, {ss_res})"
            )
            # Advanced indexing: look up each voxel's visibility from the 3D grid
            sample.extra_kwargs["visibility_mask"] = frame_mask[
                coords[:, 0], coords[:, 1], coords[:, 2]
            ].to(self.accelerator.device)  # (N_b,) float per-voxel visibility

    # -- prepare_feedback helpers: visibility dispatch + logging -------------------

    @torch.no_grad()
    def _build_visibility(self, samples: List[BaseSample]) -> None:
        """Dispatch to sparse/dense visibility mask builder (no-op if disabled)."""
        if not self.training_args.use_visibility_mask:
            return
        build = (
            self._build_visibility_masks_sparse
            if self._training_stage != "dense"
            else self._build_visibility_masks_dense
        )
        build(samples)

    def _log_train_samples(self, samples: List[BaseSample]) -> None:
        """Attach diagnostic panels and log the first 2 samples to the logger."""
        if not self.accelerator.is_main_process:
            return
        # Attach visibility heatmap overlay before logging
        if self.training_args.use_visibility_mask:
            for s in samples:
                self._attach_visibility_heatmap(s)
        if self._use_contrastive:
            self._log_flowedit_comparison(samples)
        self.log_data({"train_samples": samples[:2]}, step=self.step)

    def _log_flowedit_comparison(self, samples: List[BaseSample]) -> None:
        """Log side-by-side wandb images: condition | raw_render (neg) | flowedit (edited)."""
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        raw_images = self._tgt_buffer.get_raw_images()
        edited_images = self._tgt_buffer.get_edited_images()
        log_dict = {}

        for i, s in enumerate(samples[:4]):
            if s.condition_images:
                ci = s.condition_images[0]
                cond_pil = ci if isinstance(ci, Image.Image) else to_pil_image(ci.clamp(0, 1))
            else:
                cond_pil = None
            raw_pil = raw_images[i] if i < len(raw_images) else None
            edited_pil = edited_images[i] if i < len(edited_images) else None
            if cond_pil is None or raw_pil is None or edited_pil is None:
                continue
            w, h = cond_pil.size
            raw_pil = raw_pil.resize((w, h), Image.Resampling.LANCZOS)
            edited_pil = edited_pil.resize((w, h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (w * 3, h))
            canvas.paste(cond_pil.convert("RGB"), (0, 0))
            canvas.paste(raw_pil.convert("RGB"), (w, 0))
            canvas.paste(edited_pil.convert("RGB"), (w * 2, 0))
            log_dict[f"flowedit_comparison/{i}"] = wandb.Image(
                canvas, caption="condition | raw_render (neg) | flowedit (pos)"
            )

        if log_dict:
            self.log_data(log_dict, step=self.step)

    def _attach_visibility_heatmap(self, sample: BaseSample) -> None:
        """Generate binary heatmap showing target-visible voxels across all views.

        Calls adapter.generate_target_heatmap() which does ray-cube intersection
        from the target camera to find visible voxels, then renders them in
        viridis (yellow = target-visible, purple = not) from every render view.
        Result is appended to _log_panels for LogFormatter to pick up.
        """
        tgt_idx = sample.extra_kwargs.get("c_tgt_frame_idx")
        if tgt_idx is None:
            return
        num_frames = sample.video.shape[0] if sample.video is not None else 24
        heatmap = self.adapter.generate_target_heatmap(sample, tgt_idx, num_frames)
        if heatmap is not None:
            sample.extra_kwargs.setdefault("_log_panels", []).append(heatmap)

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

        is_sparse = self._training_stage != "dense"
        _store_mu = self._store_mu_cache_sparse if is_sparse else self._store_mu_cache_dense

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

                    # Positive target: teacher under c_tgt (edited frame if contrastive)
                    cond_override_pos = {"image_cond": batch["image_cond_tgt"]}
                    mu_teacher_steps = [
                        self._forward_step(batch, timestep_index, cond_override=cond_override_pos)[0]
                        .detach()
                        for timestep_index in train_timesteps
                    ]

                    # Negative target: teacher under raw render (contrastive only)
                    mu_neg_steps = None
                    if self._use_contrastive:
                        cond_override_neg = {"image_cond": batch["image_cond_neg"]}
                        mu_neg_steps = [
                            self._forward_step(batch, timestep_index, cond_override=cond_override_neg)[0]
                            .detach()
                            for timestep_index in train_timesteps
                        ]

                    # Reference KL anchor: ref model under c_ref (no cond_override)
                    mu_ref_steps = None
                    if self.training_args.ref_kl_beta > 0:
                        mu_ref_steps = [
                            self._forward_step(batch, timestep_index)[0]
                            .detach()
                            for timestep_index in train_timesteps
                        ]

                    _store_mu(mu_teacher_steps, mu_ref_steps, micro_batch_samples, batch, mu_neg_steps)
        torch.clear_autocast_cache()

    def _store_mu_cache_dense(self, mu_teacher_steps, mu_ref_steps, micro_batch_samples, batch, mu_neg_steps=None):
        mu_T = torch.stack(mu_teacher_steps, dim=1)  # (B, T, C, D, H, W)
        mu_ref = torch.stack(mu_ref_steps, dim=1) if mu_ref_steps else None
        mu_neg = torch.stack(mu_neg_steps, dim=1) if mu_neg_steps else None
        for i, sample in enumerate(micro_batch_samples):
            sample.extra_kwargs["mu_teacher"] = mu_T[i].to(self._mu_store_device).clone()
            if mu_ref is not None:
                sample.extra_kwargs["mu_ref"] = mu_ref[i].to(self._mu_store_device).clone()
            if mu_neg is not None:
                sample.extra_kwargs["mu_neg"] = mu_neg[i].to(self._mu_store_device).clone()

    def _store_mu_cache_sparse(self, mu_teacher_steps, mu_ref_steps, micro_batch_samples, batch, mu_neg_steps=None):
        mu_T = torch.stack(mu_teacher_steps, dim=0)  # (T, N_total, C)
        mu_ref = torch.stack(mu_ref_steps, dim=0) if mu_ref_steps else None
        mu_neg = torch.stack(mu_neg_steps, dim=0) if mu_neg_steps else None
        batch_idx = batch["sparse_coords"][:, 0].long()
        counts = [(batch_idx == b).sum().item() for b in range(len(micro_batch_samples))]
        splits_T = mu_T.split(counts, dim=1)
        splits_ref = mu_ref.split(counts, dim=1) if mu_ref is not None else [None] * len(counts)
        splits_neg = mu_neg.split(counts, dim=1) if mu_neg is not None else [None] * len(counts)
        for i, sample in enumerate(micro_batch_samples):
            sample.extra_kwargs["mu_teacher"] = splits_T[i].to(self._mu_store_device).clone()
            if splits_ref[i] is not None:
                sample.extra_kwargs["mu_ref"] = splits_ref[i].to(self._mu_store_device).clone()
            if splits_neg[i] is not None:
                sample.extra_kwargs["mu_neg"] = splits_neg[i].to(self._mu_store_device).clone()

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

        is_sparse = self._training_stage != "dense"
        if is_sparse:
            _load_mu = self._load_mu_targets_sparse
            _get_mu = self._get_mu_at_step_sparse
            _compute_mse = self._compute_mse_sparse
            _load_vis = self._load_vis_mask_sparse
        else:
            _load_mu = self._load_mu_targets_dense
            _get_mu = self._get_mu_at_step_dense
            _compute_mse = self._compute_mse_dense
            _load_vis = self._load_vis_mask_dense

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
            mse_pos_sum = torch.zeros(1, device=device)
            mse_neg_sum = torch.zeros(1, device=device)
            grad_norm = None

            use_ref_kl = self.training_args.ref_kl_beta > 0
            use_vis_mask = self.training_args.use_visibility_mask
            use_contrastive = self._use_contrastive

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
                    B = len(batch_samples)
                    mu_teacher_all, mu_ref_all, mu_neg_all, ctx = _load_mu(batch_samples, device)
                    vis_mask = _load_vis(batch_samples, device) if use_vis_mask else None

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
                            mu_T = _get_mu(mu_teacher_all, idx)
                            per_sample_mse_pos = _compute_mse(mu_S, mu_T, ctx, B, vis_mask)

                            denom = self.adapter.scheduler.get_kl_divergence_denominator(
                                std_dev_t, dt
                            )
                            if isinstance(denom, torch.Tensor) and denom.numel() > 1:
                                denom = denom.reshape(B, -1).mean(dim=1)

                            if use_contrastive:
                                mu_neg = _get_mu(mu_neg_all, idx)
                                per_sample_mse_neg = _compute_mse(mu_S, mu_neg, ctx, B, vis_mask)
                                per_sample_kl = 0.5 * (per_sample_mse_pos - per_sample_mse_neg) / denom
                            else:
                                per_sample_mse_neg = torch.zeros_like(per_sample_mse_pos)
                                per_sample_kl = 0.5 * (per_sample_mse_pos / denom)

                            distill_loss = per_sample_kl.mean()

                            ref_kl_loss = torch.zeros(1, device=device)
                            if use_ref_kl:
                                mu_ref = _get_mu(mu_ref_all, idx)
                                ref_mse = _compute_mse(mu_S, mu_ref, ctx, B, None)
                                ref_kl_loss = self.training_args.ref_kl_beta * ref_mse.mean()

                            loss = distill_loss + ref_kl_loss
                            self.accelerator.backward(loss)

                            with torch.no_grad():
                                kl_sum += per_sample_kl.detach().sum()
                                kl_count += B
                                if use_ref_kl:
                                    ref_kl_sum += ref_kl_loss.detach()
                                if vis_mask is not None:
                                    mask_cov_sum += vis_mask.mean()
                                    mask_cov_count += 1
                                if use_contrastive:
                                    mse_pos_sum += per_sample_mse_pos.detach().mean()
                                    mse_neg_sum += per_sample_mse_neg.detach().mean()

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
                                    mse_pos_sum, mse_neg_sum,
                                )
                                self.step += 1
                                kl_sum.zero_()
                                kl_count.zero_()
                                ref_kl_sum.zero_()
                                mask_cov_sum.zero_()
                                mask_cov_count.zero_()
                                mse_pos_sum.zero_()
                                mse_neg_sum.zero_()

    # =============================== Dense/Sparse helpers ===============================
    #
    # Dense vs Sparse: all differences stem from one fact — sparse latents have no
    # aligned batch dimension. Dense latents are regular 3D grids (B, C, 16, 16, 16)
    # where every sample has the same shape. Sparse latents are variable-length
    # per-voxel features (N_b, C), concatenated into (N_total, C) with a separate
    # batch_idx vector to track which voxel belongs to which sample.
    #
    # This causes 5 code-level differences:
    #
    # 1. Combining samples:
    #    Dense: torch.stack → (B, T, C, D, H, W) — aligned, indexable by batch dim
    #    Sparse: torch.cat  → (T, N_total, C)    — flat, needs batch_idx to separate
    #
    # 2. Indexing a timestep:
    #    Dense: mu_all[:, idx] — batch dim first, timestep second
    #    Sparse: mu_all[idx]   — timestep dim first (batch is implicit in N_total)
    #
    # 3. Per-sample reduction (MSE):
    #    Dense: flatten(1).mean(1) — batch dim naturally separates samples
    #    Sparse: scatter_add_(batch_idx, ...) — must explicitly group by sample index
    #
    # 4. Splitting cached mu back to individual samples:
    #    Dense: mu_T[i] — direct batch indexing
    #    Sparse: mu_T.split(counts, dim=1) — split by per-sample voxel counts
    #
    # 5. Visibility mask shape:
    #    Dense: (B, 1, 16, 16, 16) — 3D grid at fixed 16³ resolution
    #    Sparse: (N_total,) — one scalar weight per voxel, resolution from _infer_ss_resolution()

    def _load_mu_targets_dense(self, batch_samples, device):
        """Dense: stack → (B, T, C, D, H, W). ctx=None (batch dim handles separation)."""
        mu_T = torch.stack([s.extra_kwargs["mu_teacher"].to(device) for s in batch_samples])
        mu_ref = None
        if self.training_args.ref_kl_beta > 0:
            mu_ref = torch.stack([s.extra_kwargs["mu_ref"].to(device) for s in batch_samples])
        mu_neg = None
        if self._use_contrastive:
            mu_neg = torch.stack([s.extra_kwargs["mu_neg"].to(device) for s in batch_samples])
        return mu_T, mu_ref, mu_neg, None  # ctx=None for dense

    def _load_mu_targets_sparse(self, batch_samples, device):
        """Sparse: cat → (T, N_total, C). ctx=batch_idx to map voxels back to samples."""
        mu_T = torch.cat([s.extra_kwargs["mu_teacher"].to(device) for s in batch_samples], dim=1)
        mu_ref = None
        if self.training_args.ref_kl_beta > 0:
            mu_ref = torch.cat([s.extra_kwargs["mu_ref"].to(device) for s in batch_samples], dim=1)
        mu_neg = None
        if self._use_contrastive:
            mu_neg = torch.cat([s.extra_kwargs["mu_neg"].to(device) for s in batch_samples], dim=1)
        batch = BaseSample.stack(batch_samples)
        batch_idx = batch["sparse_coords"][:, 0].long()
        return mu_T, mu_ref, mu_neg, batch_idx

    def _get_mu_at_step_dense(self, mu_all, idx):
        # mu_all: (B, T, C, D, H, W) → index timestep dim
        return mu_all[:, idx]  # (B, C, D, H, W)

    def _get_mu_at_step_sparse(self, mu_all, idx):
        # mu_all: (T, N_total, C) → index timestep dim
        return mu_all[idx]  # (N_total, C)

    def _compute_mse_dense(self, mu_S, mu_T, ctx, B, vis_mask):
        """Dense: batch dim is aligned, so flatten(1).mean(1) gives per-sample MSE."""
        diff_sq = (mu_S.float() - mu_T.float()).pow(2)  # (B, C, D, H, W)
        if vis_mask is not None:
            diff_sq = diff_sq * vis_mask  # (B, C, D, H, W) * (B, 1, 16, 16, 16) broadcast
            C = mu_S.shape[1]
            # Denominator = number of visible voxels * channels
            mask_count = vis_mask.flatten(1).sum(dim=1) * C  # (B,)
            return diff_sq.flatten(1).sum(dim=1) / mask_count.clamp(min=1.0)  # (B,)
        return diff_sq.flatten(1).mean(dim=1)  # (B,)

    def _compute_mse_sparse(self, mu_S, mu_T, ctx, B, vis_mask):
        """Sparse: no batch dim, so scatter_add_(batch_idx, ...) groups voxels by sample."""
        batch_idx = ctx  # (N_total,) long — maps each voxel to its sample index
        diff_sq = (mu_S.float() - mu_T.float()).pow(2)  # (N_total, C)
        if vis_mask is not None:
            diff_sq = diff_sq * vis_mask.unsqueeze(1)  # (N_total, C) * (N_total, 1)
            # per_point sums over C channels, so denominator = sum(vis_weight) * C
            per_point = diff_sq.sum(dim=1)  # (N_total,)
            C = mu_S.shape[1]
            sums = mu_S.new_zeros(B)  # (B,)
            mask_sums = mu_S.new_zeros(B)  # (B,)
            sums.scatter_add_(0, batch_idx, per_point)
            mask_sums.scatter_add_(0, batch_idx, vis_mask)  # total visible weight per sample
            return sums / (mask_sums * C).clamp(min=1.0)  # (B,)
        per_point = diff_sq.mean(dim=1)  # (N_total,) — mean over C channels
        sums = mu_S.new_zeros(B)  # (B,)
        counts = mu_S.new_zeros(B)  # (B,)
        sums.scatter_add_(0, batch_idx, per_point)
        counts.scatter_add_(0, batch_idx, torch.ones_like(per_point))
        return sums / counts.clamp(min=1.0)  # (B,) per-sample mean MSE

    def _load_vis_mask_dense(self, batch_samples, device):
        """Dense: stack → (B, 1, 16, 16, 16). Fixed 16³ resolution matching the latent grid."""
        if not self.training_args.use_visibility_mask:
            return None
        masks = [s.extra_kwargs.get("visibility_mask") for s in batch_samples]  # each (1, 16, 16, 16)
        if not all(m is not None for m in masks):
            return None
        return torch.stack(masks).to(device)  # (B, 1, 16, 16, 16)

    def _load_vis_mask_sparse(self, batch_samples, device):
        """Sparse: cat → (N_total,). One scalar weight per voxel, resolution from _infer_ss_resolution()."""
        if not self.training_args.use_visibility_mask:
            return None
        masks = [s.extra_kwargs.get("visibility_mask") for s in batch_samples]  # each (N_b,)
        if not all(m is not None for m in masks):
            return None
        return torch.cat(masks).to(device)  # (N_total,) — one scalar per voxel

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
        mu = output.next_latents_mean
        if isinstance(mu, _SparseTensor):
            mu = mu.feats  # (N_total, C)
        return mu, output.std_dev_t, output.dt

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
        mse_pos_sum: Optional[torch.Tensor] = None,
        mse_neg_sum: Optional[torch.Tensor] = None,
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
        if mse_pos_sum is not None and mse_pos_sum.item() > 0:
            metrics["mse_pos"] = mse_pos_sum.squeeze()
        if mse_neg_sum is not None and mse_neg_sum.item() > 0:
            metrics["mse_neg"] = mse_neg_sum.squeeze()

        self.log_data({f"train/{k}": v for k, v in metrics.items()}, step=self.step)
