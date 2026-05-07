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

# src/flow_factory/models/trellis2/trellis2.py
"""
Trellis2 Adapter for Flow-Factory.

This adapter wraps Trellis2 image-to-3D models for RL-based post-training.
It uses a pseudo-pipeline pattern since Trellis2 is not a diffusers model.

Training Target:
    - shape_slat_flow_model: Structured Latent flow model for 3D shape generation
    - tex_slat_flow_model: Structured Latent flow model for texture generation

Key Differences from diffusers-based adapters:
    1. Uses SparseTensor instead of dense tensors for latents
    2. Image conditioning via DINOv2/v3 instead of text encoders
    3. Custom Euler sampler instead of diffusers scheduler
    4. Outputs 3D meshes instead of images/videos
"""
from __future__ import annotations

import os
import sys
from typing import Union, List, Dict, Any, Optional, Tuple, ClassVar, Literal
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from accelerate import Accelerator

from ..abc import BaseAdapter
from ...samples import I2VSample
from ...hparams import Arguments
from ...scheduler import SDESchedulerOutput
from ...utils.trajectory_collector import (
    TrajectoryCollector,
    CallbackCollector,
    TrajectoryIndicesType, 
    create_trajectory_collector,
    create_callback_collector,
)
from ...utils.base import filter_kwargs
from ...utils.image import MultiImageBatch, standardize_image_batch
from ...utils.logger_utils import setup_logger

from .pipeline import Trellis2PseudoPipeline

logger = setup_logger(__name__)


# ======================== Ensure Trellis2 is importable ========================
def _setup_trellis_path():
    """Add Trellis2 to sys.path if not already present and return the path."""
    trellis_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', '..', 
        'third_party', 'TRELLIS.2'
    )
    trellis_path = os.path.abspath(trellis_path)
    if trellis_path not in sys.path:
        sys.path.insert(0, trellis_path)
    return trellis_path

_trellis_path = _setup_trellis_path()

from trellis2.modules.sparse import SparseTensor
from trellis2.representations import Mesh, MeshWithVoxel
from trellis2.renderers.pbr_mesh_renderer import EnvMap
from trellis2.utils import render_utils
from o_voxel.convert import flexible_dual_grid_to_mesh

from .chunked_mixin import ChunkedDecoderMixin
from .pbr_mesh_renderer_chunked import render_frames_chunked


def _composite_rgba_pil(
    img: Image.Image,
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Image.Image:
    """Alpha-composite an RGBA PIL image onto solid *bg_color*, return RGB PIL.

    Non-RGBA images are returned as-is (converted to RGB if needed).
    *bg_color* uses float [0, 1] per channel, matching ``render_bg_color``.
    """
    if not isinstance(img, Image.Image):
        return img
    if img.mode != 'RGBA':
        return img.convert('RGB') if img.mode != 'RGB' else img
    bg_rgba = tuple(int(round(c * 255)) for c in bg_color) + (255,)
    bg = Image.new('RGBA', img.size, bg_rgba)
    return Image.alpha_composite(bg, img).convert('RGB')


def _apply_bg_to_condition_images(
    samples: List['Trellis2Sample'],
    condition_images_rgba: List[Optional[List[Image.Image]]],
    bg_color: Tuple[float, float, float],
) -> None:
    """Overwrite each sample's ``condition_images`` by compositing the original
    RGBA images onto *bg_color*, so the background matches the rendered video."""
    for b, s in enumerate(samples):
        raw_imgs = condition_images_rgba[b]
        if raw_imgs is not None:
            raw_imgs = standardize_image_batch(raw_imgs, 'pil')
            s.condition_images = [
                _composite_rgba_pil(img, bg_color) for img in raw_imgs
            ]


def _compute_adaptive_distance(
    fov_deg: float,
    fill_ratio: float = 0.9,
    object_half_size: float = 0.5,
) -> float:
    """Compute camera distance from FOV and fill ratio.

    TRELLIS meshes are normalized to ``[-0.5, 0.5]^3``, so
    ``object_half_size=0.5`` is the natural default.
    Derivation: ``r = object_half_size / (fill_ratio * tan(FOV / 2))``.
    """
    import math
    tan_half_fov = math.tan(math.radians(fov_deg) / 2)
    return object_half_size / (fill_ratio * tan_half_fov)


# ======================== Sample Dataclass ========================

@dataclass
class Trellis2Sample(I2VSample):
    """
    Sample output for Trellis2 models.
    
    Unlike image/video samples, Trellis2 outputs 3D data in SparseTensor format
    (coords + features). The sample supports flexible single-stage and multi-stage
    training scenarios.
    
    Design:
        - **Single-stage training** (e.g., only shape): The trained stage's data is
          stored in its per-stage fields AND copied to the standard ``all_latents`` /
          ``log_probs`` / ``image_cond`` fields. This allows existing Trainers
          (GRPO, NFT, AWM) to work with zero modifications.
        - **Multi-stage training** (e.g., shape + tex): Each stage's data is stored
          in its per-stage fields. A multi-stage Trainer iterates over stages and
          projects per-stage fields to standard fields before the inner loop.
    
    Stage overview:
        - **dense** (Stage 1): Sparse structure generation. Cond @ 512px.
        - **shape** (Stage 2A): Shape flow model. Cond @ 1024px.
        - **tex** (Stage 2B): Texture flow model. Cond @ 1024px. Also needs
          ``tex_concat_cond`` (normalized shape features) as cross-stage dependency.
    """
    # Class-level: fields shared across batch (identical for all samples)
    _shared_fields: ClassVar[frozenset[str]] = frozenset({
        'latent_index_map', 'log_prob_index_map', 'resolution',
    })

    # Sparse coordinate fields: per-sample shape (N_b, 4) → (N_total, 4).
    # _stack_values() rewrites col 0 to the global batch index before cat.
    _SPARSE_COORD_FIELDS: ClassVar[frozenset[str]] = frozenset({
        'sparse_coords',
    })

    # Sparse trajectory fields: per-sample shape (T, N_b, C).
    # _stack_values() cat-concats them along N → (N_total, T, C) instead of
    # torch.stack which would require identical N_b across samples.
    _SPARSE_LATENT_FIELDS: ClassVar[frozenset[str]] = frozenset({
        'all_latents', 'shape_all_latents', 'tex_all_latents',
    })

    # Sparse conditioning fields: per-sample shape (N_b, C) → (N_total, C).
    _SPARSE_CONDITION_FIELDS: ClassVar[frozenset[str]] = frozenset({
        'tex_concat_cond',
    })
    
    # ============ Global invariants (shared by all stages) ============
    sparse_coords: Optional[torch.Tensor] = None   # (N, 4) — from Stage 1, fixed for Stage 2A/2B
    resolution: Optional[int] = None
    
    # ============ Standard fields for single-stage compatibility ============
    # In single-stage mode, these are populated by inference() directly.
    # In multi-stage mode, the Trainer projects per-stage fields here.
    image_cond: Optional[torch.Tensor] = None       # (seq_len, D)
    neg_image_cond: Optional[torch.Tensor] = None   # (seq_len, D)
    # NOTE: all_latents, log_probs, latent_index_map, log_prob_index_map
    # are inherited from BaseSample and serve as the standard training interface.
    
    # ============ Per-stage: Dense (Stage 1, cond @ 512) ============
    dense_image_cond: Optional[torch.Tensor] = None     # (seq_512, D)
    dense_neg_image_cond: Optional[torch.Tensor] = None  # (seq_512, D)
    dense_final_latent: Optional[torch.Tensor] = None    # (B, C, D, H, W) — final denoised latent (always set)
    dense_all_latents: Optional[torch.Tensor] = None     # (T, B, C, D, H, W) — trajectory for training (may be None)
    dense_log_probs: Optional[torch.Tensor] = None       # (T,)
    dense_latent_index_map: Optional[torch.Tensor] = None
    dense_log_prob_index_map: Optional[torch.Tensor] = None
    dense_timesteps: Optional[torch.Tensor] = None
    
    # ============ Per-stage: Shape (Stage 2A, cond @ 1024) ============
    shape_image_cond: Optional[torch.Tensor] = None     # (seq_1024, D)
    shape_neg_image_cond: Optional[torch.Tensor] = None  # (seq_1024, D)
    shape_final_latent: Optional[torch.Tensor] = None    # (N, C_shape) — final denoised latent (always set)
    shape_all_latents: Optional[torch.Tensor] = None     # (T, N, C_shape) — trajectory for training (may be None)
    shape_log_probs: Optional[torch.Tensor] = None       # (T,)
    shape_latent_index_map: Optional[torch.Tensor] = None
    shape_log_prob_index_map: Optional[torch.Tensor] = None
    shape_timesteps: Optional[torch.Tensor] = None
    
    # ============ Per-stage: Tex (Stage 2B, cond @ 1024) ============
    tex_image_cond: Optional[torch.Tensor] = None       # (seq_1024, D)
    tex_neg_image_cond: Optional[torch.Tensor] = None    # (seq_1024, D)
    tex_final_latent: Optional[torch.Tensor] = None      # (N, C_tex) — final denoised latent (always set)
    tex_all_latents: Optional[torch.Tensor] = None       # (T, N, C_tex) — trajectory for training (may be None)
    tex_log_probs: Optional[torch.Tensor] = None         # (T,)
    tex_concat_cond: Optional[torch.Tensor] = None       # (N, C_shape) — normalized shape features
    tex_latent_index_map: Optional[torch.Tensor] = None
    tex_log_prob_index_map: Optional[torch.Tensor] = None
    tex_timesteps: Optional[torch.Tensor] = None
    
    # ============ Output (for reward / eval) ============
    mesh: Optional[Any] = None

    _STAGE_FIELD_MAP = {
        'all_latents', 'log_probs', 'image_cond', 'neg_image_cond',
        'latent_index_map', 'log_prob_index_map', 'timesteps',
    }

    _STAGE_BROADCAST_FIELDS: ClassVar[Dict[str, tuple]] = {
        'dense': ('sparse_coords', 'dense_final_latent'),
        'shape': ('shape_final_latent',),
    }
    _STAGE_METADATA_FIELDS: ClassVar[Dict[str, tuple]] = {
        'dense': (
            'dense_all_latents', 'dense_log_probs',
            'dense_image_cond', 'dense_neg_image_cond',
            'dense_latent_index_map', 'dense_log_prob_index_map', 'dense_timesteps',
        ),
        'shape': (
            'shape_all_latents', 'shape_log_probs',
            'shape_image_cond', 'shape_neg_image_cond',
            'shape_latent_index_map', 'shape_log_prob_index_map', 'shape_timesteps',
        ),
    }

    def __post_init__(self):
        if self.condition_images is not None:
            images = (self.condition_images if isinstance(self.condition_images, list)
                      else [self.condition_images])
            for i, img in enumerate(images):
                if isinstance(img, Image.Image) and img.mode == 'RGBA':
                    images[i] = _composite_rgba_pil(img, (0.0, 0.0, 0.0))
            self.condition_images = images
        super().__post_init__()

    def copy_stage_metadata_from(
        self, source: 'Trellis2Sample', stages: Union[str, List[str]],
    ) -> None:
        """Copy non-broadcast upstream metadata from *source* into this sample.

        Only copies auxiliary fields (trajectories, log-probs, cond tensors,
        index maps, timesteps) that are **not** broadcast across ranks.  The
        broadcast fields (``sparse_coords``, ``dense_final_latent``,
        ``shape_final_latent``) must be set separately by the caller after
        ``dist.broadcast``.
        """
        if isinstance(stages, str):
            stages = [stages]
        for stage in stages:
            for field in self._STAGE_METADATA_FIELDS[stage]:
                setattr(self, field, getattr(source, field))

    def activate_stage(self, stage: str) -> None:
        """Project per-stage fields to standard fields for training.

        For single-stage training, called once after inference.
        For multi-stage training, the Trainer calls this before each
        stage's optimize loop to switch which data the standard fields
        (``all_latents``, ``log_probs``, etc.) point to.
        """
        for field_name in self._STAGE_FIELD_MAP:
            per_stage_name = f'{stage}_{field_name}'
            value = getattr(self, per_stage_name, None)
            if value is not None:
                setattr(self, field_name, value)

    @classmethod
    def _stack_values(cls, key: str, values: list):
        """Override to handle variable-length sparse tensors across samples.

        Trellis2's SparseTensor format stores per-point data without padding:
        each sample may have a different number of occupied voxels N_b.
        Standard ``torch.stack`` requires identical shapes, so we cat instead.

        Dispatch table
        ──────────────
        ``_SPARSE_COORD_FIELDS``     → clone + rewrite col 0 to global b + cat → (N_total, 4)
        ``_SPARSE_LATENT_FIELDS``    → permute (T,N_b,C)→(N_b,T,C) + cat → (N_total, T, C)
        ``_SPARSE_CONDITION_FIELDS`` → cat along N → (N_total, C)
        everything else              → parent (handles shared fields, equal-shape tensors, lists)
        """
        if all(v is None for v in values):
            return None

        # sparse_coords: (N_b, 4) per sample → (N_total, 4) with correct batch idx
        if key in cls._SPARSE_COORD_FIELDS:
            chunks = []
            for b, v in enumerate(values):
                if v is None:
                    continue
                c = v.clone()
                c[:, 0] = b          # overwrite batch column to global index
                chunks.append(c)
            return torch.cat(chunks, dim=0)

        # sparse trajectory: (T, N_b, C) per sample → (N_total, T, C)
        if key in cls._SPARSE_LATENT_FIELDS:
            non_null = [v for v in values if v is not None]
            if not non_null:
                return None
            # (T, N_b, C) → (N_b, T, C), then cat along dim 0
            return torch.cat([v.permute(1, 0, 2) for v in non_null], dim=0)

        # sparse conditioning: (N_b, C) per sample → (N_total, C)
        if key in cls._SPARSE_CONDITION_FIELDS:
            non_null = [v for v in values if v is not None]
            if not non_null:
                return None
            return torch.cat(non_null, dim=0)

        return super()._stack_values(key, values)


# ======================== Adapter Class ========================

class Trellis2Adapter(BaseAdapter):
    """
    Adapter for Trellis2 Image-to-3D models.
    
    This adapter enables RL-based post-training of Trellis2's flow models:
    - shape_slat_flow_model: For improving 3D shape quality
    - tex_slat_flow_model: For improving texture quality
    
    The adapter follows the pseudo-pipeline pattern since Trellis2
    is not a diffusers model.
    """
    
    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: Trellis2PseudoPipeline
    
    # Mapping for special model directory names
    _LOCAL_MODEL_DIR_MAP = {
        'facebook/dinov3-vitl16-pretrain-lvd1689m': [
            'dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m',
            'dinov3-vitl16-pretrain-lvd1689m',
        ],
        'briaai/RMBG-2.0': [
            'rmbg2/RMBG-2.0',
            'RMBG-2.0',
        ],
    }
    
    def _find_local_model_path(self, model_name: str, base_path: str) -> Optional[str]:
        """
        Find local model path for HuggingFace models.
        
        Args:
            model_name: HuggingFace model name (e.g., 'facebook/dinov3-vitl16-pretrain-lvd1689m')
            base_path: Base path (model_name_or_path) to infer project root from
        
        Returns:
            Local path if found, None otherwise
        """
        # Get project root - base_path is like: /path/to/Flow-Factory/pretrained_weights/TRELLIS.2-4B
        if 'pretrained_weights' in base_path:
            idx = base_path.find('pretrained_weights')
            pretrained_dir = base_path[:idx + len('pretrained_weights')]
        else:
            # Fallback: use this file's location to find project root
            this_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(this_file))))
            pretrained_dir = os.path.join(project_root, 'pretrained_weights')
        
        # Try special mappings first
        if model_name in self._LOCAL_MODEL_DIR_MAP:
            for rel_path in self._LOCAL_MODEL_DIR_MAP[model_name]:
                full_path = os.path.join(pretrained_dir, rel_path)
                if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, 'config.json')):
                    return full_path
        
        # Generic search
        if '/' in model_name:
            org, name = model_name.split('/', 1)
        else:
            org, name = None, model_name
        
        search_paths = []
        if org:
            search_paths.append(os.path.join(pretrained_dir, name, org, name))
        search_paths.append(os.path.join(pretrained_dir, name))
        
        for path in search_paths:
            if os.path.isdir(path) and os.path.exists(os.path.join(path, 'config.json')):
                return path
        
        return None
    
    def load_pipeline(self) -> Trellis2PseudoPipeline:
        """Load the Trellis2 pseudo-pipeline."""
        # Determine target flow model from config
        target_flow_model = self.model_args.extra_kwargs.get('target_flow_model')
        assert target_flow_model is not None, (
            "extra_kwargs.target_flow_model is required. "
            "Set it in your YAML, e.g.: target_flow_model: shape_slat_1024"
        )
        
        model_path = self.model_args.model_name_or_path
        
        # Try to find local paths for external models
        image_cond_model_path = self._find_local_model_path(
            'facebook/dinov3-vitl16-pretrain-lvd1689m',
            model_path
        )
        rembg_model_path = self._find_local_model_path(
            'briaai/RMBG-2.0',
            model_path
        )
        
        if image_cond_model_path:
            logger.info(f"Using local DINOv3 model: {image_cond_model_path}")
        if rembg_model_path:
            logger.info(f"Using local RMBG model: {rembg_model_path}")
        
        pipeline = Trellis2PseudoPipeline.from_pretrained(
            model_path,
            target_flow_model=target_flow_model,
            low_cpu_mem_usage=False,
            image_cond_model_path=image_cond_model_path,
            rembg_model_path=rembg_model_path,
        )
        for decoder in filter(None, [pipeline.shape_decoder, pipeline.tex_decoder]):
            decoder.convert_to_fp32()
            decoder.dtype = torch.float32
            decoder.use_fp16 = False
            ChunkedDecoderMixin.inject_to(decoder)
        return pipeline
    
    def load_scheduler(self):
        """
        Build three independent stage schedulers (dense / shape / tex) from the
        Trellis2 sampler configs.

        Dense uses FlowMatchEulerDiscreteSDEScheduler (from scheduler/ package).
        Shape/tex use SparseFlowMatchEulerSDEScheduler (SparseTensor-native).

        Returns the shape scheduler as the default ``self.scheduler`` for
        single-stage Trainer compatibility.
        """
        from ...scheduler.flow_match_euler_discrete import FlowMatchEulerDiscreteSDEScheduler
        from .flow_match_euler_discrete import SparseFlowMatchEulerSDEScheduler

        sched_args = self.config.scheduler_args
        extra = self.model_args.extra_kwargs or {}

        def _stage_sde_kwargs(stage: str) -> dict:
            stage_cfg = extra.get(f'{stage}_sde', {})
            return {
                "dynamics_type": stage_cfg.get('dynamics_type', sched_args.dynamics_type),
                "noise_level":   stage_cfg.get('noise_level', sched_args.noise_level),
                "num_sde_steps": stage_cfg.get('num_sde_steps', sched_args.num_sde_steps),
                "seed":          sched_args.seed,
            }

        def _make_dense_scheduler() -> FlowMatchEulerDiscreteSDEScheduler:
            params  = self.pipeline.sparse_structure_sampler_params
            sampler = self.pipeline.sparse_structure_sampler
            steps     = int(params['steps'])
            rescale_t = float(params['rescale_t'])
            sigma_min = float(sampler.sigma_min)

            t_np = np.linspace(1.0, 0.0, steps + 1)            # (steps+1,) float64
            t_np = rescale_t * t_np / (1 + (rescale_t - 1) * t_np)

            sched = FlowMatchEulerDiscreteSDEScheduler(
                num_train_timesteps=1000,
                **_stage_sde_kwargs('dense'),
            )
            sched.timesteps     = torch.tensor(t_np[:-1] * 1000, dtype=torch.float32)
            sched.sigmas        = torch.tensor(t_np, dtype=torch.float32)
            sched._timesteps_np = t_np
            sched._sigma_min    = sigma_min
            return sched

        def _make_sparse_scheduler(stage: str) -> SparseFlowMatchEulerSDEScheduler:
            if stage == 'shape':
                params  = self.pipeline.shape_slat_sampler_params
                sampler = self.pipeline.shape_slat_sampler
            elif stage == 'tex':
                params  = self.pipeline.tex_slat_sampler_params
                sampler = self.pipeline.tex_slat_sampler
            else:
                raise ValueError(f"Unknown sparse stage: {stage!r}")

            sched = SparseFlowMatchEulerSDEScheduler(
                rescale_t   = float(params['rescale_t']),
                sigma_min   = float(sampler.sigma_min),
                **_stage_sde_kwargs(stage),
            )
            sched.set_timesteps(int(params['steps']), device='cpu')
            sched._sigma_min = float(sampler.sigma_min)
            return sched

        self.pipeline.scheduler_dense = _make_dense_scheduler()
        self.pipeline.scheduler_shape = _make_sparse_scheduler('shape')
        self.pipeline.scheduler_tex   = _make_sparse_scheduler('tex')

        train_stage = self.pipeline._target_flow_model.split('_')[0]
        return self._get_stage_scheduler(train_stage)

    def _precision_protected_components(self) -> set:
        """All Trellis2 flow models use a custom mixed-dtype layout
        (float32 head/tail + bf16 blocks via ``manual_cast``).
        A blanket ``.to(bf16)`` would permanently damage the float32
        weights, so we tell the base class to skip them entirely.
        The correct dtype layout comes from checkpoint loading itself:
        ``load_state_dict`` copies values into pre-typed parameters
        created by the model's ``__init__`` + ``convert_to``.
        LoRA parameters (on the target model) are initialized as
        float32 by PEFT, which already matches ``master_weight_dtype``.
        """
        return set(self.transformer_names)

    def _freeze_vae(self):
        """Trellis2 doesn't have a VAE, skip freezing."""
        pass
    
    def _freeze_text_encoders(self):
        """Trellis2 uses image conditioning, no text encoders to freeze."""
        pass

    def enable_gradient_checkpointing(self):
        """Toggle per-block ``use_checkpoint`` on every transformer flow model.

        Trellis2 flow models do not expose a top-level
        ``enable_gradient_checkpointing()`` method (which is what the base
        class' implementation looks for); instead, each transformer block
        has a ``use_checkpoint: bool`` attribute that routes its forward
        through ``torch.utils.checkpoint.checkpoint(...)`` when True (see
        ``third_party/TRELLIS.2/trellis2/modules/[sparse/]transformer``).

        We mirror that here: iterate over ``self.transformer_names``
        (target flow models) and flip ``block.use_checkpoint`` for every
        block that has the attribute.  Rollout / eval call paths run under
        ``torch.no_grad`` so the flag is a no-op there.
        """
        total = 0
        for name in self.transformer_names:
            component = getattr(self.pipeline, name, None)
            if component is None or not hasattr(component, 'blocks'):
                continue
            n_enabled = 0
            for block in component.blocks:
                if hasattr(block, 'use_checkpoint'):
                    block.use_checkpoint = True
                    n_enabled += 1
            logger.info(
                f"Enabled gradient checkpointing for {name}: "
                f"{n_enabled}/{len(component.blocks)} blocks"
            )
            total += n_enabled
        if total == 0:
            logger.warning(
                "enable_gradient_checkpointing: no transformer blocks with "
                f"`use_checkpoint` attribute on {self.transformer_names}."
            )

    # ======================== Stage Scheduler Helpers ========================

    _STAGE_SCHEDULER_ATTR = {
        'dense': 'scheduler_dense',
        'shape': 'scheduler_shape',
        'tex':   'scheduler_tex',
    }

    _STAGE_SAMPLER_PARAMS_ATTR = {
        'dense': 'sparse_structure_sampler_params',
        'shape': 'shape_slat_sampler_params',
        'tex':   'tex_slat_sampler_params',
    }

    @staticmethod
    def _as_scalar_resolution(resolution) -> int:
        """Flow-Factory passes (H, W) tuples; Trellis2 expects a single int."""
        if isinstance(resolution, (tuple, list)):
            return int(resolution[0])
        return int(resolution)

    def _get_stage_scheduler(self, stage: str):
        """Return the scheduler for the given stage."""
        attr = self._STAGE_SCHEDULER_ATTR.get(stage)
        if attr is None:
            raise ValueError(f"Unknown stage: {stage!r}. Expected 'dense', 'shape', or 'tex'.")
        return getattr(self.pipeline, attr)

    def _get_stage_guidance(self, stage: str) -> dict:
        """Read stage-specific CFG config from Trellis2 pipeline.json.

        Trellis2 ships per-stage sampler params (e.g. ``tex_slat_sampler_params``
        has ``guidance_strength=1.0`` so tex stage skips CFG by default).
        These values are the single source of truth: training_args.guidance_*
        are not used, and runtime overrides via sampler_params are not
        supported. To change CFG behavior, edit pipeline.json directly.

        Returns dict with keys ``guidance_scale``, ``guidance_interval``,
        ``guidance_rescale``.
        """
        attr = self._STAGE_SAMPLER_PARAMS_ATTR[stage]
        params = getattr(self.pipeline, attr)
        return {
            'guidance_scale': float(params['guidance_strength']),
            'guidance_interval': tuple(params['guidance_interval']),
            'guidance_rescale': float(params['guidance_rescale']),
        }

    def _resolve_conditioning(
        self,
        stages: List[str],
        *,
        images: Optional[MultiImageBatch] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        image_cond_512: Optional[List[torch.Tensor]] = None,
        neg_image_cond_512: Optional[List[torch.Tensor]] = None,
        image_cond_1024: Optional[List[torch.Tensor]] = None,
        neg_image_cond_1024: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple:
        """Resolve per-resolution conditioning tensors for the requested stages.

        Falls back to on-the-fly encoding from raw *images* when pre-encoded
        tensors are missing.  Zero-fills negative conditioning when absent.

        Returns ``(cond_512, neg_512, cond_1024, neg_1024, condition_images, batch_size)``.
        """
        needs_512  = 'dense' in stages
        needs_1024 = bool({'shape', 'tex'} & set(stages))

        cond_512, neg_512   = image_cond_512, neg_image_cond_512
        cond_1024, neg_1024 = image_cond_1024, neg_image_cond_1024

        if (needs_512 and cond_512 is None) or (needs_1024 and cond_1024 is None):
            if images is None:
                raise ValueError(
                    "Missing conditioning. Provide pre-encoded image_cond_512 / "
                    "image_cond_1024 (from preprocess_func) or raw images."
                )
            encoded = self.preprocess_func(images=images)
            cond_512  = cond_512  if cond_512  is not None else encoded.get('image_cond_512')
            neg_512   = neg_512   if neg_512   is not None else encoded.get('neg_image_cond_512')
            cond_1024 = cond_1024 if cond_1024 is not None else encoded.get('image_cond_1024')
            neg_1024  = neg_1024  if neg_1024  is not None else encoded.get('neg_image_cond_1024')
            condition_images = condition_images if condition_images is not None else encoded.get('condition_images')

        ref_cond = cond_1024 if cond_1024 is not None else cond_512
        if ref_cond is None:
            raise ValueError("No conditioning tensors available.")
        batch_size = len(ref_cond)

        if needs_512  and neg_512  is None and cond_512  is not None:
            neg_512  = [torch.zeros_like(c) for c in cond_512]
        if needs_1024 and neg_1024 is None and cond_1024 is not None:
            neg_1024 = [torch.zeros_like(c) for c in cond_1024]

        return cond_512, neg_512, cond_1024, neg_1024, condition_images, batch_size

    def _reduce_sparse_log_prob(
        self,
        log_prob: torch.Tensor,         # (N_total,) per-point log-prob
        sparse_coords: torch.Tensor,    # (N_total, 4), col 0 = batch index
        batch_size: int,                # explicit B; do not infer from coords max
    ) -> torch.Tensor:                  # (B,)
        """Reduce per-point log_prob to per-sample via mean aggregation.

        ``batch_size`` is passed in explicitly so that samples with zero points
        (or a trailing batch index absent from ``coords``) still produce a
        length-``B`` output vector.
        """
        batch_idx = sparse_coords[:, 0].long()                                             # (N_total,)
        point_sum = torch.zeros(batch_size, device=log_prob.device, dtype=log_prob.dtype)  # (B,)
        point_cnt = torch.zeros(batch_size, device=log_prob.device, dtype=log_prob.dtype)  # (B,)
        point_sum.scatter_add_(0, batch_idx, log_prob)                                     # (B,)
        point_cnt.scatter_add_(0, batch_idx, torch.ones_like(log_prob))                    # (B,)
        return point_sum / point_cnt.clamp_min(1.0)                                        # (B,)

    # ======================== Mode Sync ========================

    def eval(self):
        super().eval()
        self.pipeline.scheduler_dense.eval()
        self.pipeline.scheduler_shape.eval()
        self.pipeline.scheduler_tex.eval()

    def rollout(self, *args, **kwargs):
        super().rollout(*args, **kwargs)
        self.pipeline.scheduler_dense.rollout(*args, **kwargs)
        self.pipeline.scheduler_shape.rollout(*args, **kwargs)
        self.pipeline.scheduler_tex.rollout(*args, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        self.pipeline.scheduler_dense.train(mode)
        self.pipeline.scheduler_shape.train(mode)
        self.pipeline.scheduler_tex.train(mode)

    @contextmanager
    def low_vram_mode(self):
        """No-op context manager kept for caller compatibility."""
        yield

    def _resolve_component_names(self, components: Optional[Union[str, List[str]]] = None) -> List[str]:
        """
        Resolve component specifiers to concrete pipeline attribute names.
        
        Trellis2 has no VAE or text encoders.  When *components* is ``None``
        ("all components"), we return only the deduplicated transformer list.
        """
        if components is None:
            return self.transformer_names
        
        if isinstance(components, str):
            components = [components]
        
        resolved = []
        for comp in components:
            if comp == 'transformers':
                resolved.extend(self.transformer_names)
            elif comp in ('vae', 'text_encoders'):
                continue
            else:
                pipeline_dict = self.pipeline.__dict__
                if comp in pipeline_dict and (isinstance(pipeline_dict[comp], nn.Module) or hasattr(pipeline_dict[comp], 'to')):
                    resolved.append(comp)
        
        return list(dict.fromkeys(resolved))
    
    # ======================== Properties ========================
    
    @property
    def transformer_names(self) -> List[str]:
        """
        Deduplicated transformer names (by object identity).
        
        Trellis2Pipeline exposes the same ``nn.Module`` under multiple names
        (e.g., ``transformer`` and ``transformer_shape_1024``).  We keep only
        the first occurrence per unique ``id()`` to prevent double FSDP / LoRA
        wrapping.
        
        NOTE: We iterate in sorted order so that the bare ``'transformer'``
        alias (which is always alphabetically first) takes priority over its
        stage-specific counterpart (e.g. ``'transformer_shape_1024'``).
        """
        seen_ids = set()
        names = []
        pipeline_dict = self.pipeline.__dict__
        for name in sorted(pipeline_dict):
            if not name.startswith('transformer') or name.startswith('_'):
                continue
            obj = pipeline_dict[name]
            if obj is None or not isinstance(obj, nn.Module):
                continue
            obj_id = id(obj)
            if obj_id not in seen_ids:
                seen_ids.add(obj_id)
                names.append(name)
        return names
    
    @property
    def default_target_modules(self) -> List[str]:
        """
        Default LoRA target modules for Trellis2 SLatFlowModel.
        
        Based on ModulatedSparseTransformerCrossBlock architecture:
        - self_attn: Uses fused to_qkv linear, plus to_out
        - cross_attn: Uses to_q, fused to_kv for context, plus to_out
        
        NOTE: mlp layers are excluded because they use SparseTensor as input,
        which standard PEFT LoRA doesn't support.
        """
        return [
            # Self-attention (uses fused qkv)
            "self_attn.to_qkv", "self_attn.to_out",
            # Cross-attention (uses separate q and fused kv)
            "cross_attn.to_q", "cross_attn.to_kv", "cross_attn.to_out",
        ]
    
    @property
    def text_encoder_names(self) -> List[str]:
        """Trellis2 uses image conditioning, no text encoders."""
        return []
    
    @property
    def text_encoders(self) -> List[nn.Module]:
        """No text encoders in Trellis2."""
        return []
    
    @property
    def preprocessing_modules(self) -> List[str]:
        """
        Modules needed during offline preprocessing (Stage 1).
        Image encoder and sparse structure model for initial structure.
        """
        return ['image_encoder', 'sparse_structure_flow_model', 'sparse_structure_decoder']
    
    @property
    def inference_modules(self) -> List[str]:
        """
        Modules needed during training loop.

        Reward computation requires rendering multi-view RGB, which needs
        the full dense -> shape -> tex -> decode pipeline.  Therefore both
        decoders and all upstream transformers are always loaded.
        """
        target = self.pipeline._target_flow_model  # e.g. 'shape_slat_1024'
        stage = target.split('_')[0]               # 'shape', 'tex', or 'dense'
        res_suffix = target.split('_')[-1]         # '1024' or '512'

        decoders = ['shape_decoder', 'tex_decoder']
        upstream = ['sparse_structure_flow_model', 'sparse_structure_decoder']

        if stage != 'shape':
            upstream.append(f'transformer_shape_{res_suffix}')
        if stage != 'tex':
            upstream.append(f'transformer_tex_{res_suffix}')

        return ['transformer', 'image_encoder'] + upstream + decoders
    
    # ======================== Encoding Methods ========================

    def preprocess_func(
        self,
        prompt: Optional[List[str]] = None,
        images: Optional[List[List[Image.Image]]] = None,
        videos: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Override BaseAdapter.preprocess_func to encode condition images at two
        resolutions in a single pass:

        - **512 px** → ``image_cond_512`` / ``neg_image_cond_512``  (dense stage)
        - **1024 px** → ``image_cond_1024`` / ``neg_image_cond_1024``  (shape/tex stages)

        ``pipeline.preprocess_image`` (which may invoke REMBG) is called only
        *once* per image.  When the source image is RGBA (e.g. loaded by
        ``Image3DDataset``), the alpha channel is used directly so REMBG is
        skipped entirely.

        The preprocessed PIL images are stored in ``condition_images`` for
        downstream reward scoring and visualisation.
        """
        if images is None:
            return {}

        device = self.device
        embed_dim = self.pipeline.image_encoder.model.config.hidden_size

        all_conds_512: List[torch.Tensor] = []
        all_neg_conds_512: List[torch.Tensor] = []
        all_conds_1024: List[torch.Tensor] = []
        all_neg_conds_1024: List[torch.Tensor] = []
        processed_images: List[List[Image.Image]] = []

        for img_list in images:
            if not img_list:
                dummy = torch.zeros(1, embed_dim, device=device)
                all_conds_512.append(dummy)
                all_neg_conds_512.append(dummy.clone())
                all_conds_1024.append(dummy)
                all_neg_conds_1024.append(dummy.clone())
                processed_images.append([])
                continue

            processed = [self.pipeline.preprocess_image(img) for img in img_list]
            processed_images.append(processed)

            # Encode at 512 px (dense stage) — get_cond handles RGBA→RGB internally
            cond_512 = self.pipeline.get_cond(processed, 512, include_neg_cond=True)
            all_conds_512.append(cond_512['cond'].squeeze(0).to(device))
            all_neg_conds_512.append(cond_512['neg_cond'].squeeze(0).to(device))

            # Encode at 1024 px (shape / tex stages)
            cond_1024 = self.pipeline.get_cond(processed, 1024, include_neg_cond=True)
            all_conds_1024.append(cond_1024['cond'].squeeze(0).to(device))
            all_neg_conds_1024.append(cond_1024['neg_cond'].squeeze(0).to(device))

        return {
            'image_cond_512': all_conds_512,
            'neg_image_cond_512': all_neg_conds_512,
            'image_cond_1024': all_conds_1024,
            'neg_image_cond_1024': all_neg_conds_1024,
            'condition_images': processed_images,
        }

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Trellis2 doesn't use text prompts. Return empty dict.
        Override to satisfy BaseAdapter interface.
        """
        return {}
    
    def encode_image(
        self,
        images: MultiImageBatch,
        resolution: int = 512,
        include_neg_cond: bool = True,
        **kwargs,
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """
        Encode condition images using DINOv2/v3 image encoder.
        c
        Args:
            images: Multi-image batch - List[List[Image.Image]]
            resolution: Resolution for image encoder (512 or 1024)
            include_neg_cond: Whether to include negative conditioning
        
        Returns:
            Dict with:
            - 'image_cond': List of conditioning tensors
            - 'neg_image_cond': List of negative conditioning tensors
            - 'condition_images': Preprocessed images
        """
        # Handle various input formats for multi-image batch
        # Expected: List[List[PIL.Image]] - batch of samples, each with list of condition images
        if isinstance(images, Image.Image):
            # Single image -> batch of 1 sample with 1 condition image
            images = [[images]]
        elif isinstance(images, list) and len(images) > 0:
            if isinstance(images[0], Image.Image):
                # List[PIL.Image] -> batch of 1 sample with N condition images
                images = [images]
            elif not isinstance(images[0], list):
                # Try to convert using standardize_image_batch for tensor/ndarray
                images = [[img] for img in standardize_image_batch(images, 'pil')]
        
        batch_size = len(images)
        
        device = self.device
        
        # Preprocess images and extract conditioning
        all_conds = []
        all_neg_conds = []
        processed_images = []
        
        for img_list in images:
            if not img_list:
                # No condition image - use zeros
                # Get embedding dim from image encoder
                dummy_cond = torch.zeros(1, self.pipeline.image_encoder.model.embed_dim).to(device)
                all_conds.append(dummy_cond)
                if include_neg_cond:
                    all_neg_conds.append(dummy_cond.clone())
                processed_images.append([])
                continue
            
            processed = [self.pipeline.preprocess_image(img) for img in img_list]
            processed_images.append(processed)
            
            # Get conditioning — get_cond handles RGBA→RGB internally
            cond_dict = self.pipeline.get_cond(processed, resolution, include_neg_cond)
            # Squeeze batch dim since we process one sample at a time
            # cond has shape (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            all_conds.append(cond_dict['cond'].squeeze(0))
            if include_neg_cond:
                all_neg_conds.append(cond_dict['neg_cond'].squeeze(0))
        
        result = {
            'image_cond': all_conds,
            'condition_images': processed_images,
        }
        if include_neg_cond:
            result['neg_image_cond'] = all_neg_conds
        
        return result
    
    def encode_video(
        self,
        videos: Any,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Trellis2 doesn't use video. Return None."""
        return None
    
    # ======================== Forward (Unified Entry Point) ========================
    
    def _build_sparse_inputs(
        self,
        latents,
        sparse_coords: Optional[torch.Tensor],
        next_latents=None,
        tex_concat_cond=None,
    ) -> dict:
        """Normalise sparse-stage inputs to SparseTensor (idempotent).

        If ``latents`` is already a SparseTensor the other arguments are
        assumed to also be SparseTensor (or None) and are passed through
        unchanged.  If ``latents`` is a plain tensor it is assembled into
        a SparseTensor using ``sparse_coords``.

        All sparse feats are kept in float32 to match the official Trellis2
        sample contract (``pipelines/trellis2_image_to_3d.py::sample_shape_slat``
        constructs noise as ``torch.randn(...).to(device)`` — fp32 by default).
        SLatFlowModel handles bf16 internally via ``manual_cast`` on transformer
        blocks and casts the output back to ``x.dtype``. Casting feats to bf16
        here would be a no-op under LoRA (where ``next(model.parameters()).dtype``
        is fp32) but would silently downcast and break log_prob reproducibility
        under full fine-tuning.

        Returns dict with keys ``x_t``, ``next_latents``, ``concat_cond``.
        """
        if isinstance(latents, SparseTensor):
            assert next_latents is None or isinstance(next_latents, SparseTensor)
            assert tex_concat_cond is None or isinstance(tex_concat_cond, SparseTensor)
            return {'x_t': latents, 'next_latents': next_latents, 'concat_cond': tex_concat_cond}

        device = self.device

        x_t = SparseTensor(
            feats=latents.to(device=device, dtype=torch.float32),  # (N, C) fp32
            coords=sparse_coords.to(device),
        )
        concat_cond = None
        if tex_concat_cond is not None:
            concat_cond = x_t.replace(
                feats=tex_concat_cond.to(device=device, dtype=torch.float32)  # (N, C) fp32
            )
        next_latents_st = None
        if next_latents is not None:
            next_latents_st = x_t.replace(
                feats=next_latents.to(device=device, dtype=torch.float32)  # (N, C) fp32
            )

        return {'x_t': x_t, 'next_latents': next_latents_st, 'concat_cond': concat_cond}

    @property
    def _training_stage(self) -> str:
        """Derive the training stage from pipeline config."""
        return self.pipeline._target_flow_model.split('_')[0]

    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        image_cond: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        sparse_coords: Optional[torch.Tensor] = None,
        tex_concat_cond: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
        stage_resolution: int = 1024,
        neg_image_cond: Optional[torch.Tensor] = None,
        compute_log_prob: bool = True,
        noise_level: Optional[float] = None,
        **kwargs,
    ) -> SDESchedulerOutput:
        """Trainer-compatible single denoising step for all Trellis2 stages.

        ``latents`` may be a plain tensor (from Trainer batch) or a
        SparseTensor (from inference loops).  ``_build_sparse_inputs``
        handles both cases idempotently.

        ``t`` / ``t_next`` are in the framework-standard ``[0, 1000]``
        scheduler scale and are normalised to Trellis2-native ``[0, 1]``
        internally.  They may arrive as:
          * 0-d tensor — adapter-internal callers (single shared timestep)
          * ``(B,)`` tensor, all equal — GRPO passes ``batch['timesteps'][:, step_idx]``
          * ``(B,)`` tensor, per-sample — NFT passes per-sample timesteps

        All three forms are normalised to a ``(B,)`` tensor.  The
        sparse scheduler's ``_expand_to_points`` handles both scalar and
        per-sample ``(B,)`` uniformly.

        When ``stage`` is None (the typical optimize path where batch
        dict has no ``stage`` key), it is derived from
        ``pipeline._target_flow_model`` so the correct flow model is
        always selected.

        Per-stage CFG (strength / interval / rescale) is read only from
        ``pipeline.json`` via ``_get_stage_guidance`` — not from YAML
        ``training_args.guidance_*``.
        """
        if stage is None:
            stage = self._training_stage

        # Normalise t / t_next to (B,) tensors.
        if t.ndim == 0:
            t = t.unsqueeze(0)                                                # (1,)
        if t_next is not None and t_next.ndim == 0:
            t_next = t_next.unsqueeze(0)                                      # (1,)

        # Convert from framework-standard [0, 1000] to Trellis2-native [0, 1].
        t = t / 1000.0
        if t_next is not None:
            t_next = t_next / 1000.0

        t_repr = float(t.float().mean().item())
        t_next_repr = float(t_next.float().mean().item()) if t_next is not None else 0.0

        g = self._get_stage_guidance(stage)
        guidance_scale = g['guidance_scale']
        guidance_interval = g['guidance_interval']
        guidance_rescale = g['guidance_rescale']

        apply_cfg = (guidance_interval[0] <= t_repr <= guidance_interval[1]
                     and neg_image_cond is not None and guidance_scale != 1.0)

        if stage == 'dense':
            scheduler = self._get_stage_scheduler('dense')
            sigma_min = scheduler._sigma_min
            pred_pos = self._forward_dense(t_repr, latents, image_cond)
            if apply_cfg:
                pred_neg = self._forward_dense(t_repr, latents, neg_image_cond)
                pred_v = self._apply_cfg_dense(
                    pred_pos.float(), pred_neg.float(), latents.float(),
                    t_repr, guidance_scale, guidance_rescale, sigma_min,
                )
            else:
                pred_v = pred_pos.float()

            return scheduler.step(
                noise_pred=pred_v, timestep=t_repr * 1000, latents=latents,
                next_latents=next_latents, timestep_next=t_next_repr * 1000,
                noise_level=noise_level, compute_log_prob=compute_log_prob,
            )

        elif stage in ('shape', 'tex'):
            scheduler = self._get_stage_scheduler(stage)
            sigma_min = scheduler._sigma_min
            effective_concat_cond = tex_concat_cond if stage == 'tex' else None
            sparse = self._build_sparse_inputs(
                latents, sparse_coords,
                next_latents=next_latents, tex_concat_cond=effective_concat_cond,
            )
            x_t = sparse['x_t']
            concat_cond = sparse['concat_cond']
            B_forward = int(x_t.coords[:, 0].max().item()) + 1                   # scalar int

            pred_pos = self._forward_sparse(
                t, x_t, image_cond, concat_cond=concat_cond,
                stage=stage, stage_resolution=stage_resolution,
            )
            if apply_cfg:
                pred_neg = self._forward_sparse(
                    t, x_t, neg_image_cond, concat_cond=concat_cond,
                    stage=stage, stage_resolution=stage_resolution,
                )
                pred_v = self._apply_cfg_sparse(
                    pred_pos, pred_neg, x_t, t_repr, guidance_scale, guidance_rescale, sigma_min,
                    batch_size=B_forward,
                )
            else:
                pred_v = pred_pos

            pred_v = pred_v.replace(feats=pred_v.feats.float())  # (N, C) fp32

            output = scheduler.step(
                pred_v, t, t_next if t_next is not None else 0.0, x_t,
                next_latents=sparse['next_latents'],
                noise_level=noise_level, compute_log_prob=compute_log_prob,
            )
            if output.log_prob is not None:
                output.log_prob = self._reduce_sparse_log_prob(
                    output.log_prob, x_t.coords, batch_size=B_forward,
                )
            return output

        else:
            raise ValueError(f"Unknown stage: {stage!r}. Expected 'dense', 'shape', or 'tex'.")
    
    # ---------------------- Dense path ----------------------

    def _forward_dense(
        self, t: float, latents: torch.Tensor, cond: torch.Tensor,
    ) -> torch.Tensor:
        """Pure model forward for dense stage. Returns ``pred_v`` tensor.

        Autocast is explicitly disabled so the model's own ``manual_cast``
        handles dtype transitions between float32 head/tail and bf16 blocks.
        Inputs should be float32 (matching official pipeline behavior).
        """
        device = self.device
        if 'dense' == self._training_stage:
            flow_model = self.transformer
        else:
            flow_model = self.pipeline.get_flow_model('dense')

        latents = latents.to(device=device, dtype=torch.float32)
        cond = cond.to(device=device, dtype=torch.float32)
        t_tensor = torch.full((latents.shape[0],), 1000 * t, device=device, dtype=torch.float32)  # (B,)

        with torch.autocast('cuda', enabled=False):
            pred_v = flow_model(latents, t_tensor, cond)
        return pred_v

    # ---------------------- CFG helpers ----------------------

    @staticmethod
    def _apply_cfg_dense(
        pred_pos: torch.Tensor,
        pred_neg: torch.Tensor,
        x_t: torch.Tensor,
        t_val: float,
        guidance_scale: float,
        guidance_rescale: float,
        sigma_min: float,
    ) -> torch.Tensor:
        """CFG blend + optional rescale for the dense stage. Returns ``pred_v``.

        Aligned with ``trellis2_cfg_dense`` in reference rollout/base.py.
        Both ``pred_pos`` / ``pred_neg`` and ``x_t`` are plain float tensors
        of shape ``(B, C, D, H, W)``.
        """
        pred_v = guidance_scale * pred_pos + (1.0 - guidance_scale) * pred_neg  # (B, C, D, H, W)

        if guidance_rescale > 0:
            alpha = 1.0 - sigma_min
            beta  = sigma_min + alpha * t_val
            x_0_pos = alpha * x_t - beta * pred_pos                        # (B, C, D, H, W)
            x_0_cfg = alpha * x_t - beta * pred_v                          # (B, C, D, H, W)
            reduce_dims = list(range(1, x_0_pos.ndim))                     # [1,2,3,4] for 5D
            std_pos = x_0_pos.std(dim=reduce_dims, keepdim=True)           # (B, 1, 1, 1, 1)
            std_cfg = x_0_cfg.std(dim=reduce_dims, keepdim=True)           # (B, 1, 1, 1, 1)
            x_0_rescaled = x_0_cfg * (std_pos / (std_cfg + 1e-8))
            x_0 = guidance_rescale * x_0_rescaled + (1.0 - guidance_rescale) * x_0_cfg
            pred_v = (alpha * x_t - x_0) / beta
        return pred_v

    @staticmethod
    def _apply_cfg_sparse(
        pred_pos: 'SparseTensor',
        pred_neg: 'SparseTensor',
        x_t: 'SparseTensor',
        t_val: float,
        guidance_scale: float,
        guidance_rescale: float,
        sigma_min: float,
        batch_size: int,
    ) -> 'SparseTensor':
        """CFG blend + optional rescale for sparse stages. Returns ``pred_v`` SparseTensor.

        Aligned with official ``classifier_free_guidance_mixin._inference_model``:
        the rescale ratio is a per-sample std over all (N_b * C) elements of that
        sample (equivalent to ``x_0.std(dim=list(range(1, ndim)))`` for dense).

        ``batch_size`` is passed explicitly; empty samples (no coords) still get
        a ratio slot of 1.0 so broadcasting stays length-correct.
        """
        pred_v = guidance_scale * pred_pos + (1.0 - guidance_scale) * pred_neg  # SparseTensor

        if guidance_rescale > 0:
            alpha = 1.0 - sigma_min
            beta  = sigma_min + alpha * t_val
            x_0_pos = alpha * x_t - beta * pred_pos                              # SparseTensor
            x_0_cfg = alpha * x_t - beta * pred_v                                # SparseTensor

            feats_pos_f32 = x_0_pos.feats.float()                                # (N_total, C) fp32
            feats_cfg_f32 = x_0_cfg.feats.float()                                # (N_total, C) fp32
            batch_idx = x_0_pos.coords[:, 0].long()                              # (N_total,)

            ratio = torch.ones(
                batch_size, device=feats_pos_f32.device, dtype=torch.float32
            )                                                                    # (B,) fp32
            for b in range(batch_size):
                mask_b = batch_idx == b                                          # (N_total,)
                if mask_b.any():
                    std_pos_b = feats_pos_f32[mask_b].std()                      # scalar fp32
                    std_cfg_b = feats_cfg_f32[mask_b].std()                      # scalar fp32
                    ratio[b] = std_pos_b / (std_cfg_b + 1e-8)                    # scalar

            ratio_per_point = ratio[batch_idx].unsqueeze(-1).to(
                x_0_cfg.feats.dtype
            )                                                                    # (N_total, 1)
            x_0_rescaled = x_0_cfg.replace(feats=x_0_cfg.feats * ratio_per_point)
            x_0 = guidance_rescale * x_0_rescaled + (1.0 - guidance_rescale) * x_0_cfg
            pred_v = (alpha * x_t - x_0) / beta                                  # SparseTensor
        return pred_v

    # ---------------------- Shape / Tex path ----------------------

    def _forward_sparse(
        self,
        t: Union[float, torch.Tensor],
        x_t: 'SparseTensor',
        cond: torch.Tensor,
        concat_cond: Optional['SparseTensor'] = None,
        stage: str = 'shape',
        stage_resolution: int = 1024,
    ) -> 'SparseTensor':
        """Pure model forward for sparse stage. Returns ``pred_v`` SparseTensor.

        ``t`` may be a Python float (shared timestep) or a ``(B,)`` tensor
        (per-sample timestep).  The flow model already accepts a ``(B,)``
        timestep tensor, so both cases are handled uniformly.

        Autocast is explicitly disabled so the model's own ``manual_cast``
        handles dtype transitions between float32 head/tail and bf16 blocks.
        No CFG, no scheduler step — the caller handles both.
        """
        device = self.device
        B = x_t.shape[0]  # batch size from SparseTensor.coords
        if stage == self._training_stage:
            flow_model = self.transformer
        else:
            flow_model = self.pipeline.get_flow_model(stage, stage_resolution)

        cond = cond.to(device=device, dtype=torch.float32)

        if isinstance(t, (int, float)):
            t_tensor = torch.full((B,), 1000 * t, device=device, dtype=torch.float32)  # (B,)
        else:
            t_1000 = t.float().to(device) * 1000                                       # (B,) or (1,)
            t_tensor = t_1000.expand(B)                                                # (B,)

        with torch.autocast('cuda', enabled=False):
            pred_v = flow_model(x=x_t, t=t_tensor, cond=cond, concat_cond=concat_cond)
        return pred_v

    def _get_stage_conditioning(self, stage, cond_512, neg_512, cond_1024, neg_1024):
        """Return the (cond, neg_cond) pair appropriate for *stage*."""
        if stage == 'dense':
            return cond_512, neg_512
        return cond_1024, neg_1024

    def _run_stage_inference(
        self,
        stage: str,
        samples: List['Trellis2Sample'],
        image_cond: List[torch.Tensor],
        neg_image_cond: List[torch.Tensor],
        *,
        resolution: int,
        num_inference_steps: int,
        generator: Optional[torch.Generator],
        trajectory_indices: 'TrajectoryIndicesType',
        extra_call_back_kwargs: List[str],
        ss_resolution: int,
        is_training_stage: bool,
        compute_log_prob: bool,
    ) -> None:
        """Dispatch a single stage to the corresponding private method."""
        g = self._get_stage_guidance(stage)
        if stage == 'dense':
            self._inference_dense(
                samples=samples, image_cond=image_cond,
                neg_image_cond=neg_image_cond,
                num_inference_steps=num_inference_steps,
                guidance_scale=g['guidance_scale'],
                guidance_interval=g['guidance_interval'],
                guidance_rescale=g['guidance_rescale'],
                generator=generator,
                trajectory_indices=trajectory_indices,
                extra_call_back_kwargs=extra_call_back_kwargs,
                ss_resolution=ss_resolution,
                is_training_stage=is_training_stage,
                compute_log_prob=compute_log_prob,
            )
        elif stage == 'shape':
            self._inference_shape(
                samples=samples, image_cond=image_cond,
                neg_image_cond=neg_image_cond,
                resolution=resolution,
                guidance_scale=g['guidance_scale'],
                guidance_interval=g['guidance_interval'],
                guidance_rescale=g['guidance_rescale'],
                generator=generator,
                trajectory_indices=trajectory_indices,
                extra_call_back_kwargs=extra_call_back_kwargs,
                is_training_stage=is_training_stage,
                compute_log_prob=compute_log_prob,
            )
        elif stage == 'tex':
            self._inference_tex(
                samples=samples, image_cond=image_cond,
                neg_image_cond=neg_image_cond,
                resolution=resolution,
                guidance_scale=g['guidance_scale'],
                guidance_interval=g['guidance_interval'],
                guidance_rescale=g['guidance_rescale'],
                generator=generator,
                trajectory_indices=trajectory_indices,
                extra_call_back_kwargs=extra_call_back_kwargs,
                is_training_stage=is_training_stage,
                compute_log_prob=compute_log_prob,
            )
        else:
            raise ValueError(
                f"Unknown stage: {stage!r}. Expected 'dense', 'shape', or 'tex'."
            )

    @torch.no_grad()
    def inference(
        self,
        # Raw inputs
        prompt: Optional[List[str]] = None,  # Unused, for interface compatibility
        images: Optional[MultiImageBatch] = None,
        # Pre-encoded inputs (output of preprocess_func)
        image_cond_512: Optional[List[torch.Tensor]] = None,
        neg_image_cond_512: Optional[List[torch.Tensor]] = None,
        image_cond_1024: Optional[List[torch.Tensor]] = None,
        neg_image_cond_1024: Optional[List[torch.Tensor]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        # Stage(s) selection — single string or ordered list for multi-stage
        stages: Union[str, List[str], None] = None,
        training_stage: Optional[str] = None,
        # Generation parameters
        resolution: int = 1024,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        # RL-specific parameters
        compute_log_prob: bool = True,
        trajectory_indices: TrajectoryIndicesType = 'all',
        extra_call_back_kwargs: List[str] = [],
        # Trellis2-specific
        decode_output: bool = False,
        # Render parameters (used when decode_output=True)
        render_num_frames: int = 24,
        render_resolution: int = 512,
        render_bg_color: Tuple[float, float, float] = (0, 0, 0),
        envmap_path: Optional[str] = None,
        # Pre-created samples (skip stub creation; enables stage-skip)
        samples: Optional[List['Trellis2Sample']] = None,
        **kwargs,
    ) -> List[Trellis2Sample]:
        """Multi-stage inference dispatcher for Trellis2.

        Accepts either raw images or pre-encoded conditioning tensors
        (output of ``preprocess_func``).

        When *samples* is provided the sample-stub creation step is skipped
        and stages whose outputs are already present on the samples (e.g.
        ``sparse_coords`` for dense) are automatically skipped.

        Stage definitions:
            - **dense**: Runs ``sparse_structure_flow_model`` on 5D dense latents.
              Decodes the final denoised tensor to sparse coordinates stored on each
              sample.  Run this first if you want to train the structure itself.
            - **shape**: Runs ``shape_slat_flow_model`` on SparseTensor latents.
              Requires ``sample.sparse_coords`` (set by dense stage or auto-sampled).
            - **tex**: Runs ``tex_slat_flow_model`` on SparseTensor latents.
              Requires ``sample.sparse_coords`` AND ``sample.shape_all_latents``
              (the normalised final-step shape features become ``tex_concat_cond``).

        After each stage the corresponding per-stage fields
        (``dense_all_latents``, ``shape_all_latents``, ``tex_all_latents``, …) are
        populated.  The **last** stage's fields are also copied to the standard
        ``all_latents / image_cond / timesteps`` fields for single-stage Trainer
        (GRPO / NFT) compatibility.

        Args:
            images: Raw PIL images; encoded on-the-fly when pre-encoded conds absent.
            image_cond_512 / neg_image_cond_512: Pre-encoded 512 px cond (dense stage).
            image_cond_1024 / neg_image_cond_1024: Pre-encoded 1024 px cond
                (shape / tex stages).
            stages: Stage(s) to run in order, e.g. ``'shape'``,
                ``['dense', 'shape', 'tex']``.  ``None`` infers from
                ``pipeline._target_flow_model``.
            training_stage: Which stage's results to mirror into the standard
                ``all_latents / log_probs / image_cond`` fields consumed by
                Trainers.  ``None`` defaults to the last stage in *stages*.
            resolution: Output resolution for shape/tex (512 or 1024).
            decode_output: If ``True``, decode each sample to mesh after generation.

        Returns:
            ``List[Trellis2Sample]`` — one element per conditioning input.
        """
        resolution = self._as_scalar_resolution(resolution)

        # ── 1. Normalise stage list ────────────────────────────────────────────
        if stages is None:
            stages_list = [self.pipeline._target_flow_model.split('_')[0]]
        elif isinstance(stages, str):
            stages_list = [stages]
        else:
            stages_list = list(stages)

        # Determine which stage mirrors its results to the standard Trainer fields
        if training_stage is None:
            _mirror_stage = stages_list[-1]
        else:
            if training_stage not in stages_list:
                raise ValueError(
                    f"training_stage={training_stage!r} is not in stages={stages_list}. "
                    "The training stage must be included in the stages list."
                )
            _mirror_stage = training_stage

        # ── 2. Resolve conditioning ────────────────────────────────────────────
        cond_512, neg_512, cond_1024, neg_1024, condition_images, batch_size = \
            self._resolve_conditioning(
                stages_list, images=images, condition_images=condition_images,
                image_cond_512=image_cond_512, neg_image_cond_512=neg_image_cond_512,
                image_cond_1024=image_cond_1024, neg_image_cond_1024=neg_image_cond_1024,
            )

        # ── 3. Initialise sample stubs (skip if pre-created) ─────────────────
        if samples is None:
            samples = [
                Trellis2Sample(
                    condition_images=condition_images[b] if condition_images is not None else None,
                    resolution=resolution,
                    prompt=prompt[b] if prompt else None,
                )
                for b in range(batch_size)
            ]

        # ── 4. Dispatch each stage (skip stages already computed) ─────────────
        ss_resolution = 32 if resolution <= 512 else 64

        for stage in stages_list:
            if stage == 'dense' and samples[0].sparse_coords is not None:
                continue
            if stage == 'shape' and samples[0].shape_final_latent is not None:
                continue
            if stage == 'tex' and samples[0].tex_final_latent is not None:
                continue

            is_training = (stage == _mirror_stage)
            stage_cond, stage_neg = self._get_stage_conditioning(
                stage, cond_512, neg_512, cond_1024, neg_1024,
            )
            self._run_stage_inference(
                stage, samples, stage_cond, stage_neg,
                resolution=resolution,
                num_inference_steps=num_inference_steps,
                generator=generator,
                trajectory_indices=trajectory_indices,
                extra_call_back_kwargs=extra_call_back_kwargs,
                ss_resolution=ss_resolution,
                is_training_stage=is_training,
                compute_log_prob=compute_log_prob,
            )

        if decode_output:
            if condition_images is not None:
                _apply_bg_to_condition_images(
                    samples, condition_images_rgba=condition_images,
                    bg_color=render_bg_color,
                )
            envmap = self._build_envmap(envmap_path)
            samples = [
                self.render_latents(
                    s,
                    num_frames=render_num_frames,
                    resolution=render_resolution,
                    bg_color=render_bg_color,
                    envmap=envmap,
                )
                for s in samples
            ]

        return samples

    def inference_with_shared_dense(self, **kwargs) -> List[Trellis2Sample]:
        """Deprecated: delegates to ``inference(training_stage='shape')``."""
        kwargs.setdefault('stages', ['dense', 'shape', 'tex'])
        kwargs.setdefault('training_stage', 'shape')
        return self.inference(**kwargs)

    def inference_with_shared_dense_shape(self, **kwargs) -> List[Trellis2Sample]:
        """Deprecated: delegates to ``inference(training_stage='tex')``."""
        kwargs.setdefault('stages', ['dense', 'shape', 'tex'])
        kwargs.setdefault('training_stage', 'tex')
        return self.inference(**kwargs)

    def _decode_dense_to_coords(
        self,
        z_s: torch.Tensor,        # (B, C, D, H, W) dense structure latent
        ss_resolution: int = 32,  # target coord resolution
        max_num_coords: int = 49152,
    ) -> List[torch.Tensor]:
        """Decode dense structure latent to per-sample sparse coordinates.

        Mirrors the decode step in ``Trellis2ImageTo3DPipeline.sample_sparse_structure``
        and the reference ``binarize_structure``::

            decoder(z_s) > 0  →  optional max_pool3d  →  argwhere per sample

        When the number of occupied voxels exceeds *max_num_coords*, a random
        subset is selected (matching the official ``1024_cascade`` pipeline
        which caps at 49152 tokens to avoid decoder OOM).

        Returns:
            List of ``(N_b, 4)`` int32 coord tensors, one per batch element.
            Each tensor's ``[:, 0]`` column is set to 0 (single-sample convention).
        """
        decoder = self.pipeline.sparse_structure_decoder
        decoded = decoder(z_s) > 0          # (B, 1, D, H, W) bool
        if ss_resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // ss_resolution
            decoded = torch.nn.functional.max_pool3d(
                decoded.float(), ratio, ratio, 0
            ) > 0.5                          # (B, 1, ss_res, ss_res, ss_res)
        # argwhere → (N_total, 5): [batch, chan, d, h, w]; keep [batch, d, h, w]
        all_coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()  # (N_total, 4)
        B = z_s.shape[0]
        result = []
        for b in range(B):
            coords_b = all_coords[all_coords[:, 0] == b].clone().contiguous()  # (N_b, 4)
            N_b = coords_b.shape[0]
            if max_num_coords > 0 and N_b > max_num_coords:
                perm = torch.randperm(N_b, device=coords_b.device)[:max_num_coords]  # (max_num_coords,)
                coords_b = coords_b[perm]                                             # (max_num_coords, 4)
            result.append(coords_b)
        return result

    @torch.no_grad()
    def _inference_dense(
        self,
        samples: List['Trellis2Sample'],
        image_cond: List[torch.Tensor],      # per-sample (seq, D)
        neg_image_cond: List[torch.Tensor],  # per-sample (seq, D)
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        guidance_rescale: float = 0.0,
        generator: Optional[torch.Generator] = None,
        trajectory_indices: TrajectoryIndicesType = 'all',
        extra_call_back_kwargs: List[str] = [],
        ss_resolution: int = 32,
        is_training_stage: bool = True,
        compute_log_prob: bool = True,
    ) -> None:
        """Run the dense sparse-structure flow model.

        Generates one 5D dense latent per conditioning input, decodes the final
        denoised tensor to sparse coordinates, and populates ``dense_*`` fields
        and ``sparse_coords`` on each sample in-place.
        """
        device = self.device
        B = len(samples)
        flow_model = self.pipeline.get_flow_model('dense')

        cond_batched = (image_cond if isinstance(image_cond, torch.Tensor)
                        else torch.stack(image_cond)).to(device=device)      # (B, seq, D)
        neg_batched  = (neg_image_cond if isinstance(neg_image_cond, torch.Tensor)
                        else torch.stack(neg_image_cond)).to(device=device)

        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        gen = generator[0] if isinstance(generator, list) else generator
        noise = torch.randn(
            B, in_channels, reso, reso, reso, generator=gen
        ).to(device)                                                  # (B, C, D, H, W)

        scheduler = self._get_stage_scheduler('dense')
        t_np = scheduler._timesteps_np                                # float64, (steps+1,)
        num_inference_steps = len(t_np) - 1
        timesteps = torch.tensor(t_np, device=device, dtype=torch.float32) * 1000

        latent_collector    = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latent_collector.collect(noise.unsqueeze(0), step_idx=0)
        callback_collector  = create_callback_collector(trajectory_indices, num_inference_steps)
        log_prob_collector  = create_trajectory_collector(trajectory_indices, num_inference_steps)

        sigma_min = scheduler._sigma_min

        current = noise
        for i in range(num_inference_steps):
            t_val      = float(t_np[i])
            t_next_val = float(t_np[i + 1])
            noise_level_i = scheduler.get_noise_level_for_sigma(t_val)
            step_compute_lp = compute_log_prob and (noise_level_i > 0 if isinstance(noise_level_i, (int, float)) else (noise_level_i > 0).any().item())

            pred_pos = self._forward_dense(t_val, current, cond_batched)

            apply_cfg = guidance_interval[0] <= t_val <= guidance_interval[1]
            if apply_cfg and guidance_scale != 1.0:
                pred_neg = self._forward_dense(t_val, current, neg_batched)
                pred_v = self._apply_cfg_dense(
                    pred_pos.float(), pred_neg.float(), current.float(),
                    t_val, guidance_scale, guidance_rescale, sigma_min,
                )
            else:
                pred_v = pred_pos

            pred_v = pred_v.float()

            output = scheduler.step(
                noise_pred=pred_v,
                timestep=t_val * 1000,
                latents=current,
                next_latents=None,
                timestep_next=t_next_val * 1000,
                noise_level=noise_level_i,
                compute_log_prob=step_compute_lp,
            )
            current = output.next_latents

            latent_collector.collect(current.unsqueeze(0), i + 1)
            callback_collector.collect_step(i, output, extra_call_back_kwargs)
            if step_compute_lp and output.log_prob is not None:
                log_prob_collector.collect(output.log_prob.detach().unsqueeze(0), i)

        all_latents        = latent_collector.get_result()         # List[(1, B, C, D, H, W)]
        latent_index_map   = latent_collector.get_index_map()
        extra_callback_res = callback_collector.get_result()
        callback_index_map = callback_collector.get_index_map()
        all_log_probs      = log_prob_collector.get_result()
        log_prob_index_map = log_prob_collector.get_index_map()

        coords_list = self._decode_dense_to_coords(current, ss_resolution)

        for b, sample in enumerate(samples):
            dense_latents_b = (
                torch.stack([lat[0][b] for lat in all_latents], dim=0)
                if all_latents else None
            )                                                          # (T', C, D, H, W)
            dense_log_probs_b = (
                torch.stack([lp[0][b] for lp in all_log_probs], dim=0)
                if all_log_probs else None
            )                                                          # (T_log,)
            coords_b = coords_list[b]
            coords_b[:, 0] = 0

            sample.sparse_coords             = coords_b
            sample.dense_final_latent        = current[b]
            sample.dense_all_latents         = dense_latents_b
            sample.dense_log_probs           = dense_log_probs_b
            sample.dense_image_cond          = image_cond[b]
            sample.dense_neg_image_cond      = neg_image_cond[b]
            sample.dense_latent_index_map    = latent_index_map
            sample.dense_log_prob_index_map  = log_prob_index_map
            sample.dense_timesteps           = timesteps
            if is_training_stage:
                sample.activate_stage('dense')
                sample.extra_kwargs = {
                    **{k: (v[b] if isinstance(v, (list, torch.Tensor)) else v)
                       for k, v in extra_callback_res.items()},
                    'callback_index_map': callback_index_map,
                }

    @torch.no_grad()
    def _inference_shape(
        self,
        samples: List['Trellis2Sample'],
        image_cond: List[torch.Tensor],      # per-sample (seq, D)
        neg_image_cond: List[torch.Tensor],  # per-sample (seq, D)
        resolution: int = 1024,
        guidance_scale: float = 7.5,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        guidance_rescale: float = 0.0,
        generator: Optional[torch.Generator] = None,
        trajectory_indices: TrajectoryIndicesType = 'all',
        extra_call_back_kwargs: List[str] = [],
        is_training_stage: bool = True,
        compute_log_prob: bool = True,
    ) -> None:
        """Run the shape SLat flow model as a single batched pass.

        All samples are merged into one batched ``SparseTensor`` and
        denoised together, then results are split back per sample.

        Requires: ``sample.sparse_coords is not None`` for every sample.
        """
        device = self.device
        B = len(samples)
        flow_model  = self.pipeline.get_flow_model('shape', resolution)
        in_channels = flow_model.in_channels

        scheduler = self._get_stage_scheduler('shape')
        num_inference_steps = len(scheduler.get_timesteps_for_loop())
        timesteps = scheduler.timesteps

        # ── Batch assembly ──────────────────────────────────────────────
        # Collect per-sample coords and noise, then merge into a single
        # batched SparseTensor via from_tensor_list (which offsets batch
        # indices automatically).  This lets the denoising loop run once
        # for all B samples instead of looping per sample.
        coords_list = []
        noise_list = []
        for b, sample in enumerate(samples):
            if sample.sparse_coords is None:
                raise ValueError(
                    f"sample[{b}].sparse_coords is None. "
                    "Run the dense stage first or pre-assign coords via the dispatcher."
                )
            coords_b = sample.sparse_coords.to(device)                             # (N_b, 4)
            noise_b = torch.randn(coords_b.shape[0], in_channels,
                                  device=device, dtype=torch.float32)              # (N_b, C)
            coords_list.append(coords_b)
            noise_list.append(noise_b)

        x_t = SparseTensor.from_tensor_list(noise_list, coords_list)               # batch_size=B

        latent_collector   = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latent_collector.collect(x_t.feats.unsqueeze(0), step_idx=0)               # (1, N_total, C)
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)
        log_prob_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)

        cond = torch.stack(
            [c.to(device=device, dtype=torch.float32) for c in image_cond], dim=0,
        )                                                                          # (B, seq, D)
        neg_cond = torch.stack(
            [c.to(device=device, dtype=torch.float32) for c in neg_image_cond], dim=0,
        )                                                                          # (B, seq, D)
        sigma_min = scheduler._sigma_min

        # ── Batched denoising loop ─────────────────────────────────────
        # All B samples share the same timestep schedule.  _forward_sparse
        # and scheduler.step both operate on the batched SparseTensor
        # transparently.  log_prob is reduced from per-point (N_total,) to
        # per-sample (B,) via _reduce_sparse_log_prob.
        for step_idx in scheduler.get_timesteps_for_loop():
            t_val = scheduler.get_precise_t(step_idx)
            t_next_val = scheduler.get_precise_t(step_idx + 1)

            if self._mode != 'eval':
                noise_level_i = scheduler.get_noise_level_for_sigma(t_val)
                step_compute_lp = compute_log_prob and (noise_level_i > 0)
            else:
                noise_level_i = 0.0
                step_compute_lp = False

            pred_pos = self._forward_sparse(t_val, x_t, cond, stage='shape', stage_resolution=resolution)

            apply_cfg = guidance_interval[0] <= t_val <= guidance_interval[1]
            if apply_cfg and guidance_scale != 1.0:
                pred_neg = self._forward_sparse(t_val, x_t, neg_cond, stage='shape', stage_resolution=resolution)
                pred_v = self._apply_cfg_sparse(
                    pred_pos, pred_neg, x_t, t_val, guidance_scale, guidance_rescale, sigma_min,
                    batch_size=B,
                )
            else:
                pred_v = pred_pos

            output = scheduler.step(
                pred_v, t_val, t_next_val, x_t,
                noise_level=noise_level_i,
                compute_log_prob=step_compute_lp,
            )
            if output.log_prob is not None:
                output.log_prob = self._reduce_sparse_log_prob(
                    output.log_prob, x_t.coords, batch_size=B,
                )                                                                  # (B,)

            x_t = output.next_latents                                              # SparseTensor
            latent_collector.collect(x_t.feats.unsqueeze(0), step_idx + 1)         # (1, N_total, C)
            callback_collector.collect_step(step_idx, output, extra_call_back_kwargs)
            if step_compute_lp and output.log_prob is not None:
                log_prob_collector.collect(output.log_prob.detach().unsqueeze(0), step_idx)  # (1, B)

        # ── Result splitting ────────────────────────────────────────────
        # Collectors accumulated batched data; now split back per sample.
        # - stacked_latents: (T', N_total, C) → per sample via layout slice
        # - stacked_log_probs: (T_log, B) → per sample via column index
        # - extra_callback_res: values indexed by [b] (list/tensor) or
        #   kept as-is (scalars), following the dense-stage convention.
        all_latents        = latent_collector.get_result()
        latent_index_map   = latent_collector.get_index_map()
        extra_callback_res = callback_collector.get_result()
        callback_index_map = callback_collector.get_index_map()
        all_log_probs      = log_prob_collector.get_result()
        log_prob_index_map = log_prob_collector.get_index_map()

        stacked_latents = (
            torch.stack([lat[0] for lat in all_latents], dim=0)
            if all_latents else None
        )                                                                          # (T', N_total, C)
        stacked_log_probs = (
            torch.stack([lp[0] for lp in all_log_probs], dim=0)
            if all_log_probs else None
        )                                                                          # (T_log, B)

        feats_list, _ = x_t.to_tensor_list()                                       # List of (N_b, C)

        for b, sample in enumerate(samples):
            slc = x_t.layout[b]
            sample.shape_final_latent        = feats_list[b]                       # (N_b, C)
            sample.shape_all_latents         = (
                stacked_latents[:, slc, :] if stacked_latents is not None else None
            )                                                                      # (T', N_b, C)
            sample.shape_log_probs           = (
                stacked_log_probs[:, b] if stacked_log_probs is not None else None
            )                                                                      # (T_log,)
            sample.shape_image_cond          = image_cond[b]
            sample.shape_neg_image_cond      = neg_image_cond[b]
            sample.shape_latent_index_map    = latent_index_map
            sample.shape_log_prob_index_map  = log_prob_index_map
            sample.shape_timesteps           = timesteps
            if is_training_stage:
                sample.activate_stage('shape')
                sample.extra_kwargs = {
                    **{k: (v[b] if isinstance(v, (list, torch.Tensor)) else v)
                       for k, v in extra_callback_res.items()},
                    'callback_index_map': callback_index_map,
                }

    @torch.no_grad()
    def _inference_tex(
        self,
        samples: List['Trellis2Sample'],
        image_cond: List[torch.Tensor],      # per-sample (seq, D)
        neg_image_cond: List[torch.Tensor],  # per-sample (seq, D)
        resolution: int = 1024,
        guidance_scale: float = 7.5,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        guidance_rescale: float = 0.0,
        generator: Optional[torch.Generator] = None,
        trajectory_indices: TrajectoryIndicesType = 'all',
        extra_call_back_kwargs: List[str] = [],
        is_training_stage: bool = True,
        compute_log_prob: bool = True,
    ) -> None:
        """Run the texture SLat flow model as a single batched pass.

        Requires: ``sample.sparse_coords`` and ``sample.shape_final_latent``
        for every sample.
        """
        device = self.device
        B = len(samples)
        flow_model  = self.pipeline.get_flow_model('tex', resolution)

        scheduler = self._get_stage_scheduler('tex')
        num_inference_steps = len(scheduler.get_timesteps_for_loop())
        timesteps = scheduler.timesteps

        # ── Batch assembly ──────────────────────────────────────────────
        # Same pattern as _inference_shape, plus an extra concat_cond
        # built from each sample's shape_final_latent.  The concat_cond
        # SparseTensor shares the same coords/layout as x_t so its feats
        # can simply be cat-ed in the same order.
        coords_list = []
        noise_list = []
        concat_cond_list = []
        for b, sample in enumerate(samples):
            if sample.sparse_coords is None:
                raise ValueError(f"sample[{b}].sparse_coords is None.")
            if sample.shape_final_latent is None:
                raise ValueError(
                    f"sample[{b}].shape_final_latent is None. Run shape stage first."
                )
            coords_b = sample.sparse_coords.to(device)                             # (N_b, 4)
            tex_concat_cond_b = sample.shape_final_latent.to(
                device=device, dtype=torch.float32,
            )                                                                      # (N_b, C_shape)

            noise_channels = flow_model.in_channels - tex_concat_cond_b.shape[-1]
            if noise_channels <= 0:
                raise ValueError(
                    f"tex flow_model.in_channels ({flow_model.in_channels}) must exceed "
                    f"tex_concat_cond channels ({tex_concat_cond_b.shape[-1]})."
                )
            noise_b = torch.randn(coords_b.shape[0], noise_channels,
                                  device=device, dtype=torch.float32)              # (N_b, C_noise)
            coords_list.append(coords_b)
            noise_list.append(noise_b)
            concat_cond_list.append(tex_concat_cond_b)

        x_t = SparseTensor.from_tensor_list(noise_list, coords_list)               # batch_size=B
        concat_cond_sparse = x_t.replace(
            feats=torch.cat(concat_cond_list, dim=0),                              # (N_total, C_shape)
        )

        latent_collector   = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latent_collector.collect(x_t.feats.unsqueeze(0), step_idx=0)               # (1, N_total, C_noise)
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)
        log_prob_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)

        cond = torch.stack(
            [c.to(device=device, dtype=torch.float32) for c in image_cond], dim=0,
        )                                                                          # (B, seq, D)
        neg_cond = torch.stack(
            [c.to(device=device, dtype=torch.float32) for c in neg_image_cond], dim=0,
        )                                                                          # (B, seq, D)
        sigma_min = scheduler._sigma_min

        # ── Batched denoising loop ─────────────────────────────────────
        for step_idx in scheduler.get_timesteps_for_loop():
            t_val = scheduler.get_precise_t(step_idx)
            t_next_val = scheduler.get_precise_t(step_idx + 1)

            if self._mode != 'eval':
                noise_level_i = scheduler.get_noise_level_for_sigma(t_val)
                step_compute_lp = compute_log_prob and (noise_level_i > 0)
            else:
                noise_level_i = 0.0
                step_compute_lp = False

            pred_pos = self._forward_sparse(
                t_val, x_t, cond, concat_cond=concat_cond_sparse,
                stage='tex', stage_resolution=resolution,
            )

            apply_cfg = guidance_interval[0] <= t_val <= guidance_interval[1]
            if apply_cfg and guidance_scale != 1.0:
                pred_neg = self._forward_sparse(
                    t_val, x_t, neg_cond, concat_cond=concat_cond_sparse,
                    stage='tex', stage_resolution=resolution,
                )
                pred_v = self._apply_cfg_sparse(
                    pred_pos, pred_neg, x_t, t_val, guidance_scale, guidance_rescale, sigma_min,
                    batch_size=B,
                )
            else:
                pred_v = pred_pos

            output = scheduler.step(
                pred_v, t_val, t_next_val, x_t,
                noise_level=noise_level_i,
                compute_log_prob=step_compute_lp,
            )
            if output.log_prob is not None:
                output.log_prob = self._reduce_sparse_log_prob(
                    output.log_prob, x_t.coords, batch_size=B,
                )                                                                  # (B,)

            x_t = output.next_latents                                              # SparseTensor
            latent_collector.collect(x_t.feats.unsqueeze(0), step_idx + 1)         # (1, N_total, C_noise)
            callback_collector.collect_step(step_idx, output, extra_call_back_kwargs)
            if step_compute_lp and output.log_prob is not None:
                log_prob_collector.collect(output.log_prob.detach().unsqueeze(0), step_idx)  # (1, B)

        # ── Result splitting ────────────────────────────────────────────
        # Same splitting logic as _inference_shape.
        all_latents        = latent_collector.get_result()
        latent_index_map   = latent_collector.get_index_map()
        extra_callback_res = callback_collector.get_result()
        callback_index_map = callback_collector.get_index_map()
        all_log_probs      = log_prob_collector.get_result()
        log_prob_index_map = log_prob_collector.get_index_map()

        stacked_latents = (
            torch.stack([lat[0] for lat in all_latents], dim=0)
            if all_latents else None
        )                                                                          # (T', N_total, C_noise)
        stacked_log_probs = (
            torch.stack([lp[0] for lp in all_log_probs], dim=0)
            if all_log_probs else None
        )                                                                          # (T_log, B)

        feats_list, _ = x_t.to_tensor_list()                                       # List of (N_b, C_noise)

        for b, sample in enumerate(samples):
            slc = x_t.layout[b]
            sample.tex_final_latent          = feats_list[b]                       # (N_b, C_noise)
            sample.tex_all_latents           = (
                stacked_latents[:, slc, :] if stacked_latents is not None else None
            )                                                                      # (T', N_b, C_noise)
            sample.tex_log_probs             = (
                stacked_log_probs[:, b] if stacked_log_probs is not None else None
            )                                                                      # (T_log,)
            sample.tex_image_cond            = image_cond[b]
            sample.tex_neg_image_cond        = neg_image_cond[b]
            sample.tex_concat_cond           = concat_cond_list[b]
            sample.tex_latent_index_map      = latent_index_map
            sample.tex_log_prob_index_map    = log_prob_index_map
            sample.tex_timesteps             = timesteps
            if is_training_stage:
                sample.activate_stage('tex')
                sample.extra_kwargs = {
                    **{k: (v[b] if isinstance(v, (list, torch.Tensor)) else v)
                       for k, v in extra_callback_res.items()},
                    'callback_index_map': callback_index_map,
                }

    # ======================== Decoding Methods ========================
    
    def _get_denormalized_features(
        self,
        sample: Trellis2Sample,
        stage: str,
    ) -> Optional[torch.Tensor]:
        """
        Get denormalized final features (x_0) for a given stage.
        
        The trajectory stores normalized latents. This method retrieves the
        final step and applies denormalization (x_0 * std + mean).
        
        Args:
            sample: Trellis2Sample with per-stage trajectory data.
            stage: One of 'dense', 'shape', 'tex'.
        
        Returns:
            Denormalized features tensor, or None if stage data is absent.
        """
        if stage == 'dense':
            features = sample.dense_final_latent
        elif stage == 'shape':
            features = sample.shape_final_latent
        elif stage == 'tex':
            features = sample.tex_final_latent
        else:
            raise ValueError(f"Unknown stage: {stage!r}")

        if features is None:
            return None
        
        # Denormalize
        device = features.device
        mean, std = self.pipeline.get_normalization_tensors(stage, device)
        return features * std + mean

    @torch.no_grad()
    def decode_shape(
        self,
        sample: Trellis2Sample,
        return_subs: bool = False,
    ) -> Any:
        """
        Decode shape structured latent to mesh.
        
        Uses forward_chunked to run the VAE decoder with adaptive chunking,
        then applies FlexiDualGridVaeDecoder mesh conversion manually
        (sigmoid → intersected → quad_lerp → flexible_dual_grid_to_mesh).

        Args:
            sample: Trellis2Sample with shape stage data
            return_subs: Whether to return subdivisions (needed for texture decoding)
        
        Returns:
            Mesh object, or (mesh, subs) if return_subs=True
        """
        device = self.device
        
        features = self._get_denormalized_features(sample, 'shape')
        if features is None:
            raise ValueError("No shape features available for decoding")
        
        coords = sample.sparse_coords
        if coords is None:
            raise ValueError("No sparse_coords available for decoding")
        
        decoder = self.pipeline.shape_decoder
        decoder.eval()
        decoder.set_resolution(sample.resolution)

        slat = SparseTensor(
            feats=features.to(device=device, dtype=torch.float32),
            coords=coords.to(device),
        )

        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            h, subs = decoder.forward_chunked(slat, return_subs=True)

        # FlexiDualGridVaeDecoder post-processing: raw SparseTensor → Mesh
        voxel_margin = decoder.voxel_margin
        vertices = h.replace(
            (1 + 2 * voxel_margin) * torch.sigmoid(h.feats[..., 0:3]) - voxel_margin
        )
        intersected = h.replace(h.feats[..., 3:6] > 0)
        quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
        meshes = [
            Mesh(*flexible_dual_grid_to_mesh(
                v.coords[:, 1:], v.feats, i.feats, q.feats,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=decoder.resolution,
                train=False,
            ))
            for v, i, q in zip(vertices, intersected, quad_lerp)
        ]
        mesh = meshes[0] if meshes else None
        
        if return_subs:
            return mesh, subs
        return mesh
    
    @torch.no_grad()
    def decode_texture(
        self,
        sample: Trellis2Sample,
        subs: List[Any],
    ) -> Any:
        """
        Decode texture structured latent to texture voxels.
        
        Args:
            sample: Trellis2Sample with tex stage data
            subs: Subdivisions from shape decoder (required for texture decoding)
        
        Returns:
            SparseTensor of texture voxels
        """
        features = self._get_denormalized_features(sample, 'tex')
        if features is None:
            return None
        
        coords = sample.sparse_coords
        if coords is None:
            return None

        device = self.device
        
        decoder = self.pipeline.tex_decoder
        if decoder is None:
            return None

        slat = SparseTensor(
            feats=features.to(device=device, dtype=torch.float32),
            coords=coords.to(device),
        )

        with torch.autocast('cuda', enabled=False):
            tex_voxels = decoder.forward_chunked(slat, guide_subs=subs) * 0.5 + 0.5
        
        return tex_voxels
    
    @torch.no_grad()
    def decode_latents(
        self,
        latents: Union[Trellis2Sample, torch.Tensor],
        **kwargs,
    ) -> Any:
        """
        Decode latents to textured 3D mesh.
        
        This method decodes both shape and texture structured latents
        and combines them into a MeshWithVoxel object.
        
        Args:
            latents: Trellis2Sample containing shape_slat and tex_slat data,
                     or a tensor (for interface compatibility, will raise error)
            **kwargs: Additional arguments (resolution override, etc.)
        
        Returns:
            MeshWithVoxel object with geometry and texture
        """
        if not isinstance(latents, Trellis2Sample):
            raise TypeError(
                f"Trellis2 decode_latents expects Trellis2Sample, got {type(latents)}. "
                "Use decode_shape() for shape-only decoding."
            )
        
        sample = latents
        resolution = kwargs.get('resolution', sample.resolution) or 1024

        mesh, subs = self.decode_shape(sample, return_subs=True)
        if mesh is None:
            return None

        tex_voxels = self.decode_texture(sample, subs)

        # subs hold large spatial caches (layout tensors) from decode_shape;
        # they are no longer needed after texture decoding.
        for sub in subs:
            sub.clear_spatial_cache()
        del subs
        torch.cuda.empty_cache()

        if tex_voxels is None:
            return mesh

        textured_mesh = MeshWithVoxel(
            mesh.vertices,
            mesh.faces,
            origin=[-0.5, -0.5, -0.5],
            voxel_size=1 / resolution,
            coords=tex_voxels.coords[:, 1:],
            attrs=tex_voxels.feats,
            voxel_shape=torch.Size([*tex_voxels.shape, *tex_voxels.spatial_shape]),
            layout=self.pipeline.pbr_attr_layout if hasattr(self.pipeline, 'pbr_attr_layout') else None,
        )
        return textured_mesh

    def _build_envmap(self, envmap_path: Optional[str] = None) -> Any:
        """Load an HDR environment map and return an EnvMap object.

        Args:
            envmap_path: Path to an .exr HDR file.  When ``None``, falls back
                to the default ``forest.exr`` shipped with Trellis2.

        Returns:
            An ``EnvMap`` instance ready for ``PbrMeshRenderer.render()``.
        """
        import cv2

        if envmap_path is None:
            envmap_path = os.path.join(_trellis_path, 'assets', 'hdri', 'forest.exr')
        assert os.path.exists(envmap_path), f"EnvMap HDR file not found: {envmap_path}"

        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        hdr_bgr = cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED)          # (H, W, 3) float32 BGR
        assert hdr_bgr is not None, f"Failed to load EnvMap HDR file: {envmap_path}"
        hdr_rgb = cv2.cvtColor(hdr_bgr, cv2.COLOR_BGR2RGB)               # (H, W, 3) float32 RGB
        hdr_tensor = torch.tensor(hdr_rgb, dtype=torch.float32, device=self.device)  # (H, W, 3) float32
        return EnvMap(hdr_tensor)

    @torch.no_grad()
    @torch.autocast('cuda', enabled=False)
    def render_latents(
        self,
        sample: Trellis2Sample,
        num_frames: int = 24,
        resolution: int = 512,
        bg_color: Tuple[float, float, float] = (0, 0, 0),
        envmap: Optional[Any] = None,
        envmap_path: Optional[str] = None,
        render_mode: Literal['shaded', 'clay', 'normal'] = 'shaded',
        **render_kwargs,
    ) -> Trellis2Sample:
        """Decode latents to mesh and render deterministic multiview frames.

        Uses uniform-yaw + fixed-pitch + adaptive-distance camera placement
        for reproducible evaluation and reward computation.

        Args:
            sample: A ``Trellis2Sample`` with shape + tex latents.
            num_frames: Number of views to render.
            resolution: Render resolution in pixels.
            bg_color: Background color as ``(R, G, B)`` floats in [0, 1].
            envmap: Pre-built ``EnvMap`` object (avoids repeated disk I/O
                when called in a loop).
            envmap_path: Path to ``.exr`` HDR file; used only when *envmap*
                is ``None``.
            render_mode: Foreground source for ``sample.video``.

                - ``'shaded'`` (default): full PBR-shaded RGB. Reward sees
                  geometry **and** texture/material — current behaviour.
                - ``'clay'``: SSAO occlusion broadcast to RGB. Geometry-only,
                  zero dependency on tex-stage outputs (base_color / metallic
                  / roughness). Useful for isolating shape attribution.
                - ``'normal'``: view-space surface normals as RGB. Renderer
                  already maps normals into ``[0, 1]`` and fills the background
                  with white. Same geometry-only property as ``'clay'``;
                  richer detail on smooth / convex shapes where SSAO is too
                  flat.
            **render_kwargs: Forwarded to ``render_frames``.

        Returns:
            The same *sample* with ``sample.video`` set to a
            ``(T, C, H, W)`` float32 tensor in [0, 1].
        """
        mesh = self.decode_latents(sample)
        if mesh is None:
            return sample

        torch.cuda.empty_cache()

        if envmap is None:
            envmap = self._build_envmap(envmap_path)

        _FOV_DEG    = 40.0
        _PITCH_DEG  = 20.0
        _FILL_RATIO = 0.9
        _START_YAW  = np.pi  # front-facing view first (180° from default)
        r = _compute_adaptive_distance(_FOV_DEG, fill_ratio=_FILL_RATIO)
        yaws_rad   = torch.linspace(
            _START_YAW, _START_YAW + 2 * np.pi, num_frames + 1,
        )[:-1].tolist()
        pitchs_rad = [np.deg2rad(_PITCH_DEG)] * num_frames
        extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws_rad, pitchs_rad, r, _FOV_DEG,
        )
        # nvdiffrast 在 faces > ~16M (2^24) 时会触发 subtriangle count overflow，
        # 改用 chunked 渲染器按 4M faces 分块跑 + 跨 chunk 深度合成（无上限），
        # 不再依赖 mesh.simplify 做面数限流。
        ret = render_frames_chunked(
            mesh, extrinsics, intrinsics,
            {'resolution': resolution},
            envmap=envmap,
            verbose=render_kwargs.pop('verbose', False),
            **render_kwargs,
        )
        alpha = ret['alpha']                                         # (T, 1, H, W) cuda float [0, 1]

        if render_mode == 'shaded':
            fg = ret['shaded']                                       # (T, 3, H, W) cuda float [0, 1]
        elif render_mode == 'clay':
            clay = ret['clay']                                       # (T, 1, H, W) cuda float [0, 1] SSAO occlusion
            fg = clay.expand(-1, 3, -1, -1).contiguous()             # (T, 3, H, W) cuda float [0, 1] gray
        elif render_mode == 'normal':
            # renderer already maps view-space normals into [0, 1] RGB and
            # fills background with 1.0 (see pbr_mesh_renderer_chunked.py
            # `out_normal = -gb_cam_normal * 0.5 + 0.5`). No extra rescaling.
            fg = ret['normal'].clamp(0, 1)                           # (T, 3, H, W) cuda float [0, 1] RGB
        else:
            raise ValueError(
                f"render_latents: unsupported render_mode={render_mode!r}; "
                "expected one of {'shaded', 'clay', 'normal'}"
            )

        bg = torch.tensor(
            bg_color, dtype=fg.dtype, device=fg.device,
        ).reshape(1, 3, 1, 1)                                        # (1, 3, 1, 1)
        frames = (fg + bg * (1 - alpha)).clamp(0, 1).cpu()           # (T, 3, H, W) float32

        sample.video = frames
        return sample
