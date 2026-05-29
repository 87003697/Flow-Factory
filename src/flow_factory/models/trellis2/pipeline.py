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

# src/flow_factory/models/trellis2/pipeline.py
"""
Pseudo-Pipeline for Trellis2 Image-to-3D models.

This module wraps the Trellis2 models into a flat component structure
compatible with Flow-Factory's BaseAdapter interface.

Trellis2 Architecture:
    - sparse_structure_flow_model: Generates 3D occupancy structure
    - shape_slat_flow_model_512/1024: Generates shape structured latent
    - tex_slat_flow_model_512/1024: Generates texture structured latent
    - *_decoder: Decodes latents to meshes/textures
    - image_cond_model: DINOv2/v3 image encoder for conditioning

This pseudo-pipeline exposes components as flat attributes for BaseAdapter:
    - self.transformer: The trainable flow model (shape_slat_flow_model by default)
    - self.image_encoder: The image conditioning model (DINOv2/v3)
    - self.scheduler: None (to be set by BaseAdapter.load_scheduler)
"""
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

import torch
import torch.nn as nn
from PIL import Image
import numpy as np


class Trellis2PseudoPipeline:
    """
    Pseudo-Pipeline for Trellis2 models.
    
    This class serves as a component container that mimics the diffusers Pipeline
    interface just enough for BaseAdapter to work. It does NOT inherit from
    DiffusionPipeline but exposes components as direct attributes.
    
    Key Design Decisions:
        1. `self.transformer` points to the primary trainable model 
           (shape_slat_flow_model_1024 by default for shape training)
        2. All Trellis2 models are stored in `self._models` dict for utility access
        3. Scheduler is initially None, set by BaseAdapter.load_scheduler()
    
    Args:
        models: Dict of Trellis2 nn.Module models
        image_cond_model: Image conditioning model (DINOv2/v3)
        sparse_structure_sampler: Sampler for sparse structure
        shape_slat_sampler: Sampler for shape SLat
        tex_slat_sampler: Sampler for texture SLat
        sampler_params: Dict of sampler parameters
        normalization_params: Dict of normalization parameters
        rembg_model: Background removal model (optional)
        target_flow_model: Which flow model to use as transformer
            Options: 'shape_slat_1024', 'shape_slat_512', 'tex_slat_1024', 'tex_slat_512'
    """
    
    REQUIRED_MODELS = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]
    
    # 512-resolution variants are only needed for low-res training
    OPTIONAL_MODELS = [
        'shape_slat_flow_model_512',
        'tex_slat_flow_model_512',
    ]
    
    # Mapping from target name to model key
    TARGET_MODEL_MAP = {
        'dense':           'sparse_structure_flow_model',
        'shape_slat_1024': 'shape_slat_flow_model_1024',
        'shape_slat_512':  'shape_slat_flow_model_512',
        'tex_slat_1024':   'tex_slat_flow_model_1024',
        'tex_slat_512':    'tex_slat_flow_model_512',
    }
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        image_cond_model: nn.Module,
        sparse_structure_sampler: Any = None,
        shape_slat_sampler: Any = None,
        tex_slat_sampler: Any = None,
        sparse_structure_sampler_params: Dict = None,
        shape_slat_sampler_params: Dict = None,
        tex_slat_sampler_params: Dict = None,
        shape_slat_normalization: Dict = None,
        tex_slat_normalization: Dict = None,
        rembg_model: Optional[Any] = None,
        target_flow_model: str = 'shape_slat_1024',
    ):
        # Store all models in internal dict
        self._models = models
        
        # ============= Flat Component Attributes (for BaseAdapter) =============
        # Primary trainable model - this is what BaseAdapter will prepare/LoRA
        target_model_key = self.TARGET_MODEL_MAP.get(target_flow_model, 'shape_slat_flow_model_1024')
        self.transformer = models.get(target_model_key)
        if self.transformer is None:
            raise ValueError(f"Target flow model '{target_flow_model}' not found in models. "
                           f"Available: {list(models.keys())}")
        
        # Image encoder for conditioning
        self.image_encoder = image_cond_model
        
        # VAE - None for Trellis2 (no VAE-based encoding/decoding)
        self.vae = None
        
        # Decoders
        self.shape_decoder = models.get('shape_slat_decoder')
        self.tex_decoder = models.get('tex_slat_decoder')
        self.sparse_structure_decoder = models.get('sparse_structure_decoder')
        
        # Sparse structure flow model (typically frozen)
        self.sparse_structure_flow_model = models.get('sparse_structure_flow_model')
        
        # ============= Stage-specific flow models =============
        # Named with 'transformer_' prefix so BaseAdapter.transformer_names
        # discovers them for LoRA / FSDP preparation.
        self.transformer_dense = models.get('sparse_structure_flow_model')
        self.transformer_shape_512 = models.get('shape_slat_flow_model_512')
        self.transformer_shape_1024 = models.get('shape_slat_flow_model_1024')
        self.transformer_tex_512 = models.get('tex_slat_flow_model_512')
        self.transformer_tex_1024 = models.get('tex_slat_flow_model_1024')
        
        # Scheduler - None, to be set by BaseAdapter.load_scheduler()
        self.scheduler = None
        
        # ============= Samplers (Trellis2's own sampling logic) =============
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        
        # Sampler parameters
        self.sparse_structure_sampler_params = sparse_structure_sampler_params or {}
        self.shape_slat_sampler_params = shape_slat_sampler_params or {}
        self.tex_slat_sampler_params = tex_slat_sampler_params or {}
        
        # Normalization parameters
        self.shape_slat_normalization = shape_slat_normalization or {'mean': [0.0], 'std': [1.0]}
        self.tex_slat_normalization = tex_slat_normalization or {'mean': [0.0], 'std': [1.0]}
        
        # PBR attribute layout for texture decoding
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        
        # Background removal model
        self.rembg_model = rembg_model
        
        # Track which model is the training target
        self._target_flow_model = target_flow_model
        self._device = 'cpu'
    
    @staticmethod
    def _instantiate(module: Any, cfg: dict, path_override: Optional[str] = None) -> Any:
        """Instantiate a component from a ``{name, args}`` config dict.

        Args:
            module: The module to look up the class from (e.g. ``samplers``).
            cfg: Config dict with ``'name'`` and optional ``'args'`` keys.
            path_override: If given, sets ``cfg['args']['model_name']`` to this value.
        """
        name = cfg['name']
        args = cfg.get('args', {}).copy()
        if path_override is not None:
            args['model_name'] = path_override
        return getattr(module, name)(**args)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config_file: str = "pipeline.json",
        target_flow_model: str = 'shape_slat_1024',
        image_cond_model_path: Optional[str] = None,
        rembg_model_path: Optional[str] = None,
        **kwargs,
    ) -> "Trellis2PseudoPipeline":
        """Load a pretrained Trellis2 model from a local directory or HF repo.

        Args:
            path: Local directory or HuggingFace repo id.
            config_file: Name of the pipeline config JSON inside *path*.
            target_flow_model: Which flow model to designate as the trainable
                transformer (e.g. ``'shape_slat_1024'``).
            image_cond_model_path: Override the DINOv3 model path from config.
            rembg_model_path: Override the rembg model path from config.
        """
        import sys
        trellis_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'third_party', 'TRELLIS.2')
        if trellis_path not in sys.path:
            sys.path.insert(0, os.path.abspath(trellis_path))

        from trellis2 import models as trellis_models
        from trellis2.pipelines import samplers, rembg
        from trellis2.modules import image_feature_extractor

        config_path = os.path.join(path, config_file)
        if not os.path.exists(config_path):
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(path, config_file)

        with open(config_path, 'r') as f:
            args = json.load(f)['args']

        allowed = set(cls.REQUIRED_MODELS + cls.OPTIONAL_MODELS)
        _models = {}
        for k, v in args['models'].items():
            if k not in allowed:
                continue
            local_path = os.path.join(path, v)
            if os.path.exists(f"{local_path}.json"):
                _models[k] = trellis_models.from_pretrained(local_path)
            else:
                _models[k] = trellis_models.from_pretrained(v)

        sparse_structure_sampler = cls._instantiate(samplers, args['sparse_structure_sampler'])
        shape_slat_sampler       = cls._instantiate(samplers, args['shape_slat_sampler'])
        tex_slat_sampler         = cls._instantiate(samplers, args['tex_slat_sampler'])

        image_cond_model = cls._instantiate(
            image_feature_extractor, args['image_cond_model'],
            path_override=image_cond_model_path,
        )

        rembg_model = None
        if args.get('rembg_model'):
            rembg_cfg = args['rembg_model']
            if isinstance(rembg_cfg, str):
                rembg_cfg = {'name': rembg_cfg, 'args': {}}
            try:
                rembg_model = cls._instantiate(rembg, rembg_cfg, path_override=rembg_model_path)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to load rembg model (not needed for RGBA training): {e}"
                )
                rembg_model = None

        return cls(
            models=_models,
            image_cond_model=image_cond_model,
            sparse_structure_sampler=sparse_structure_sampler,
            shape_slat_sampler=shape_slat_sampler,
            tex_slat_sampler=tex_slat_sampler,
            sparse_structure_sampler_params=args['sparse_structure_sampler']['params'],
            shape_slat_sampler_params=args['shape_slat_sampler'].get('params', {}),
            tex_slat_sampler_params=args['tex_slat_sampler'].get('params', {}),
            shape_slat_normalization=args.get('shape_slat_normalization'),
            tex_slat_normalization=args.get('tex_slat_normalization'),
            rembg_model=rembg_model,
            target_flow_model=target_flow_model,
        )
    
    @property
    def device(self) -> torch.device:
        """Get device of the models."""
        if hasattr(self, '_device') and self._device != 'cpu':
            return torch.device(self._device)
        if self.transformer is not None:
            return next(self.transformer.parameters()).device
        return torch.device('cpu')
    
    def to(self, device: Union[str, torch.device]) -> "Trellis2PseudoPipeline":
        """Move all models to specified device."""
        self._device = device
        for model in self._models.values():
            if model is not None:
                model.to(device)
        if self.image_encoder is not None:
            self.image_encoder.to(device)
        if self.rembg_model is not None:
            self.rembg_model.to(device)
        return self
    
    def cuda(self) -> "Trellis2PseudoPipeline":
        return self.to('cuda')
    
    def cpu(self) -> "Trellis2PseudoPipeline":
        return self.to('cpu')
    
    # ======================== Utility Methods ========================
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess input image: remove background and crop to square.
        
        Copied from Trellis2ImageTo3DPipeline.preprocess_image()
        """
        # Check if image has alpha channel
        has_alpha = False
        if image.mode == 'RGBA':
            alpha = np.array(image)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        
        # Resize if too large
        max_size = max(image.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.Resampling.LANCZOS
            )
        
        # Remove background if needed
        if has_alpha:
            output = image
        else:
            image = image.convert('RGB')
            if self.rembg_model is not None:
                output = self.rembg_model(image)
            else:
                # No rembg, just add alpha channel
                output = image.convert('RGBA')
        
        # Crop to square around object
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        if len(bbox) == 0:
            return output
        
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = (
            center[0] - size // 2,
            center[1] - size // 2,
            center[0] + size // 2,
            center[1] + size // 2
        )
        output = output.crop(bbox)
        return output  # RGBA — downstream (Trellis2Sample) handles bg-color compositing
    
    def get_cond(
        self,
        images: List[Image.Image],
        resolution: int,
        include_neg_cond: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Get conditioning embeddings from images.
        
        Args:
            images: List of PIL images (RGB or RGBA; RGBA is composited onto black)
            resolution: Resolution for image encoder
            include_neg_cond: Whether to include negative conditioning
        
        Returns:
            Dict with 'cond' and optionally 'neg_cond' tensors
        """
        rgb_images = [
            Image.alpha_composite(
                Image.new('RGBA', img.size, (0, 0, 0, 255)), img,
            ).convert('RGB') if img.mode == 'RGBA' else img
            for img in images
        ]

        if hasattr(self.image_encoder, 'image_size'):
            self.image_encoder.image_size = resolution
        
        cond = self.image_encoder(rgb_images)
        
        if not include_neg_cond:
            return {'cond': cond}
        
        neg_cond = torch.zeros_like(cond)
        return {'cond': cond, 'neg_cond': neg_cond}
    
    def get_normalization_tensors(
        self,
        latent_type: str,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get mean and std tensors for latent normalization.
        
        Args:
            latent_type: 'shape' or 'tex'
            device: Target device
        
        Returns:
            (mean, std) tensors
        """
        if latent_type == 'shape':
            norm = self.shape_slat_normalization
        else:
            norm = self.tex_slat_normalization
        
        mean = torch.tensor(norm['mean'])[None].to(device)
        std = torch.tensor(norm['std'])[None].to(device)
        return mean, std

    # ======================== Stage → Model Resolution ========================

    # Mapping from (stage, resolution) to pipeline attribute name
    _STAGE_MODEL_MAP: Dict[Tuple, str] = {
        ('shape', 1024): 'transformer_shape_1024',
        ('shape', 512):  'transformer_shape_512',
        ('tex', 1024):   'transformer_tex_1024',
        ('tex', 512):    'transformer_tex_512',
        ('dense', None):  'transformer_dense',
    }

    def get_flow_model(self, stage: Optional[str] = None, resolution: int = 1024) -> nn.Module:
        """
        Resolve a stage name to the corresponding flow model.

        Args:
            stage: One of 'shape', 'tex', 'dense', or None.
                   None → ``self.transformer`` (backward compatible with single-stage).
            resolution: Model resolution variant (512 or 1024). Only matters for
                        shape/tex stages. Can be a tuple (H, W).

        Returns:
            The nn.Module for the requested stage.

        Raises:
            ValueError: If the requested model is not available.
        """
        if isinstance(resolution, (tuple, list)):
            resolution = resolution[0]
        if stage is None:
            return self.transformer

        key = (stage, resolution if stage != 'dense' else None)
        attr_name = self._STAGE_MODEL_MAP.get(key)
        if attr_name is None:
            raise ValueError(f"Unknown stage/resolution: stage={stage!r}, resolution={resolution}")

        model = getattr(self, attr_name, None)
        if model is None:
            raise ValueError(
                f"Flow model for stage={stage!r}, resolution={resolution} not loaded. "
                f"Expected pipeline attribute '{attr_name}'."
            )
        return model

    # ======================== Sparse Structure Sampling ========================

    @torch.no_grad()
    def sample_sparse_structure(
        self,
        cond: Dict[str, torch.Tensor],
        resolution: int = 32,
        num_samples: int = 1,
        sampler_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Sample sparse 3D occupancy structure.

        This uses the sparse_structure_flow_model to generate initial
        occupancy coordinates that define where the 3D object exists.

        Args:
            cond: Conditioning dict with 'cond' and optionally 'neg_cond'
            resolution: Output resolution (32 or 64)
            num_samples: Number of samples to generate
            sampler_params: Override sampler parameters

        Returns:
            coords: Tensor of shape (N, 4) - [batch_idx, x, y, z]
        """
        device = self.device
        sampler_params = {**self.sparse_structure_sampler_params, **(sampler_params or {})}

        flow_model = self.sparse_structure_flow_model
        reso = flow_model.resolution
        in_channels = flow_model.in_channels

        # Initial noise
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(device)

        # Sample
        flow_model.to(device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples

        # Decode to binary occupancy
        decoder = self.sparse_structure_decoder
        decoder.to(device)
        decoded = decoder(z_s) > 0

        # Adjust resolution if needed
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5

        # Extract coordinates
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        return coords

    # ======================== CFG Rescale ========================

    @staticmethod
    def apply_cfg_rescale(
        x: torch.Tensor,
        t: float,
        pred_cond: torch.Tensor,
        pred_cfg: torch.Tensor,
        guidance_rescale: float,
        sigma_min: float,
    ) -> torch.Tensor:
        """Rescale CFG velocity to match per-token std of the conditional prediction.

        Implements the std-rescaling from Common Diffusion Noise Schedules
        (Lin et al. 2024, sec. 3.4), adapted for flow-matching (sigma_min=0).
        """
        alpha = 1.0 - sigma_min
        beta  = sigma_min + alpha * t
        x_0_cond = alpha * x - beta * pred_cond
        x_0_cfg  = alpha * x - beta * pred_cfg
        scale = x_0_cond.std(dim=-1, keepdim=True) / (x_0_cfg.std(dim=-1, keepdim=True) + 1e-8)
        x_0 = guidance_rescale * (x_0_cfg * scale) + (1 - guidance_rescale) * x_0_cfg
        return (alpha * x - x_0) / beta
