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

# src/flow_factory/inference.py
"""Inference entry point for Flow-Factory.

Supports batch inference for all registered model adapters
(T2I / I2I / T2V / I2V / V2V).

Parameter routing works via ``extra_kwargs`` + ``filter_kwargs``: any
model-specific key in the YAML config (e.g. ``negative_prompt``,
``num_frames``, ``condition_image_size``, ``strength``) is collected into
``InferenceArguments.extra_kwargs`` and forwarded to the adapter's
``inference()`` method after filtering against its actual signature.
This means **this file never needs to change when a new adapter is added**.

Input records (from ``prompts_file`` or ``--prompts``) may contain arbitrary
per-sample keys (``images``, ``videos``, ``negative_prompt``, …) that are
merged into the per-call kwargs after the shared base kwargs.
"""
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import imageio
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator

from .hparams import ModelArguments
from .hparams.args import Arguments
from .models.loader import load_model
from .samples import BaseSample
from .utils.base import filter_kwargs


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger("flow_factory.inference")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class InferenceArguments:
    """Inference configuration.

    Known fields are validated explicitly.  Any extra YAML keys (e.g.
    ``negative_prompt``, ``num_frames``, ``condition_image_size``,
    ``strength``) are collected in ``extra_kwargs`` and forwarded to the
    adapter unchanged, so this class never needs to be modified for
    model-specific parameters.
    """

    # Model
    model_name_or_path: str = None
    model_type: str = None
    finetune_type: str = "lora"
    checkpoint_path: str = None

    # Data
    prompts_file: Optional[str] = None
    prompts: Optional[List[str]] = None

    # Inference settings
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    height: int = 1024
    width: int = 1024
    num_samples_per_prompt: int = 1
    seed: int = 42

    # Output
    output_dir: str = "outputs"
    save_format: str = "png"
    video_fps: int = 16
    save_metadata: bool = True

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Model-specific pass-through parameters populated from unknown YAML keys
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "InferenceArguments":
        """Load configuration from YAML file.

        Standard sections (``model``, ``data``, ``inference``, ``output``,
        ``device``) are unpacked and matched to known fields.  Any remaining
        keys are stored in ``extra_kwargs`` and logged so the user knows they
        are being forwarded to the adapter.
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Flatten nested config sections (only expand dict-valued sections)
        flat_config: Dict[str, Any] = {}
        for section in ('model', 'data', 'inference', 'output'):
            if section in config and isinstance(config[section], dict):
                flat_config.update(config[section])

        # Also absorb top-level keys that are not section headers
        _section_keys = {'model', 'data', 'inference', 'output', 'device'}
        for k, v in config.items():
            if k not in _section_keys:
                flat_config.setdefault(k, v)

        known_fields = {
            f_obj.name for f_obj in cls.__dataclass_fields__.values()
            if f_obj.name != 'extra_kwargs'
        }
        known  = {k: v for k, v in flat_config.items() if k in known_fields}
        extras = {k: v for k, v in flat_config.items() if k not in known_fields}

        if extras:
            logger.info(f"Model-specific kwargs will be forwarded to adapter: {sorted(extras)}")

        return cls(**known, extra_kwargs=extras)


# =============================================================================
# I/O Helpers  (module-level so they are easily testable in isolation)
# =============================================================================

def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_video(path: str) -> List[Image.Image]:
    """Load all frames of a video file as a list of RGB PIL Images."""
    return [Image.fromarray(frame) for frame in imageio.v3.imread(path, plugin="pyav")]


def _load_records(config: InferenceArguments) -> List[Dict[str, Any]]:
    """Return a list of inference records.

    Each record is a plain dict that may contain any subset of::

        {
            "prompt":          str,
            "negative_prompt": str,               # optional
            "images":          List[PIL.Image],   # optional – for I2I / I2V
            "videos":          List[List[PIL.Image]], # optional – for V2V
            <any other per-sample keys>,
        }

    Sources (in priority order):

    1. ``config.prompts`` (CLI inline list) → list of ``{prompt: …}`` records.
    2. ``config.prompts_file``:

       - ``.txt``   → one prompt per line
       - ``.json``  → list of strings *or* list/dict of full records
       - ``.jsonl`` → one record per line (string or dict)
    """
    records: List[Dict[str, Any]] = []

    if config.prompts:
        records = [{"prompt": p} for p in config.prompts]
    elif config.prompts_file:
        pf = Path(config.prompts_file)
        if pf.suffix == '.jsonl':
            with open(pf) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        records.append(item if isinstance(item, dict) else {"prompt": item})
        elif pf.suffix == '.json':
            with open(pf) as f:
                data = json.load(f)
            if isinstance(data, list):
                records = [d if isinstance(d, dict) else {"prompt": d} for d in data]
            elif isinstance(data, dict):
                records = [{"prompt": p} for p in data.get("prompts", [])]
        else:  # plain text
            with open(pf) as f:
                records = [{"prompt": ln.strip()} for ln in f if ln.strip()]

    if not records:
        raise ValueError("No prompts provided. Use --prompts or set prompts_file in config.")

    # Resolve relative image / video paths relative to the prompts file
    base_dir = str(Path(config.prompts_file).parent) if config.prompts_file else "."
    for rec in records:
        if "images" in rec:
            resolved = [p if os.path.isabs(p) else os.path.join(base_dir, p)
                        for p in rec["images"]]
            rec["images"] = [_load_image(p) for p in resolved]
        if "videos" in rec:
            resolved = [p if os.path.isabs(p) else os.path.join(base_dir, p)
                        for p in rec["videos"]]
            rec["videos"] = [_load_video(p) for p in resolved]

    return records


# ------------------------------------------------------------------
# Sample saving
# ------------------------------------------------------------------

def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert a float tensor to a uint8 numpy array.

    Handles both [0, 1] and [-1, 1] value ranges automatically.
    """
    arr = t.float().cpu().numpy()
    if arr.min() < -0.01:      # [-1, 1] range – normalise first
        arr = (arr + 1.0) / 2.0
    return (arr.clip(0.0, 1.0) * 255).astype('uint8')


def _save_image_sample(sample: BaseSample, output_dir: Path, stem: str, fmt: str) -> str:
    """Save ``sample.image`` (Tensor C×H×W) as a PIL image file."""
    arr = _tensor_to_uint8(sample.image)   # (C, H, W) uint8
    arr = arr.transpose(1, 2, 0)           # → (H, W, C)
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    path = output_dir / f"{stem}.{fmt}"
    Image.fromarray(arr).save(path)
    return str(path)


def _save_video_sample(sample: BaseSample, output_dir: Path, stem: str, fps: int) -> str:
    """Save ``sample.video`` (Tensor T×C×H×W) as an mp4 file."""
    arr = _tensor_to_uint8(sample.video)   # (T, C, H, W) uint8
    arr = arr.transpose(0, 2, 3, 1)        # → (T, H, W, C)
    frames = [arr[i] for i in range(arr.shape[0])]
    path = str(output_dir / f"{stem}.mp4")
    imageio.mimwrite(path, frames, fps=fps, format='FFMPEG',
                     codec='libx264', pixelformat='yuv420p')
    return path


def _save_sample(
    sample: BaseSample,
    output_dir: Path,
    stem: str,
    save_format: str,
    video_fps: int,
    save_metadata: bool,
    meta: Dict[str, Any],
) -> Optional[str]:
    """Dispatch to image or video save based on which field is populated."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if sample.image is not None:
        path = _save_image_sample(sample, output_dir, stem, save_format)
    elif sample.video is not None:
        path = _save_video_sample(sample, output_dir, stem, video_fps)
    else:
        logger.warning(f"Sample '{stem}' has neither .image nor .video — skipping.")
        return None

    if save_metadata:
        with open(output_dir / f"{stem}.json", 'w') as f:
            json.dump(meta, f, indent=2)

    return path


# =============================================================================
# Inference Pipeline
# =============================================================================

class InferencePipeline:
    """Adapter-agnostic batch inference pipeline.

    Parameter routing::

        base_kwargs   (config fixed params)
          + extra_kwargs  (config model-specific params)
          + per-record overrides  (from prompts_file record fields)
          + runtime values        (generator, compute_log_prob)
        ──────────────────────────────────────────────────
                filter_kwargs(adapter.inference, …)
                            │
                adapter.inference(**filtered)
    """

    def __init__(self, config: InferenceArguments):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        self.accelerator = Accelerator()
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _read_adapter_config(checkpoint_path: str) -> Dict[str, Any]:
        """Read LoRA hyper-parameters from a checkpoint's adapter_config.json."""
        cfg_path = os.path.join(checkpoint_path, "adapter_config.json")
        if not os.path.exists(cfg_path):
            return {}
        with open(cfg_path) as f:
            return json.load(f)

    def _build_arguments(self) -> Arguments:
        """Build a minimal ``Arguments`` wrapper accepted by ``load_model``."""
        target_modules = "default"
        lora_rank = 8

        if self.config.finetune_type == "lora" and self.config.checkpoint_path:
            ac = self._read_adapter_config(self.config.checkpoint_path)
            target_modules = ac.get("target_modules", target_modules)
            lora_rank = ac.get("r", lora_rank)

        model_args = ModelArguments(
            model_name_or_path=self.config.model_name_or_path,
            model_type=self.config.model_type,
            finetune_type=self.config.finetune_type,
            target_modules=target_modules,
            lora_rank=lora_rank,
        )

        _dtype_to_mp = {"bfloat16": "bf16", "float16": "fp16", "float32": "no"}
        mp_key = _dtype_to_mp.get(self.config.dtype, "no")

        args = Arguments()
        args.model_args = model_args
        args.training_args.seed = self.config.seed
        args.training_args.latent_storage_dtype = mp_key
        args.mixed_precision = mp_key
        return args

    def _load_model(self):
        """Load model adapter, restore checkpoint, and move to device."""
        if not self.config.model_name_or_path or not self.config.model_type:
            raise ValueError("`model_name_or_path` and `model_type` are required in config.")

        logger.info(f"Loading model: {self.config.model_type}")
        args = self._build_arguments()
        self.adapter = load_model(args, self.accelerator)

        if self.config.checkpoint_path:
            logger.info(f"Loading checkpoint: {self.config.checkpoint_path}")
            self.adapter.load_checkpoint(self.config.checkpoint_path)

        if hasattr(self.adapter, "pipeline"):
            self.adapter.pipeline.to(self.device, dtype=self.dtype)

        self.adapter.eval()
        logger.info("Model loaded successfully")

    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(self) -> List[str]:
        """Run batch inference and return saved output paths."""
        records = _load_records(self.config)
        logger.info(
            f"Running inference on {len(records)} records × "
            f"{self.config.num_samples_per_prompt} sample(s) each"
        )

        output_dir = Path(self.config.output_dir)
        output_paths: List[str] = []

        # Kwargs shared across all records; model-specific params live in extra_kwargs
        base_kwargs: Dict[str, Any] = {
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "height": self.config.height,
            "width": self.config.width,
            "compute_log_prob": False,
            **self.config.extra_kwargs,
        }

        for idx, record in enumerate(tqdm(records, desc="Generating")):
            prompt = record.get("prompt", "")

            for s_idx in range(self.config.num_samples_per_prompt):
                seed = self.config.seed + idx * self.config.num_samples_per_prompt + s_idx
                generator = torch.Generator(device=self.device).manual_seed(seed)

                # Merge layers: base config  <  per-record overrides  <  runtime
                inference_kwargs: Dict[str, Any] = {
                    **base_kwargs,
                    **record,
                    "generator": generator,
                }

                samples = self.adapter.inference(
                    **filter_kwargs(self.adapter.inference, **inference_kwargs)
                )

                for sample_idx, sample in enumerate(samples):
                    stem = f"{idx:05d}_{s_idx:02d}_{sample_idx:02d}"
                    meta: Dict[str, Any] = {
                        "prompt": prompt,
                        "index": idx,
                        "sample_idx": s_idx,
                        "seed": seed,
                        **{k: v for k, v in base_kwargs.items()
                           if isinstance(v, (int, float, str, bool))},
                    }
                    path = _save_sample(
                        sample, output_dir, stem,
                        self.config.save_format,
                        self.config.video_fps,
                        self.config.save_metadata,
                        meta,
                    )
                    if path:
                        output_paths.append(path)

        logger.info(f"Saved {len(output_paths)} outputs to {self.config.output_dir}")
        return output_paths


# =============================================================================
# Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Flow-Factory Inference")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--prompts", nargs="+", help="Override prompts")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--checkpoint_path", type=str, help="Override checkpoint path")
    return parser.parse_args()


def main():
    args = parse_args()
    config = InferenceArguments.load_from_yaml(args.config)

    if args.prompts:
        config.prompts = args.prompts
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.checkpoint_path:
        config.checkpoint_path = args.checkpoint_path

    logger.info("=" * 60)
    logger.info("Flow-Factory Inference")
    logger.info(f"  Config:       {args.config}")
    logger.info(f"  Model:        {config.model_type}")
    logger.info(f"  Checkpoint:   {config.checkpoint_path}")
    logger.info(f"  Output:       {config.output_dir}")
    logger.info(f"  Extra kwargs: {sorted(config.extra_kwargs) or 'none'}")
    logger.info("=" * 60)

    pipeline = InferencePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
