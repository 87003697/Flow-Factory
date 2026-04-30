#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visual debug for Trellis2 geometry-only render modes.

Phase 0 of the clay-geometry-reward-shape plan: before wiring `render_mode` into
the trainer / yaml / reward, dump a few frames in each candidate mode so we can
eyeball whether `clay` (SSAO occlusion) or `normal` is geometry-readable for
typical Trellis2 condition images.

Outputs (under `--output_dir`):
    {mode}_grid.png            — N-frame grid for a single render mode.
    {mode}_side_by_side.png    — single condition+render side-by-side tile, mimicking
                                  the QwenVLSideBySideReward reward input layout.
    all_modes_comparison.png   — one row per mode, evenly sampled views, for
                                  cross-mode visual comparison.

Usage:
    python scripts/debug_clay_render.py
    python scripts/debug_clay_render.py --image dataset/trellis2/images/<hash>.webp
    python scripts/debug_clay_render.py --render_modes shaded,clay
"""
import argparse
import gc
import math
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "third_party", "TRELLIS.2"))

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_CACHE"] = os.path.join(
    project_root, "pretrained_weights/dinov3-vitl16-pretrain-lvd1689m"
)

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from accelerate import Accelerator

from src.flow_factory.hparams import (
    Arguments,
    ModelArguments,
    SchedulerArguments,
    TrainingArguments,
)
from src.flow_factory.hparams.training_args import EvaluationArguments
from src.flow_factory.models.trellis2.trellis2 import Trellis2Adapter

LOCAL_MODEL_PATH = os.path.join(project_root, "pretrained_weights/TRELLIS.2-4B")
DEFAULT_IMAGE_PATH = os.path.join(
    project_root, "third_party/TRELLIS.2/assets/example_image/T.png"
)
DEFAULT_OUTPUT_DIR = os.path.join(project_root, "scripts/outputs/debug_clay")


def _build_config() -> Arguments:
    """Minimal Arguments for instantiating Trellis2Adapter in inference mode."""
    model_args = ModelArguments(
        model_name_or_path=LOCAL_MODEL_PATH,
        model_type="trellis2",
        finetune_type="lora",
        lora_rank=16,
        lora_alpha=32,
        target_modules="default",
        target_components=["transformer"],
        resume_path=None,
        extra_kwargs={"target_flow_model": "shape_slat_1024"},
    )
    return Arguments(
        model_args=model_args,
        training_args=TrainingArguments(
            enable_gradient_checkpointing=False,
            ema_decay=0.0,
        ),
        eval_args=EvaluationArguments(),
        scheduler_args=SchedulerArguments(dynamics_type="ODE"),
        mixed_precision="bf16",
    )


def _setup_adapter() -> Trellis2Adapter:
    config = _build_config()
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    print("[setup] Loading Trellis2Adapter ...")
    adapter = Trellis2Adapter(config=config, accelerator=accelerator)
    adapter.pipeline.image_encoder.to(adapter.device)
    return adapter


def _frames_to_grid(
    video: torch.Tensor, ncols: int | None = None
) -> Image.Image:
    """Tile (T, 3, H, W) float [0,1] frames into a single grid PIL image."""
    T, C, H, W = video.shape  # (T, 3, H, W)
    if ncols is None:
        ncols = max(1, int(math.ceil(math.sqrt(T))))
    nrows = int(math.ceil(T / ncols))

    canvas = np.zeros((nrows * H, ncols * W, 3), dtype=np.uint8)  # (nrows*H, ncols*W, 3)
    arr = (video.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255).astype(
        np.uint8
    )  # (T, H, W, 3) uint8
    for idx in range(T):
        r, c = divmod(idx, ncols)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W, :] = arr[idx]
    return Image.fromarray(canvas)


def _resize_pil(img: Image.Image, side: int) -> Image.Image:
    return img.convert("RGB").resize((side, side), Image.Resampling.BILINEAR)


def _frame_to_pil(frame: torch.Tensor) -> Image.Image:
    """Convert a single (3, H, W) float[0,1] CUDA/CPU tensor to PIL."""
    arr = (frame.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(
        np.uint8
    )  # (H, W, 3) uint8
    return Image.fromarray(arr, mode="RGB")


def _side_by_side(
    cond_pil: Image.Image, render_frame: torch.Tensor, tile_resolution: int
) -> Image.Image:
    """Mimic `QwenVLSideBySideReward._build_side_by_side_frames` for one frame."""
    cond_tile = _resize_pil(cond_pil, tile_resolution)  # (S, S, 3) PIL
    render_tile = _frame_to_pil(render_frame).resize(
        (tile_resolution, tile_resolution), Image.Resampling.BILINEAR
    )  # (S, S, 3) PIL
    cond_arr = np.asarray(cond_tile, dtype=np.uint8)  # (S, S, 3) uint8
    rend_arr = np.asarray(render_tile, dtype=np.uint8)  # (S, S, 3) uint8
    sbs = np.concatenate([cond_arr, rend_arr], axis=1)  # (S, 2S, 3) uint8
    return Image.fromarray(sbs, mode="RGB")


def _all_modes_comparison(
    videos: dict[str, torch.Tensor], num_views: int = 6
) -> Image.Image:
    """One row per mode, num_views evenly sampled frames, with a label gutter."""
    labels = list(videos.keys())
    T = next(iter(videos.values())).shape[0]
    indices = torch.linspace(0, T - 1, num_views).long().tolist()
    label_width = 180

    rows = []
    for label in labels:
        video = videos[label]                              # (T, 3, H, W)
        sel = video[indices]                               # (num_views, 3, H, W)
        sel_np = (
            sel.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255
        ).astype(np.uint8)                                 # (num_views, H, W, 3)
        row_img = np.concatenate(list(sel_np), axis=1)     # (H, num_views*W, 3)
        h, w_total, _ = row_img.shape
        labeled = np.zeros((h, w_total + label_width, 3), dtype=np.uint8)
        labeled[:, label_width:, :] = row_img
        pil_row = Image.fromarray(labeled)
        draw = ImageDraw.Draw(pil_row)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
            )
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((10, h // 2 - 16), label, fill=(255, 255, 255), font=font)
        rows.append(np.array(pil_row))

    grid = np.concatenate(rows, axis=0)
    return Image.fromarray(grid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH,
                        help="Condition image path (RGBA or RGB).")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render_modes", default="shaded,clay,normal",
                        help="CSV of render modes. Subset of {shaded,clay,normal}.")
    parser.add_argument("--tile_resolution", type=int, default=384,
                        help="Side length used for the side-by-side tile (matches "
                             "QwenVLSideBySideReward default).")
    parser.add_argument("--side_by_side_frame", type=int, default=0,
                        help="Which frame index to use for the side-by-side tile.")
    parser.add_argument("--num_inference_steps", type=int, default=12)
    args = parser.parse_args()

    render_modes = [m.strip() for m in args.render_modes.split(",") if m.strip()]
    valid_modes = {"shaded", "clay", "normal"}
    bad = set(render_modes) - valid_modes
    if bad:
        raise ValueError(
            f"Unsupported render_mode(s): {sorted(bad)}; expected subset of {sorted(valid_modes)}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    assert os.path.isfile(args.image), f"Condition image not found: {args.image}"

    image = Image.open(args.image)
    print(f"[input] {args.image}  size={image.size}  mode={image.mode}")
    t0 = time.time()

    adapter = _setup_adapter()
    print(f"[setup] adapter loaded  [{time.time()-t0:.0f}s]")

    print("[preprocess] Encoding image conditioning ...")
    encoded = adapter.preprocess_func(images=[[image]])
    print(f"[preprocess] done  [{time.time()-t0:.0f}s]")

    adapter.pipeline.image_encoder.cpu()
    torch.cuda.empty_cache()
    envmap = adapter._build_envmap()

    for stage, res in [("dense", 1024), ("shape", 1024), ("tex", 1024)]:
        adapter.pipeline.get_flow_model(stage, res).to(adapter.device)
    for attr in ["sparse_structure_decoder", "shape_decoder", "tex_decoder"]:
        getattr(adapter.pipeline, attr).to(adapter.device)

    torch.manual_seed(args.seed)
    print("[inference] dense + shape + tex (ODE) ...")
    t_inf = time.time()
    with adapter.low_vram_mode():
        sample = adapter.inference(
            image_cond_512=encoded["image_cond_512"],
            neg_image_cond_512=encoded["neg_image_cond_512"],
            image_cond_1024=encoded["image_cond_1024"],
            neg_image_cond_1024=encoded["neg_image_cond_1024"],
            condition_images=encoded.get("condition_images"),
            stages=["dense", "shape", "tex"],
            num_inference_steps=args.num_inference_steps,
            compute_log_prob=False,
            decode_output=False,
        )[0]
    print(f"[inference] done  coords={sample.sparse_coords.shape[0]}  [{time.time()-t_inf:.0f}s]")

    cond_for_sbs = (
        encoded["condition_images"][0][0]
        if encoded.get("condition_images")
        else image
    )

    videos: dict[str, torch.Tensor] = {}
    with adapter.low_vram_mode():
        for mode in render_modes:
            t_r = time.time()
            adapter.render_latents(
                sample,
                num_frames=args.num_frames,
                resolution=args.resolution,
                envmap=envmap,
                render_mode=mode,
            )
            assert sample.video is not None, f"render_latents returned no video for mode={mode}"
            video = sample.video.detach().clone().cpu()  # (T, 3, H, W) float32 [0,1]
            videos[mode] = video
            print(f"[render] mode={mode:<6s}  shape={tuple(video.shape)}  "
                  f"min={video.min().item():.3f}  max={video.max().item():.3f}  "
                  f"mean={video.mean().item():.3f}  [{time.time()-t_r:.1f}s]")

            grid = _frames_to_grid(video)
            grid_path = os.path.join(args.output_dir, f"{mode}_grid.png")
            grid.save(grid_path)
            print(f"[save] {grid_path}")

            if 0 <= args.side_by_side_frame < video.shape[0]:
                sbs = _side_by_side(
                    cond_for_sbs,
                    video[args.side_by_side_frame],
                    args.tile_resolution,
                )
                sbs_path = os.path.join(args.output_dir, f"{mode}_side_by_side.png")
                sbs.save(sbs_path)
                print(f"[save] {sbs_path}")

    if len(videos) >= 2:
        comp = _all_modes_comparison(videos)
        comp_path = os.path.join(args.output_dir, "all_modes_comparison.png")
        comp.save(comp_path)
        print(f"[save] {comp_path}")

    del sample, adapter, envmap
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nDone. Total {time.time()-t0:.0f}s. Outputs at {args.output_dir}")


if __name__ == "__main__":
    main()
