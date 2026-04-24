#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Render comparison test for Trellis2Adapter: Official ODE vs Local SDE rollout.

Paths:
    A  Official ODE  (Trellis2ImageTo3DPipeline.run)
    B  Local   SDE   (adapter.rollout, all stages SDE)

Flow:
    Phase 1: Path A — official pipeline ODE inference → render → save A.pt → release
    Phase 2: Load adapter, precompute image conditioning
    Phase 3: Path B — adapter SDE rollout inference → render → save B_sde.pt → release
    Phase 4: CPU only — load A/B_sde.pt → comparison grid + numerical diff + log_prob stats

Usage:
    conda activate grpo3d_trellis2
    cd Flow-Factory
    python scripts/test_trellis2_inference_sde.py
"""
import argparse
import os
import sys
import gc
import math
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "third_party", "TRELLIS.2"))

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_CACHE"] = os.path.join(
    project_root, "pretrained_weights/dinov3-vitl16-pretrain-lvd1689m"
)

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from accelerate import Accelerator

from src.flow_factory.hparams import (
    Arguments,
    ModelArguments,
    TrainingArguments,
    SchedulerArguments,
)
from src.flow_factory.hparams.training_args import EvaluationArguments
from src.flow_factory.models.trellis2.trellis2 import Trellis2Adapter

LOCAL_MODEL_PATH = os.path.join(project_root, "pretrained_weights/TRELLIS.2-4B")
TEST_IMAGE_PATH = os.path.join(
    project_root, "third_party/TRELLIS.2/assets/example_image/T.png"
)
ENVMAP_PATH = os.path.join(
    project_root, "third_party/TRELLIS.2/assets/hdri/forest.exr"
)
OUTPUT_DIR = os.path.join(project_root, "scripts/outputs/render_comparison")

_FOV_DEG = 40.0
_PITCH_DEG = 20.0
_FILL_RATIO = 0.9
_SEED = 42
_NUM_FRAMES = 24
_RESOLUTION = 512


# ──────────────────────── helpers ────────────────────────────────

def _compute_adaptive_distance(fov_deg: float, fill_ratio: float = 0.9,
                               object_half_size: float = 0.5) -> float:
    tan_half_fov = math.tan(math.radians(fov_deg) / 2)
    return object_half_size / (fill_ratio * tan_half_fov)


def build_envmap(device: str = "cuda"):
    from trellis2.renderers.pbr_mesh_renderer import EnvMap

    hdr_bgr = cv2.imread(ENVMAP_PATH, cv2.IMREAD_UNCHANGED)
    hdr_rgb = cv2.cvtColor(hdr_bgr, cv2.COLOR_BGR2RGB)
    hdr_tensor = torch.tensor(hdr_rgb, dtype=torch.float32, device=device)  # (H, W, 3)
    return EnvMap(hdr_tensor)


def render_mesh(mesh, envmap, num_frames: int = _NUM_FRAMES,
                resolution: int = _RESOLUTION) -> torch.Tensor:
    from trellis2.utils import render_utils

    r = _compute_adaptive_distance(_FOV_DEG, fill_ratio=_FILL_RATIO)
    yaws_rad = torch.linspace(0, 2 * np.pi, num_frames + 1)[:-1].tolist()
    pitchs_rad = [np.deg2rad(_PITCH_DEG)] * num_frames
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws_rad, pitchs_rad, r, _FOV_DEG,
    )
    ret = render_utils.render_frames(
        mesh, extrinsics, intrinsics,
        {"resolution": resolution, "bg_color": (0, 0, 0)},
        envmap=envmap, verbose=False,
    )
    frames_np = np.stack(ret["shaded"])                    # (T, H, W, C) uint8
    frames = torch.from_numpy(frames_np).float() / 255.0  # (T, H, W, C) float32 [0,1]
    frames = frames.permute(0, 3, 1, 2)                   # (T, C, H, W) float32
    return frames


def make_comparison_image(tensors: dict, num_views: int = 6) -> Image.Image:
    labels = list(tensors.keys())
    T = next(iter(tensors.values())).shape[0]
    indices = torch.linspace(0, T - 1, num_views).long().tolist()

    label_width = 180
    rows = []
    for label in labels:
        video = tensors[label]                             # (T, C, H, W)
        selected = video[indices]                          # (num_views, C, H, W)
        selected_np = (selected.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        row_img = np.concatenate(list(selected_np), axis=1)

        h, w_total, c = row_img.shape
        labeled = np.zeros((h, w_total + label_width, c), dtype=np.uint8)
        labeled[:, label_width:, :] = row_img
        pil_row = Image.fromarray(labeled)
        draw = ImageDraw.Draw(pil_row)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((8, h // 2 - 14), label, fill=(255, 255, 255), font=font)
        rows.append(np.array(pil_row))

    grid = np.concatenate(rows, axis=0)
    return Image.fromarray(grid)


# ──────────────────── config / setup ─────────────────────────────

def create_test_config(noise_level: float = 0.7, num_sde_steps: int = 1) -> Arguments:
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
    training_args = TrainingArguments(
        enable_gradient_checkpointing=False,
        ema_decay=0.0,
    )
    return Arguments(
        model_args=model_args,
        training_args=training_args,
        eval_args=EvaluationArguments(),
        scheduler_args=SchedulerArguments(
            noise_level=noise_level,
            num_sde_steps=num_sde_steps,
        ),
        mixed_precision="bf16",
    )


def setup_adapter(noise_level: float = 0.7, num_sde_steps: int = 1):
    config = create_test_config(noise_level=noise_level, num_sde_steps=num_sde_steps)
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    print("[setup] Loading Trellis2Adapter ...")
    adapter = Trellis2Adapter(config=config, accelerator=accelerator)
    adapter.pipeline.image_encoder.to(adapter.device)
    image = Image.open(TEST_IMAGE_PATH)
    return adapter, image


# ──────────────────────── main ───────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_level", type=float, default=0.7,
                        help="SDE noise level (default: 0.7)")
    parser.add_argument("--num_sde_steps", type=int, default=1,
                        help="Number of SDE steps per rollout (default: 1)")
    parser.add_argument("--skip_path_a", action="store_true",
                        help="Skip Path A (official ODE) if A.pt already exists")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image = Image.open(TEST_IMAGE_PATH)
    assert image is not None, f"Test image not found: {TEST_IMAGE_PATH}"
    t0 = time.time()

    nl_tag = f"nl{args.noise_level:g}_ns{args.num_sde_steps}"  # e.g. nl1.5_ns1
    sde_pt_name = f"B_sde_{nl_tag}.pt"
    sde_comp_name = f"render_comparison_sde_{nl_tag}.png"

    a_pt_path = os.path.join(OUTPUT_DIR, "A.pt")
    skip_a = args.skip_path_a and os.path.exists(a_pt_path)

    # ═══════════════ Phase 1: Path A (official pipeline) ═════════
    if skip_a:
        print(f"\n[Phase 1 SKIPPED — A.pt already exists at {a_pt_path}]")
    else:
        print("\n" + "=" * 60)
        print("PATH A: Official ODE (Trellis2ImageTo3DPipeline.run)")
        print("=" * 60)

        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        from trellis2.modules import image_feature_extractor
        import o_voxel  # noqa: F401

        local_dinov3 = os.path.join(
            project_root,
            "pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/"
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
        )
        _orig_init = image_feature_extractor.DinoV3FeatureExtractor.__init__
        def _patched_init(self, model_name: str, image_size=512):
            _orig_init(self, local_dinov3, image_size)
        image_feature_extractor.DinoV3FeatureExtractor.__init__ = _patched_init
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(LOCAL_MODEL_PATH)
        image_feature_extractor.DinoV3FeatureExtractor.__init__ = _orig_init
        pipeline.cuda()

        torch.manual_seed(_SEED)
        mesh_a = pipeline.run(image, seed=_SEED, pipeline_type='1024')[0]
        mesh_a.simplify(16777216)
        print(f"  Mesh: {mesh_a.vertices.shape[0]} verts, {mesh_a.faces.shape[0]} faces")

        envmap_a = build_envmap("cuda")
        video_a = render_mesh(mesh_a, envmap_a)
        torch.save(video_a.cpu(), a_pt_path)
        print(f"  video: {video_a.shape}  [{time.time()-t0:.0f}s]")

        del pipeline, mesh_a, envmap_a, video_a
        gc.collect()
        torch.cuda.empty_cache()

    # ═══════════════ Phase 2: Load adapter + precompute cond ═════
    print("\n" + "=" * 60)
    print(f"Phase 2: Load adapter + precompute image conditioning"
          f"  (noise_level={args.noise_level}, num_sde_steps={args.num_sde_steps})")
    print("=" * 60)

    adapter, image = setup_adapter(
        noise_level=args.noise_level,
        num_sde_steps=args.num_sde_steps,
    )

    print("[preprocess] Encoding image conditioning (512 + 1024) ...")
    encoded = adapter.preprocess_func(images=[[image]])
    print(f"  done  [{time.time()-t0:.0f}s]")

    adapter.pipeline.image_encoder.cpu()
    torch.cuda.empty_cache()

    envmap = adapter._build_envmap()

    # ═══════════════ Phase 3: Path B (Local SDE rollout) ═════════
    print(f"\n{'=' * 60}")
    print(f"PATH B: Local SDE rollout (adapter.rollout)  [{nl_tag}]")
    print("=" * 60)

    torch.manual_seed(_SEED)
    adapter.rollout()

    # Report which steps will actually be SDE for each stage
    for stage_name, sched in [
        ('dense', adapter.pipeline.scheduler_dense),
        ('shape', adapter.pipeline.scheduler_shape),
        ('tex',   adapter.pipeline.scheduler_tex),
    ]:
        print(f"  [{stage_name}] sde_steps={sched.current_sde_steps.tolist()}, "
              f"num={sched.num_sde_steps}, noise_level={sched.noise_level}")

    for stage, res in [('dense', 1024), ('shape', 1024), ('tex', 1024)]:
        adapter.pipeline.get_flow_model(stage, res).to(adapter.device)
    for attr in ['sparse_structure_decoder', 'shape_decoder', 'tex_decoder']:
        getattr(adapter.pipeline, attr).to(adapter.device)

    torch.manual_seed(_SEED)

    t_path = time.time()
    with adapter.low_vram_mode():
        sample = adapter.inference(
            image_cond_512=encoded['image_cond_512'],
            neg_image_cond_512=encoded['neg_image_cond_512'],
            image_cond_1024=encoded['image_cond_1024'],
            neg_image_cond_1024=encoded['neg_image_cond_1024'],
            condition_images=encoded.get('condition_images'),
            stages=["dense", "shape", "tex"],
            compute_log_prob=True,
            decode_output=False,
        )[0]
        print(f"  inference done  [{time.time()-t_path:.0f}s]")
        print(f"  coords: {sample.sparse_coords.shape[0]}")

        # Log log_prob statistics for each stage
        for stage_name, lp_field in [
            ("dense", "dense_log_probs"),
            ("shape", "shape_log_probs"),
            ("tex",   "tex_log_probs"),
        ]:
            lp = getattr(sample, lp_field, None)
            if lp is not None:
                print(f"  {stage_name} log_prob: mean={lp.mean():.4f}, min={lp.min():.4f}, "
                      f"max={lp.max():.4f}, has_nan={lp.isnan().any()}, has_inf={lp.isinf().any()}, "
                      f"shape={tuple(lp.shape)}")
            else:
                print(f"  {stage_name} log_prob: None")

        t_render = time.time()
        try:
            adapter.render_latents(
                sample, num_frames=_NUM_FRAMES, resolution=_RESOLUTION,
                envmap=envmap, verbose=False,
            )
        except torch.cuda.OutOfMemoryError:
            print("  !! OOM during render — skipping path B")
            adapter.pipeline.shape_decoder.cpu()
            adapter.pipeline.tex_decoder.cpu()
            torch.cuda.empty_cache()
            sample.video = None

    sde_pt_path = os.path.join(OUTPUT_DIR, sde_pt_name)
    if sample.video is not None:
        torch.save(sample.video.cpu(), sde_pt_path)
        print(f"  render done  video={sample.video.shape}  [{time.time()-t_render:.0f}s]")
    else:
        print(f"  render SKIPPED (OOM)  [{time.time()-t_render:.0f}s]")
    print(f"  total path B: {time.time()-t_path:.0f}s  [{time.time()-t0:.0f}s elapsed]")

    del sample, adapter, envmap
    gc.collect()
    torch.cuda.empty_cache()

    # ═══════════════ Phase 4: Comparison (CPU only) ═══════════════
    print("\n" + "=" * 60)
    print(f"Phase 4: Comparison — Official ODE vs Local SDE ({nl_tag})")
    print("=" * 60)

    all_videos = {}
    ode_label = "A (Official ODE)"
    sde_label = f"B (Local SDE, {nl_tag})"
    for pt_path, label in [(a_pt_path, ode_label), (sde_pt_path, sde_label)]:
        if os.path.exists(pt_path):
            all_videos[label] = torch.load(pt_path, weights_only=True)
        else:
            print(f"  {os.path.basename(pt_path)}: not found, skipping")

    if len(all_videos) == 2:
        comp = make_comparison_image(all_videos)
        comp_path = os.path.join(OUTPUT_DIR, sde_comp_name)
        comp.save(comp_path)
        print(f"Comparison image saved to {comp_path}")

        ref = all_videos[ode_label]
        local = all_videos[sde_label]
        diff = (local - ref).abs()                         # (T, C, H, W)
        print(f"  B_sde[{nl_tag}] vs A — mean_abs_diff={diff.mean():.4f}, max={diff.max():.4f}")
    else:
        print("  Not enough results for comparison")

    print(f"\nDone. Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
