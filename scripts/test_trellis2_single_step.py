#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-step diagnostic test for Trellis2Adapter.

Compares official pipeline (float32) vs adapter (bf16) at 5 diagnostic
cross-sections for dense and shape stages:

  A. pred_pos  — model forward with positive conditioning only
  B. pred_cfg  — CFG blend (no rescale)
  C. pred_final — CFG + guidance_rescale
  D. next_latents — Euler step output
  E. occupancy — dense decode → voxel count

Usage:
    conda activate grpo3d_trellis2
    cd Flow-Factory
    python scripts/test_trellis2_single_step.py
"""
import os
import sys
import gc
import json
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
from PIL import Image
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
_SEED = 42
_T_DENSE = 0.9  # within guidance_interval [0.6, 1.0]


def _diff(a, b, label=""):
    a_f, b_f = a.float(), b.float()
    mad = (a_f - b_f).abs().mean().item()
    maxd = (a_f - b_f).abs().max().item()
    print(f"  {label:30s}  mean_abs_diff={mad:.6f}  max_abs_diff={maxd:.6f}")
    return mad


# ═══════════════════════════════════════════════════════════════════
# Phase 0: Dump pipeline.json sampler params
# ═══════════════════════════════════════════════════════════════════
def phase0_dump_params():
    print("\n" + "=" * 60)
    print("Phase 0: pipeline.json per-stage sampler params")
    print("=" * 60)
    with open(os.path.join(LOCAL_MODEL_PATH, "pipeline.json")) as f:
        cfg = json.load(f)["args"]
    for key in ["sparse_structure_sampler", "shape_slat_sampler", "tex_slat_sampler"]:
        print(f"\n  {key}:")
        print(f"    args:   {cfg[key]['args']}")
        print(f"    params: {cfg[key]['params']}")


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Load official pipeline (float32)
# ═══════════════════════════════════════════════════════════════════
def phase1_load_official():
    print("\n" + "=" * 60)
    print("Phase 1: Loading official pipeline (float32)")
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
    return pipeline


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Load adapter (bf16)
# ═══════════════════════════════════════════════════════════════════
def phase2_load_adapter():
    print("\n" + "=" * 60)
    print("Phase 2: Loading adapter (bf16)")
    print("=" * 60)
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
    config = Arguments(
        model_args=model_args,
        training_args=training_args,
        eval_args=EvaluationArguments(),
        scheduler_args=SchedulerArguments(),
        mixed_precision="bf16",
    )
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    adapter = Trellis2Adapter(config=config, accelerator=accelerator)
    adapter.eval()
    return adapter


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Dense single-step comparison
# ═══════════════════════════════════════════════════════════════════
def phase3_dense(official_pipeline, adapter):
    print("\n" + "=" * 60)
    print("Phase 3: Dense single-step comparison")
    print("=" * 60)

    device = torch.device("cuda")
    dense_params = official_pipeline.sparse_structure_sampler_params
    sigma_min = official_pipeline.sparse_structure_sampler.sigma_min
    guidance_strength = dense_params['guidance_strength']
    guidance_rescale = dense_params['guidance_rescale']
    rescale_t = dense_params['rescale_t']
    print(f"  guidance_strength={guidance_strength}, guidance_rescale={guidance_rescale}, "
          f"rescale_t={rescale_t}, sigma_min={sigma_min}")

    image = Image.open(TEST_IMAGE_PATH)
    image_processed = official_pipeline.preprocess_image(image)
    torch.manual_seed(_SEED)
    cond_official = official_pipeline.get_cond([image_processed], resolution=512)

    flow_model_off = official_pipeline.models['sparse_structure_flow_model']
    flow_model_off.to(device)
    reso = flow_model_off.resolution
    in_ch = flow_model_off.in_channels
    torch.manual_seed(_SEED)
    noise = torch.randn(1, in_ch, reso, reso, reso, device=device)  # (1, C, D, H, W)

    # Rescaled t
    t_raw = _T_DENSE
    t_rescaled = rescale_t * t_raw / (1 + (rescale_t - 1) * t_raw)
    t_next_raw = t_raw - (1.0 / 12.0)
    t_next_rescaled = rescale_t * t_next_raw / (1 + (rescale_t - 1) * t_next_raw)
    print(f"  t_raw={t_raw}, t_rescaled={t_rescaled:.6f}, t_next_rescaled={t_next_rescaled:.6f}")

    t_model = torch.tensor([1000 * t_rescaled], device=device, dtype=torch.float32)

    # ─── Official: single step ───
    print("\n  --- Official (float32) ---")
    cond_tensor = cond_official['cond'].to(device)     # (1, seq, D)
    neg_tensor = cond_official['neg_cond'].to(device)   # (1, seq, D)

    with torch.no_grad():
        off_pred_pos = flow_model_off(noise, t_model, cond_tensor)          # (1, C, D, H, W)
        off_pred_neg = flow_model_off(noise, t_model, neg_tensor)           # (1, C, D, H, W)

    off_pred_cfg = guidance_strength * off_pred_pos + (1 - guidance_strength) * off_pred_neg

    alpha = 1.0 - sigma_min
    beta_val = sigma_min + alpha * t_rescaled
    off_x0_pos = alpha * noise - beta_val * off_pred_pos
    off_x0_cfg = alpha * noise - beta_val * off_pred_cfg
    reduce_dims = list(range(1, off_x0_pos.ndim))
    std_pos = off_x0_pos.std(dim=reduce_dims, keepdim=True)
    std_cfg = off_x0_cfg.std(dim=reduce_dims, keepdim=True)
    off_x0_rescaled = off_x0_cfg * (std_pos / (std_cfg + 1e-8))
    off_x0 = guidance_rescale * off_x0_rescaled + (1.0 - guidance_rescale) * off_x0_cfg
    off_pred_final = (alpha * noise - off_x0) / beta_val

    delta = t_rescaled - t_next_rescaled
    off_next_latents = noise - delta * off_pred_final

    ss_decoder = official_pipeline.models['sparse_structure_decoder']
    ss_decoder.to(device)
    off_decoded = ss_decoder(off_next_latents) > 0
    off_occupancy = off_decoded.sum().item()
    print(f"  occupancy={off_occupancy}")

    # ─── Adapter: single step ───
    print("\n  --- Adapter (bf16) ---")
    adapter_flow = adapter.pipeline.get_flow_model('dense')
    adapter_flow.to(device)
    adapter_dtype = next(adapter_flow.parameters()).dtype
    print(f"  model dtype: {adapter_dtype}")

    adapter.pipeline.image_encoder.to(device)
    encoded = adapter.preprocess_func(images=[[image]])
    cond_adp = encoded['image_cond_512'][0].to(device=device, dtype=adapter_dtype).unsqueeze(0)
    neg_adp = encoded['neg_image_cond_512'][0].to(device=device, dtype=adapter_dtype).unsqueeze(0)


    noise_adp = noise.clone()
    t_model_adp = t_model.clone()

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=adapter_dtype, enabled=(adapter_dtype != torch.float32)):
        adp_pred_pos = adapter_flow(noise_adp, t_model_adp, cond_adp)
        adp_pred_neg = adapter_flow(noise_adp, t_model_adp, neg_adp)

    adp_pred_cfg = guidance_strength * adp_pred_pos.float() + (1 - guidance_strength) * adp_pred_neg.float()

    noise_f = noise.float()
    adp_x0_pos = alpha * noise_f - beta_val * adp_pred_pos.float()
    adp_x0_cfg = alpha * noise_f - beta_val * adp_pred_cfg
    std_pos_a = adp_x0_pos.std(dim=reduce_dims, keepdim=True)
    std_cfg_a = adp_x0_cfg.std(dim=reduce_dims, keepdim=True)
    adp_x0_rescaled = adp_x0_cfg * (std_pos_a / (std_cfg_a + 1e-8))
    adp_x0 = guidance_rescale * adp_x0_rescaled + (1.0 - guidance_rescale) * adp_x0_cfg
    adp_pred_final = (alpha * noise_f - adp_x0) / beta_val

    adp_next_latents = noise_f - delta * adp_pred_final

    adp_ss_decoder = adapter.pipeline.sparse_structure_decoder
    adp_ss_decoder.to(device)
    adp_decoded = adp_ss_decoder(adp_next_latents) > 0
    adp_occupancy = adp_decoded.sum().item()
    print(f"  occupancy={adp_occupancy}")

    # ─── Compare ───
    print("\n  --- Comparison ---")
    _diff(off_pred_pos, adp_pred_pos.float(), "A. pred_pos")
    _diff(off_pred_cfg, adp_pred_cfg, "B. pred_cfg")
    _diff(off_pred_final, adp_pred_final, "C. pred_final (with rescale)")
    _diff(off_next_latents, adp_next_latents, "D. next_latents")
    print(f"  {'E. occupancy':30s}  official={off_occupancy}  adapter={adp_occupancy}  diff={abs(off_occupancy - adp_occupancy)}")


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Shape single-step comparison
# ═══════════════════════════════════════════════════════════════════
def phase4_shape(official_pipeline, adapter, encoded_adp):
    print("\n" + "=" * 60)
    print("Phase 4: Shape single-step comparison")
    print("=" * 60)

    device = torch.device("cuda")
    shape_params = official_pipeline.shape_slat_sampler_params
    sigma_min = official_pipeline.shape_slat_sampler.sigma_min
    guidance_strength = shape_params['guidance_strength']
    guidance_rescale = shape_params['guidance_rescale']
    rescale_t = shape_params['rescale_t']
    print(f"  guidance_strength={guidance_strength}, guidance_rescale={guidance_rescale}, "
          f"rescale_t={rescale_t}, sigma_min={sigma_min}")

    image = Image.open(TEST_IMAGE_PATH)
    image_processed = official_pipeline.preprocess_image(image)

    # Get coords from official pipeline
    torch.manual_seed(_SEED)
    cond_off = official_pipeline.get_cond([image_processed], resolution=512)
    torch.manual_seed(_SEED)
    coords = official_pipeline.sample_sparse_structure(cond_off, resolution=64)
    N = coords.shape[0]
    print(f"  coords: {N} voxels")

    flow_model_off = official_pipeline.models.get(
        'shape_slat_flow_model_1024',
        official_pipeline.models.get('shape_slat_flow_model_512')
    )
    flow_model_off.to(device)
    in_ch = flow_model_off.in_channels
    model_dtype_off = next(flow_model_off.parameters()).dtype
    print(f"  official shape model dtype: {model_dtype_off}")

    from trellis2.modules.sparse import SparseTensor

    torch.manual_seed(_SEED)
    noise_feats = torch.randn(N, in_ch, device=device)  # (N, C)

    t_raw = 0.9
    t_rescaled = rescale_t * t_raw / (1 + (rescale_t - 1) * t_raw)
    t_next_raw = t_raw - (1.0 / 12.0)
    t_next_rescaled = rescale_t * t_next_raw / (1 + (rescale_t - 1) * t_next_raw)
    print(f"  t_rescaled={t_rescaled:.6f}, t_next_rescaled={t_next_rescaled:.6f}")

    x_t = SparseTensor(feats=noise_feats.to(dtype=model_dtype_off), coords=coords.to(device))
    t_model = torch.tensor([1000 * t_rescaled], device=device, dtype=torch.float32)

    cond_1024 = official_pipeline.get_cond([image_processed], resolution=1024)
    cond_tensor = cond_1024['cond'].to(device)
    neg_tensor = cond_1024['neg_cond'].to(device)

    # ─── Official forward ───
    print("\n  --- Official (float32) ---")
    with torch.no_grad():
        off_pred_pos = flow_model_off(x=x_t, t=t_model, cond=cond_tensor, concat_cond=None)
        off_pred_neg = flow_model_off(x=x_t, t=t_model, cond=neg_tensor, concat_cond=None)

    off_pred_cfg = guidance_strength * off_pred_pos + (1 - guidance_strength) * off_pred_neg

    alpha = 1.0 - sigma_min
    beta_val = sigma_min + alpha * t_rescaled
    off_x0_pos = alpha * x_t - beta_val * off_pred_pos
    off_x0_cfg = alpha * x_t - beta_val * off_pred_cfg
    off_std_pos = off_x0_pos.feats.std()
    off_std_cfg = off_x0_cfg.feats.std()
    ratio = off_std_pos / (off_std_cfg + 1e-8)
    off_x0_rescaled = off_x0_cfg.replace(feats=off_x0_cfg.feats * ratio)
    off_x0 = guidance_rescale * off_x0_rescaled + (1.0 - guidance_rescale) * off_x0_cfg
    off_pred_final = (alpha * x_t - off_x0) / beta_val

    delta = t_rescaled - t_next_rescaled
    off_next_feats = (x_t - delta * off_pred_final).feats

    # ─── Adapter forward ───
    print("\n  --- Adapter (bf16) ---")
    adapter_flow = adapter.pipeline.get_flow_model('shape', 1024)
    adapter_flow.to(device)
    adapter_dtype = next(adapter_flow.parameters()).dtype
    print(f"  adapter shape model dtype: {adapter_dtype}")

    cond_adp = encoded_adp['image_cond_1024'][0].to(device=device, dtype=adapter_dtype).unsqueeze(0)
    neg_adp = encoded_adp['neg_image_cond_1024'][0].to(device=device, dtype=adapter_dtype).unsqueeze(0)

    x_t_adp = SparseTensor(feats=noise_feats.to(dtype=adapter_dtype), coords=coords.to(device))

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=adapter_dtype, enabled=(adapter_dtype != torch.float32)):
        adp_pred_pos = adapter_flow(x=x_t_adp, t=t_model, cond=cond_adp, concat_cond=None)
        adp_pred_neg = adapter_flow(x=x_t_adp, t=t_model, cond=neg_adp, concat_cond=None)

    adp_pred_cfg = guidance_strength * adp_pred_pos + (1 - guidance_strength) * adp_pred_neg

    x_t_f = SparseTensor(feats=noise_feats.float(), coords=coords.to(device))
    adp_x0_pos = alpha * x_t_f - beta_val * adp_pred_pos.replace(feats=adp_pred_pos.feats.float())
    adp_x0_cfg_st = alpha * x_t_f - beta_val * adp_pred_cfg.replace(feats=adp_pred_cfg.feats.float())
    adp_std_pos = adp_x0_pos.feats.std()
    adp_std_cfg = adp_x0_cfg_st.feats.std()
    adp_ratio = adp_std_pos / (adp_std_cfg + 1e-8)
    adp_x0_rescaled = adp_x0_cfg_st.replace(feats=adp_x0_cfg_st.feats * adp_ratio)
    adp_x0 = guidance_rescale * adp_x0_rescaled + (1.0 - guidance_rescale) * adp_x0_cfg_st
    adp_pred_final = (alpha * x_t_f - adp_x0) / beta_val

    adp_next_feats = (x_t_f - delta * adp_pred_final).feats

    # ─── Compare ───
    print("\n  --- Comparison ---")
    _diff(off_pred_pos.feats, adp_pred_pos.feats.float(), "A. pred_pos")
    _diff(off_pred_cfg.feats, adp_pred_cfg.feats.float(), "B. pred_cfg")
    _diff(off_pred_final.feats, adp_pred_final.feats.float(), "C. pred_final (with rescale)")
    _diff(off_next_feats, adp_next_feats, "D. next_feats")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    phase0_dump_params()

    official = phase1_load_official()
    adapter = phase2_load_adapter()

    phase3_dense(official, adapter)

    # Encode adapter conditioning for phase 4 before offloading
    adapter.pipeline.image_encoder.to(torch.device("cuda"))
    image = Image.open(TEST_IMAGE_PATH)
    encoded_adp = adapter.preprocess_func(images=[[image]])
    adapter.pipeline.image_encoder.cpu()

    official.models['sparse_structure_flow_model'].cpu()
    official.models['sparse_structure_decoder'].cpu()
    adapter.pipeline.get_flow_model('dense').cpu()
    adapter.pipeline.sparse_structure_decoder.cpu()
    torch.cuda.empty_cache()

    phase4_shape(official, adapter, encoded_adp)

    print(f"\n{'=' * 60}")
    print(f"Done. Total time: {time.time() - t0:.0f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
