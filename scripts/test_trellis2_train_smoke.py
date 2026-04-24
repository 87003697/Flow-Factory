#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pre-GRPO smoke tests for Trellis2.

Two gates:
  A. **Gradient flow** — run ``adapter.forward(stage='shape')`` with bs=2 SDE
     inputs, ``compute_log_prob=True`` and ``guidance_rescale>0`` so the
     per-sample std loop in ``_apply_cfg_sparse`` is exercised.  Call
     ``log_prob.sum().backward()`` and verify (i) no autograd error, (ii) all
     LoRA parameters have non-None, non-NaN ``.grad`` tensors.
  B. **Sample stacking** — build two mock ``Trellis2Sample`` instances with
     per-stage fields populated, call ``BaseSample.stack([s0, s1])`` and
     check that
        - ``sparse_coords`` is ``(N_total, 4)`` with col 0 in ``{0, 1}``;
        - ``all_latents`` (stage-mirrored shape field) is ``(N_total, T, C)``;
        - ``log_probs`` is ``(bs, T_log)``;
        - ``timesteps`` is ``(bs, steps+1)``;
        - ``latent_index_map`` / ``log_prob_index_map`` are shared single tensors.

The script exits non-zero if either gate fails.

Usage:
    conda activate grpo3d_trellis2
    cd Flow-Factory
    python scripts/test_trellis2_train_smoke.py
"""
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
from src.flow_factory.models.trellis2.trellis2 import Trellis2Adapter, Trellis2Sample
from src.flow_factory.samples import BaseSample

LOCAL_MODEL_PATH = os.path.join(project_root, "pretrained_weights/TRELLIS.2-4B")
TEST_IMAGE_PATH = os.path.join(
    project_root, "third_party/TRELLIS.2/assets/example_image/T.png"
)

_SEED = 42
_T_VAL = 0.7
_T_NEXT = 0.65


def _create_config(enable_gradient_checkpointing: bool = False) -> Arguments:
    sde_cfg = {"dynamics_type": "Flow-SDE", "noise_level": 0.7, "num_sde_steps": 1}
    model_args = ModelArguments(
        model_name_or_path=LOCAL_MODEL_PATH,
        model_type="trellis2",
        finetune_type="lora",
        lora_rank=16,
        lora_alpha=32,
        target_modules="default",
        target_components=["transformer"],
        resume_path=None,
        extra_kwargs={
            "target_flow_model": "shape_slat_1024",
            "shape_sde": sde_cfg,
        },
    )
    training_args = TrainingArguments(
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        ema_decay=0.0,
    )
    return Arguments(
        model_args=model_args,
        training_args=training_args,
        eval_args=EvaluationArguments(),
        scheduler_args=SchedulerArguments(
            noise_level=0.7, num_sde_steps=1, seed=_SEED,
        ),
        mixed_precision="bf16",
    )


def _setup_adapter(enable_gradient_checkpointing: bool = False) -> Trellis2Adapter:
    config = _create_config(enable_gradient_checkpointing=enable_gradient_checkpointing)
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    print(f"[setup] Loading Trellis2Adapter (gradient_checkpointing={enable_gradient_checkpointing}) ...")
    adapter = Trellis2Adapter(config=config, accelerator=accelerator)
    adapter.pipeline.image_encoder.to(adapter.device)
    return adapter


def _make_rand(shape, device, dtype, seed: int) -> torch.Tensor:
    g = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn(shape, generator=g, device='cpu').to(device=device, dtype=dtype)


def _get_sparse_coords(adapter: Trellis2Adapter, image, n_samples: int) -> tuple:
    """Run dense ODE inference once to obtain real sparse_coords for both samples."""
    encoded = adapter.preprocess_func(images=[[image]] * n_samples)
    adapter.pipeline.image_encoder.cpu()
    torch.cuda.empty_cache()

    adapter.eval()
    for stage in ['dense', 'shape', 'tex']:
        adapter.pipeline.get_flow_model(stage, 1024).to(adapter.device)
    for attr in ['sparse_structure_decoder', 'shape_decoder', 'tex_decoder']:
        getattr(adapter.pipeline, attr).to(adapter.device)

    torch.manual_seed(_SEED)
    with adapter.low_vram_mode():
        samples = adapter.inference(
            image_cond_512=encoded['image_cond_512'],
            neg_image_cond_512=encoded['neg_image_cond_512'],
            image_cond_1024=encoded['image_cond_1024'],
            neg_image_cond_1024=encoded['neg_image_cond_1024'],
            condition_images=encoded.get('condition_images'),
            stages=["dense"],
            compute_log_prob=False,
            decode_output=False,
        )
    return encoded, [s.sparse_coords.clone() for s in samples]


def _test_gradient_flow(
    adapter: Trellis2Adapter,
    encoded: dict,
    coords_list: list,
    truncate_to: int = 0,
    use_batched_t: bool = True,
) -> bool:
    """Gate A: gradient through bs=2 shape forward with per-sample CFG rescale.

    When ``truncate_to > 0`` the coords are sliced to that many points per
    sample (fallback for when VRAM can't host the full token count).  When
    ``use_batched_t`` is True, ``t`` / ``t_next`` are passed as ``(B,)``
    tensors to exercise the trainer-style calling convention.
    """
    print("\n" + "=" * 70)
    print(
        "Gate A: Gradient flow through _apply_cfg_sparse per-sample std "
        f"(bs=2, truncate_to={truncate_to or 'full'}, batched_t={use_batched_t})"
    )
    print("=" * 70)

    adapter.train()
    torch.cuda.empty_cache()

    for stage_name in ['dense', 'tex']:
        adapter.pipeline.get_flow_model(stage_name, 1024).cpu()
    shape_flow = adapter.pipeline.get_flow_model('shape', 1024).to(adapter.device)
    torch.cuda.empty_cache()

    device = adapter.device
    in_channels = shape_flow.in_channels

    coords_0 = coords_list[0].to(device).clone()                                      # (N_0, 4)
    coords_0[:, 0] = 0
    coords_1 = coords_list[1].to(device).clone()                                      # (N_1, 4)
    coords_1[:, 0] = 0
    if truncate_to > 0:
        coords_0 = coords_0[:truncate_to]                                             # (n, 4)
        coords_1 = coords_1[:truncate_to]                                             # (n, 4)
    N_A, N_B = coords_0.shape[0], coords_1.shape[0]
    coords_b = torch.cat([coords_0, coords_1.clone()], dim=0)                         # (N_A+N_B, 4)
    coords_b[N_A:, 0] = 1
    print(f"  sparse_coords: N_A={N_A}, N_B={N_B}, N_total={N_A+N_B}")

    feats_b = _make_rand((N_A + N_B, in_channels), device, torch.float32, _SEED + 1)  # (N_total, C)
    next_feats_b = _make_rand(
        (N_A + N_B, in_channels), device, torch.float32, _SEED + 2,
    )                                                                                 # (N_total, C)

    cond_0 = encoded['image_cond_1024'][0].to(device=device, dtype=torch.float32)     # (seq, D)
    cond_1 = encoded['image_cond_1024'][1].to(device=device, dtype=torch.float32)     # (seq, D)
    neg_0  = encoded['neg_image_cond_1024'][0].to(device=device, dtype=torch.float32) # (seq, D)
    neg_1  = encoded['neg_image_cond_1024'][1].to(device=device, dtype=torch.float32) # (seq, D)
    cond_b = torch.stack([cond_0, cond_1], dim=0)                                     # (2, seq, D)
    neg_b  = torch.stack([neg_0,  neg_1],  dim=0)                                     # (2, seq, D)

    if use_batched_t:
        # Trainer-style: batch['timesteps'][:, step_idx] is (bs,) with the
        # same value for every sample.
        t_val  = torch.full((2,), _T_VAL,  device=device, dtype=torch.float32)       # (2,)
        t_next = torch.full((2,), _T_NEXT, device=device, dtype=torch.float32)       # (2,)
    else:
        t_val  = torch.tensor(_T_VAL,  device=device, dtype=torch.float32)           # scalar
        t_next = torch.tensor(_T_NEXT, device=device, dtype=torch.float32)           # scalar

    trainable_params = [p for p in shape_flow.parameters() if p.requires_grad]
    print(f"  trainable params: {len(trainable_params)}")
    assert len(trainable_params) > 0, "no trainable (LoRA) params — test can't verify gradient"

    for p in trainable_params:
        p.grad = None

    out = adapter.forward(
        stage='shape', stage_resolution=1024,
        t=t_val, t_next=t_next,
        latents=feats_b, sparse_coords=coords_b,
        next_latents=next_feats_b,
        image_cond=cond_b, neg_image_cond=neg_b,
        guidance_scale=7.5, guidance_rescale=0.5, guidance_interval=(0.0, 1.0),
        compute_log_prob=True, noise_level=0.7,
    )

    log_prob = out.log_prob                                                           # (2,)
    print(f"  log_prob shape={tuple(log_prob.shape)}, values={log_prob.detach().tolist()}")
    assert log_prob.shape == (2,), f"log_prob shape mismatch: {log_prob.shape}"

    loss = log_prob.sum()
    print(f"  loss (sum log_prob) = {float(loss):.4f}")
    loss.backward()

    n_with_grad = 0
    n_nan = 0
    n_inf = 0
    max_abs_grad = 0.0
    for p in trainable_params:
        if p.grad is None:
            continue
        n_with_grad += 1
        g = p.grad
        if g.isnan().any():
            n_nan += 1
        if g.isinf().any():
            n_inf += 1
        max_abs_grad = max(max_abs_grad, float(g.abs().max().item()))

    print(f"  params with non-None grad: {n_with_grad} / {len(trainable_params)}")
    print(f"  params with NaN grad:      {n_nan}")
    print(f"  params with Inf grad:      {n_inf}")
    print(f"  max |grad|:                {max_abs_grad:.3e}")

    ok = (
        n_with_grad == len(trainable_params)
        and n_nan == 0
        and n_inf == 0
        and max_abs_grad > 0.0
    )
    print(f"  [{'PASS' if ok else 'FAIL'}] gradient flow")
    return ok


def _build_mock_sample(
    batch_idx: int,
    coords: torch.Tensor,
    num_steps: int = 12,
    num_sde_steps: int = 1,
    seq_len: int = 4101,
    hidden: int = 1024,
    channels: int = 8,
    device: str = 'cpu',
) -> Trellis2Sample:
    """Synthetic Trellis2Sample with shape-stage fields filled for stacking."""
    coords = coords.clone()                                                           # (N_b, 4)
    coords[:, 0] = 0
    N_b = coords.shape[0]

    g = torch.Generator(device='cpu').manual_seed(1000 + batch_idx)
    all_latents   = torch.randn(num_steps + 1, N_b, channels, generator=g)            # (T, N_b, C)
    log_probs     = torch.randn(num_sde_steps, generator=g)                           # (T_log,)
    image_cond    = torch.randn(seq_len, hidden, generator=g)                         # (seq, D)
    neg_image_cond = torch.randn(seq_len, hidden, generator=g)                        # (seq, D)

    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)                               # (T+1,) shared across batch
    latent_index_map   = torch.arange(num_steps + 1, dtype=torch.long)                # (T+1,)
    log_prob_index_map = torch.full((num_steps,), -1, dtype=torch.long)               # (T,)
    log_prob_index_map[6] = 0

    s = Trellis2Sample(
        sparse_coords=coords,
        shape_all_latents=all_latents,
        shape_log_probs=log_probs,
        shape_image_cond=image_cond,
        shape_neg_image_cond=neg_image_cond,
        all_latents=all_latents,
        log_probs=log_probs,
        image_cond=image_cond,
        neg_image_cond=neg_image_cond,
        timesteps=timesteps,
        latent_index_map=latent_index_map,
        log_prob_index_map=log_prob_index_map,
    )
    return s


def _test_sample_stacking() -> bool:
    """Gate B: BaseSample.stack correctly merges two bs=1 Trellis2Samples."""
    print("\n" + "=" * 70)
    print("Gate B: Trellis2Sample stacking into bs=2 batch (mock data, CPU only)")
    print("=" * 70)

    N_A, N_B = 17239, 19305
    coords_A = torch.zeros(N_A, 4, dtype=torch.int32)
    coords_A[:, 1:] = torch.randint(0, 64, (N_A, 3), dtype=torch.int32)
    coords_B = torch.zeros(N_B, 4, dtype=torch.int32)
    coords_B[:, 1:] = torch.randint(0, 64, (N_B, 3), dtype=torch.int32)

    s0 = _build_mock_sample(0, coords_A)
    s1 = _build_mock_sample(1, coords_B)

    batch = BaseSample.stack([s0, s1])
    print(f"  batch keys: {sorted(batch.keys())}")

    checks = []

    sc = batch.get('sparse_coords')
    if sc is None:
        checks.append(("sparse_coords present", False))
    else:
        ok_shape = sc.shape == (N_A + N_B, 4)
        col0 = sc[:, 0]
        ok_col0 = bool((col0[:N_A] == 0).all().item()) and bool((col0[N_A:] == 1).all().item())
        print(f"  sparse_coords: shape={tuple(sc.shape)} (expect ({N_A+N_B}, 4))  "
              f"col0 split OK={ok_col0}")
        checks.append(("sparse_coords shape", ok_shape))
        checks.append(("sparse_coords col0 = {0, 1}", ok_col0))

    al = batch.get('all_latents')
    T = 13
    C = 8
    if al is None:
        checks.append(("all_latents present", False))
    else:
        ok = al.shape == (N_A + N_B, T, C)
        print(f"  all_latents:   shape={tuple(al.shape)} (expect ({N_A+N_B}, {T}, {C}))")
        checks.append(("all_latents shape", ok))

    lp = batch.get('log_probs')
    if lp is None:
        checks.append(("log_probs present", False))
    else:
        ok = lp.shape == (2, 1)
        print(f"  log_probs:     shape={tuple(lp.shape)} (expect (2, 1))")
        checks.append(("log_probs shape", ok))

    ts = batch.get('timesteps')
    if ts is None:
        checks.append(("timesteps present", False))
    else:
        ok = ts.shape == (2, T)
        print(f"  timesteps:     shape={tuple(ts.shape)} (expect (2, {T}))")
        checks.append(("timesteps shape", ok))

    lim = batch.get('latent_index_map')
    if lim is None:
        checks.append(("latent_index_map present", False))
    else:
        ok_shape = lim.shape == (T,)
        ok_shared = bool((lim == torch.arange(T, dtype=torch.long)).all().item())
        print(f"  latent_index_map: shape={tuple(lim.shape)} shared={ok_shared}")
        checks.append(("latent_index_map shape", ok_shape))
        checks.append(("latent_index_map shared", ok_shared))

    lpm = batch.get('log_prob_index_map')
    if lpm is None:
        checks.append(("log_prob_index_map present", False))
    else:
        ok = lpm.shape == (T - 1,)
        print(f"  log_prob_index_map: shape={tuple(lpm.shape)}")
        checks.append(("log_prob_index_map shape", ok))

    ic = batch.get('image_cond')
    if ic is None:
        checks.append(("image_cond present", False))
    else:
        ok = ic.shape == (2, 4101, 1024)
        print(f"  image_cond:    shape={tuple(ic.shape)} (expect (2, 4101, 1024))")
        checks.append(("image_cond shape", ok))

    all_ok = True
    for name, ok in checks:
        flag = 'PASS' if ok else 'FAIL'
        print(f"    [{flag}] {name}")
        all_ok = all_ok and ok

    return all_ok


def _create_dense_config(enable_gradient_checkpointing: bool = False) -> Arguments:
    """Config with target_flow_model='dense' for dense gradient testing."""
    sde_cfg = {"dynamics_type": "Flow-SDE", "noise_level": 0.7, "num_sde_steps": 1}
    model_args = ModelArguments(
        model_name_or_path=LOCAL_MODEL_PATH,
        model_type="trellis2",
        finetune_type="lora",
        lora_rank=16,
        lora_alpha=32,
        target_modules="default",
        target_components=["transformer"],
        resume_path=None,
        extra_kwargs={
            "target_flow_model": "dense",
            "dense_sde": sde_cfg,
        },
    )
    training_args = TrainingArguments(
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        ema_decay=0.0,
    )
    return Arguments(
        model_args=model_args,
        training_args=training_args,
        eval_args=EvaluationArguments(),
        scheduler_args=SchedulerArguments(
            noise_level=0.7, num_sde_steps=1, seed=_SEED,
        ),
        mixed_precision="bf16",
    )


def _test_dense_gradient_flow() -> bool:
    """Gate C: gradient flow through dense forward (bs=2, stage='dense')."""
    print("\n" + "=" * 70)
    print("Gate C: Gradient flow through dense forward (bs=2, stage='dense')")
    print("=" * 70)

    config = _create_dense_config(enable_gradient_checkpointing=True)
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    print("[setup] Loading Trellis2Adapter (target_flow_model=dense) ...")
    adapter = Trellis2Adapter(config=config, accelerator=accelerator)
    adapter.pipeline.image_encoder.to(adapter.device)

    encoded = adapter.preprocess_func(
        images=[[Image.open(TEST_IMAGE_PATH)], [Image.open(TEST_IMAGE_PATH)]]
    )
    adapter.pipeline.image_encoder.cpu()
    torch.cuda.empty_cache()

    dense_flow = adapter.pipeline.get_flow_model('dense')
    dense_flow.to(adapter.device)
    adapter.train()

    device = adapter.device
    reso = dense_flow.resolution
    in_channels = dense_flow.in_channels

    cond_b = torch.stack([
        encoded['image_cond_512'][0].to(device=device, dtype=torch.float32),
        encoded['image_cond_512'][1].to(device=device, dtype=torch.float32),
    ], dim=0)                                                                  # (2, seq, D)
    neg_b = torch.stack([
        encoded['neg_image_cond_512'][0].to(device=device, dtype=torch.float32),
        encoded['neg_image_cond_512'][1].to(device=device, dtype=torch.float32),
    ], dim=0)                                                                  # (2, seq, D)

    latents = _make_rand(
        (2, in_channels, reso, reso, reso), device, torch.float32, _SEED + 10,
    )                                                                          # (2, C, D, H, W)
    next_latents = _make_rand(
        (2, in_channels, reso, reso, reso), device, torch.float32, _SEED + 11,
    )                                                                          # (2, C, D, H, W)
    t_val  = torch.full((2,), _T_VAL,  device=device, dtype=torch.float32)     # (2,)
    t_next = torch.full((2,), _T_NEXT, device=device, dtype=torch.float32)     # (2,)

    trainable_params = [p for p in dense_flow.parameters() if p.requires_grad]
    print(f"  trainable params: {len(trainable_params)}")
    assert len(trainable_params) > 0, "no trainable (LoRA) params for dense flow model"

    for p in trainable_params:
        p.grad = None

    out = adapter.forward(
        stage='dense',
        t=t_val, t_next=t_next,
        latents=latents, next_latents=next_latents,
        image_cond=cond_b, neg_image_cond=neg_b,
        guidance_scale=7.5, guidance_rescale=0.5, guidance_interval=(0.0, 1.0),
        compute_log_prob=True, noise_level=0.7,
    )

    log_prob = out.log_prob                                                    # (2,)
    print(f"  log_prob shape={tuple(log_prob.shape)}, values={log_prob.detach().tolist()}")

    loss = log_prob.sum()
    print(f"  loss (sum log_prob) = {float(loss):.4f}")
    loss.backward()

    n_with_grad = 0
    n_nan = 0
    n_inf = 0
    max_abs_grad = 0.0
    for p in trainable_params:
        if p.grad is None:
            continue
        n_with_grad += 1
        g = p.grad
        if g.isnan().any():
            n_nan += 1
        if g.isinf().any():
            n_inf += 1
        max_abs_grad = max(max_abs_grad, float(g.abs().max().item()))

    print(f"  params with non-None grad: {n_with_grad} / {len(trainable_params)}")
    print(f"  params with NaN grad:      {n_nan}")
    print(f"  params with Inf grad:      {n_inf}")
    print(f"  max |grad|:                {max_abs_grad:.3e}")

    ok = (
        n_with_grad == len(trainable_params)
        and n_nan == 0
        and n_inf == 0
        and max_abs_grad > 0.0
    )
    print(f"  [{'PASS' if ok else 'FAIL'}] dense gradient flow")

    del adapter, dense_flow
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return ok


def main():
    t0 = time.time()
    image = Image.open(TEST_IMAGE_PATH)
    assert image is not None, f"Test image not found: {TEST_IMAGE_PATH}"

    gate_b_ok = _test_sample_stacking()
    print(f"\nGate B elapsed: {time.time()-t0:.0f}s")

    # Production config: gradient checkpointing ON (required for bs=2 + full
    # N~36k tokens + bf16 activations + gradient graph) + trainer-style
    # ``(B,)`` timestep tensors.
    adapter = _setup_adapter(enable_gradient_checkpointing=True)
    print("[setup] Building sparse_coords via dense ODE on 2 samples ...")
    t_setup = time.time()
    encoded, coords_list = _get_sparse_coords(adapter, image, n_samples=2)
    print(f"  sparse_coords[0] N_A={coords_list[0].shape[0]}  "
          f"sparse_coords[1] N_B={coords_list[1].shape[0]}  [{time.time()-t_setup:.0f}s]")

    gate_a_ok = _test_gradient_flow(
        adapter, encoded, coords_list,
        truncate_to=0,           # full N per sample
        use_batched_t=True,      # (B,) timestep tensor (Trainer-style)
    )

    del adapter
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    gate_c_ok = _test_dense_gradient_flow()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Gate A (gradient flow, full N, (B,) t, ckpt ON): "
          f"{'PASS' if gate_a_ok else 'FAIL'}")
    print(f"  Gate B (sample stacking):                        "
          f"{'PASS' if gate_b_ok else 'FAIL'}")
    print(f"  Gate C (dense gradient flow):                    "
          f"{'PASS' if gate_c_ok else 'FAIL'}")

    all_ok = gate_a_ok and gate_b_ok and gate_c_ok
    print(f"\n{'ALL PASS' if all_ok else 'FAILED'}   total: {time.time()-t0:.0f}s")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
