"""Verify FlowEdit premise: edited renders are closer to condition images than raw renders.

Computes DINOv2 cosine similarity between:
  - sim(raw_render, condition_image)
  - sim(flowedit_edited, condition_image)

Success criterion: delta > 0 for > 80% of samples.

Usage (in debug pod with FlowEdit server running):
    python scripts/verify_flowedit_premise.py \
        --server http://localhost:8092 \
        --dataset /local-ssd/alphaimages_v2_formatted/train \
        --weights /local-ssd/pretrained_weights \
        --num-samples 20
"""
import argparse
import base64
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="http://localhost:8092")
    parser.add_argument("--dataset", type=str, default="/local-ssd/alphaimages_v2_formatted/train")
    parser.add_argument("--weights", type=str, default="/local-ssd/pretrained_weights")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--prompt", type=str, default="Rotate the camera. White background.")
    parser.add_argument("--cfg-tgt", type=float, default=7.5)
    parser.add_argument("--cfg-src", type=float, default=-7.5)
    parser.add_argument("--n-max", type=int, default=28)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-views", type=int, default=16,
                        help="Number of rendered views per sample (pick random frame)")
    return parser.parse_args()


# ============================================================================
# FlowEdit API (copied from vllm-omni eval script)
# ============================================================================

def img_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call_flowedit(server: str, source: Image.Image, condition: Image.Image,
                  prompt: str, cfg_tgt: float, cfg_src: float,
                  n_max: int, steps: int, seed: int) -> Image.Image:
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_to_b64(source)}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_to_b64(condition)}"}},
            ]
        }],
        "extra_body": {
            "num_inference_steps": steps,
            "guidance_scale": 1,
            "true_cfg_scale": cfg_tgt,
            "true_cfg_scale_src": cfg_src,
            "n_max": n_max,
            "seed": seed,
        }
    }
    resp = requests.post(f"{server}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    b64_url = data["choices"][0]["message"]["content"][0]["image_url"]["url"]
    b64_str = b64_url.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")


# ============================================================================
# DINOv2 similarity
# ============================================================================

@torch.no_grad()
def load_dino(weights_dir: str, device: str = "cuda"):
    """Load DINOv2 ViT-L/14 from local weights."""
    from transformers import AutoModel, AutoImageProcessor

    model_path = Path(weights_dir) / "TRELLIS.2-4B" / "dinov2_vitl14_reg"
    if not model_path.exists():
        model_path = "facebook/dinov2-large"
        print(f"Local DINOv2 not found, using HuggingFace: {model_path}")

    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).eval().to(device)
    return processor, model


@torch.no_grad()
def dino_embed(processor, model, img: Image.Image, device: str = "cuda") -> torch.Tensor:
    """Get CLS token embedding from DINOv2."""
    inputs = processor(images=img, return_tensors="pt").to(device)
    outputs = model(**inputs)
    return F.normalize(outputs.last_hidden_state[:, 0], dim=-1)  # (1, D)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a * b).sum().item()


# ============================================================================
# Data loading: simulate what TargetImageBuffer does
# ============================================================================

def load_sample_pair(dataset_dir: Path, sample_idx: int, rng: np.random.Generator):
    """Load condition image and a random rendered frame for one sample.

    Expected dataset layout:
        <dataset_dir>/<uid>/rgba.png          (condition image)
        <dataset_dir>/<uid>/renders/           (multi-view renders, optional)

    If no renders exist, we generate a synthetic "raw render" by rotating/cropping.
    For this verification we need the ACTUAL training pipeline renders — run this
    after a rollout has been saved, or use pre-rendered data.
    """
    dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if sample_idx >= len(dirs):
        return None, None, None

    sample_dir = dirs[sample_idx]
    uid = sample_dir.name

    # Condition image
    cond_path = sample_dir / "rgba.png"
    if not cond_path.exists():
        cond_path = sample_dir / "image.png"
    if not cond_path.exists():
        return None, None, uid

    cond = Image.open(cond_path).convert("RGB")

    # Rendered frames (from training rollout output or pre-rendered)
    renders_dir = sample_dir / "renders"
    if renders_dir.exists():
        frames = sorted(renders_dir.glob("*.png"))
        if frames:
            frame_idx = rng.integers(0, len(frames))
            raw = Image.open(frames[frame_idx]).convert("RGB")
            return cond, raw, uid

    # Fallback: use condition image as raw (this means delta should be ~0)
    return cond, None, uid


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading DINOv2 from {args.weights}...")
    processor, model = load_dino(args.weights, device)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"ERROR: Dataset not found at {dataset_dir}")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    results = []

    sample_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    indices = rng.choice(len(sample_dirs), size=min(args.num_samples, len(sample_dirs)), replace=False)

    for i, idx in enumerate(tqdm(indices, desc="Evaluating")):
        cond, raw, uid = load_sample_pair(dataset_dir, idx, rng)
        if cond is None or raw is None:
            continue

        # Compute embeddings
        emb_cond = dino_embed(processor, model, cond, device)

        # Raw render similarity
        emb_raw = dino_embed(processor, model, raw, device)
        sim_raw = cosine_sim(emb_cond, emb_raw)

        # FlowEdit edited
        try:
            edited = call_flowedit(
                args.server, raw, cond,
                args.prompt, args.cfg_tgt, args.cfg_src,
                args.n_max, args.steps, args.seed + i,
            )
        except Exception as e:
            print(f"  FlowEdit failed for {uid}: {e}")
            continue

        emb_edited = dino_embed(processor, model, edited, device)
        sim_edited = cosine_sim(emb_cond, emb_edited)
        delta = sim_edited - sim_raw

        results.append({
            "uid": uid,
            "sim_raw": sim_raw,
            "sim_edited": sim_edited,
            "delta": delta,
        })
        print(f"  [{i+1}/{args.num_samples}] {uid}: "
              f"raw={sim_raw:.4f}, edited={sim_edited:.4f}, delta={delta:+.4f}")

    # Summary
    if not results:
        print("\nERROR: No valid results. Check dataset path and FlowEdit server.")
        sys.exit(1)

    deltas = [r["delta"] for r in results]
    positive_ratio = sum(1 for d in deltas if d > 0) / len(deltas)
    mean_delta = np.mean(deltas)

    print(f"\n{'='*60}")
    print(f"Results ({len(results)} samples):")
    print(f"  Mean delta:      {mean_delta:+.4f}")
    print(f"  Positive ratio:  {positive_ratio:.1%} ({sum(1 for d in deltas if d > 0)}/{len(deltas)})")
    print(f"  Mean sim_raw:    {np.mean([r['sim_raw'] for r in results]):.4f}")
    print(f"  Mean sim_edited: {np.mean([r['sim_edited'] for r in results]):.4f}")
    print(f"{'='*60}")

    if positive_ratio >= 0.8:
        print("✓ PASS: FlowEdit premise holds (>80% positive delta)")
    else:
        print("✗ FAIL: FlowEdit premise NOT confirmed (<80% positive delta)")
        print("  Consider tuning cfg_tgt, n_max, steps parameters.")
        sys.exit(1)


if __name__ == "__main__":
    main()
