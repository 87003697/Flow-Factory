#!/usr/bin/env python
# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# scripts/data/score_ltx_syn_data.py
"""Offline scoring for LTX2 synthesised samples (ltx_syn_data subset).

Each sample stores cached video / audio latents under ``latents/`` and
``audio_latents/``; this script decodes them through the LTX2 ``vae`` /
``audio_vae`` / ``vocoder`` and scores the resulting video+audio with
ImageBind, CLAP and (optionally) UnifiedReward 2.0 APS via vLLM.

Run via ``scripts/data/score_ltx_syn_data.sh`` which wraps ``torchrun``;
each rank shards the dataset with ``samples[rank::world_size]`` and
writes its own ``scores_rank{r}.jsonl``.  Rank 0 then merges into
``scores.jsonl`` + ``summary.json`` after ``dist.barrier()``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline

from flow_factory.hparams import MultiRewardArguments, RewardArguments
from flow_factory.rewards.abc import BaseRewardModel, RewardModelOutput
from flow_factory.rewards.registry import get_reward_model_class
from flow_factory.samples import T2AVSample
from flow_factory.utils.base import filter_kwargs, move_tensors_to_device
from flow_factory.utils.image import standardize_image_batch
from flow_factory.utils.video import standardize_video_batch
from flow_factory.utils.audio import standardize_audio_batch


# ============================== Config & CLI ==============================


@dataclass
class RuntimeConfig:
    """Runtime view derived from yaml + CLI args."""

    dataset_root: Path
    out_dir: Path
    prompt_field: str
    ltx2_repo: str
    vae_dtype: torch.dtype
    rewards_yaml: List[Dict[str, Any]]
    requested_rewards: List[str]
    limit: Optional[int]
    keep_decoded: str
    keep_decoded_first_n: int
    batch_size: int
    save_first_frame_png: bool
    decode_timestep: float
    decode_noise_scale: Optional[float]


_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score LTX2 synthesised samples by decoding cached latents and "
        "running ImageBind / CLAP / UnifiedReward APS rewards.",
    )
    parser.add_argument("--config", required=True, type=str, help="Path to yaml config.")
    parser.add_argument(
        "--rewards",
        type=str,
        default=None,
        help="Comma-separated reward names to enable (subset of yaml `rewards`). "
        "If omitted, all rewards in yaml are used.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many samples (applied before rank sharding).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-sample is the natural unit; this is reserved for future "
        "microbatching and currently must be 1.",
    )
    parser.add_argument(
        "--keep_decoded",
        choices=["all", "first_n", "none"],
        default="first_n",
        help="Disk policy for decoded mp4/wav. 'first_n' keeps only the first "
        "N samples per rank (controlled by --keep_decoded_first_n).",
    )
    parser.add_argument(
        "--keep_decoded_first_n",
        type=int,
        default=4,
        help="When --keep_decoded=first_n, keep this many decoded samples per rank.",
    )
    parser.add_argument(
        "--vae_dtype",
        choices=["bfloat16", "float16", "float32"],
        default=None,
        help="Override yaml model.vae_dtype.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Override yaml data.out_dir.",
    )
    parser.add_argument(
        "--save_first_frame_png",
        action="store_true",
        help="Save the first decoded frame as PNG for visual verification.",
    )
    return parser.parse_args(argv)


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config yaml not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(
            f"expected mapping at top of config yaml {cfg_path}, got {type(cfg).__name__}"
        )
    return cfg


def build_runtime_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> RuntimeConfig:
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    rewards_yaml = cfg.get("rewards", [])

    dataset_root = Path(data_cfg["dataset_root"]).resolve()
    out_dir = Path(args.out_dir or data_cfg["out_dir"]).resolve()
    prompt_field = data_cfg.get("prompt_field", "generation_prompt")

    vae_dtype_str = args.vae_dtype or model_cfg.get("vae_dtype", "bfloat16")
    if vae_dtype_str not in _DTYPE_MAP:
        raise ValueError(
            f"unsupported vae_dtype {vae_dtype_str!r}; expected one of {sorted(_DTYPE_MAP)}"
        )

    if not isinstance(rewards_yaml, list) or not rewards_yaml:
        raise ValueError("`rewards` in yaml must be a non-empty list")

    yaml_names = [r["name"] for r in rewards_yaml]
    if args.rewards:
        requested = [n.strip() for n in args.rewards.split(",") if n.strip()]
        unknown = [n for n in requested if n not in yaml_names]
        if unknown:
            raise ValueError(
                f"--rewards refers to names not in yaml: {unknown}. "
                f"Available: {yaml_names}"
            )
    else:
        requested = list(yaml_names)

    if args.batch_size != 1:
        raise NotImplementedError(
            "Per-sample microbatching is the v1 design; --batch_size > 1 not supported."
        )

    decode_cfg = model_cfg.get("decode", {}) or {}
    return RuntimeConfig(
        dataset_root=dataset_root,
        out_dir=out_dir,
        prompt_field=prompt_field,
        ltx2_repo=model_cfg.get("ltx2_repo", "Lightricks/LTX-2"),
        vae_dtype=_DTYPE_MAP[vae_dtype_str],
        rewards_yaml=rewards_yaml,
        requested_rewards=requested,
        limit=args.limit,
        keep_decoded=args.keep_decoded,
        keep_decoded_first_n=args.keep_decoded_first_n,
        batch_size=args.batch_size,
        save_first_frame_png=args.save_first_frame_png,
        decode_timestep=float(decode_cfg.get("decode_timestep", 0.0)),
        decode_noise_scale=decode_cfg.get("decode_noise_scale"),
    )


# ============================== Distributed Setup ==============================


def setup_distributed() -> Tuple[int, int, torch.device]:
    """Init NCCL process group when run via torchrun, fall back to single-rank otherwise."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LTX2 VAE decode.")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices visible. "
            "Did you forget CUDA_VISIBLE_DEVICES?"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank, world_size, device


# ============================== Self-checks ==============================


def selfcheck_dependencies(rewards_yaml: List[Dict[str, Any]], requested: Set[str]) -> None:
    """Fail fast on missing system deps before loading the heavy LTX2 pipeline."""
    import importlib

    for module_name in ("av", "soundfile"):
        if importlib.util.find_spec(module_name) is None:
            raise ImportError(
                f"missing required module {module_name!r}; "
                f"install with `pip install {module_name}` in the flow-factory env."
            )

    unified_cfgs = [
        r for r in rewards_yaml
        if r["name"] in requested and "unified_reward" in r.get("reward_model", "")
    ]
    if not unified_cfgs:
        return

    if importlib.util.find_spec("openai") is None:
        raise ImportError(
            "unified_reward_* requires the `openai` package; "
            "install with `pip install openai`."
        )

    for cfg in unified_cfgs:
        api_base = cfg.get("api_base_url", "http://localhost:8080/v1")
        url = api_base.rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            raise RuntimeError(
                f"vLLM server for reward {cfg['name']!r} is not reachable at {url}. "
                f"Start it via `bash scripts/data/start_unified_reward_server.sh` first.\n"
                f"Underlying error: {exc!r}"
            ) from exc

        served = {item["id"] for item in payload.get("data", [])}
        wanted = cfg.get("vlm_model", "UnifiedReward")
        if wanted not in served:
            raise RuntimeError(
                f"vLLM at {url} does not serve model {wanted!r} (got {sorted(served)}). "
                f"Update yaml `vlm_model` or restart the server with --served-model-name."
            )


# ============================== Pipeline ==============================


def build_ltx2_pipeline(
    repo: str,
    vae_dtype: torch.dtype,
    device: torch.device,
) -> LTX2Pipeline:
    """Load only the VAE + audio_vae + vocoder + scheduler from LTX2.

    Skipped modules (passed as ``None`` to ``from_pretrained``):
      - ``transformer`` (~few-tens-of-GB LTX-2 video transformer)
      - ``text_encoder`` / ``tokenizer`` / ``processor`` (Gemma3 27B family)
      - ``connectors`` (depends on text_encoder hidden size)

    Passing ``None`` here is the documented way to tell diffusers to neither
    download nor instantiate those subfolders, which avoids ~140 GB of
    unrelated weights for an offline decoder-only run.
    """
    pipeline = LTX2Pipeline.from_pretrained(
        repo,
        low_cpu_mem_usage=False,
        transformer=None,
        text_encoder=None,
        tokenizer=None,
        processor=None,
        connectors=None,
    )
    pipeline.vae = pipeline.vae.to(device=device, dtype=vae_dtype)  # vae on rank GPU
    pipeline.audio_vae = pipeline.audio_vae.to(device=device, dtype=vae_dtype)  # audio_vae on rank GPU
    pipeline.vocoder = pipeline.vocoder.to(device=device, dtype=torch.float32)  # vocoder fp32 to avoid clipping
    pipeline.vae.enable_tiling()
    pipeline.vae.use_framewise_decoding = True
    return pipeline


# ============================== Latent Loading & Geometry ==============================


def _check_geometry(
    sample_id: str,
    video_dict: Dict[str, Any],
    audio_dict: Dict[str, Any],
    vae_spatial: int,
    vae_temporal: int,
) -> Tuple[int, int, int, float]:
    """Resolve pixel-space (height, width, num_frames, fps) and validate divisibility."""
    latents = video_dict.get("latents")
    if latents is None or latents.ndim != 4:
        raise ValueError(
            f"sample {sample_id!r}: video latent must be (C,F,H,W) tensor, "
            f"got shape {None if latents is None else tuple(latents.shape)}"
        )
    _, latent_f, latent_h, latent_w = latents.shape
    declared_f = video_dict.get("num_frames")
    declared_h = video_dict.get("height")
    declared_w = video_dict.get("width")
    if declared_f != latent_f or declared_h != latent_h or declared_w != latent_w:
        raise ValueError(
            f"sample {sample_id!r}: latent shape {(latent_f, latent_h, latent_w)} "
            f"disagrees with declared (num_frames, height, width)="
            f"{(declared_f, declared_h, declared_w)}"
        )

    fps = float(video_dict.get("fps", 24.0))
    if fps <= 0:
        raise ValueError(f"sample {sample_id!r}: invalid fps={fps}")

    height_px = latent_h * vae_spatial
    width_px = latent_w * vae_spatial
    num_frames_px = (latent_f - 1) * vae_temporal + 1

    if (num_frames_px - 1) % vae_temporal != 0:
        raise ValueError(
            f"sample {sample_id!r}: derived num_frames_px={num_frames_px} not "
            f"compatible with vae_temporal={vae_temporal}"
        )
    if height_px % vae_spatial != 0 or width_px % vae_spatial != 0:
        raise ValueError(
            f"sample {sample_id!r}: derived (h,w)=({height_px},{width_px}) not "
            f"divisible by vae_spatial={vae_spatial}"
        )

    audio_latents = audio_dict.get("latents")
    if audio_latents is None or audio_latents.ndim != 3:
        raise ValueError(
            f"sample {sample_id!r}: audio latent must be (C,L,M) tensor, "
            f"got shape {None if audio_latents is None else tuple(audio_latents.shape)}"
        )
    return height_px, width_px, num_frames_px, fps


def load_sample_latents(
    sample_meta: Dict[str, Any],
    dataset_root: Path,
    device: torch.device,
    vae_dtype: torch.dtype,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Load video + audio latents for one sample and place them on ``device``."""
    media_path = sample_meta.get("media_path")
    if not media_path:
        raise KeyError("sample_meta missing 'media_path'")
    sample_id = Path(media_path).stem

    video_path = dataset_root / sample_meta["latent_path"]
    audio_path = dataset_root / sample_meta["audio_latent_path"]
    if not video_path.is_file():
        raise FileNotFoundError(f"sample {sample_id}: video latent missing at {video_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"sample {sample_id}: audio latent missing at {audio_path}")

    video_dict = torch.load(video_path, map_location="cpu", weights_only=False)  # dict on CPU
    audio_dict = torch.load(audio_path, map_location="cpu", weights_only=False)  # dict on CPU
    video_dict["latents"] = video_dict["latents"].to(device=device, dtype=vae_dtype)  # (C,F,H,W) on rank GPU
    audio_dict["latents"] = audio_dict["latents"].to(device=device, dtype=vae_dtype)  # (C,L,M) on rank GPU
    return video_dict, audio_dict, sample_id


# ============================== Decoding ==============================


@torch.no_grad()
def decode_ltx2_latent_pair(
    pipeline: LTX2Pipeline,
    video_dict: Dict[str, Any],
    audio_dict: Dict[str, Any],
    sample_id: str,
    decode_timestep: float = 0.0,
    decode_noise_scale: Optional[float] = None,
) -> Tuple[np.ndarray, torch.Tensor, int, int, int, int, int]:
    """Decode cached latents for one sample.

    The video is returned as a numpy ndarray with the (F, H, W, C) layout that
    ``standardize_video_batch`` auto-recognises for ndarrays (the equivalent
    auto-detection only fires for ndarrays — torch tensors are assumed to be
    (T, C, H, W) by that helper).

    Returns:
        video_uint8: (F, H, W, C) uint8 [0,255] ndarray on CPU
        audio_stereo: (C_audio, T_samples) float32 on CPU
        sample_rate: int (vocoder output rate, typically 24000)
        fps: int (rounded from latent dict)
        num_frames_px: int
        height_px: int
        width_px: int
    """
    vae = pipeline.vae
    audio_vae = pipeline.audio_vae
    vocoder = pipeline.vocoder
    vae_spatial = pipeline.vae_spatial_compression_ratio  # 32
    vae_temporal = pipeline.vae_temporal_compression_ratio  # 8

    height_px, width_px, num_frames_px, fps = _check_geometry(
        sample_id, video_dict, audio_dict, vae_spatial, vae_temporal,
    )

    video_latents_5d = video_dict["latents"].unsqueeze(0)  # (1, C, F, H, W) unpacked + normalized
    audio_latents_4d = audio_dict["latents"].unsqueeze(0)  # (1, C, L, M) unpacked + normalized

    # Video decoding follows pipeline_ltx2.py L1444-1468 exactly:
    #   noise injection on STILL-NORMALIZED latents -> denormalize -> vae.decode.
    # ``_denormalize_latents`` reshapes mean/std as (1,C,1,1,1) so the unpacked
    # 5D form is the right input; we never need to pack/unpack for video.
    vid = video_latents_5d  # (1, C, F, H, W) normalized, float
    if vae.config.timestep_conditioning:
        noise = torch.randn_like(vid)  # (1, C, F, H, W) decode noise
        vae_timestep = torch.tensor([decode_timestep], device=vid.device, dtype=vid.dtype)  # (1,)
        dns = decode_timestep if decode_noise_scale is None else decode_noise_scale
        dns_t = torch.tensor([dns], device=vid.device, dtype=vid.dtype)[:, None, None, None, None]  # (1,1,1,1,1)
        vid = (1 - dns_t) * vid + dns_t * noise  # (1, C, F, H, W) noisy normalized
    else:
        vae_timestep = None
    vid = pipeline._denormalize_latents(
        vid, vae.latents_mean, vae.latents_std, vae.config.scaling_factor,
    )  # (1, C, F, H, W) denormalized
    vid = vid.to(vae.dtype)
    video = vae.decode(vid, vae_timestep, return_dict=False)[0]  # (1, C_pix, F, H_px, W_px)
    video_pp = pipeline.video_processor.postprocess_video(video, output_type="np")  # (1, F, H_px, W_px, C) float [0,1]

    # Audio: ``_denormalize_audio_latents`` expects packed shape (B, L, C*M)
    # because audio_vae.latents_mean/std are stored as a flat (C*M,) tensor.
    # The official pipeline order (pipeline_ltx2.py L1432-1435) is therefore
    # pack -> denormalize -> unpack -> decode -> vocoder.
    audio_packed = pipeline._pack_audio_latents(audio_latents_4d)  # (1, L, C*M) packed
    audio_packed = pipeline._denormalize_audio_latents(
        audio_packed, audio_vae.latents_mean, audio_vae.latents_std,
    )  # (1, L, C*M) denormalized
    _, _, latent_l, latent_m = audio_latents_4d.shape  # (1, C, L, M)
    aud = pipeline._unpack_audio_latents(audio_packed, latent_l, num_mel_bins=latent_m)  # (1, C, L, M)
    aud = aud.to(audio_vae.dtype)
    mel = audio_vae.decode(aud, return_dict=False)[0]  # (1, C_audio, num_mel, T_mel)
    waveform = vocoder(mel.to(vocoder.dtype))  # (1, C_audio, T_samples)
    sample_rate = int(vocoder.config.output_sampling_rate)

    video_uint8 = (np.clip(video_pp[0], 0.0, 1.0) * 255.0).round().astype(np.uint8)  # (F, H_px, W_px, C) ndarray
    audio_stereo = waveform[0].detach().cpu().float().clamp_(-1.0, 1.0)  # (C_audio, T_samples)

    return (
        video_uint8,
        audio_stereo,
        sample_rate,
        int(round(fps)),
        num_frames_px,
        height_px,
        width_px,
    )


def save_decoded_media(
    video_uint8: np.ndarray,
    audio_stereo: torch.Tensor,
    sample_rate: int,
    fps: int,
    out_mp4: Path,
    out_wav: Path,
    save_first_frame_png: bool,
) -> None:
    """Mux mp4 (with stereo audio) via PyAV and write a mono wav for reuse."""
    import soundfile as sf

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        video=video_uint8,  # (F, H, W, C) uint8 ndarray
        fps=fps,
        audio=audio_stereo,
        audio_sample_rate=sample_rate,
        output_path=str(out_mp4),
    )

    if audio_stereo.ndim == 2 and audio_stereo.shape[0] >= 2:
        wave_mono = audio_stereo.mean(dim=0)  # (T_samples,)
    elif audio_stereo.ndim == 2:
        wave_mono = audio_stereo[0]  # (T_samples,)
    else:
        wave_mono = audio_stereo  # (T_samples,)
    sf.write(str(out_wav), wave_mono.numpy(), sample_rate, subtype="PCM_16")

    if save_first_frame_png:
        from PIL import Image as PILImage

        png_path = out_mp4.with_suffix(".first_frame.png")
        PILImage.fromarray(video_uint8[0]).save(png_path)


def audio_stereo_to_mono(audio_stereo: torch.Tensor) -> torch.Tensor:
    """(C, T) -> (T,) by averaging channels; passes through if already mono/1D."""
    if audio_stereo.ndim == 1:
        return audio_stereo
    if audio_stereo.shape[0] == 1:
        return audio_stereo[0]
    return audio_stereo.mean(dim=0)


# ============================== Reward Models ==============================


def _rewrite_device(reward_cfg_dict: Dict[str, Any], local_rank: int) -> Dict[str, Any]:
    """Replace device='cuda' with the rank-specific cuda:{local_rank}."""
    out = dict(reward_cfg_dict)
    dev = out.get("device", "cuda")
    if dev == "cuda":
        out["device"] = f"cuda:{local_rank}"
    return out


def build_reward_models(
    cfg_rewards_yaml: List[Dict[str, Any]],
    requested_names: List[str],
    local_rank: int,
    accelerator: Accelerator,
) -> Tuple[Dict[str, BaseRewardModel], Dict[str, RewardArguments]]:
    """Instantiate the requested reward models on this rank's GPU."""
    by_name = {r["name"]: r for r in cfg_rewards_yaml}
    selected = [_rewrite_device(by_name[n], local_rank) for n in requested_names]
    multi_args = MultiRewardArguments.from_dict(selected)

    reward_models: Dict[str, BaseRewardModel] = {}
    reward_configs: Dict[str, RewardArguments] = {}
    for cfg in multi_args.reward_configs:
        cls = get_reward_model_class(cfg.reward_model)
        model = cls(config=cfg, accelerator=accelerator)
        reward_models[cfg.name] = model
        reward_configs[cfg.name] = cfg
    return reward_models, reward_configs


# ============================== Sample Assembly & Scoring ==============================


_MEDIA_FIELDS = {"image", "video", "audio", "condition_images", "condition_videos"}


def _convert_media_for_model(batch_input: Dict[str, Any], model: BaseRewardModel) -> Dict[str, Any]:
    """Mirror RewardProcessor._convert_media_format for a single batch."""
    output_type = "pt" if getattr(model, "use_tensor_inputs", False) else "pil"
    result: Dict[str, Any] = {}
    for k, v in batch_input.items():
        if k not in _MEDIA_FIELDS or v is None:
            result[k] = v
            continue
        if k == "image":
            result[k] = standardize_image_batch(v, output_type=output_type)
        elif k == "video":
            result[k] = standardize_video_batch(v, output_type=output_type)
        elif k == "audio":
            audio_out = "pt" if output_type == "pt" else "np"
            result[k] = standardize_audio_batch(v, output_type=audio_out)
        elif k == "condition_images":
            result[k] = [standardize_image_batch(imgs, output_type=output_type) for imgs in v]
        elif k == "condition_videos":
            result[k] = [standardize_video_batch(videos, output_type=output_type) for videos in v]
    return result


def build_t2av_sample(
    prompt: str,
    video_uint8: np.ndarray,
    audio_mono: torch.Tensor,
    sample_rate: int,
) -> T2AVSample:
    """Build a single T2AVSample suitable for reward inputs.

    The video is passed in as a ``(F, H, W, C) uint8`` ndarray so that
    ``standardize_video_batch`` (used inside the reward processors) auto-
    detects the (T, H, W, C) layout. Torch tensors are NOT used here because
    that helper assumes torch tensors are already (T, C, H, W).
    """
    sample = T2AVSample(
        prompt=prompt,
        video=video_uint8,  # (F, H, W, C) uint8 ndarray -> standardized later per-reward
        audio=audio_mono.unsqueeze(0),  # (1, T_samples) — promoted to mono channel
        audio_sample_rate=sample_rate,
    )
    sample.extra_kwargs["audio_sample_rate"] = [sample_rate]
    return sample


@torch.no_grad()
def score_sample(
    sample: T2AVSample,
    reward_models: Dict[str, BaseRewardModel],
) -> Dict[str, Any]:
    """Score a single sample with each enabled reward model.

    For UnifiedReward we collect axis sub-scores (e.g. alignment / physics /
    style) from ``RewardModelOutput.extra_info`` and stash them under
    ``"<name>_axes"``.
    """
    results: Dict[str, Any] = {}
    for name, model in reward_models.items():
        fields = filter_kwargs(model.__call__, **sample)
        batch_input: Dict[str, Any] = {}
        for k in fields:
            value = getattr(sample, k, None)
            if value is None:
                continue
            batch_input[k] = [value]
        if "audio_sample_rate" in fields:
            batch_input["audio_sample_rate"] = [sample.audio_sample_rate]
        batch_input = _convert_media_for_model(batch_input, model)
        batch_input = move_tensors_to_device(batch_input, model.device)

        output = model(**batch_input)
        rewards = output.rewards if isinstance(output, RewardModelOutput) else output
        rewards = torch.as_tensor(rewards, dtype=torch.float32).flatten().cpu()
        if rewards.numel() != 1:
            raise RuntimeError(
                f"reward {name!r} returned {rewards.numel()} scores for a single sample; expected 1."
            )
        score_value = float(rewards.item())
        if math.isnan(score_value):
            results[name] = None  # NaN serialised as null; merged with global mean later
        else:
            results[name] = score_value

        if isinstance(output, RewardModelOutput) and output.extra_info:
            axes: Dict[str, Optional[float]] = {}
            for axis_key, axis_tensor in output.extra_info.items():
                if not isinstance(axis_tensor, torch.Tensor):
                    continue
                axis_value = float(axis_tensor.flatten()[0].item())
                axes[axis_key] = None if math.isnan(axis_value) else axis_value
            if axes:
                results[f"{name}_axes"] = axes
    return results


# ============================== JSONL & Merge ==============================


def append_jsonl_line(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def merge_rank_jsonls(out_dir: Path, world_size: int) -> Path:
    merged: List[Dict[str, Any]] = []
    for r in range(world_size):
        rank_path = out_dir / f"scores_rank{r}.jsonl"
        if not rank_path.is_file():
            raise FileNotFoundError(
                f"missing rank shard {rank_path}; cannot merge results. "
                f"Re-run rank {r} only or rerun the whole job."
            )
        with rank_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    merged.append(json.loads(line))

    merged.sort(key=lambda rec: rec["sample_id"])
    merged_path = out_dir / "scores.jsonl"
    with merged_path.open("w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
    return merged_path


def write_summary(scores_jsonl: Path, summary_path: Path) -> Dict[str, Any]:
    """Aggregate per-reward statistics, filling NaNs with the global valid-sample mean."""
    records: List[Dict[str, Any]] = []
    with scores_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    reward_names: Set[str] = set()
    for rec in records:
        reward_names.update(rec.get("scores", {}).keys())
    reward_names = {n for n in reward_names if not n.endswith("_axes")}

    summary: Dict[str, Any] = {"num_samples": len(records), "rewards": {}}
    for name in sorted(reward_names):
        raw_values: List[Optional[float]] = []
        for rec in records:
            v = rec.get("scores", {}).get(name)
            raw_values.append(None if v is None else float(v))
        valid = [v for v in raw_values if v is not None]
        n_valid = len(valid)
        n_filled = len(raw_values) - n_valid
        if n_valid == 0:
            summary["rewards"][name] = {
                "n_total": len(raw_values), "n_valid": 0, "n_filled": n_filled,
                "mean": None, "std": None, "min": None, "max": None,
            }
            continue
        valid_arr = np.asarray(valid, dtype=np.float64)
        fill_value = float(valid_arr.mean())
        for rec in records:
            scores = rec.setdefault("scores", {})
            if scores.get(name) is None:
                scores[name] = fill_value
                rec.setdefault("filled", {})[name] = True
        all_arr = np.asarray(
            [rec["scores"][name] for rec in records], dtype=np.float64,
        )
        summary["rewards"][name] = {
            "n_total": int(all_arr.size),
            "n_valid": int(n_valid),
            "n_filled": int(n_filled),
            "mean": float(all_arr.mean()),
            "std": float(all_arr.std()),
            "min": float(all_arr.min()),
            "max": float(all_arr.max()),
        }

    with scores_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


# ============================== Main ==============================


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    runtime = build_runtime_config(args, cfg)

    rank, world_size, device = setup_distributed()
    is_main = rank == 0

    if is_main:
        runtime.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[rank {rank}/{world_size}] device={device} out_dir={runtime.out_dir}")

    selfcheck_dependencies(runtime.rewards_yaml, set(runtime.requested_rewards))

    dataset_json = runtime.dataset_root / "dataset.json"
    if not dataset_json.is_file():
        raise FileNotFoundError(f"dataset.json not found under {runtime.dataset_root}")
    with dataset_json.open("r", encoding="utf-8") as f:
        full_meta: List[Dict[str, Any]] = json.load(f)
    if runtime.limit is not None:
        full_meta = full_meta[: runtime.limit]
    local_meta = full_meta[rank::world_size]

    if is_main:
        print(
            f"[rank {rank}] dataset total={len(full_meta)} -> local={len(local_meta)} "
            f"(rewards={runtime.requested_rewards}, vae_dtype={runtime.vae_dtype})"
        )

    pipeline = build_ltx2_pipeline(runtime.ltx2_repo, runtime.vae_dtype, device)

    # Each rank holds its own Accelerator() instance. Even with NCCL initialised
    # globally, we never call any cross-rank gather here (all rewards used are
    # pointwise) and the only barrier is the explicit dist.barrier() below.
    accelerator = Accelerator()
    reward_models, _reward_configs = build_reward_models(
        runtime.rewards_yaml, runtime.requested_rewards, int(os.environ.get("LOCAL_RANK", "0")),
        accelerator,
    )

    rank_out_path = runtime.out_dir / f"scores_rank{rank}.jsonl"
    if rank_out_path.is_file():
        rank_out_path.unlink()  # v1: no resume; nuke stale shards.

    decoded_dir = runtime.out_dir / "decoded"
    decoded_dir.mkdir(parents=True, exist_ok=True)

    for local_idx, sample_meta in enumerate(local_meta):
        video_dict, audio_dict, sample_id = load_sample_latents(
            sample_meta, runtime.dataset_root, device, runtime.vae_dtype,
        )
        prompt = sample_meta.get(runtime.prompt_field) or sample_meta.get("generation_prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(
                f"sample {sample_id!r}: prompt field {runtime.prompt_field!r} is empty"
            )

        video_uint8, audio_stereo, sample_rate, fps, num_frames_px, height_px, width_px = (
            decode_ltx2_latent_pair(
                pipeline,
                video_dict,
                audio_dict,
                sample_id,
                decode_timestep=runtime.decode_timestep,
                decode_noise_scale=runtime.decode_noise_scale,
            )
        )

        if local_idx == 0:
            print(
                f"[rank {rank}] first sample {sample_id}: "
                f"video={tuple(video_uint8.shape)} audio={tuple(audio_stereo.shape)} "
                f"sr={sample_rate} fps={fps} (h,w,F)=({height_px},{width_px},{num_frames_px})"
            )

        keep = (
            runtime.keep_decoded == "all"
            or (runtime.keep_decoded == "first_n" and local_idx < runtime.keep_decoded_first_n)
        )
        out_mp4 = decoded_dir / f"{sample_id}.mp4"
        out_wav = decoded_dir / f"{sample_id}.wav"
        if keep:
            save_decoded_media(
                video_uint8, audio_stereo, sample_rate, fps,
                out_mp4, out_wav,
                save_first_frame_png=runtime.save_first_frame_png and local_idx == 0,
            )
        decoded_video_str = str(out_mp4) if keep else None
        decoded_audio_str = str(out_wav) if keep else None

        audio_mono = audio_stereo_to_mono(audio_stereo)  # (T_samples,)
        sample = build_t2av_sample(prompt, video_uint8, audio_mono, sample_rate)
        scores = score_sample(sample, reward_models)

        record = {
            "sample_id": sample_id,
            "rank": rank,
            "prompt": prompt,
            "media_path": sample_meta.get("media_path"),
            "decoded_video": decoded_video_str,
            "decoded_audio": decoded_audio_str,
            "fps": fps,
            "sample_rate": sample_rate,
            "num_frames": num_frames_px,
            "height": height_px,
            "width": width_px,
            "scores": scores,
        }
        append_jsonl_line(rank_out_path, record)
        print(
            f"[rank {rank}] {local_idx + 1}/{len(local_meta)} {sample_id} -> "
            f"{ {k: v for k, v in scores.items() if not k.endswith('_axes')} }"
        )

        del video_dict, audio_dict, video_uint8, audio_stereo, audio_mono, sample
        torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()

    if is_main:
        merged_path = merge_rank_jsonls(runtime.out_dir, world_size)
        summary = write_summary(merged_path, runtime.out_dir / "summary.json")
        print(f"[rank 0] merged -> {merged_path}")
        print(f"[rank 0] summary -> {runtime.out_dir / 'summary.json'}")
        for name, stats in summary["rewards"].items():
            print(
                f"  {name}: mean={stats['mean']!r} std={stats['std']!r} "
                f"n_valid={stats['n_valid']} n_filled={stats['n_filled']}"
            )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
