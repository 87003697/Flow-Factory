"""
Integration tests for pairwise win-rate reward models.

Requires a running vLLM server at localhost:8080 serving the
``UnifiedReward`` model.  Run with:

    python tests/test_pairwise_reward_integration.py
"""
from __future__ import annotations

import asyncio
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

sys.path.insert(0, "src")

from flow_factory.hparams import RewardArguments
from flow_factory.rewards.abc import GroupwiseRewardModel, RewardModelOutput
from flow_factory.rewards.registry import get_reward_model_class


API_BASE_URL = "http://localhost:8080/v1"
VLM_MODEL = "UnifiedReward"
GROUP_SIZE = 4


def make_dummy_accelerator():
    """Minimal mock accelerator for non-distributed tests."""

    @dataclass
    class FakeAccelerator:
        device: torch.device = torch.device("cpu")
        process_index: int = 0
        num_processes: int = 1

        def is_local_main_process(self):
            return True

    return FakeAccelerator()


def make_reward_config(**extra) -> RewardArguments:
    """Build a RewardArguments with the given extra_kwargs."""
    defaults = {
        "api_base_url": API_BASE_URL,
        "api_key": "EMPTY",
        "vlm_model": VLM_MODEL,
        "max_concurrent": 16,
        "max_retries": 3,
        "timeout": 120,
        "max_tokens": 8192,
    }
    defaults.update(extra)
    return RewardArguments(extra_kwargs=defaults)


def make_test_images(n: int, size: int = 64) -> List[Image.Image]:
    """Generate n distinct solid-color test images."""
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "white", "orange"]
    return [Image.new("RGB", (size, size), color=colors[i % len(colors)]) for i in range(n)]


def make_test_videos(n: int, num_frames: int = 4, size: int = 64) -> List[List[Image.Image]]:
    """Generate n test videos (each is a list of solid-color frames)."""
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "white", "orange"]
    videos = []
    for i in range(n):
        frames = [
            Image.new("RGB", (size, size), color=colors[(i + f) % len(colors)])
            for f in range(num_frames)
        ]
        videos.append(frames)
    return videos


def validate_output(output: RewardModelOutput, group_size: int, test_name: str):
    """Common assertions for reward output."""
    assert isinstance(output, RewardModelOutput), (
        f"[{test_name}] Expected RewardModelOutput, got {type(output)}"
    )
    rewards = output.rewards
    assert rewards.shape == (group_size,), (
        f"[{test_name}] Expected rewards shape ({group_size},), got {rewards.shape}"
    )
    assert not torch.isnan(rewards).any(), (
        f"[{test_name}] Rewards contain NaN: {rewards}"
    )
    assert (rewards >= 0).all() and (rewards <= 1).all(), (
        f"[{test_name}] Rewards out of [0,1] range: {rewards}"
    )


def test_registry_resolution():
    """All 4 pairwise models resolve from registry."""
    names = [
        "unified_reward_think_image_pairwise",
        "unified_reward_think_video_pairwise",
        "unified_reward_flex_image_pairwise",
        "unified_reward_flex_video_pairwise",
    ]
    for name in names:
        cls = get_reward_model_class(name)
        assert issubclass(cls, GroupwiseRewardModel), (
            f"{name} should be a GroupwiseRewardModel subclass, got {cls.__bases__}"
        )
    print("[PASS] test_registry_resolution")


def test_think_image_pairwise():
    """Think-mode image pairwise with live VLM API."""
    config = make_reward_config()
    accelerator = make_dummy_accelerator()

    cls = get_reward_model_class("unified_reward_think_image_pairwise")
    model = cls(config, accelerator)
    assert model.required_fields == ("prompt", "image")

    prompts = ["A cute cat sitting on a windowsill"] * GROUP_SIZE
    images = make_test_images(GROUP_SIZE)

    t0 = time.time()
    output = model(prompt=prompts, image=images)
    elapsed = time.time() - t0

    validate_output(output, GROUP_SIZE, "think_image")
    print(f"[PASS] test_think_image_pairwise  rewards={output.rewards.tolist()}  elapsed={elapsed:.1f}s")


def test_think_video_pairwise():
    """Think-mode video pairwise with live VLM API."""
    config = make_reward_config(max_frames=4)
    accelerator = make_dummy_accelerator()

    cls = get_reward_model_class("unified_reward_think_video_pairwise")
    model = cls(config, accelerator)
    assert model.required_fields == ("prompt", "video")

    prompts = ["A dog running in a park"] * GROUP_SIZE
    videos = make_test_videos(GROUP_SIZE, num_frames=6)

    t0 = time.time()
    output = model(prompt=prompts, video=videos)
    elapsed = time.time() - t0

    validate_output(output, GROUP_SIZE, "think_video")
    print(f"[PASS] test_think_video_pairwise  rewards={output.rewards.tolist()}  elapsed={elapsed:.1f}s")


def test_flex_image_pairwise():
    """Flex-mode image pairwise with live VLM API."""
    config = make_reward_config(
        overall_weight=1.0,
        dim_weight=1.0,
        category_weights=[1.0, 1.0, 0.5],
    )
    accelerator = make_dummy_accelerator()

    cls = get_reward_model_class("unified_reward_flex_image_pairwise")
    model = cls(config, accelerator)
    assert model.required_fields == ("prompt", "image")

    prompts = ["A beautiful sunset over the ocean"] * GROUP_SIZE
    images = make_test_images(GROUP_SIZE)

    t0 = time.time()
    output = model(prompt=prompts, image=images)
    elapsed = time.time() - t0

    validate_output(output, GROUP_SIZE, "flex_image")
    assert output.extra_info is not None, "[flex_image] extra_info should not be None"
    assert "overall_win_rate" in output.extra_info, "[flex_image] missing overall_win_rate"
    assert "dim_mean_win_rate" in output.extra_info, "[flex_image] missing dim_mean_win_rate"
    print(f"[PASS] test_flex_image_pairwise  rewards={output.rewards.tolist()}  elapsed={elapsed:.1f}s")
    print(f"       extra_info keys: {list(output.extra_info.keys())}")


def test_flex_video_pairwise():
    """Flex-mode video pairwise with live VLM API."""
    config = make_reward_config(
        max_frames=4,
        overall_weight=1.0,
        dim_weight=1.0,
    )
    accelerator = make_dummy_accelerator()

    cls = get_reward_model_class("unified_reward_flex_video_pairwise")
    model = cls(config, accelerator)
    assert model.required_fields == ("prompt", "video")

    prompts = ["A bird flying across a blue sky"] * GROUP_SIZE
    videos = make_test_videos(GROUP_SIZE, num_frames=6)

    t0 = time.time()
    output = model(prompt=prompts, video=videos)
    elapsed = time.time() - t0

    validate_output(output, GROUP_SIZE, "flex_video")
    assert output.extra_info is not None, "[flex_video] extra_info should not be None"
    assert "overall_win_rate" in output.extra_info, "[flex_video] missing overall_win_rate"
    print(f"[PASS] test_flex_video_pairwise  rewards={output.rewards.tolist()}  elapsed={elapsed:.1f}s")


def test_max_pairs():
    """max_pairs limits the number of comparisons."""
    config = make_reward_config(max_pairs=2)
    accelerator = make_dummy_accelerator()

    cls = get_reward_model_class("unified_reward_think_image_pairwise")
    model = cls(config, accelerator)

    prompts = ["A mountain landscape"] * GROUP_SIZE
    images = make_test_images(GROUP_SIZE)

    t0 = time.time()
    output = model(prompt=prompts, image=images)
    elapsed = time.time() - t0

    validate_output(output, GROUP_SIZE, "max_pairs")
    print(f"[PASS] test_max_pairs  rewards={output.rewards.tolist()}  elapsed={elapsed:.1f}s")


def test_pointwise_client_factory_fix():
    """Verify the pointwise model builds per-run async clients."""
    config = make_reward_config()
    accelerator = make_dummy_accelerator()

    cls = get_reward_model_class("unified_reward_image_acs")
    model = cls(config, accelerator)

    assert hasattr(model, "_make_client"), "pointwise model should expose _make_client"
    assert not hasattr(model, "client"), "pointwise model should NOT have bare self.client"
    assert not hasattr(model, "semaphore"), "pointwise model should NOT have bare self.semaphore"

    client = model._make_client()
    assert client is not None
    asyncio.run(client.close())
    print("[PASS] test_pointwise_client_factory_fix")


ALL_TESTS = [
    test_registry_resolution,
    test_pointwise_client_factory_fix,
    test_think_image_pairwise,
    test_think_video_pairwise,
    test_flex_image_pairwise,
    test_flex_video_pairwise,
    test_max_pairs,
]


if __name__ == "__main__":
    print(f"Running {len(ALL_TESTS)} integration tests against vLLM at {API_BASE_URL}")
    print(f"Model: {VLM_MODEL}, group_size: {GROUP_SIZE}")
    print("=" * 60)

    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception:
            failed += 1
            print(f"[FAIL] {test_fn.__name__}")
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(ALL_TESTS)} total")
    sys.exit(1 if failed else 0)
