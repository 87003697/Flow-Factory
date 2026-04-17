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

# src/flow_factory/rewards/unified_reward.py
"""
UnifiedReward 2.0 API-based reward models for image and video generation.

Provides the **structured pointwise family** aligned with UnifiedReward
2.0's upstream ACS/APS API prompts: extracts multiple axis scores
(e.g. Alignment/Coherence/Style) and aggregates them into a single
reward via configurable weights.  All models share the same API
transport layer (``UnifiedRewardAPIBase``).

Recommended YAML config (image generation ACS):
    rewards:
      - name: "unified_reward_image_acs"
        reward_model: "unified_reward_image_acs"
        batch_size: 8
        api_base_url: "http://localhost:8080/v1"
        vlm_model: "UnifiedReward"
        alignment_weight: 1.0
        coherence_weight: 0.5
        style_weight: 0.5

Recommended YAML config (video generation APS):
    rewards:
      - name: "unified_reward_video_aps"
        reward_model: "unified_reward_video_aps"
        batch_size: 4
        api_base_url: "http://localhost:8080/v1"
        vlm_model: "UnifiedReward"
        alignment_weight: 1.0
        physics_weight: 1.0
        style_weight: 1.0
        max_frames: 16
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from ..utils.image import pil_image_to_base64
from .abc import PointwiseRewardModel, RewardModelOutput

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


class UnifiedRewardAPIBase(PointwiseRewardModel):
    """
    API transport layer for UnifiedReward models.

    Connects to a vLLM/SGLang server via the OpenAI-compatible chat
    completions endpoint.  Returns **raw response text** — parsing is
    delegated to the scalar or structured pointwise base classes.

    Extra kwargs (passed via YAML config):
        api_base_url (str): Default ``"http://localhost:8080/v1"``
        api_key (str): Default ``"EMPTY"``
        vlm_model (str): Served model name. Default ``"UnifiedReward"``
        max_concurrent (int): Max concurrent API requests. Default 100
        max_retries (int): Max retries per API call. Default 10
        timeout (int): Timeout in seconds. Default 120
        max_tokens (int): Max generation tokens. Default 4096
        max_cache_size (int): Max cache entries. Default 1024
    """

    required_fields = ("prompt",)
    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "UnifiedReward requires the `openai` package. "
                "Install it with: pip install openai"
            )

        self.api_base_url = config.extra_kwargs.get("api_base_url", "http://localhost:8080/v1")
        self.api_key = config.extra_kwargs.get("api_key", "EMPTY")
        self.vlm_model = config.extra_kwargs.get("vlm_model", "UnifiedReward")
        self.max_concurrent = config.extra_kwargs.get("max_concurrent", 100)
        self.max_retries = config.extra_kwargs.get("max_retries", 10)
        self.timeout = config.extra_kwargs.get("timeout", 120)
        self.max_tokens = config.extra_kwargs.get("max_tokens", 4096)
        self.max_cache_size = config.extra_kwargs.get("max_cache_size", 1024)

        self.client = AsyncOpenAI(
            base_url=self.api_base_url,
            api_key=self.api_key,
        )
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        self._text_cache: dict[str, str] = {}

    def _add_to_cache(self, key: str, value: str):
        """Add entry to text cache with FIFO eviction."""
        if len(self._text_cache) >= self.max_cache_size:
            self._text_cache.pop(next(iter(self._text_cache)))
        self._text_cache[key] = value

    @staticmethod
    def _replace_nan_with_mean(rewards: torch.Tensor) -> torch.Tensor:
        """Replace NaN entries with the mean of valid entries.

        If all entries are NaN, replaces with 0.0 as a safe fallback.
        Logs a warning when NaN replacement occurs.

        Note:
            Intentional design per Constraint #26 (fail-fast exemption
            for documented auto-fallback): partial API failures are
            tolerated by filling with the batch mean rather than
            raising, to avoid aborting a training step over a transient
            network error against a remote vLLM server.  A warning is
            always emitted (see below) so silent data corruption is
            impossible.  Upstream Pref-GRPO follows the same policy.
        """
        nan_mask = torch.isnan(rewards)
        nan_count = nan_mask.sum().item()
        if nan_count == 0:
            return rewards

        valid = rewards[~nan_mask]
        fill_value = valid.mean().item() if valid.numel() > 0 else 0.0
        logger.warning(
            "UnifiedReward: %d/%d samples failed API scoring and were "
            "replaced with batch mean (%.4f)",
            nan_count,
            rewards.numel(),
            fill_value,
        )
        rewards = rewards.clone()
        rewards[nan_mask] = fill_value
        return rewards

    async def _query_api_text(self, messages: list, cache_key: str) -> str:
        """Send a single request to the VLM and return the raw response text.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    completion = await self.client.chat.completions.create(
                        model=self.vlm_model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                    )

                text = completion.choices[0].message.content
                self._add_to_cache(cache_key, text)
                return text

            except Exception as e:
                last_error = e
                logger.warning(
                    "UnifiedReward API error on attempt %d/%d: %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries - 1:
                    sleep_time = min(2**attempt + random.uniform(0, 1), 30)
                    await asyncio.sleep(sleep_time)

        raise RuntimeError(
            f"UnifiedReward API failed after {self.max_retries} retries: " f"{last_error}"
        )


class UnifiedRewardStructuredPointwiseBase(UnifiedRewardAPIBase):
    """
    Structured pointwise family: extracts multiple axis scores from VLM
    response text and aggregates them into a single reward via
    configurable weights.

    Subclasses define ``AXIS_KEYS``, ``AXIS_REGEX``, ``AXIS_MAX``,
    ``DEFAULT_AXIS_WEIGHTS``, and ``PROMPT_TEMPLATE``.

    Per-axis weights can be overridden in YAML config via
    ``<axis>_weight`` (e.g. ``alignment_weight: 2.0``).

    Returns:
        ``RewardModelOutput`` with:
        - ``rewards``: aggregated scalar, shape ``(batch_size,)``
        - ``extra_info``: per-axis normalized scores as tensors
    """

    AXIS_KEYS: List[str] = []
    AXIS_REGEX: Dict[str, str] = {}
    AXIS_MAX: Dict[str, float] = {}
    DEFAULT_AXIS_WEIGHTS: Dict[str, float] = {}

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        self.axis_weights: Dict[str, float] = {
            k: config.extra_kwargs.get(f"{k}_weight", self.DEFAULT_AXIS_WEIGHTS.get(k, 1.0))
            for k in self.AXIS_KEYS
        }

        for k, w in self.axis_weights.items():
            if w < 0:
                raise ValueError(f"Axis weight for '{k}' must be non-negative, got {w}")

        total_weight = sum(self.axis_weights.values())
        if total_weight <= 0:
            raise ValueError(
                f"Sum of axis weights must be positive, got {total_weight}. "
                f"Weights: {self.axis_weights}"
            )

    def _extract_structured_scores(self, text: str) -> Dict[str, float]:
        """Extract raw numeric scores for each axis from VLM response."""
        scores: Dict[str, float] = {}
        for key in self.AXIS_KEYS:
            pattern = self.AXIS_REGEX.get(key, "")
            match = re.search(pattern, text)
            scores[key] = float(match.group(1)) if match else 0.0
        return scores

    def _normalize_structured_scores(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Normalize each axis score to [0, 1]."""
        return {k: v / self.AXIS_MAX.get(k, 1.0) for k, v in raw.items()}

    def _aggregate_scores(self, normalized: Dict[str, float]) -> float:
        """Weighted average of normalized axis scores."""
        total_weight = sum(self.axis_weights.values())
        return (
            sum(self.axis_weights[k] * normalized.get(k, 0.0) for k in self.AXIS_KEYS)
            / total_weight
        )

    def _scores_from_text(self, text: str) -> Tuple[float, Dict[str, float]]:
        """Full pipeline: raw text -> (aggregated reward, per-axis normalized)."""
        raw = self._extract_structured_scores(text)
        normalized = self._normalize_structured_scores(raw)
        aggregated = self._aggregate_scores(normalized)
        return aggregated, normalized

    def _pack_results(
        self,
        results: List[Tuple[float, Dict[str, float]]],
    ) -> RewardModelOutput:
        """Pack async per-sample scoring results into a ``RewardModelOutput``.

        Args:
            results: Per-sample ``(aggregated_reward, per_axis_normalized)``
                tuples in batch order.  ``nan`` in either slot signals an
                API failure for that sample (replaced with batch mean on
                the aggregated axis; kept as-is in ``extra_info``).

        Returns:
            ``RewardModelOutput`` whose ``rewards`` is a ``(batch_size,)``
            tensor with NaN filled via ``_replace_nan_with_mean``, and
            ``extra_info`` contains per-axis normalized scores as
            ``(batch_size,)`` tensors keyed by ``"{axis}_scores"``.
        """
        rewards = [r[0] for r in results]
        per_axis = {
            f"{k}_scores": torch.tensor(
                [r[1][k] for r in results], dtype=torch.float32
            )  # (batch_size,)
            for k in self.AXIS_KEYS
        }
        reward_tensor = self._replace_nan_with_mean(
            torch.tensor(rewards, dtype=torch.float32)  # (batch_size,)
        )
        return RewardModelOutput(rewards=reward_tensor, extra_info=per_axis)


class UnifiedRewardImageGenACSRewardModel(UnifiedRewardStructuredPointwiseBase):
    """
    UnifiedReward 2.0 structured scoring for image generation.

    Evaluates three axes: Alignment, Coherence, Style (ACS).
    Each axis range 1-5, normalized to [0, 1].  Final reward is a
    weighted average controlled by ``alignment_weight``,
    ``coherence_weight``, ``style_weight`` in the YAML config.

    Usage in YAML config:
        rewards:
          - name: "unified_reward_image_acs"
            reward_model: "unified_reward_image_acs"
            batch_size: 8
            api_base_url: "http://localhost:8080/v1"
            vlm_model: "UnifiedReward"
            alignment_weight: 1.0
            coherence_weight: 0.5
            style_weight: 0.5
    """

    required_fields = ("prompt", "image")

    PROMPT_TEMPLATE = (
        "You are presented with a generated image and its associated text caption. "
        "Your task is to analyze the image across multiple dimensions in relation "
        "to the caption. Specifically:\n"
        "Provide overall assessments for the image along the following axes "
        "(each rated from 1 to 5):\n"
        "- Alignment Score: How well the image matches the caption in terms of content.\n"
        "- Coherence Score: How logically consistent the image is "
        "(absence of visual glitches, object distortions, etc.).\n"
        "- Style Score: How aesthetically appealing the image looks, "
        "regardless of caption accuracy.\n\n"
        "Output your evaluation using the format below:\n\n"
        "Alignment Score (1-5): X\n"
        "Coherence Score (1-5): Y\n"
        "Style Score (1-5): Z\n\n"
        "Your task is provided as follows:\n"
        "Text Caption: [__PROMPT__]"
    )

    AXIS_KEYS = ["alignment", "coherence", "style"]
    AXIS_REGEX = {
        "alignment": r"Alignment Score \(1-5\):\s*(\d+(?:\.\d+)?)",
        "coherence": r"Coherence Score \(1-5\):\s*(\d+(?:\.\d+)?)",
        "style": r"Style Score \(1-5\):\s*(\d+(?:\.\d+)?)",
    }
    AXIS_MAX = {"alignment": 5.0, "coherence": 5.0, "style": 5.0}
    DEFAULT_AXIS_WEIGHTS = {"alignment": 1.0, "coherence": 1.0, "style": 1.0}

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        if image is None and video is not None:
            image = [v[0] for v in video]

        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided")

        if len(prompt) != len(image):
            raise ValueError(f"Mismatch: {len(prompt)} prompts vs {len(image)} images")

        results = asyncio.run(self._async_score_batch(prompt, image))
        return self._pack_results(results)

    async def _async_score_batch(
        self, prompts: List[str], images: List[Image.Image]
    ) -> List[Tuple[float, Dict[str, float]]]:
        tasks = [self._score_single(p, img) for p, img in zip(prompts, images)]
        return list(await asyncio.gather(*tasks))

    def _build_cache_key(self, prompt: str, image: Image.Image) -> str:
        """Compute a deterministic cache key from prompt and image bytes."""
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        return hashlib.md5((prompt + image_hash).encode()).hexdigest()

    def _build_messages(self, prompt: str, image: Image.Image) -> list:
        """Assemble the OpenAI-compatible chat messages for an image sample."""
        question = self.PROMPT_TEMPLATE.replace("__PROMPT__", prompt)
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_image_to_base64(image)},
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

    async def _score_single(
        self, prompt: str, image: Image.Image
    ) -> Tuple[float, Dict[str, float]]:
        """Score a single image sample via VLM API."""
        cache_key = self._build_cache_key(prompt, image)
        messages = self._build_messages(prompt, image)
        try:
            text = await self._query_api_text(messages, cache_key)
        except RuntimeError:
            nan = float("nan")
            return nan, {k: nan for k in self.AXIS_KEYS}
        return self._scores_from_text(text)


class UnifiedRewardVideoGenAPSRewardModel(UnifiedRewardStructuredPointwiseBase):
    """
    UnifiedReward 2.0 structured scoring for video generation.

    Evaluates three axes: Alignment, Physics, Style (APS).
    Each axis range 1-5, normalized to [0, 1].  Final reward is a
    weighted average controlled by ``alignment_weight``,
    ``physics_weight``, ``style_weight`` in the YAML config.

    Video frames are sent as individual images through the API.  Long
    clips are uniformly sub-sampled down to ``max_frames`` frames to
    respect the upstream vLLM server's ``--limit-mm-per-prompt.image``
    bound.  When ``condition_images`` is provided (I2V), the first
    reference image of each sample is prepended to the frame sequence
    so the scorer can evaluate subject/style fidelity.

    Extra kwargs (in addition to those from ``UnifiedRewardAPIBase``):
        max_frames (int): Maximum frames sent to the VLM API. Frames
            are uniformly sampled when the video exceeds this limit.
            Default: 16 (matches the upstream APS reference script).

    Usage in YAML config:
        rewards:
          - name: "unified_reward_video_aps"
            reward_model: "unified_reward_video_aps"
            batch_size: 4
            api_base_url: "http://localhost:8080/v1"
            vlm_model: "UnifiedReward"
            alignment_weight: 1.0
            physics_weight: 1.0
            style_weight: 1.0
            max_frames: 16
    """

    required_fields = ("prompt", "video", "condition_images")

    PROMPT_TEMPLATE = (
        "You are presented with a generated video and its associated text caption. "
        "Your task is to analyze the video across multiple dimensions in relation "
        "to the caption. Specifically:\n"
        "Provide overall assessments for the video along the following axes "
        "(each rated from 1 to 5):\n"
        "- Alignment Score: How well the video matches the caption in terms of content.\n"
        "- Physics Score: How well the gravity, movements, collisions, and "
        "interactions make physical sense.\n"
        "- Style Score: How visually appealing the video looks, "
        "regardless of caption accuracy.\n\n"
        "Output your evaluation using the format below:\n\n"
        "Alignment Score (1-5): X\n"
        "Physics Score (1-5): Y\n"
        "Style Score (1-5): Z\n\n"
        "Your task is provided as follows:\n"
        "Text Caption: [__PROMPT__]"
    )

    AXIS_KEYS = ["alignment", "physics", "style"]
    AXIS_REGEX = {
        "alignment": r"Alignment Score \(1-5\):\s*(\d+(?:\.\d+)?)",
        "physics": r"Physics Score \(1-5\):\s*(\d+(?:\.\d+)?)",
        "style": r"Style Score \(1-5\):\s*(\d+(?:\.\d+)?)",
    }
    AXIS_MAX = {"alignment": 5.0, "physics": 5.0, "style": 5.0}
    DEFAULT_AXIS_WEIGHTS = {"alignment": 1.0, "physics": 1.0, "style": 1.0}

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.max_frames = config.extra_kwargs.get("max_frames", 16)

    def _sample_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Uniformly sample ``frames`` down to ``self.max_frames`` if needed.

        Args:
            frames: Full per-sample frame list (PIL Images).

        Returns:
            A sub-sampled list whose length is at most ``self.max_frames``.
            When the input already fits, the original list is returned
            unchanged.
        """
        if len(frames) <= self.max_frames:
            return frames
        indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)  # (max_frames,)
        return [frames[i] for i in indices]

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        if video is None:
            raise ValueError("'video' must be provided")

        if len(prompt) != len(video):
            raise ValueError(f"Mismatch: {len(prompt)} prompts vs {len(video)} videos")

        sampled_videos, cond_imgs = self._prepare_video_inputs(video, condition_images)
        results = asyncio.run(self._async_score_batch(prompt, sampled_videos, cond_imgs))
        return self._pack_results(results)

    def _prepare_video_inputs(
        self,
        video: List[List[Image.Image]],
        condition_images: Optional[List[List[Image.Image]]] = None,
    ) -> Tuple[List[List[Image.Image]], List[Optional[Image.Image]]]:
        """Sub-sample frames and resolve per-sample condition images.

        Args:
            video: Per-sample frame sequences.
            condition_images: Optional per-sample lists of condition images
                (I2V).  When provided, the first image of each inner list
                is taken as the reference image for that sample.

        Returns:
            Tuple ``(sampled_videos, cond_imgs)`` where ``sampled_videos``
            has each clip sub-sampled to at most ``self.max_frames``
            frames, and ``cond_imgs`` is aligned with the batch (``None``
            for pure T2V samples).
        """
        cond_imgs: List[Optional[Image.Image]] = [None] * len(video)
        if condition_images is not None:
            if len(condition_images) != len(video):
                raise ValueError(
                    f"Mismatch: {len(condition_images)} condition_images " f"vs {len(video)} videos"
                )
            cond_imgs = [imgs[0] if imgs else None for imgs in condition_images]

        sampled_videos = [self._sample_frames(v) for v in video]
        return sampled_videos, cond_imgs

    async def _async_score_batch(
        self,
        prompts: List[str],
        videos: List[List[Image.Image]],
        cond_imgs: List[Optional[Image.Image]],
    ) -> List[Tuple[float, Dict[str, float]]]:
        tasks = [self._score_single(p, v, c) for p, v, c in zip(prompts, videos, cond_imgs)]
        return list(await asyncio.gather(*tasks))

    def _build_cache_key(
        self,
        prompt: str,
        frames: List[Image.Image],
        condition_image: Optional[Image.Image] = None,
    ) -> str:
        """Compute a deterministic cache key from prompt and all visual inputs.

        The condition image (if any) is hashed first so that different
        reference images produce distinct cache keys even when the frame
        sequence is identical.
        """
        hash_parts: List[bytes] = []
        if condition_image is not None:
            hash_parts.append(hashlib.md5(condition_image.tobytes()).digest())
        for f in frames:
            hash_parts.append(hashlib.md5(f.tobytes()).digest())
        visual_hash = hashlib.md5(b"".join(hash_parts)).hexdigest()
        return hashlib.md5((prompt + visual_hash).encode()).hexdigest()

    def _build_messages(
        self,
        prompt: str,
        frames: List[Image.Image],
        condition_image: Optional[Image.Image] = None,
    ) -> list:
        """Assemble the OpenAI-compatible chat messages for a video sample.

        When a condition image is present (I2V), it is placed at the front
        of the image sequence so the scorer can evaluate subject/style
        fidelity against the reference before the generated frames.
        """
        content: list = []
        if condition_image is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": pil_image_to_base64(condition_image)},
                }
            )
        for frame in frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": pil_image_to_base64(frame)},
                }
            )
        content.append({"type": "text", "text": self.PROMPT_TEMPLATE.replace("__PROMPT__", prompt)})
        return [{"role": "user", "content": content}]

    async def _score_single(
        self,
        prompt: str,
        frames: List[Image.Image],
        condition_image: Optional[Image.Image] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Score a single video sample via VLM API."""
        cache_key = self._build_cache_key(prompt, frames, condition_image)
        messages = self._build_messages(prompt, frames, condition_image)
        try:
            text = await self._query_api_text(messages, cache_key)
        except RuntimeError:
            nan = float("nan")
            return nan, {k: nan for k in self.AXIS_KEYS}
        return self._scores_from_text(text)
