# Copyright 2026 Flow-Factory Contributors
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
UnifiedReward API-based reward models for image and video generation.

Provides two pointwise families:

- **Structured (recommended)**: extracts multiple axis scores
  (e.g. Alignment/Coherence/Style) and aggregates them into a single
  reward via configurable weights.  Aligned with UnifiedReward 2.0.
- **Scalar (deprecated)**: extracts a single ``Final Score:`` from the
  VLM response.  No corresponding upstream 2.0 API prompt exists;
  prefer the structured variants instead.

Both families share the same API transport layer (``UnifiedRewardAPIBase``).

Recommended YAML config (structured, image generation ACS):
    rewards:
      - name: "unified_reward_image_acs"
        reward_model: "unified_reward_image_acs"
        batch_size: 8
        api_base_url: "http://localhost:8080/v1"
        vlm_model: "UnifiedReward"
        alignment_weight: 1.0
        coherence_weight: 0.5
        style_weight: 0.5

Recommended YAML config (structured, video generation APS):
    rewards:
      - name: "unified_reward_video_aps"
        reward_model: "unified_reward_video_aps"
        batch_size: 4
        api_base_url: "http://localhost:8080/v1"
        vlm_model: "UnifiedReward"
        alignment_weight: 1.0
        physics_weight: 1.0
        style_weight: 1.0
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
import warnings
from typing import Dict, List, Optional, Tuple

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


class UnifiedRewardScalarPointwiseBase(UnifiedRewardAPIBase):
    """
    Scalar pointwise family: extracts a single numeric score from VLM
    response text via ``SCORE_REGEX`` and normalizes by ``SCORE_MAX``.

    Subclasses override ``SCORE_REGEX``, ``SCORE_MAX``, and
    ``PROMPT_TEMPLATE``.
    """

    SCORE_REGEX: str = r"Final Score:\s*(\d+(?:\.\d+)?)"
    SCORE_MAX: float = 1.0

    def _extract_score(self, text: str) -> float:
        """Extract raw numeric score from VLM response text."""
        match = re.search(self.SCORE_REGEX, text)
        if match:
            return float(match.group(1))
        return 0.0

    def _normalize_score(self, raw_score: float) -> float:
        """Normalize raw score to [0, 1]."""
        return raw_score / self.SCORE_MAX

    def _score_from_text(self, text: str) -> float:
        """Full pipeline: raw text -> normalized scalar reward."""
        return self._normalize_score(self._extract_score(text))


class UnifiedRewardImageGenRewardModel(UnifiedRewardScalarPointwiseBase):
    """
    .. deprecated::
        Use ``UnifiedRewardImageGenACSRewardModel``
        (``unified_reward_image_acs``) instead.  The scalar prompt has
        no corresponding upstream UnifiedReward 2.0 API template.

    UnifiedReward for image generation quality assessment.

    Evaluates caption alignment and overall image quality.
    Raw score range 1-5, normalized to [0, 1] via ``score / 5``.
    """

    required_fields = ("prompt", "image")

    PROMPT_TEMPLATE = (
        "You are given a text caption and a generated image based on "
        "that caption. Your task is to evaluate this image based on two "
        "key criteria:\n"
        "1. Alignment with the Caption: Assess how well this image "
        "aligns with the provided caption. Consider the accuracy of "
        "depicted objects, their relationships, and attributes as "
        "described in the caption.\n"
        "2. Overall Image Quality: Examine the visual quality of this "
        "image, including clarity, detail preservation, color accuracy, "
        "and overall aesthetic appeal.\n"
        "Based on the above criteria, assign a score from 1 to 5 after "
        "'Final Score:'.\n"
        "Your task is provided as follows:\n"
        "Text Caption: [{prompt}]"
    )
    SCORE_REGEX = r"Final Score:\s*([1-5](?:\.\d+)?)"
    SCORE_MAX = 5.0

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
    ) -> RewardModelOutput:
        warnings.warn(
            "unified_reward_image is deprecated and has no matching "
            "UnifiedReward 2.0 API prompt. Use unified_reward_image_acs "
            "(UnifiedRewardImageGenACSRewardModel) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if image is None and video is not None:
            image = [v[0] for v in video]

        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided")

        if len(prompt) != len(image):
            raise ValueError(f"Mismatch: {len(prompt)} prompts vs {len(image)} images")

        rewards = asyncio.run(self._async_score_batch(prompt, image))
        reward_tensor = self._replace_nan_with_mean(torch.tensor(rewards, dtype=torch.float32))
        return RewardModelOutput(rewards=reward_tensor, extra_info={})

    async def _async_score_batch(
        self, prompts: List[str], images: List[Image.Image]
    ) -> List[float]:
        tasks = [self._score_single(p, img) for p, img in zip(prompts, images)]
        return list(await asyncio.gather(*tasks))

    async def _score_single(self, prompt: str, image: Image.Image) -> float:
        cache_key = hashlib.md5(
            (prompt + hashlib.md5(image.tobytes()).hexdigest()).encode()
        ).hexdigest()

        question = self.PROMPT_TEMPLATE.format(prompt=prompt)
        messages = [
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
        try:
            text = await self._query_api_text(messages, cache_key)
        except RuntimeError:
            return float("nan")
        return self._score_from_text(text)


class UnifiedRewardVideoGenRewardModel(UnifiedRewardScalarPointwiseBase):
    """
    .. deprecated::
        Use ``UnifiedRewardVideoGenAPSRewardModel``
        (``unified_reward_video_aps``) instead.  The scalar prompt has
        no corresponding upstream UnifiedReward 2.0 API template.

    UnifiedReward for video generation quality assessment.

    Evaluates 5 dimensions: visual quality, temporal consistency,
    dynamic degree, text-to-video alignment, and factual consistency.
    Raw score range 1-10, normalized to [0, 1] via ``score / 10``.

    Video frames are sent as individual images through the API.
    """

    required_fields = ("prompt", "video")

    PROMPT_TEMPLATE = (
        "Suppose you are an expert in judging and evaluating the quality "
        "of AI-generated videos, please watch the frames of a given "
        "video and see the text prompt for generating the video.\n"
        "Then give scores from 5 different dimensions:\n"
        "(1) visual quality: the quality of the video in terms of "
        "clearness, resolution, brightness, and color\n"
        "(2) temporal consistency, the consistency of objects or humans "
        "in video\n"
        "(3) dynamic degree, the degree of dynamic changes\n"
        "(4) text-to-video alignment, the alignment between the text "
        "prompt and the video content\n"
        "(5) factual consistency, the consistency of the video content "
        "with the common-sense and factual knowledge\n\n"
        "For each dimension, output a number from [1,2,3,4], \n"
        "in which '1' means 'Bad', '2' means 'Average', '3' means "
        "'Good', \n"
        "'4' means 'Real' or 'Perfect' (the video is like a real "
        "video)\n"
        "Finally, based on above 5 dimensions, assign a score from 1 "
        "to 10 after 'Final Score:'\n"
        "Here is an output example:\n"
        "visual quality: 4\n"
        "temporal consistency: 4\n"
        "dynamic degree: 3\n"
        "text-to-video alignment: 1\n"
        "factual consistency: 2\n"
        "Final Score: 6\n\n"
        "**Note: In the example above, scores are placeholders meant "
        "only to demonstrate the format. Your actual evaluation should "
        "be based on the quality of the given video.**\n"
        "Your task is provided as follows: Text Prompt: [{prompt}]"
    )
    SCORE_REGEX = r"Final Score:\s*(\d+(?:\.\d+)?)"
    SCORE_MAX = 10.0

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        video: Optional[List[List[Image.Image]]] = None,
    ) -> RewardModelOutput:
        warnings.warn(
            "unified_reward_video is deprecated and has no matching "
            "UnifiedReward 2.0 API prompt. Use unified_reward_video_aps "
            "(UnifiedRewardVideoGenAPSRewardModel) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if video is None:
            raise ValueError("'video' must be provided")

        if len(prompt) != len(video):
            raise ValueError(f"Mismatch: {len(prompt)} prompts vs {len(video)} videos")

        rewards = asyncio.run(self._async_score_batch(prompt, video))
        reward_tensor = self._replace_nan_with_mean(torch.tensor(rewards, dtype=torch.float32))
        return RewardModelOutput(rewards=reward_tensor, extra_info={})

    async def _async_score_batch(
        self,
        prompts: List[str],
        videos: List[List[Image.Image]],
    ) -> List[float]:
        tasks = [self._score_single(p, v) for p, v in zip(prompts, videos)]
        return list(await asyncio.gather(*tasks))

    async def _score_single(self, prompt: str, frames: List[Image.Image]) -> float:
        frames_hash = hashlib.md5(
            b"".join(hashlib.md5(f.tobytes()).digest() for f in frames)
        ).hexdigest()
        cache_key = hashlib.md5((prompt + frames_hash).encode()).hexdigest()

        question = self.PROMPT_TEMPLATE.format(prompt=prompt)
        content: list = [
            {
                "type": "image_url",
                "image_url": {"url": pil_image_to_base64(frame)},
            }
            for frame in frames
        ]
        content.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": content}]
        try:
            text = await self._query_api_text(messages, cache_key)
        except RuntimeError:
            return float("nan")
        return self._score_from_text(text)


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
        "Text Caption: [{prompt}]"
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
    ) -> RewardModelOutput:
        if image is None and video is not None:
            image = [v[0] for v in video]

        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided")

        if len(prompt) != len(image):
            raise ValueError(f"Mismatch: {len(prompt)} prompts vs {len(image)} images")

        results = asyncio.run(self._async_score_batch(prompt, image))
        rewards = [r[0] for r in results]
        per_axis = {
            f"{k}_scores": torch.tensor([r[1][k] for r in results], dtype=torch.float32)
            for k in self.AXIS_KEYS
        }
        reward_tensor = self._replace_nan_with_mean(torch.tensor(rewards, dtype=torch.float32))
        return RewardModelOutput(rewards=reward_tensor, extra_info=per_axis)

    async def _async_score_batch(
        self, prompts: List[str], images: List[Image.Image]
    ) -> List[Tuple[float, Dict[str, float]]]:
        tasks = [self._score_single(p, img) for p, img in zip(prompts, images)]
        return list(await asyncio.gather(*tasks))

    async def _score_single(
        self, prompt: str, image: Image.Image
    ) -> Tuple[float, Dict[str, float]]:
        cache_key = hashlib.md5(
            (prompt + hashlib.md5(image.tobytes()).hexdigest()).encode()
        ).hexdigest()

        question = self.PROMPT_TEMPLATE.format(prompt=prompt)
        messages = [
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

    Video frames are sent as individual images through the API.

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
    """

    required_fields = ("prompt", "video")

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
        "Text Caption: [{prompt}]"
    )

    AXIS_KEYS = ["alignment", "physics", "style"]
    AXIS_REGEX = {
        "alignment": r"Alignment Score \(1-5\):\s*(\d+(?:\.\d+)?)",
        "physics": r"Physics Score \(1-5\):\s*(\d+(?:\.\d+)?)",
        "style": r"Style Score \(1-5\):\s*(\d+(?:\.\d+)?)",
    }
    AXIS_MAX = {"alignment": 5.0, "physics": 5.0, "style": 5.0}
    DEFAULT_AXIS_WEIGHTS = {"alignment": 1.0, "physics": 1.0, "style": 1.0}

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        video: Optional[List[List[Image.Image]]] = None,
    ) -> RewardModelOutput:
        if video is None:
            raise ValueError("'video' must be provided")

        if len(prompt) != len(video):
            raise ValueError(f"Mismatch: {len(prompt)} prompts vs {len(video)} videos")

        results = asyncio.run(self._async_score_batch(prompt, video))
        rewards = [r[0] for r in results]
        per_axis = {
            f"{k}_scores": torch.tensor([r[1][k] for r in results], dtype=torch.float32)
            for k in self.AXIS_KEYS
        }
        reward_tensor = self._replace_nan_with_mean(torch.tensor(rewards, dtype=torch.float32))
        return RewardModelOutput(rewards=reward_tensor, extra_info=per_axis)

    async def _async_score_batch(
        self,
        prompts: List[str],
        videos: List[List[Image.Image]],
    ) -> List[Tuple[float, Dict[str, float]]]:
        tasks = [self._score_single(p, v) for p, v in zip(prompts, videos)]
        return list(await asyncio.gather(*tasks))

    async def _score_single(
        self, prompt: str, frames: List[Image.Image]
    ) -> Tuple[float, Dict[str, float]]:
        frames_hash = hashlib.md5(
            b"".join(hashlib.md5(f.tobytes()).digest() for f in frames)
        ).hexdigest()
        cache_key = hashlib.md5((prompt + frames_hash).encode()).hexdigest()

        question = self.PROMPT_TEMPLATE.format(prompt=prompt)
        content: list = [
            {
                "type": "image_url",
                "image_url": {"url": pil_image_to_base64(frame)},
            }
            for frame in frames
        ]
        content.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": content}]
        try:
            text = await self._query_api_text(messages, cache_key)
        except RuntimeError:
            nan = float("nan")
            return nan, {k: nan for k in self.AXIS_KEYS}
        return self._scores_from_text(text)
