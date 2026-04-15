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

# src/flow_factory/rewards/vllm_evaluate.py
"""
VLM Evaluate Reward Models for image and video generation.

Evaluates generation quality by querying a VLM (e.g., Qwen3-VL) via
OpenAI-compatible API with a Yes/No question, then extracting
P(Yes) / (P(Yes) + P(No)) from the VLM's logprobs as the reward.

Provides two pointwise variants:

- **VLMEvaluateImageRewardModel** (``vllm_evaluate`` / ``vllm_evaluate_image``):
  Evaluates a single image per sample.
- **VLMEvaluateVideoRewardModel** (``vllm_evaluate_video``):
  Evaluates a video (sequence of frames) per sample, with optional
  condition image support for image-to-video (i2v) tasks.

Usage in YAML config (image):
    rewards:
      - name: "vllm_evaluate"
        reward_model: "vllm_evaluate"
        batch_size: 8
        api_base_url: "http://localhost:8000/v1"
        api_key: "EMPTY"
        vlm_model: "Qwen3-VL"

Usage in YAML config (video):
    rewards:
      - name: "vllm_evaluate_video"
        reward_model: "vllm_evaluate_video"
        batch_size: 4
        api_base_url: "http://localhost:8000/v1"
        vlm_model: "Qwen3-VL"
        max_frames: 8
        timeout: 180
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from ..utils.image import pil_image_to_base64
from .abc import PointwiseRewardModel, RewardModelOutput

logger = logging.getLogger(__name__)

# Suppress verbose HTTP/retry logs from openai client and httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# =====================================================================
# Helper functions
# =====================================================================


def _get_yes_cond_prob(completion, canonicalize: bool = False) -> float:
    """
    Extract P(Yes) / (P(Yes) + P(No)) from a VLM completion's logprobs.

    Args:
        completion: OpenAI-compatible ChatCompletion response.
        canonicalize: If True, aggregate probabilities for all case
            variations of "yes" and "no" (e.g., "Yes", "yes", "YES").

    Returns:
        Conditional probability of "Yes". Returns 0.0 on failure.
    """
    if completion is None:
        return 0.0

    logprobs = completion.choices[0].logprobs
    if not logprobs:
        return 0.0

    if not canonicalize:
        token_logprobs = {t.token: t.logprob for t in logprobs.content[0].top_logprobs}
        yes_logprob = token_logprobs.get("Yes", float("-inf"))
        no_logprob = token_logprobs.get("No", float("-inf"))

        if yes_logprob == float("-inf") and no_logprob == float("-inf"):
            return 0.0

        diff = torch.tensor(yes_logprob - no_logprob, dtype=torch.float64)
        return torch.sigmoid(diff).item()
    else:
        # Aggregate all case variations
        token_probs = {t.token: np.exp(t.logprob) for t in logprobs.content[0].top_logprobs}
        tokens = np.array(list(token_probs.keys()))
        probs = np.array(list(token_probs.values()), dtype=np.float64)  # (num_tokens,)
        tokens_stripped = np.array([token.strip().lower() for token in tokens])

        yes_prob_sum = probs[tokens_stripped == "yes"].sum()
        no_prob_sum = probs[tokens_stripped == "no"].sum()
        total = yes_prob_sum + no_prob_sum

        if total == 0.0:
            return 0.0
        return float(yes_prob_sum / total)


class VLMEvaluateBase(PointwiseRewardModel):
    """Shared API transport and logprob scoring layer for VLM evaluate models.

    Handles OpenAI client initialization, concurrency control, FIFO caching,
    and the retry+logprob-extraction pipeline.  Subclasses only need to
    implement ``__call__`` and assemble per-sample messages/cache-keys.

    Extra kwargs (passed via YAML config):
        api_base_url (str): Base URL for the OpenAI-compatible API.
            Default: "http://localhost:8000/v1"
        api_key (str): API key. Default: "EMPTY"
        vlm_model (str): VLM model name. Default: "Qwen3-VL"
        max_concurrent (int): Max concurrent API requests. Default: 100
        max_retries (int): Max retries per API call. Default: 10
        timeout (int): Timeout in seconds per API call. Default: 60
        top_logprobs (int): Number of top logprobs to request. Default: 20
        canonicalize (bool): Aggregate Yes/No case variants. Default: False
        max_cache_size (int): Max FIFO cache entries. Default: 1024
    """

    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "VLMEvaluateBase requires the `openai` package. "
                "Install it with: pip install openai"
            )

        # Read extra kwargs with defaults
        self.api_base_url = config.extra_kwargs.get("api_base_url", "http://localhost:8000/v1")
        self.api_key = config.extra_kwargs.get("api_key", "EMPTY")
        self.vlm_model = config.extra_kwargs.get("vlm_model", "Qwen3-VL")
        self.max_concurrent = config.extra_kwargs.get("max_concurrent", 100)
        self.max_retries = config.extra_kwargs.get("max_retries", 10)
        self.timeout = config.extra_kwargs.get("timeout", 60)
        self.top_logprobs = config.extra_kwargs.get("top_logprobs", 20)
        self.canonicalize = config.extra_kwargs.get("canonicalize", False)
        self.max_cache_size = config.extra_kwargs.get("max_cache_size", 1024)

        # Initialize async OpenAI client
        self.client = AsyncOpenAI(
            base_url=self.api_base_url,
            api_key=self.api_key,
        )
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        # Simple FIFO cache: img_hash -> score
        self._cache: dict[str, float] = {}

    def _add_to_cache(self, key: str, value: float):
        """Add entry to cache with FIFO eviction."""
        if len(self._cache) >= self.max_cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    async def _request_score(self, messages: list, cache_key: str) -> float:
        """Send a single VLM request and return P(Yes|Yes,No).

        Encapsulates retry logic, concurrency gating, logprob extraction,
        and cache read/write.  Returns 0.0 if all retries are exhausted.
        """
        if cache_key in self._cache:
            return self._cache[cache_key]

        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    completion = await self.client.chat.completions.create(
                        model=self.vlm_model,
                        messages=messages,
                        temperature=0.0,
                        max_completion_tokens=1,
                        logprobs=True,
                        top_logprobs=self.top_logprobs,
                        timeout=self.timeout,
                    )

                score = _get_yes_cond_prob(completion, canonicalize=self.canonicalize)
                self._add_to_cache(cache_key, score)
                return score

            except Exception as e:
                logger.warning(
                    "VLM API error on attempt %d/%d: %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)

        return 0.0


class VLMEvaluateImageRewardModel(VLMEvaluateBase):
    """VLM-based image evaluation reward model.

    Sends each image to a VLM with a comprehensive quality assessment
    prompt and derives a reward from the logprob-based P(Yes|Yes,No).

    Registered as ``vllm_evaluate`` (backward-compatible) and
    ``vllm_evaluate_image``.
    """

    required_fields = ("prompt", "image")

    EVALUATE_PROMPT = (
        "You are an expert image quality assessor. "
        "Evaluate this AI-generated image by considering ALL of the following criteria:\n"
        "1. Naturalness: Does the scene look realistic with correct perspective, "
        "shadows, and lighting?\n"
        "2. Artifacts: Is the image free from distortions, blurriness, watermarks, "
        "deformed faces, unusual body parts, or unharmonized subjects?\n"
        "3. Aesthetic Appeal: Does the image exhibit pleasing composition, "
        "color harmony, and visual balance?\n"
        "4. Detail & Clarity: Are textures, edges, and fine details rendered "
        "sharply and coherently without noise or smearing?\n"
        "5. Overall Coherence: Is the image semantically consistent, with all "
        "elements logically fitting together in a unified scene?\n\n"
        "Considering all the above criteria holistically, is this a high-quality image? "
        "Answer Yes or No."
    )

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
        """
        Compute VLM evaluation rewards for a batch of images.

        Args:
            prompt: List of text prompts (not used for evaluation question,
                but kept for interface compatibility).
            image: List of generated images.
            video: Not used; falls back to first frame if image is None.

        Returns:
            RewardModelOutput with per-sample scores in [0, 1].
        """
        # Handle video input (use first frame)
        if image is None and video is not None:
            image = [v[0] for v in video]

        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided")

        assert len(prompt) == len(image), f"Mismatch: {len(prompt)} prompts vs {len(image)} images"

        # Run async scoring
        rewards = asyncio.run(self._async_score_batch(image))

        return RewardModelOutput(
            rewards=torch.tensor(rewards, dtype=torch.float32),  # (batch_size,)
            extra_info={},
        )

    async def _async_score_batch(
        self,
        images: List[Image.Image],
    ) -> List[float]:
        """Score all images in the batch concurrently."""
        tasks = [self._score_single(img) for img in images]
        return list(await asyncio.gather(*tasks))

    async def _score_single(
        self,
        image: Image.Image,
    ) -> float:
        """
        Query the VLM for a single image and return P(Yes|Yes,No).

        Uses caching to avoid redundant API calls and retries on failure.
        """
        cache_key = hashlib.md5(image.tobytes()).hexdigest()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image)}},
                    {"type": "text", "text": self.EVALUATE_PROMPT},
                ],
            }
        ]
        return await self._request_score(messages, cache_key)


VLMEvaluateRewardModel = VLMEvaluateImageRewardModel


class VLMEvaluateVideoRewardModel(VLMEvaluateBase):
    """VLM-based video evaluation reward model.

    Sends video frames (and optionally a condition image for i2v tasks)
    to a VLM with a quality assessment prompt, deriving a reward from
    the logprob-based P(Yes|Yes,No).

    Extra kwargs (in addition to those from VLMEvaluateBase):
        max_frames (int): Maximum number of frames to send. Frames are
            uniformly sampled when the video exceeds this limit.
            Default: 16
    """

    required_fields = ("prompt", "video", "condition_images")

    VIDEO_EVALUATE_PROMPT = (
        "You are an expert video quality assessor. "
        "Evaluate this AI-generated video (shown as a sequence of frames) by considering "
        "ALL of the following criteria:\n"
        "1. Temporal Consistency: Are objects, subjects, and backgrounds consistent "
        "and coherent across all frames?\n"
        "2. Motion Quality: Are movements smooth and natural, free from jitter, "
        "warping, flickering, or unnatural transitions?\n"
        "3. Artifacts: Is each frame free from distortions, blurriness, "
        "deformed faces, or visual glitches?\n"
        "4. Aesthetic Appeal: Does the video exhibit pleasing composition, "
        "color harmony, and visual balance throughout?\n"
        "5. Overall Coherence: Is the video semantically consistent as a "
        "unified clip, with all elements logically fitting together?\n\n"
        "Considering all the above criteria holistically, is this a high-quality video? "
        "Answer Yes or No."
    )

    I2V_EVALUATE_PROMPT = (
        "You are an expert video quality assessor. "
        "You are given a reference image (the first image) and an AI-generated video "
        "(the subsequent frames) that should be derived from it. "
        "Evaluate the video by considering ALL of the following criteria:\n"
        "1. Reference Fidelity: Does the video faithfully preserve the subject, style, "
        "and scene from the reference image?\n"
        "2. Temporal Consistency: Are objects, subjects, and backgrounds consistent "
        "and coherent across all frames?\n"
        "3. Motion Quality: Are movements smooth and natural, free from jitter, "
        "warping, flickering, or unnatural transitions?\n"
        "4. Artifacts: Is each frame free from distortions, blurriness, "
        "deformed faces, or visual glitches?\n"
        "5. Overall Coherence: Is the video semantically consistent as a "
        "unified clip, logically extending the reference image?\n\n"
        "Considering all the above criteria holistically, is this a high-quality video "
        "that faithfully extends the reference image? Answer Yes or No."
    )

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.max_frames = config.extra_kwargs.get("max_frames", 8)

    def _sample_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Uniformly sample frames down to ``max_frames`` if needed."""
        if len(frames) <= self.max_frames:
            return frames
        indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
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
        """Compute VLM evaluation rewards for a batch of videos.

        Args:
            prompt: List of text prompts.
            video: List of frame sequences. Each element is a list of PIL
                Images representing one video clip.
            condition_images: Optional list of condition image lists for i2v.
                When present, the first image of each inner list is used as
                the reference image and the i2v-specific prompt is selected.

        Returns:
            RewardModelOutput with per-sample scores in [0, 1].
        """
        if video is None:
            raise ValueError("'video' must be provided for VLMEvaluateVideoRewardModel")

        if len(prompt) != len(video):
            raise ValueError(f"Mismatch: {len(prompt)} prompts vs {len(video)} videos")

        cond_imgs: List[Optional[Image.Image]] = [None] * len(video)
        if condition_images is not None:
            if len(condition_images) != len(video):
                raise ValueError(
                    f"Mismatch: {len(condition_images)} condition_images " f"vs {len(video)} videos"
                )
            cond_imgs = [imgs[0] if imgs else None for imgs in condition_images]

        sampled_videos = [self._sample_frames(v) for v in video]
        rewards = asyncio.run(self._async_score_batch(sampled_videos, cond_imgs))

        return RewardModelOutput(
            rewards=torch.tensor(rewards, dtype=torch.float32),  # (batch_size,)
            extra_info={},
        )

    async def _async_score_batch(
        self,
        videos: List[List[Image.Image]],
        cond_imgs: List[Optional[Image.Image]],
    ) -> List[float]:
        """Score all videos in the batch concurrently."""
        tasks = [
            self._score_single(frames, cond_img) for frames, cond_img in zip(videos, cond_imgs)
        ]
        return list(await asyncio.gather(*tasks))

    def _build_cache_key(
        self,
        frames: List[Image.Image],
        condition_image: Optional[Image.Image] = None,
    ) -> str:
        """Compute a deterministic cache key from all visual inputs."""
        hash_parts = []
        if condition_image is not None:
            hash_parts.append(hashlib.md5(condition_image.tobytes()).digest())
        for f in frames:
            hash_parts.append(hashlib.md5(f.tobytes()).digest())
        return hashlib.md5(b"".join(hash_parts)).hexdigest()

    def _build_messages(
        self,
        frames: List[Image.Image],
        condition_image: Optional[Image.Image] = None,
    ) -> list:
        """Assemble the OpenAI-compatible chat messages for a video sample.

        When a condition image is present (i2v), it is placed first and
        the i2v-specific prompt is used; otherwise the t2v prompt is used.
        """
        content: list = []
        if condition_image is not None:
            content.append(
                {"type": "image_url", "image_url": {"url": pil_image_to_base64(condition_image)}}
            )
            eval_prompt = self.I2V_EVALUATE_PROMPT
        else:
            eval_prompt = self.VIDEO_EVALUATE_PROMPT

        for frame in frames:
            content.append({"type": "image_url", "image_url": {"url": pil_image_to_base64(frame)}})
        content.append({"type": "text", "text": eval_prompt})

        return [{"role": "user", "content": content}]

    async def _score_single(
        self,
        frames: List[Image.Image],
        condition_image: Optional[Image.Image] = None,
    ) -> float:
        """Score a single video sample via VLM API."""
        cache_key = self._build_cache_key(frames, condition_image)
        messages = self._build_messages(frames, condition_image)
        return await self._request_score(messages, cache_key)
