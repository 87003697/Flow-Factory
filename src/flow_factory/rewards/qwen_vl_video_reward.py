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

# src/flow_factory/rewards/qwen_vl_video_reward.py
"""
Qwen-VL side-by-side scalar reward for image-conditioned 3D / video generation.

Each rendered frame is concatenated with the input condition image horizontally
(LEFT = condition, RIGHT = render). The stack is sent as ``image_url`` items to a
vLLM OpenAI-compatible server. Scoring matches ``vllm_evaluate.VLMEvaluateRewardModel``:
a holistic Yes/No question and ``P(Yes) / (P(Yes) + P(No))`` from first-token
logprobs (already in ``[0, 1]``).

When ``enable_reason=True`` is set in ``extra_kwargs``, scoring becomes
**thinking-conditioned**: each sample first generates a visible reason via the
VLM with ``enable_thinking=True``, and the same reason is then placed back into
the chat context before issuing the 1-token Yes/No question. The reason string
is also returned via ``RewardModelOutput.extra_info['reasons']`` so the
downstream logger can display it in wandb captions.

Recommended YAML extra_kwargs:
    api_base_url, api_key, vlm_model, max_concurrent, max_retries, timeout,
    max_frames, tile_resolution, top_logprobs, canonicalize,
    enable_reason, reason_thinking_token_budget, prompt_preset
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from ..utils.image import pil_image_to_base64
from .abc import RewardModelOutput
from .unified_reward import UnifiedRewardAPIBase
from .vllm_evaluate import _get_yes_cond_prob

logger = logging.getLogger(__name__)


# Stage-specific rubric presets. The ``tex`` preset assumes the side-by-side
# RIGHT panel is a fully textured RGB render aligned with the colored condition;
# the ``shape`` preset assumes the RIGHT panel is a matte gray clay render with
# texture/color absent BY DESIGN (only geometry is being optimized in that
# stage). Selected at construction via ``extra_kwargs.prompt_preset``.

_TEX_EVALUATION_FRAMEWORK = (
    "You are evaluating a 3D generation result. Each attached image is one "
    "frame shown side-by-side: the LEFT half is the input condition image; "
    "the RIGHT half is one rendered view of the generated 3D asset (split "
    "at the vertical center; width:height is 2:1).\n\n"
    "Consider ALL of the following across the frames:\n"
    "1. Identity: same object/subject as the condition.\n"
    "2. Geometry: shape, proportions, structural fidelity.\n"
    "3. Texture & color: alignment with the condition.\n"
    "4. Multi-view consistency: the asset stays coherent across frames.\n"
    "5. Artifacts: absence of broken geometry, holes, color bleeding, or "
    "solid blank renders.\n\n"
)

_TEX_YES_NO_DECISION = (
    "Considering every frame holistically, does the rendered 3D asset "
    "match the condition well enough to count as a successful generation? "
    "Answer Yes or No."
)

_TEX_REASON_BODY = (
    "Briefly analyze the frames in 1-3 sentences, citing concrete visual "
    "evidence for each of the five aspects above (identity, geometry, "
    "texture & color, multi-view consistency, artifacts). Focus on details "
    "that distinguish this specific render from other plausible ones (e.g. "
    "geometry errors, texture misalignment, color mismatch, holes, blank "
    "faces, view inconsistency). Do not output a score."
)

_SHAPE_EVALUATION_FRAMEWORK = (
    "You are evaluating a 3D shape generation result. Each attached image "
    "is one frame shown side-by-side: the LEFT half is the input condition "
    "image (in full color); the RIGHT half is one rendered view of the "
    "generated 3D mesh, shown as matte gray clay (no texture, no color). "
    "This color/texture difference is EXPECTED — only geometry is being "
    "optimized in this stage. Judge geometry only; do NOT penalize for "
    "missing color, missing texture, or material mismatch.\n\n"
    "Consider ALL of the following across the frames:\n"
    "1. Identity: the rendered shape depicts the same object/subject as the "
    "condition (recognizable from silhouette and prominent geometric "
    "features alone).\n"
    "2. Silhouette & contour: the render's outline matches the condition "
    "object's shape from this view.\n"
    "3. Geometry: proportions, structural fidelity, and presence of the key "
    "parts visible in the condition.\n"
    "4. Multi-view consistency: the asset stays coherent across frames "
    "(no shape that flips or morphs between views).\n"
    "5. Artifacts: absence of holes, floaters, non-manifold spikes, missing "
    "parts, or solid blank renders.\n\n"
)

_SHAPE_YES_NO_DECISION = (
    "Does the rendered 3D mesh match the condition object's geometry well "
    "enough to count as a successful shape generation? Answer Yes or No."
)

_SHAPE_REASON_BODY = (
    "Briefly analyze the frames in 1-3 sentences, citing concrete visual "
    "evidence for each of the five aspects above (identity, silhouette, "
    "geometry, multi-view consistency, artifacts). Ignore color/texture "
    "differences. Do not output a score."
)

_PROMPT_PRESETS: dict[str, dict[str, str]] = {
    "tex": {
        "framework": _TEX_EVALUATION_FRAMEWORK,
        "yes_no_decision": _TEX_YES_NO_DECISION,
        "reason_body": _TEX_REASON_BODY,
    },
    "shape": {
        "framework": _SHAPE_EVALUATION_FRAMEWORK,
        "yes_no_decision": _SHAPE_YES_NO_DECISION,
        "reason_body": _SHAPE_REASON_BODY,
    },
}


class QwenVLSideBySideReward(UnifiedRewardAPIBase):
    """Side-by-side frames + Yes/No logprob reward (see ``vllm_evaluate``).

    Extra kwargs (beyond ``UnifiedRewardAPIBase``):
        max_frames (int): Max frames to send. Default 16.
        tile_resolution (int): Square tile side before horizontal concat. Default 384.
        top_logprobs (int): Passed to chat completions for Yes/No extraction. Default 20.
        canonicalize (bool): Aggregate yes/YES for logprob aggregation. Default False.
        enable_reason (bool): If True, generate a thinking-conditioned reason
            per sample and use it as additional context for the Yes/No
            question. Default False.
        reason_thinking_token_budget (int): vLLM ``thinking_token_budget`` for
            the reason request — the only knob controlling Qwen3.5 thinking
            effort. Defaults to 1024. The OpenAI-side ``max_tokens`` is
            internally derived as ``thinking_token_budget +
            REASON_FINAL_TOKEN_MARGIN`` so the model always has room to emit
            ``</think>`` and final content after the thinking
            section is forced to close. vLLM must be started with
            ``--reasoning-parser qwen3 --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'``.
    """

    required_fields = ("prompt", "video", "condition_images")
    use_tensor_inputs = False

    REASON_FINAL_TOKEN_MARGIN = 1024

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        self.max_frames = int(config.extra_kwargs.get("max_frames", 16))
        self.tile_resolution = int(config.extra_kwargs.get("tile_resolution", 384))
        self.top_logprobs = int(config.extra_kwargs.get("top_logprobs", 20))
        self.canonicalize = bool(config.extra_kwargs.get("canonicalize", False))
        self.enable_reason = bool(config.extra_kwargs.get("enable_reason", False))
        self.reason_thinking_token_budget = int(
            config.extra_kwargs.get("reason_thinking_token_budget", 1024)
        )
        self.reason_max_tokens = (
            self.reason_thinking_token_budget + self.REASON_FINAL_TOKEN_MARGIN
        )

        self._configure_prompts(config.extra_kwargs.get("prompt_preset", "tex"))

        if self.top_logprobs < 1:
            raise ValueError(
                f"QwenVLSideBySideReward: top_logprobs must be >= 1, got {self.top_logprobs}"
            )
        if self.max_frames <= 0:
            raise ValueError(
                f"QwenVLSideBySideReward: max_frames must be positive, got {self.max_frames}"
            )
        if self.tile_resolution <= 0:
            raise ValueError(
                "QwenVLSideBySideReward: tile_resolution must be positive, "
                f"got {self.tile_resolution}"
            )
        if self.reason_thinking_token_budget <= 0:
            raise ValueError(
                "QwenVLSideBySideReward: reason_thinking_token_budget must be "
                f"positive, got {self.reason_thinking_token_budget}"
            )

    # ============================== Prompt Configuration ==============================

    def _configure_prompts(self, preset_name: str) -> None:
        """Validate ``preset_name`` and bind the resolved prompt strings.

        Sets ``self.prompt_preset`` (the preset name) plus five UPPERCASE
        instance attributes that act as set-once constants for scoring:
        ``self.EVALUATION_FRAMEWORK``, ``self.YES_NO_DECISION``,
        ``self.YES_NO_PROMPT``, ``self.REASON_PROMPT``, and
        ``self.REASON_CONDITIONED_YES_NO_PROMPT``. ``YES_NO_DECISION`` is
        shared verbatim between direct and reason-conditioned yes/no, so
        the prompt-only ablation between the two paths differs ONLY by the
        intermediate assistant reason turn.

        Raises ``ValueError`` if ``preset_name`` is not registered in
        ``_PROMPT_PRESETS``.
        """
        if preset_name not in _PROMPT_PRESETS:
            raise ValueError(
                "QwenVLSideBySideReward: prompt_preset must be one of "
                f"{sorted(_PROMPT_PRESETS)}, got {preset_name!r}"
            )
        preset = _PROMPT_PRESETS[preset_name]
        framework = preset["framework"]
        decision = preset["yes_no_decision"]
        self.prompt_preset = preset_name
        self.EVALUATION_FRAMEWORK = framework
        self.YES_NO_DECISION = decision
        self.YES_NO_PROMPT = framework + decision
        self.REASON_PROMPT = framework + preset["reason_body"]
        self.REASON_CONDITIONED_YES_NO_PROMPT = decision

    # ============================== API Queries ==============================

    async def _query_yes_no_logprob(
        self,
        messages: list,
        semaphore: asyncio.Semaphore | None,
        *,
        client,
    ) -> float:
        """Yes/No first-token logprob — same as ``vllm_evaluate.VLMEvaluateRewardModel``."""
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        create_kwargs = dict(
            model=self.vlm_model,
            messages=messages,
            temperature=0.0,
            max_completion_tokens=1,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            timeout=self.timeout,
            extra_body=extra_body,
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if semaphore is not None:
                    async with semaphore:
                        completion = await client.chat.completions.create(**create_kwargs)
                else:
                    completion = await client.chat.completions.create(**create_kwargs)

                return float(_get_yes_cond_prob(completion, canonicalize=self.canonicalize))

            except Exception as e:
                last_error = e
                logger.warning(
                    "QwenVLSideBySideReward API error on attempt %d/%d: %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries - 1:
                    sleep_time = min(2**attempt + random.uniform(0, 1), 30)
                    await asyncio.sleep(sleep_time)

        raise RuntimeError(
            f"QwenVLSideBySideReward API failed after {self.max_retries} retries: "
            f"{last_error}"
        )

    REASON_FIELD_PRIORITY: Tuple[str, ...] = (
        "content",
        "reasoning",
        "reasoning_content",
    )

    async def _query_reason_text(
        self,
        messages: list,
        semaphore: asyncio.Semaphore | None,
        *,
        client,
    ) -> str:
        """Generate a thinking-conditioned reason for one sample.

        Uses the same VLM endpoint as ``_query_yes_no_logprob`` but with
        ``enable_thinking=True`` and a higher ``max_tokens`` budget.

        Reads the final content first via an explicit field priority list to stay
        compatible with vLLM's evolving schema:

        1. ``content`` — the final answer after the model's thinking block.
        2. ``reasoning`` — current vLLM field (>= 0.17 / Qwen3.5 standard,
           tracked by RFC vllm-project/vllm#27755).
        3. ``reasoning_content`` — deprecated but still emitted on
           older vLLM (<= 0.16) and by ``deepseek_r1`` parser builds.

        All three empty raises ``ValueError``. The caller catches this
        and substitutes ``REASON_FAILED_SENTINEL`` so the sample still
        flows through the reason-conditioned yes/no path (uniform batch
        semantics, no retry, sentinel surfaces in the wandb caption).
        """
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_token_budget": self.reason_thinking_token_budget,
        }
        create_kwargs = dict(
            model=self.vlm_model,
            messages=messages,
            temperature=0.0,
            max_tokens=self.reason_max_tokens,
            timeout=self.timeout,
            extra_body=extra_body,
        )

        if semaphore is not None:
            async with semaphore:
                completion = await client.chat.completions.create(**create_kwargs)
        else:
            completion = await client.chat.completions.create(**create_kwargs)

        message_dict = completion.choices[0].message.model_dump()
        content = None
        used_field = None
        for field_name in self.REASON_FIELD_PRIORITY:
            if field_name in message_dict and message_dict[field_name]:
                content = message_dict[field_name]
                used_field = field_name
                break
        if content is None:
            raise ValueError(
                "QwenVLSideBySideReward reason API returned empty "
                f"{'/'.join(self.REASON_FIELD_PRIORITY)}"
            )
        if not isinstance(content, str):
            raise TypeError(
                "QwenVLSideBySideReward reason API returned non-str "
                f"{used_field}: {type(content).__name__}"
            )
        reason = content.strip()
        if not reason:
            raise ValueError(
                "QwenVLSideBySideReward reason API returned blank reason"
            )
        return reason

    # ============================== Frame Preparation ==============================

    def _sample_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        if len(frames) <= self.max_frames:
            return frames
        indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)  # (max_frames,)
        return [frames[i] for i in indices]

    def _resize_to_tile(self, img: Image.Image) -> Image.Image:
        tile_image = img.convert("RGB").resize(
            (self.tile_resolution, self.tile_resolution),
            Image.Resampling.BILINEAR,
        )  # (tile_res, tile_res, 3) RGB
        return tile_image

    def _build_side_by_side_frames(
        self,
        cond_pil: Image.Image,
        frames_pil: List[Image.Image],
    ) -> List[Image.Image]:
        cond_tile = self._resize_to_tile(cond_pil)  # (tile_res, tile_res, 3) PIL
        cond_arr = np.asarray(cond_tile, dtype=np.uint8)  # (tile_res, tile_res, 3) uint8

        sbs_frames: List[Image.Image] = []
        for frame in frames_pil:
            frame_tile = self._resize_to_tile(frame)  # (tile_res, tile_res, 3) PIL
            frame_arr = np.asarray(frame_tile, dtype=np.uint8)  # (tile_res, tile_res, 3) uint8
            sbs_arr = np.concatenate([cond_arr, frame_arr], axis=1)  # (tile_res, 2*tile_res, 3)
            sbs_frames.append(Image.fromarray(sbs_arr, mode="RGB"))
        return sbs_frames

    # ============================== Message Builders ==============================

    def _build_user_text(self, prompt: str) -> str:
        caption_block = f"Caption: {prompt.strip()}\n\n" if prompt.strip() else ""
        return caption_block + self.YES_NO_PROMPT

    def _build_messages(self, prompt: str, sbs_frames: List[Image.Image]) -> list:
        text = self._build_user_text(prompt)
        content: list = [
            {
                "type": "image_url",
                "image_url": {"url": pil_image_to_base64(frame)},
            }
            for frame in sbs_frames
        ]
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _build_reason_user_text(self, prompt: str) -> str:
        caption_block = f"Caption: {prompt.strip()}\n\n" if prompt.strip() else ""
        return caption_block + self.REASON_PROMPT

    def _build_reason_messages(
        self, prompt: str, sbs_frames: List[Image.Image]
    ) -> list:
        """Build messages for the thinking-reason request.

        Same image payload as ``_build_messages`` but the trailing text
        block asks for a brief visual analysis instead of Yes/No.
        """
        text = self._build_reason_user_text(prompt)
        content: list = [
            {
                "type": "image_url",
                "image_url": {"url": pil_image_to_base64(frame)},
            }
            for frame in sbs_frames
        ]
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _build_reason_conditioned_messages(
        self, prompt: str, sbs_frames: List[Image.Image], reason: str
    ) -> list:
        """Place the generated reason back into the chat as an assistant turn.

        The yes/no question is asked with ``enable_thinking=False`` and
        ``max_completion_tokens=1`` by ``_query_yes_no_logprob``, so this
        list directly drives the second-pass scoring.
        """
        messages = self._build_reason_messages(prompt, sbs_frames)
        messages.append({"role": "assistant", "content": reason})
        messages.append(
            {
                "role": "user",
                "content": self.REASON_CONDITIONED_YES_NO_PROMPT,
            }
        )
        return messages

    # ============================== Scoring ==============================

    REASON_FAILED_SENTINEL = "[reason failed]"

    async def _score_single(
        self,
        prompt: str,
        sbs_frames: List[Image.Image],
        semaphore: asyncio.Semaphore,
        client,
    ) -> Tuple[float, str]:
        """Score one sample. Returns ``(reward, reason)``.

        - ``enable_reason=False``: directly issues 1-token Yes/No on raw
          messages; reason is the empty string.
        - ``enable_reason=True``: first generates a thinking reason, then
          asks Yes/No conditioned on that same reason.

          On a reason API failure the assistant turn falls back to the
          sentinel ``REASON_FAILED_SENTINEL``. The Yes/No request still
          runs through ``_build_reason_conditioned_messages`` so the
          message structure stays uniform across the batch (no NaN
          handling, no mixing with direct yes/no semantics) and the
          sentinel surfaces in the wandb caption for easy diagnosis.
        """
        reason = ""
        if self.enable_reason:
            try:
                reason = await self._query_reason_text(
                    self._build_reason_messages(prompt, sbs_frames),
                    semaphore,
                    client=client,
                )
            except (RuntimeError, TimeoutError, ValueError, TypeError) as e:
                logger.warning(
                    "QwenVLSideBySideReward reason failed for prompt %r "
                    "(%s: %s); using sentinel %r and continuing with "
                    "reason-conditioned yes/no",
                    prompt[:64],
                    type(e).__name__,
                    e,
                    self.REASON_FAILED_SENTINEL,
                )
                reason = self.REASON_FAILED_SENTINEL
            messages = self._build_reason_conditioned_messages(
                prompt, sbs_frames, reason
            )
        else:
            messages = self._build_messages(prompt, sbs_frames)

        try:
            score = await self._query_yes_no_logprob(messages, semaphore, client=client)
        except RuntimeError:
            score = float("nan")
        return score, reason

    async def _async_score_batch(
        self,
        prompts: List[str],
        sbs_frames_batch: List[List[Image.Image]],
        client,
    ) -> List[Tuple[float, str]]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            self._score_single(p, frames, semaphore, client)
            for p, frames in zip(prompts, sbs_frames_batch)
        ]
        return list(await asyncio.gather(*tasks))

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
            raise ValueError("QwenVLSideBySideReward: 'video' must be provided")
        if condition_images is None:
            raise ValueError(
                "QwenVLSideBySideReward: 'condition_images' must be provided "
                "(this reward needs a reference image to compare against)"
            )

        batch_size = len(prompt)
        if len(video) != batch_size or len(condition_images) != batch_size:
            raise ValueError(
                "QwenVLSideBySideReward batch length mismatch: "
                f"prompt={len(prompt)}, video={len(video)}, "
                f"condition_images={len(condition_images)}"
            )

        sbs_frames_batch: List[List[Image.Image]] = []
        for idx in range(batch_size):
            cond_list = condition_images[idx]
            if cond_list is None or len(cond_list) == 0:
                raise ValueError(
                    f"QwenVLSideBySideReward: condition_images[{idx}] is empty; "
                    "expected at least one reference image per sample"
                )
            cond_pil = cond_list[0]

            frames = video[idx]
            if frames is None or len(frames) == 0:
                raise ValueError(
                    f"QwenVLSideBySideReward: video[{idx}] is empty; "
                    "expected at least one rendered frame"
                )

            sampled = self._sample_frames(list(frames))
            sbs_frames_batch.append(self._build_side_by_side_frames(cond_pil, sampled))

        async def _run() -> List[Tuple[float, str]]:
            client = self._make_client()
            try:
                return await self._async_score_batch(prompt, sbs_frames_batch, client)
            finally:
                await client.close()

        results = asyncio.run(_run())
        scores = [r[0] for r in results]  # (batch_size,)
        reasons = [r[1] for r in results]  # (batch_size,)
        rewards = self._replace_nan_with_mean(
            torch.tensor(scores, dtype=torch.float32)  # (batch_size,)
        )
        # Short-circuit to {} when reasons are not used so RewardProcessor's
        # extra_info path is identical to the pre-thinking behavior.
        extra_info = {"reasons": reasons} if self.enable_reason else {}
        return RewardModelOutput(rewards=rewards, extra_info=extra_info)
