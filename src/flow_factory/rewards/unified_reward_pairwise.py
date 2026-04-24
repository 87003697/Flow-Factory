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

# src/flow_factory/rewards/unified_reward_pairwise.py
"""
Pairwise win-rate reward models using UnifiedReward VLM API.

Implements ``GroupwiseRewardModel`` subclasses that evaluate all C(K, 2)
pairs within a group and aggregate per-sample win rates.  Two evaluation
modes are provided:

- **think**: Chain-of-thought scoring with ``<answer>Image/Video 1/2 is
  better</answer>`` output.
- **flex**: Structured JSON output with per-category winners and
  configurable overall/dimension weighting.

Each mode has an image variant and a video variant (4 concrete classes).

Concurrency model:

- A fresh ``AsyncOpenAI`` client is created inside each ``asyncio.run()``
  call so that the underlying ``httpx.AsyncClient`` connection pool is
  always bound to the active event loop (reusing a client across
  ``asyncio.run()`` boundaries causes *Connection error*).
- ``asyncio.Semaphore`` is created inside each ``asyncio.run()`` call so
  it belongs to the current event loop.
- With ``async_reward=True, num_workers=N, max_concurrent=M``, total
  in-flight API requests = N * M.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import random
import re
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from ..utils.image import pil_image_to_base64
from .abc import GroupwiseRewardModel, RewardModelOutput

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


THINK_IMAGE_TEMPLATE = """\
You are an objective and precise evaluator for image quality comparison. \
I will provide you with a text caption and two images generated based on \
this caption. You must analyze the two images carefully and determine \
which image is better.

Evaluation procedure:

1. The caption for the generated images is: 「{prompt}」. You must \
evaluate the two images across these core dimensions:
- Semantic consistency (how closely the image content aligns with the caption)
- Aesthetics (composition, color usage, artistic expression)
- Authenticity (realism and attention to detail)

2. You are also encouraged to add up to two additional evaluation \
dimensions if they are relevant to the specific caption or images \
(e.g., creativity, spatial layout, fine-grained detail). If no extra \
dimensions are relevant, just keep the three core dimensions.

3. For each evaluation dimension:
- Provide a score between 1–10 for both Image 1 and Image 2
- Provide a short rationale for each score (2–5 short sentences)
- The evaluation must follow exactly this format with line breaks and \
indentation:
    Dimension name: 
        Image 1 (x/10) - rationale; 
        Image 2 (y/10) - rationale

4. After evaluating all dimensions, calculate the total score for each \
image and show the calculation explicitly, following this exact format:
    Total score:
    Image 1: x+x+x=total
    Image 2: y+y+y=total

5. Wrap all reasoning and scoring strictly within <think> and </think> tags.

6. After </think>, output the final judgment strictly inside <answer> \
and </answer> tags, containing only one of:
- Image 1 is better
- Image 2 is better

Constraints:
- You must strictly follow the line breaks, indentation, and formatting \
shown in the example below.
- Do not merge multiple dimensions into one line. Each dimension must \
follow the 3-line block format shown below.
- Do not use Markdown formatting, bullet points, bold text, or headings.
- Do not output explanations outside <think> and <answer>.
- The <answer> tag must contain only the final string with no extra words.

Required output format:

<think>
1. Semantic consistency: 
    Image 1 (9/10) - ...; 
    Image 2 (7/10) - ...
2. Aesthetics: 
    Image 1 (8/10) - ...; 
    Image 2 (8/10) - ...
3. Authenticity: 
    Image 1 (8/10) - ...; 
    Image 2 (5/10) - ...
[Additional dimension if any]: 
    Image 1 (7/10) - ...; 
    Image 2 (8/10) - ...
[Additional dimension if any]: 
    Image 1 (6/10) - ...; 
    Image 2 (7/10) - ...
Total score:
Image 1: 9+8+8+7+6=38
Image 2: 7+8+5+8+7=35
</think>
<answer>Image 1 is better</answer>

Note: The example above is only to illustrate the exact format \
(line breaks, indentation, symbols, and style). Your actual evaluation \
must follow this format exactly, but be based on the given caption and \
images."""

THINK_VIDEO_TEMPLATE = """\
You are an objective and precise evaluator for video quality comparison. \
I will provide you with a text caption and a sequence of consecutive \
frames extracted from two generated videos based on this caption. The \
first half of the frames belong to Video 1, and the second half of the \
frames belong to Video 2. You must analyze these two videos carefully \
and determine which video is better.

Instructions (MUST follow strictly):
1. All reasoning, analysis, explanations, and scores MUST be written \
strictly inside <think> and </think> tags.
2. The <think> block must start immediately with the first evaluation \
dimension. Do NOT include any introduction, notes, or explanations \
before the first numbered dimension.
3. After </think>, output the final judgment strictly inside <answer> \
and </answer> tags, containing only one of:
- Video 1 is better
- Video 2 is better
4. Do NOT output anything outside <think> and <answer>. No extra \
explanations, notes, or prefaces.

Evaluation procedure:

1. The caption for the generated videos is: 「{prompt}」. The provided \
frames represent two candidate videos:
- First half: Video 1
- Second half: Video 2

2. You must evaluate the two videos across these core dimensions:
- Semantic consistency (how closely the video content aligns with the caption)
- Temporal coherence (smoothness and logical flow of motion across frames)
- Authenticity (realism and attention to detail)

3. You may also add up to two additional evaluation dimensions if they \
are clearly relevant (e.g., camera stability, lighting consistency, \
creativity). If no extra dimensions are relevant, keep only the three \
core dimensions.

4. For each evaluation dimension:
- Provide a score between 1–10 for both Video 1 and Video 2.
- Provide a short rationale for each score (2–5 short sentences).
- Each dimension must follow exactly this 3-line block format with \
numbering, line breaks, and indentation:
    N. Dimension name: 
        Video 1 (x/10) - rationale; 
        Video 2 (y/10) - rationale

5. After evaluating all dimensions, calculate the total score for each \
video and show the calculation explicitly, following this exact format:
    Total score:
    Video 1: x+x+x(+...)=total
    Video 2: y+y+y(+...)=total

6. All reasoning, analysis, scoring, and totals must be written strictly \
inside <think> and </think> tags.

Required output format (follow this exactly, including line breaks and \
indentation):

<think>
1. Semantic consistency: 
    Video 1 (9/10) - ...; 
    Video 2 (7/10) - ...
2. Temporal coherence: 
    Video 1 (8/10) - ...; 
    Video 2 (6/10) - ...
3. Authenticity: 
    Video 1 (7/10) - ...; 
    Video 2 (5/10) - ...
[Additional dimension if any]: 
    Video 1 (8/10) - ...; 
    Video 2 (6/10) - ...
[Additional dimension if any]: 
    Video 1 (7/10) - ...; 
    Video 2 (7/10) - ...
Total score:
Video 1: 9+8+7+8+7=39
Video 2: 7+6+5+6+7=31
</think>
<answer>Video 1 is better</answer>

Note: The example above is only to illustrate the exact format \
(numbering, line breaks, indentation, and style). Your actual evaluation \
must follow this format exactly, but be based on the given caption and \
the two provided videos (frames divided into two halves)."""

FLEX_IMAGE_TEMPLATE = """\
## Identity
You are a top-tier AI Image Content Evaluation Expert. Your task is to \
perform a hierarchical, multi-dimensional comparative analysis of \
Image 1 and Image 2 based on the provided Prompt.

## Evaluation Framework

### 1. Mandatory Starting Categories
For every evaluation, you MUST address these three core areas, but you \
should **independently define 3-5 sub-dimensions** for each based on \
what makes the images distinct:
- **A. Semantic Alignment & Accuracy**: Evaluate how well the images \
capture the prompt's subjects, actions, and constraints.
- **B. Image Quality & Realism**: Evaluate technical execution, \
physical logic, and visual clarity.
- **C. Aesthetics & Artistry**: Evaluate artistic appeal, color \
harmony, and compositional mastery.
*Note: If the prompt involves unique traits, you are encouraged to add \
a personalized Category D.*

### 2. Scoring & Reasoning Rules
- **Dynamic Dimensions**: Do not rely on a fixed list. Choose \
sub-dimensions that best highlight the differences between the two images.
- **Sum-of-10 Constraint**: For every sub-dimension, the scores for \
Image 1 and Image 2 MUST total exactly 10 (e.g., 8+2, 5+5).
- **Evidence-Based Reasoning**: Provide professional, critical analysis \
for each score. Avoid generic praise; point out specific visual evidence.

## Input Data
**Prompt:** [{prompt}]

**Content to be Evaluated:**
[Image 1] 
[Image 2] 

## Output Format
Output the results as a single, complete JSON object.

```json
{{
  "prompt": "[Original Prompt]",
  "categories": [
    {{
      "name": "[Category Name]",
      "dims": [
        {{
          "name": "[Custom Sub-dimension]",
          "reason_1": "[Specific evidence]",
          "reason_2": "[Specific evidence]",
          "score_1": 0-10,
          "score_2": 0-10
        }}
      ],
      "cat_reason": "[Category-level analysis]",
      "cat_winner": "Image 1/2"
    }}
  ],
  "reason": "[Overall analysis]",
  "winner": "Image 1/2"
}}"""

FLEX_VIDEO_TEMPLATE = """\
## Identity
You are a top-tier AI Video Evaluation Expert. Perform a hierarchical, \
multi-dimensional comparative analysis of Video 1 and Video 2 based on \
the provided Prompt.

## Evaluation Framework

### 1. Mandatory Categories
For each, independently define **3-5 specific sub-dimensions** based \
on the videos' actual content:
- **A. Semantic Alignment & Accuracy**: Accuracy of subjects, \
attributes, spatial relationships, and environment as defined by the prompt.
- **B. Video Quality & Dynamic Realism**: Technical fidelity, temporal \
stability (no flickering/warping), subject identity persistence, and \
physical plausibility of motion.
- **C. Narrative, Aesthetics & Cinematography**: Composition, color \
harmony, camera movement quality (smoothness/intent), and narrative flow.
*Note: If the prompt involves unique traits, you are encouraged to add \
a personalized Category D.*

### 2. Core Rules
- **Dynamic Selection**: Do NOT simply copy a fixed list. Choose \
sub-dimensions that most effectively differentiate the two videos.
- **Sum-of-10 Scoring**: For every sub-dimension, the total score \
(Video 1 + Video 2) MUST strictly equal 10 points (e.g., 6+4, 5+5).
- **Evidence-Based Reasoning**: Provide professional, critical analysis \
pointing to specific visual/temporal evidence.

## Input Data
**Prompt:** [{prompt}]

**Content to be Evaluated:**
[Video 1] 
[Video 2] 

## Output Format
Return a single, valid JSON object in English.

```json
{{
  "prompt": "[Original Prompt]",
  "categories": [
    {{
      "name": "[Category Name]",
      "dims": [
        {{
          "name": "[Custom Sub-dimension]",
          "reason_1": "[Specific evidence]",
          "reason_2": "[Specific evidence]",
          "score_1": 0-10,
          "score_2": 0-10
        }}
      ],
      "cat_reason": "[Category-level analysis]",
      "cat_winner": "Video 1/2"
    }}
  ],
  "reason": "[Overall analysis]",
  "winner": "Video 1/2"
}}"""


@dataclass
class PairResult:
    """Evaluation result for a single pair comparison."""

    pair: Tuple[int, int]
    overall_winner: Optional[int]
    cat_winners: List[Tuple[str, Optional[int]]] = field(default_factory=list)


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_BETTER_PATTERNS_1 = ("Image 1 is better", "Video 1 is better")
_BETTER_PATTERNS_2 = ("Image 2 is better", "Video 2 is better")


def _parse_think_winner(response_text: str) -> Optional[int]:
    """Extract winner from think-mode ``<answer>`` tags."""
    match = _ANSWER_RE.search(response_text)
    if match is None:
        return None
    answer = match.group(1).strip()
    if any(p in answer for p in _BETTER_PATTERNS_1):
        return 1
    if any(p in answer for p in _BETTER_PATTERNS_2):
        return 2
    return None


def _parse_json_payload(text: str) -> Optional[dict]:
    """Parse JSON from flex-mode response, with greedy-regex fallback."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _normalize_winner(text: str) -> Optional[int]:
    """Map winner string to 1 (first wins) or 2 (second wins)."""
    if not text:
        return None
    lowered = text.lower()
    first = re.search(r"\b(?:video|image)\s*1\b", lowered)
    second = re.search(r"\b(?:video|image)\s*2\b", lowered)
    if first and not second:
        return 1
    if second and not first:
        return 2
    return None


def _iter_category_winners(data: dict) -> List[Tuple[str, Optional[int]]]:
    """Extract per-category winners from flex JSON payload (up to 3 categories)."""
    result: List[Tuple[str, Optional[int]]] = []
    for cat in data.get("categories", [])[:3]:
        cat_name = str(cat.get("name", "")).strip() or "category"
        winner = _normalize_winner(str(cat.get("cat_winner", "")))
        result.append((cat_name, winner))
    return result


def _parse_flex_response(
    response_text: str,
) -> Tuple[Optional[int], List[Tuple[str, Optional[int]]]]:
    """Parse flex-mode JSON response into overall winner and category winners."""
    parsed = _parse_json_payload(response_text)
    if parsed is None:
        return None, []
    overall_winner = _normalize_winner(str(parsed.get("winner", "")))
    cat_winners = _iter_category_winners(parsed)
    return overall_winner, cat_winners


def _sample_frames(frames: List[Image.Image], max_frames: int) -> List[Image.Image]:
    """Uniformly sub-sample video frames down to *max_frames*."""
    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)  # (max_frames,)
    return [frames[i] for i in indices]


def _init_flex_weights(instance: object, config: RewardArguments) -> None:
    """Read flex-mode weight kwargs from *config* onto *instance*."""
    instance.overall_weight = config.extra_kwargs.get("overall_weight", 1.0)
    instance.dim_weight = config.extra_kwargs.get("dim_weight", 1.0)
    raw = config.extra_kwargs.get("category_weights", None)
    if isinstance(raw, str):
        instance.category_weights = [float(x.strip()) for x in raw.split(",") if x.strip()]
    elif isinstance(raw, list):
        instance.category_weights = [float(x) for x in raw]
    else:
        instance.category_weights = None


def _aggregate_flex_win_rate(
    group_size: int,
    results: List[PairResult],
    overall_weight: float,
    dim_weight: float,
    category_weights: Optional[List[float]],
) -> RewardModelOutput:
    """Aggregate overall + per-category win rates with configurable weights.

    Used by flex-mode subclasses.  Combines the overall winner signal
    with per-category dimension signals via a weighted average.
    """
    overall_win: Dict[int, float] = defaultdict(float)
    overall_cmp: Dict[int, int] = defaultdict(int)
    dim_win: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    dim_cmp: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    dim_key_order: List[str] = []

    for result in results:
        idx1, idx2 = result.pair
        overall_cmp[idx1] += 1
        overall_cmp[idx2] += 1
        if result.overall_winner == 1:
            overall_win[idx1] += 1.0
        elif result.overall_winner == 2:
            overall_win[idx2] += 1.0
        else:
            overall_win[idx1] += 0.5
            overall_win[idx2] += 0.5

        for dim_key, cat_winner in result.cat_winners:
            if dim_key not in dim_key_order:
                dim_key_order.append(dim_key)
            dim_cmp[dim_key][idx1] += 1
            dim_cmp[dim_key][idx2] += 1
            if cat_winner == 1:
                dim_win[dim_key][idx1] += 1.0
            elif cat_winner == 2:
                dim_win[dim_key][idx2] += 1.0
            else:
                dim_win[dim_key][idx1] += 0.5
                dim_win[dim_key][idx2] += 0.5

    overall_win_rate = [overall_win[k] / max(overall_cmp[k], 1) for k in range(group_size)]

    if not dim_key_order or dim_weight <= 0:
        return RewardModelOutput(
            rewards=torch.tensor(overall_win_rate, dtype=torch.float32),
            extra_info={"overall_win_rate": overall_win_rate},
        )

    cat_wts = category_weights or [1.0] * len(dim_key_order)
    if len(cat_wts) < len(dim_key_order):
        cat_wts = cat_wts + [1.0] * (len(dim_key_order) - len(cat_wts))
    cat_wts = cat_wts[: len(dim_key_order)]

    dim_mean_win_rate: List[float] = []
    has_dim_data: List[bool] = []
    for idx in range(group_size):
        per_dim_rates: List[float] = []
        per_dim_weights: List[float] = []
        for dim_key, w in zip(dim_key_order, cat_wts):
            if dim_cmp[dim_key][idx] > 0:
                per_dim_rates.append(dim_win[dim_key][idx] / dim_cmp[dim_key][idx])
                per_dim_weights.append(w)
        has_dim_data.append(bool(per_dim_rates))
        if per_dim_rates and sum(per_dim_weights) > 0:
            weighted = sum(r * w for r, w in zip(per_dim_rates, per_dim_weights))
            dim_mean_win_rate.append(weighted / sum(per_dim_weights))
        else:
            dim_mean_win_rate.append(0.0)

    rewards: List[float] = []
    for idx in range(group_size):
        overall = overall_win_rate[idx]
        dim_mean = dim_mean_win_rate[idx]
        if has_dim_data[idx]:
            reward = (overall_weight * overall + dim_weight * dim_mean) / (
                overall_weight + dim_weight
            )
        else:
            reward = overall
        rewards.append(reward)

    return RewardModelOutput(
        rewards=torch.tensor(rewards, dtype=torch.float32),
        extra_info={
            "overall_win_rate": overall_win_rate,
            "dim_mean_win_rate": dim_mean_win_rate,
        },
    )


class UnifiedRewardPairwiseBase(GroupwiseRewardModel):
    """Groupwise pairwise win-rate reward using the UnifiedReward VLM API.

    Enumerates all C(K, 2) pairs within a group (optionally capped by
    ``max_pairs``), queries the VLM for each pair, and aggregates
    per-sample win rates.

    Subclasses must implement ``_build_pair_messages`` and
    ``_parse_response``.

    Extra kwargs (passed via YAML ``extra_kwargs`` or top-level keys):
        api_base_url (str): Default ``"http://localhost:8080/v1"``
        api_key (str): Default ``"EMPTY"``
        vlm_model (str): Served model name.  Default ``"UnifiedReward"``
        max_concurrent (int): Per-call concurrency limit.  Default 64
        max_retries (int): Max retries per API call.  Default 10
        timeout (int): Timeout in seconds.  Default 120
        max_tokens (int): Max generation tokens.  Default 4096
        max_pairs (int | None): Cap on number of pairs to evaluate.
            ``None`` (default) evaluates all C(K, 2) pairs.
    """

    required_fields: Tuple[str, ...] = ("prompt",)
    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError(
                "UnifiedReward pairwise requires the `openai` package. "
                "Install it with: pip install openai"
            ) from None

        self.api_base_url = config.extra_kwargs.get("api_base_url", "http://localhost:8080/v1")
        self.api_key = config.extra_kwargs.get("api_key", "EMPTY")
        self.vlm_model = config.extra_kwargs.get("vlm_model", "UnifiedReward")
        self.max_concurrent = config.extra_kwargs.get("max_concurrent", 64)
        self.max_retries = config.extra_kwargs.get("max_retries", 10)
        self.timeout = config.extra_kwargs.get("timeout", 120)
        self.max_tokens = config.extra_kwargs.get("max_tokens", 4096)
        self.max_pairs: int | None = config.extra_kwargs.get("max_pairs", None)

    def _make_client(self):
        """Create a fresh ``AsyncOpenAI`` client for the current event loop.

        A new client must be created per ``asyncio.run()`` call because
        the internal ``httpx.AsyncClient`` connection pool is bound to the
        event loop that was active when the first request was made.  Reusing
        a client across ``asyncio.run()`` boundaries causes *Connection error*
        since the old event loop has been closed.
        """
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            base_url=self.api_base_url,
            api_key=self.api_key,
        )

    async def _query_api_text(
        self,
        messages: list,
        semaphore: asyncio.Semaphore,
        client,
    ) -> str:
        """Send a single chat-completion request with exponential-backoff retry.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with semaphore:
                    completion = await client.chat.completions.create(
                        model=self.vlm_model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                    )
                return completion.choices[0].message.content

            except Exception as e:
                last_error = e
                logger.warning(
                    "Pairwise API error on attempt %d/%d: %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries - 1:
                    sleep_time = min(2**attempt + random.uniform(0, 1), 30)
                    await asyncio.sleep(sleep_time)

        raise RuntimeError(f"Pairwise API failed after {self.max_retries} retries: {last_error}")

    @abstractmethod
    def _build_pair_messages(
        self,
        prompt: str,
        idx1: int,
        idx2: int,
        image: Optional[List[Image.Image]],
        video: Optional[List[List[Image.Image]]],
    ) -> list:
        """Build OpenAI-compatible chat messages for a single pair."""

    @abstractmethod
    def _parse_response(
        self, response_text: str
    ) -> Tuple[Optional[int], List[Tuple[str, Optional[int]]]]:
        """Parse VLM response into overall winner and per-category winners.

        Returns:
            Tuple of ``(overall_winner, cat_winners)`` where:

            - *overall_winner*: 0 (first wins), 1 (second wins), or
              ``None`` (tie / parse failure)
            - *cat_winners*: list of ``(category_name, winner)`` tuples;
              empty for think mode
        """

    async def _score_pair(
        self,
        semaphore: asyncio.Semaphore,
        idx1: int,
        idx2: int,
        prompt: str,
        image: Optional[List[Image.Image]],
        video: Optional[List[List[Image.Image]]],
        client,
    ) -> PairResult:
        """Score one pair via VLM API and parse the response."""
        messages = self._build_pair_messages(prompt, idx1, idx2, image, video)
        text = await self._query_api_text(messages, semaphore, client)
        overall_winner, cat_winners = self._parse_response(text)
        return PairResult(
            pair=(idx1, idx2),
            overall_winner=overall_winner,
            cat_winners=cat_winners,
        )

    def _aggregate_win_rate(
        self,
        group_size: int,
        results: List[PairResult],
    ) -> RewardModelOutput:
        """Aggregate pairwise results into per-sample overall win-rate rewards."""
        win_count: Dict[int, float] = defaultdict(float)
        compare_count: Dict[int, int] = defaultdict(int)

        for result in results:
            idx1, idx2 = result.pair
            compare_count[idx1] += 1
            compare_count[idx2] += 1
            if result.overall_winner == 1:
                win_count[idx1] += 1.0
            elif result.overall_winner == 2:
                win_count[idx2] += 1.0
            else:
                win_count[idx1] += 0.5
                win_count[idx2] += 0.5

        overall_win_rate = [win_count[k] / max(compare_count[k], 1) for k in range(group_size)]
        return RewardModelOutput(
            rewards=torch.tensor(overall_win_rate, dtype=torch.float32),
            extra_info={"overall_win_rate": overall_win_rate},
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
        group_size = len(prompt)
        shared_prompt = prompt[0]

        all_pairs = list(itertools.combinations(range(group_size), 2))
        if self.max_pairs is not None and len(all_pairs) > self.max_pairs:
            all_pairs = random.sample(all_pairs, self.max_pairs)

        async def _run() -> List[PairResult]:
            client = self._make_client()
            semaphore = asyncio.Semaphore(self.max_concurrent)
            try:
                tasks = [
                    self._score_pair(semaphore, i, j, shared_prompt, image, video, client)
                    for i, j in all_pairs
                ]
                return list(await asyncio.gather(*tasks))
            finally:
                await client.close()

        results = asyncio.run(_run())
        return self._aggregate_win_rate(group_size, results)


class UnifiedRewardThinkImagePairwise(UnifiedRewardPairwiseBase):
    """Think-mode pairwise reward for image generation."""

    required_fields = ("prompt", "image")

    def _build_pair_messages(
        self,
        prompt: str,
        idx1: int,
        idx2: int,
        image: Optional[List[Image.Image]],
        video: Optional[List[List[Image.Image]]],
    ) -> list:
        if image is None:
            raise ValueError("'image' is required for image pairwise reward")
        question = THINK_IMAGE_TEMPLATE.format(prompt=prompt)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image[idx1])}},
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image[idx2])}},
                    {"type": "text", "text": question},
                ],
            }
        ]

    def _parse_response(
        self, response_text: str
    ) -> Tuple[Optional[int], List[Tuple[str, Optional[int]]]]:
        return _parse_think_winner(response_text), []


class UnifiedRewardThinkVideoPairwise(UnifiedRewardPairwiseBase):
    """Think-mode pairwise reward for video generation."""

    required_fields = ("prompt", "video")

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.max_frames: int = config.extra_kwargs.get("max_frames", 8)

    def _build_pair_messages(
        self,
        prompt: str,
        idx1: int,
        idx2: int,
        image: Optional[List[Image.Image]],
        video: Optional[List[List[Image.Image]]],
    ) -> list:
        if video is None:
            raise ValueError("'video' is required for video pairwise reward")
        frames_1 = _sample_frames(video[idx1], self.max_frames)
        frames_2 = _sample_frames(video[idx2], self.max_frames)
        content: list = []
        for frame in frames_1 + frames_2:
            content.append({"type": "image_url", "image_url": {"url": pil_image_to_base64(frame)}})
        content.append({"type": "text", "text": THINK_VIDEO_TEMPLATE.format(prompt=prompt)})
        return [{"role": "user", "content": content}]

    def _parse_response(
        self, response_text: str
    ) -> Tuple[Optional[int], List[Tuple[str, Optional[int]]]]:
        return _parse_think_winner(response_text), []


class UnifiedRewardFlexImagePairwise(UnifiedRewardPairwiseBase):
    """Flex-mode pairwise reward for image generation."""

    required_fields = ("prompt", "image")

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        _init_flex_weights(self, config)

    def _aggregate_win_rate(self, group_size: int, results: List[PairResult]) -> RewardModelOutput:
        return _aggregate_flex_win_rate(
            group_size, results, self.overall_weight, self.dim_weight, self.category_weights
        )

    def _build_pair_messages(
        self,
        prompt: str,
        idx1: int,
        idx2: int,
        image: Optional[List[Image.Image]],
        video: Optional[List[List[Image.Image]]],
    ) -> list:
        if image is None:
            raise ValueError("'image' is required for image pairwise reward")
        question = FLEX_IMAGE_TEMPLATE.format(prompt=prompt)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image[idx1])}},
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image[idx2])}},
                    {"type": "text", "text": question},
                ],
            }
        ]

    def _parse_response(
        self, response_text: str
    ) -> Tuple[Optional[int], List[Tuple[str, Optional[int]]]]:
        return _parse_flex_response(response_text)


class UnifiedRewardFlexVideoPairwise(UnifiedRewardPairwiseBase):
    """Flex-mode pairwise reward for video generation."""

    required_fields = ("prompt", "video")

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.max_frames: int = config.extra_kwargs.get("max_frames", 8)
        _init_flex_weights(self, config)

    def _aggregate_win_rate(self, group_size: int, results: List[PairResult]) -> RewardModelOutput:
        return _aggregate_flex_win_rate(
            group_size, results, self.overall_weight, self.dim_weight, self.category_weights
        )

    def _build_pair_messages(
        self,
        prompt: str,
        idx1: int,
        idx2: int,
        image: Optional[List[Image.Image]],
        video: Optional[List[List[Image.Image]]],
    ) -> list:
        if video is None:
            raise ValueError("'video' is required for video pairwise reward")
        frames_1 = _sample_frames(video[idx1], self.max_frames)
        frames_2 = _sample_frames(video[idx2], self.max_frames)
        content: list = []
        for frame in frames_1 + frames_2:
            content.append({"type": "image_url", "image_url": {"url": pil_image_to_base64(frame)}})
        content.append({"type": "text", "text": FLEX_VIDEO_TEMPLATE.format(prompt=prompt)})
        return [{"role": "user", "content": content}]

    def _parse_response(
        self, response_text: str
    ) -> Tuple[Optional[int], List[Tuple[str, Optional[int]]]]:
        return _parse_flex_response(response_text)
