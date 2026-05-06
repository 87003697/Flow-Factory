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

from typing import Optional

import torch
from accelerate import Accelerator
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ..hparams import RewardArguments
from .abc import PointwiseRewardModel, RewardModelOutput
from .pick_score import _extract_feature_tensor


class PickScoreTextImageSumRewardModel(PointwiseRewardModel):
    required_fields = ("prompt", "video", "condition_images")

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)

    def _encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {
            key: value.to(device=self.device) for key, value in image_inputs.items()  # (N, ...)
        }

        image_embs = _extract_feature_tensor(
            self.model.get_image_features(**image_inputs)
        )  # (N, D)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)  # (N, D)
        return image_embs

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        text_inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {
            key: value.to(device=self.device) for key, value in text_inputs.items()  # (N, ...)
        }

        text_embs = _extract_feature_tensor(self.model.get_text_features(**text_inputs))  # (N, D)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)  # (N, D)
        return text_embs

    def _compute_video_scores(
        self,
        prompt: list[str],
        video: list[list[Image.Image]],
        condition_images: list[list[Image.Image]],
        batch_size: int,
    ) -> torch.Tensor:
        """Compute text+image PickScore sums across rendered video frames."""
        frame_counts = [len(clip) for clip in video]
        text_embs = self._encode_texts(prompt)  # (B, D)
        source_images = [images[0] for images in condition_images]
        source_image_embs = self._encode_images(source_images)  # (B, D)

        flat_frames = [frame for clip in video for frame in clip]
        all_frame_embs = []
        for start in range(0, len(flat_frames), batch_size):
            frame_embs = self._encode_images(flat_frames[start : start + batch_size])  # (batch, D)
            all_frame_embs.append(frame_embs)
        flat_frame_embs = torch.cat(all_frame_embs, dim=0)  # (sum_T, D)

        logit_scale = self.model.logit_scale.exp()  # ()
        frame_emb_groups = flat_frame_embs.split(frame_counts)  # tuple[(T_i, D), ...]
        scores = []
        for idx, frame_embs_for_sample in enumerate(frame_emb_groups):
            text_scores = logit_scale * (text_embs[idx] * frame_embs_for_sample).sum(
                dim=-1
            )  # (T_i,)
            image_scores = logit_scale * (source_image_embs[idx] * frame_embs_for_sample).sum(
                dim=-1
            )  # (T_i,)
            sample_score = (text_scores + image_scores).sum()  # ()
            scores.append(sample_score)
        return torch.stack(scores)  # (B,)

    @staticmethod
    def _validate_inputs(
        prompt: list[str],
        video: Optional[list[list[Image.Image]]],
        condition_images: Optional[list[list[Image.Image]]],
    ) -> None:
        if video is None:
            raise ValueError("PickScoreTextImageSumRewardModel: 'video' must be provided")
        if condition_images is None:
            raise ValueError(
                "PickScoreTextImageSumRewardModel: 'condition_images' must be "
                "provided (this reward needs a reference image to compare against)"
            )

        batch_size = len(prompt)
        if len(video) != batch_size or len(condition_images) != batch_size:
            raise ValueError(
                "PickScoreTextImageSumRewardModel batch length mismatch: "
                f"prompt={len(prompt)}, video={len(video)}, "
                f"condition_images={len(condition_images)}"
            )

        for idx in range(batch_size):
            if video[idx] is None or len(video[idx]) == 0:
                raise ValueError(
                    f"PickScoreTextImageSumRewardModel: video[{idx}] is empty; "
                    "expected at least one rendered frame"
                )
            if condition_images[idx] is None or len(condition_images[idx]) == 0:
                raise ValueError(
                    "PickScoreTextImageSumRewardModel: "
                    f"condition_images[{idx}] is empty; expected at least one "
                    "reference image per sample"
                )

    @torch.no_grad()
    def __call__(
        self,
        prompt: list[str],
        image: Optional[list[Image.Image]] = None,
        video: Optional[list[list[Image.Image]]] = None,
        condition_images: Optional[list[list[Image.Image]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        if not isinstance(prompt, list):
            prompt = [prompt]
        self._validate_inputs(prompt, video, condition_images)
        assert video is not None
        assert condition_images is not None

        batch_size = getattr(self.config, "batch_size", len(prompt))
        scores = self._compute_video_scores(
            prompt=prompt,
            video=video,
            condition_images=condition_images,
            batch_size=batch_size,
        )  # (B,)
        rewards = scores / 26  # (B,)
        return RewardModelOutput(rewards=rewards, extra_info={})
