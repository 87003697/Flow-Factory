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

from typing import List

import torch
from accelerate import Accelerator

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments


class AORewardModel(PointwiseRewardModel):
    """
    AO reward measuring clay-mask similarity for rendered 3D geometry.

    For each rendered view the reward is the foreground-masked mean of
    ``clay = 1 - SSAO``.  The per-sample score is the **minimum** across
    all views, so any single angle with geometric artefacts (holes, spikes,
    fragmented self-occluding structure) dominates the reward.
    """

    required_fields = ("clay_video", "mask_video")
    use_tensor_inputs = True

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

    @torch.no_grad()
    def __call__(
        self,
        clay_video: List[torch.Tensor],
        mask_video: List[torch.Tensor],
        **kwargs,
    ) -> RewardModelOutput:
        eps = 1e-6
        rewards = []
        for clay, mask in zip(clay_video, mask_video):
            clay = clay.to(self.device)                               # (T, 1, H, W)
            mask = mask.to(self.device)                               # (T, 1, H, W)
            per_view_num = (clay * mask).sum(dim=(-3, -2, -1))        # (T,)
            per_view_den = mask.sum(dim=(-3, -2, -1)) + eps           # (T,)
            per_view_reward = per_view_num / per_view_den             # (T,)
            rewards.append(per_view_reward.min())                     # scalar: worst-view AO

        reward_tensor = torch.stack(rewards).float().cpu()            # (batch_size,)
        return RewardModelOutput(rewards=reward_tensor)
