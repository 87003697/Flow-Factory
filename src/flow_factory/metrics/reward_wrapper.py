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

# src/flow_factory/metrics/reward_wrapper.py
"""
Wrappers to use existing reward models as evaluation metrics.
"""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from .abc import BaseMetric

logger = logging.getLogger(__name__)


class PickScoreMetric(BaseMetric):
    """
    PickScore as an evaluation metric.
    
    Wraps the PickScore reward model for standalone evaluation.
    Higher scores indicate better text-image alignment.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        
        self.model_name = config.get(
            'model_name_or_path', 
            'yuvalkirstain/PickScore_v1'
        )
        self.batch_size = config.get('batch_size', 32)
        
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """Lazy load PickScore model."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoProcessor, AutoModel
        except ImportError:
            raise ImportError(
                "PickScore requires transformers. Install with: pip install transformers"
            )
        
        logger.info(f"Loading PickScore model: {self.model_name}")
        
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).eval().to(self.device)
    
    def compute(
        self,
        samples: Union[List[Path], List[Image.Image], torch.Tensor],
        prompts: Optional[List[str]] = None,
        references: Optional[Union[List[Path], List[Image.Image], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Compute PickScore."""
        if prompts is None:
            raise ValueError("PickScore requires prompts for scoring.")
        
        self._load_model()
        
        n = min(len(samples), len(prompts))
        if len(samples) != len(prompts):
            logger.warning(
                f"Sample count ({len(samples)}) != prompt count ({len(prompts)}). "
                f"Will compute {n} scores."
            )
        
        samples = samples[:n]
        prompts = prompts[:n]
        
        if isinstance(samples[0], Path):
            scores = []
            for i in tqdm(range(0, len(samples), self.batch_size), desc="Computing PickScore"):
                batch_paths = samples[i:i + self.batch_size]
                batch_prompts = prompts[i:i + self.batch_size]
                batch_images = self._load_images(batch_paths)
                scores.extend(self.compute_batch(batch_images, batch_prompts))
        elif isinstance(samples[0], Image.Image):
            scores = self.compute_batch(list(samples), list(prompts))
        else:
            raise ValueError("PickScore expects paths or PIL images")
        
        scores = np.array(scores)
        
        return {
            'pickscore_mean': float(np.mean(scores)),
            'pickscore_std': float(np.std(scores)),
            'pickscore_min': float(np.min(scores)),
            'pickscore_max': float(np.max(scores)),
            'num_samples': len(scores),
        }
    
    def compute_batch(
        self,
        images: List[Image.Image],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        """Compute PickScore for a batch of images."""
        if prompts is None:
            raise ValueError("PickScore requires prompts")
        
        self._load_model()
        
        # Ensure alignment
        n = min(len(images), len(prompts))
        images = images[:n]
        prompts = prompts[:n]
        
        scores = []
        
        for i in tqdm(range(0, n, self.batch_size), desc="Computing PickScore"):
            batch_images = images[i:i + self.batch_size]
            batch_prompts = prompts[i:i + self.batch_size]
            
            # Process inputs — combine text + image in one processor call so that
            # model(**inputs) returns outputs.image_embeds / outputs.text_embeds as
            # plain tensors (compatible with current transformers versions).
            inputs = self._processor(
                text=batch_prompts,
                images=batch_images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            
            with torch.no_grad():
                # outputs.image_embeds and outputs.text_embeds are already projected
                # tensors of shape (batch, embed_dim); no need for get_*_features().
                # Score range: [0, ~100] due to logit_scale multiplication.
                outputs = self._model(**inputs)
                image_embs = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embs = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                
                # Compute scores
                batch_scores = (image_embs * text_embs).sum(dim=-1) * self._model.logit_scale.exp()
                scores.extend(batch_scores.cpu().numpy().tolist())
        
        return scores
