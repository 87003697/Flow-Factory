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

# src/flow_factory/metrics/clip_score.py
"""
CLIP Score metric for text-image alignment.
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


class CLIPScoreMetric(BaseMetric):
    """
    CLIP Score metric for measuring text-image alignment.
    
    Computes the cosine similarity between CLIP embeddings of
    images and their corresponding prompts.
    
    Higher is better (more aligned).
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        
        self.model_name = config.get('model', 'openai/clip-vit-large-patch14')
        self.batch_size = config.get('batch_size', 32)
        
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """Lazy load CLIP model."""
        if self._model is not None:
            return
        
        try:
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            raise ImportError(
                "CLIP Score requires transformers. Install with: pip install transformers"
            )
        
        logger.info(f"Loading CLIP model: {self.model_name}")
        
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model.eval()
    
    def compute(
        self,
        samples: Union[List[Path], List[Image.Image], torch.Tensor],
        prompts: Optional[List[str]] = None,
        references: Optional[Union[List[Path], List[Image.Image], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Compute CLIP Score."""
        if prompts is None:
            raise ValueError("CLIP Score requires prompts for text-image alignment.")
        
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
            for i in tqdm(range(0, len(samples), self.batch_size), desc="Computing CLIP Score"):
                batch_paths = samples[i:i + self.batch_size]
                batch_prompts = prompts[i:i + self.batch_size]
                batch_images = self._load_images(batch_paths)
                scores.extend(self.compute_batch(batch_images, batch_prompts))
        elif isinstance(samples[0], Image.Image):
            scores = self.compute_batch(list(samples), list(prompts))
        else:
            raise ValueError("CLIP Score expects paths or PIL images")
        
        scores = np.array(scores)
        
        return {
            'clip_score_mean': float(np.mean(scores)),
            'clip_score_std': float(np.std(scores)),
            'clip_score_min': float(np.min(scores)),
            'clip_score_max': float(np.max(scores)),
            'num_samples': len(scores),
        }
    
    def compute_batch(
        self,
        images: List[Image.Image],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        """Compute CLIP scores for a batch of images."""
        if prompts is None:
            raise ValueError("CLIP Score requires prompts")
        
        self._load_model()
        
        scores = []
        
        # Count prompts that will be truncated for observability
        _tokenizer = self._processor.tokenizer
        _too_long = sum(
            1 for p in prompts
            if len(_tokenizer.encode(p)) > _tokenizer.model_max_length
        )
        if _too_long:
            logger.warning(
                f"{_too_long}/{len(prompts)} prompts exceed the model's max token length "
                f"({_tokenizer.model_max_length}) and will be truncated. "
                "CLIP Score may be underestimated for long prompts (e.g. multi-panel descriptions)."
            )

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_prompts = prompts[i:i + self.batch_size]
            
            # Process inputs — truncation required: CLIP text encoder is capped at 77 tokens
            inputs = self._processor(
                text=batch_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                
                # Get normalized features
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity (diagonal for matched pairs)
                batch_scores = (image_embeds * text_embeds).sum(dim=-1)
                scores.extend(batch_scores.cpu().numpy().tolist())
        
        return scores
    
    def compute_from_paths(
        self,
        sample_paths: List[Path],
        reference_paths: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """Not the primary interface for CLIP Score."""
        raise NotImplementedError(
            "CLIP Score requires prompts. Use compute() with prompts parameter instead."
        )
