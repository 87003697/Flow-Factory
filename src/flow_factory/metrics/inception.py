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

# src/flow_factory/metrics/inception.py
"""
Inception Score (IS) metric.
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


class InceptionScoreMetric(BaseMetric):
    """
    Inception Score (IS) metric.
    
    Measures quality and diversity of generated images using
    InceptionV3 classifier predictions.
    
    Higher is better (more diverse and confident).
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        
        self.batch_size = config.get('batch_size', 32)
        self.splits = config.get('splits', 10)  # Number of splits for std computation
        
        self._model = None
        self._transform = None
    
    def _load_model(self):
        """Lazy load InceptionV3 model."""
        if self._model is not None:
            return
        
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            from torchvision import transforms
        except ImportError:
            raise ImportError(
                "Inception Score requires torchvision. Install with: pip install torchvision"
            )
        
        logger.info("Loading InceptionV3 for Inception Score...")
        
        self._model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self._model = self._model.to(self.device)
        self._model.eval()
        
        self._transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _get_predictions(self, image_paths: List[Path]) -> np.ndarray:
        """Get softmax predictions for all images."""
        self._load_model()
        
        all_preds = []
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Getting predictions"):
            batch_paths = image_paths[i:i + self.batch_size]
            
            batch_tensors = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert('RGB')
                    tensor = self._transform(img)
                    batch_tensors.append(tensor)
                except Exception as e:
                    logger.warning(f"Failed to load {p}: {e}")
            
            if not batch_tensors:
                continue
            
            batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                logits = self._model(batch)
                preds = torch.softmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
        
        return np.concatenate(all_preds, axis=0)
    
    def _calculate_is(self, preds: np.ndarray) -> tuple:
        """Calculate Inception Score from predictions."""
        # Compute marginal distribution
        p_y = np.mean(preds, axis=0, keepdims=True)
        
        # Compute KL divergence for each sample
        kl_divs = preds * (np.log(preds + 1e-10) - np.log(p_y + 1e-10))
        kl_divs = np.sum(kl_divs, axis=1)
        
        # Compute IS
        is_score = np.exp(np.mean(kl_divs))
        
        return is_score
    
    def compute(
        self,
        samples: Union[List[Path], List[Image.Image], torch.Tensor],
        prompts: Optional[List[str]] = None,
        references: Optional[Union[List[Path], List[Image.Image], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Compute Inception Score."""
        if isinstance(samples[0], Path):
            return self.compute_from_paths(list(samples))
        else:
            raise ValueError("Inception Score currently only supports path-based computation")
    
    def compute_from_paths(
        self,
        sample_paths: List[Path],
        reference_paths: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """Compute Inception Score from file paths."""
        logger.info(f"Computing Inception Score for {len(sample_paths)} samples...")
        
        # Get predictions
        preds = self._get_predictions(sample_paths)
        
        # Compute IS with splits for std estimation
        split_scores = []
        n = len(preds)
        split_size = n // self.splits
        
        for i in range(self.splits):
            start = i * split_size
            end = start + split_size
            split_preds = preds[start:end]
            split_is = self._calculate_is(split_preds)
            split_scores.append(split_is)
        
        is_mean = float(np.mean(split_scores))
        is_std = float(np.std(split_scores))
        
        return {
            'inception_score_mean': is_mean,
            'inception_score_std': is_std,
            'num_samples': len(sample_paths),
            'num_splits': self.splits,
        }
