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

# src/flow_factory/metrics/lpips.py
"""
LPIPS (Learned Perceptual Image Patch Similarity) metric.
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


class LPIPSMetric(BaseMetric):
    """
    LPIPS (Learned Perceptual Image Patch Similarity) metric.
    
    Computes perceptual similarity between generated and reference images.
    Lower is better (more similar).
    
    Requires reference images for pairwise comparison.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        
        self.batch_size = config.get('batch_size', 16)
        self.net = config.get('net', 'alex')  # 'alex', 'vgg', 'squeeze'
        
        self._model = None
        self._transform = None
    
    def _load_model(self):
        """Lazy load LPIPS model."""
        if self._model is not None:
            return
        
        try:
            import lpips
            from torchvision import transforms
        except ImportError:
            raise ImportError(
                "LPIPS metric requires lpips package. Install with: pip install lpips"
            )
        
        logger.info(f"Loading LPIPS model ({self.net})...")
        
        self._model = lpips.LPIPS(net=self.net).to(self.device)
        self._model.eval()
        
        # Transform: LPIPS expects [-1, 1] range
        self._transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def _load_and_transform(self, path: Path) -> Optional[torch.Tensor]:
        """Load and transform a single image."""
        try:
            img = Image.open(path).convert('RGB')
            return self._transform(img)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None
    
    def compute(
        self,
        samples: Union[List[Path], List[Image.Image], torch.Tensor],
        prompts: Optional[List[str]] = None,
        references: Optional[Union[List[Path], List[Image.Image], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Compute LPIPS score."""
        if references is None:
            raise ValueError("LPIPS requires reference images for pairwise comparison.")
        
        if isinstance(samples[0], Path) and isinstance(references[0], Path):
            return self.compute_from_paths(list(samples), list(references))
        else:
            raise ValueError("LPIPS currently only supports path-based computation")
    
    def compute_from_paths(
        self,
        sample_paths: List[Path],
        reference_paths: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """Compute LPIPS from file paths (pairwise)."""
        if reference_paths is None:
            raise ValueError("LPIPS requires reference images.")
        
        self._load_model()
        
        # Match samples to references (1:1 or cycle)
        n_samples = len(sample_paths)
        n_refs = len(reference_paths)
        
        if n_samples != n_refs:
            logger.warning(
                f"Sample count ({n_samples}) != reference count ({n_refs}). "
                f"Will compute {min(n_samples, n_refs)} pairwise scores."
            )
        
        n_pairs = min(n_samples, n_refs)
        scores = []
        
        logger.info(f"Computing LPIPS for {n_pairs} image pairs...")
        
        for i in tqdm(range(0, n_pairs, self.batch_size), desc="Computing LPIPS"):
            batch_end = min(i + self.batch_size, n_pairs)
            
            sample_tensors = []
            ref_tensors = []
            
            for j in range(i, batch_end):
                sample_t = self._load_and_transform(sample_paths[j])
                ref_t = self._load_and_transform(reference_paths[j])
                
                if sample_t is not None and ref_t is not None:
                    sample_tensors.append(sample_t)
                    ref_tensors.append(ref_t)
            
            if not sample_tensors:
                continue
            
            samples_batch = torch.stack(sample_tensors).to(self.device)
            refs_batch = torch.stack(ref_tensors).to(self.device)
            
            with torch.no_grad():
                batch_scores = self._model(samples_batch, refs_batch)
                scores.extend(batch_scores.squeeze().cpu().numpy().tolist())
        
        scores = np.array(scores)
        
        return {
            'lpips_mean': float(np.mean(scores)),
            'lpips_std': float(np.std(scores)),
            'lpips_min': float(np.min(scores)),
            'lpips_max': float(np.max(scores)),
            'num_pairs': len(scores),
        }
    
    def compute_batch(
        self,
        images: List[Image.Image],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Not applicable for LPIPS (requires pairs).
        """
        raise NotImplementedError("LPIPS requires pairwise comparison. Use compute_from_paths instead.")
