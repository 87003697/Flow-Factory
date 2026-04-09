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

# src/flow_factory/metrics/fid.py
"""
Fréchet Inception Distance (FID) metric.
"""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

import torch
import numpy as np
from PIL import Image
from scipy import linalg
from tqdm import tqdm

from .abc import BaseMetric

logger = logging.getLogger(__name__)


class FIDMetric(BaseMetric):
    """
    Fréchet Inception Distance (FID) metric.
    
    Computes the FID between generated samples and reference images
    using InceptionV3 features.
    
    Requires reference images for comparison.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        
        self.batch_size = config.get('batch_size', 32)
        self.dims = config.get('dims', 2048)  # InceptionV3 feature dimension
        
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
            raise ImportError("FID metric requires torchvision. Install with: pip install torchvision")
        
        logger.info("Loading InceptionV3 for FID computation...")
        
        # Load model
        self._model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self._model.fc = torch.nn.Identity()  # Remove classification head
        self._model = self._model.to(self.device)
        self._model.eval()
        
        # Transform
        self._transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _extract_features(self, image_paths: List[Path]) -> np.ndarray:
        """Extract InceptionV3 features from images."""
        self._load_model()
        
        features = []
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i + self.batch_size]
            
            # Load and transform images
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
                feat = self._model(batch)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def _calculate_fid(self, mu1: np.ndarray, sigma1: np.ndarray, 
                       mu2: np.ndarray, sigma2: np.ndarray) -> float:
        """Calculate FID score from statistics."""
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return float(fid)
    
    def compute(
        self,
        samples: Union[List[Path], List[Image.Image], torch.Tensor],
        prompts: Optional[List[str]] = None,
        references: Optional[Union[List[Path], List[Image.Image], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Compute FID score."""
        if references is None:
            raise ValueError("FID requires reference images. Provide 'reference_dir' in config.")
        
        # Convert to paths if needed
        if isinstance(samples[0], Path):
            sample_paths = samples
        else:
            raise ValueError("FID currently only supports path-based computation")
        
        if isinstance(references[0], Path):
            reference_paths = references
        else:
            raise ValueError("FID currently only supports path-based computation")
        
        return self.compute_from_paths(sample_paths, reference_paths)
    
    def compute_from_paths(
        self,
        sample_paths: List[Path],
        reference_paths: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """Compute FID from file paths."""
        if reference_paths is None:
            raise ValueError("FID requires reference images.")
        
        logger.info(f"Computing FID: {len(sample_paths)} samples vs {len(reference_paths)} references")
        
        # Extract features
        logger.info("Extracting features from generated samples...")
        sample_features = self._extract_features(sample_paths)
        
        logger.info("Extracting features from reference images...")
        reference_features = self._extract_features(reference_paths)
        
        # Calculate statistics
        mu_samples = np.mean(sample_features, axis=0)
        sigma_samples = np.cov(sample_features, rowvar=False)
        
        mu_ref = np.mean(reference_features, axis=0)
        sigma_ref = np.cov(reference_features, rowvar=False)
        
        # Calculate FID
        fid_score = self._calculate_fid(mu_samples, sigma_samples, mu_ref, sigma_ref)
        
        return {
            'fid': fid_score,
            'num_samples': len(sample_paths),
            'num_references': len(reference_paths),
        }
