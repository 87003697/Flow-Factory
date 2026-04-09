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

# src/flow_factory/metrics/abc.py
"""
Base class for evaluation metrics.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import torch
from PIL import Image


class BaseMetric(ABC):
    """
    Abstract base class for evaluation metrics.
    
    All metrics should inherit from this class and implement
    at least one of the compute methods.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize the metric.
        
        Args:
            config: Metric-specific configuration dict.
            device: Device to run computations on.
        """
        self.config = config
        self.device = device
    
    @abstractmethod
    def compute(
        self,
        samples: Union[List[Path], List[Image.Image], torch.Tensor],
        prompts: Optional[List[str]] = None,
        references: Optional[Union[List[Path], List[Image.Image], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Compute the metric.
        
        Args:
            samples: Generated samples (paths, PIL images, or tensor).
            prompts: Optional prompts corresponding to samples.
            references: Optional reference samples for comparison metrics.
        
        Returns:
            Dictionary containing metric results.
        """
        pass
    
    def compute_batch(
        self,
        images: List[Image.Image],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Compute metric for a batch of images.
        
        Override for efficient batch processing.
        
        Args:
            images: Batch of PIL images.
            prompts: Optional prompts for each image.
        
        Returns:
            List of scores for each image.
        """
        raise NotImplementedError("Batch computation not implemented for this metric")
    
    def compute_from_paths(
        self,
        sample_paths: List[Path],
        reference_paths: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """
        Compute metric from file paths.
        
        Override for metrics that work directly with paths (e.g., FID).
        
        Args:
            sample_paths: Paths to generated samples.
            reference_paths: Optional paths to reference samples.
        
        Returns:
            Dictionary containing metric results.
        """
        raise NotImplementedError("Path-based computation not implemented for this metric")
    
    def _load_images(self, paths: List[Path]) -> List[Image.Image]:
        """Helper to load images from paths."""
        images = []
        for p in paths:
            try:
                img = Image.open(p).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Warning: Failed to load {p}: {e}")
        return images
