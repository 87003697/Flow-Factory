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

# src/flow_factory/evaluate.py
"""
Evaluation entry point for Flow-Factory.
Compute metrics on generated samples.
"""
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import torch
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm

from .metrics import load_metric, list_registered_metrics


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger("flow_factory.evaluate")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvaluateArguments:
    """Evaluation configuration."""
    # Input samples
    samples_dir: str = None
    samples_pattern: str = "*.png"
    
    # Reference data (for FID, etc.)
    reference_dir: Optional[str] = None
    prompts_file: Optional[str] = None
    
    # Metrics configuration
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Output
    output_path: str = "results/metrics.json"
    
    # Device
    device: str = "cuda"
    batch_size: int = 32
    
    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "EvaluateArguments":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract flat config
        flat_config = {
            'samples_dir': config.get('samples_dir'),
            'samples_pattern': config.get('samples_pattern', '*.png'),
            'reference_dir': config.get('reference_dir'),
            'prompts_file': config.get('prompts_file'),
            'metrics': config.get('metrics', {}),
            'output_path': config.get('output', {}).get('output_path', 'results/metrics.json'),
            'device': config.get('device', 'cuda'),
            'batch_size': config.get('batch_size', 32),
        }
        
        return cls(**{k: v for k, v in flat_config.items() if v is not None})


# =============================================================================
# Evaluation Pipeline
# =============================================================================

class EvaluationPipeline:
    """Evaluation pipeline for computing metrics."""
    
    def __init__(self, config: EvaluateArguments):
        self.config = config
        self.device = torch.device(config.device)
        self.metrics = {}
        
        # Load metrics
        self._load_metrics()
    
    def _load_metrics(self):
        """Load and initialize metrics."""
        for metric_name, metric_config in self.config.metrics.items():
            if metric_config is None:
                metric_config = {}
            
            logger.info(f"Loading metric: {metric_name}")
            try:
                metric = load_metric(metric_name, metric_config, self.device)
                self.metrics[metric_name] = metric
            except Exception as e:
                logger.warning(f"Failed to load metric {metric_name}: {e}")
    
    def _load_samples(self) -> List[Path]:
        """Load sample image paths."""
        samples_dir = Path(self.config.samples_dir)
        if not samples_dir.exists():
            raise ValueError(f"Samples directory not found: {samples_dir}")
        
        # Find all matching files
        pattern = self.config.samples_pattern
        sample_paths = sorted(samples_dir.glob(pattern))
        
        if not sample_paths:
            raise ValueError(f"No samples found in {samples_dir} with pattern {pattern}")
        
        logger.info(f"Found {len(sample_paths)} samples")
        return sample_paths
    
    def _load_prompts(self) -> Optional[List[str]]:
        """Load prompts if available."""
        if not self.config.prompts_file:
            return None
        
        prompts_path = Path(self.config.prompts_file)
        if not prompts_path.exists():
            logger.warning(f"Prompts file not found: {prompts_path}")
            return None
        
        prompts = []
        if prompts_path.suffix == '.json':
            with open(prompts_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    prompts = [p if isinstance(p, str) else p.get('prompt', '') for p in data]
                elif isinstance(data, dict):
                    prompts = data.get('prompts', [])
        elif prompts_path.suffix == '.jsonl':
            with open(prompts_path) as f:
                for line in f:
                    item = json.loads(line.strip())
                    prompts.append(item.get('prompt', item) if isinstance(item, dict) else item)
        else:
            with open(prompts_path) as f:
                prompts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(prompts)} prompts")
        return prompts
    
    def _load_references(self) -> Optional[List[Path]]:
        """Load reference image paths if available."""
        if not self.config.reference_dir:
            return None
        
        ref_dir = Path(self.config.reference_dir)
        if not ref_dir.exists():
            logger.warning(f"Reference directory not found: {ref_dir}")
            return None
        
        ref_paths = sorted(ref_dir.glob("*.png")) + sorted(ref_dir.glob("*.jpg"))
        logger.info(f"Found {len(ref_paths)} reference images")
        return ref_paths
    
    def _load_images_batch(self, paths: List[Path], batch_size: int):
        """Generator that yields batches of images."""
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert('RGB')
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load image {p}: {e}")
            yield images
    
    def run(self) -> Dict[str, Any]:
        """Run evaluation and compute all metrics."""
        sample_paths = self._load_samples()
        prompts = self._load_prompts()
        reference_paths = self._load_references()
        
        results = {}
        
        for metric_name, metric in self.metrics.items():
            logger.info(f"Computing {metric_name}...")
            
            try:
                # Use generic compute interface
                result = metric.compute(
                    samples=sample_paths,
                    prompts=prompts,
                    references=reference_paths,
                )
                
                results[metric_name] = result
                logger.info(f"  {metric_name}: {result}")
                
            except Exception as e:
                logger.error(f"Error computing {metric_name}: {e}")
                results[metric_name] = {'error': str(e)}
        
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        output = {
            'samples_dir': str(self.config.samples_dir),
            'reference_dir': str(self.config.reference_dir) if self.config.reference_dir else None,
            'num_samples': len(list(Path(self.config.samples_dir).glob(self.config.samples_pattern))),
            'metrics': results,
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


# =============================================================================
# Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Flow-Factory Evaluation")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--samples_dir", type=str, help="Override samples directory")
    parser.add_argument("--output_path", type=str, help="Override output path")
    parser.add_argument("--list_metrics", action="store_true", help="List available metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # List metrics if requested
    if args.list_metrics:
        print("Available metrics:")
        for name, path in list_registered_metrics().items():
            print(f"  - {name}: {path}")
        return
    
    # Load config
    config = EvaluateArguments.load_from_yaml(args.config)
    
    # Apply CLI overrides
    if args.samples_dir:
        config.samples_dir = args.samples_dir
    if args.output_path:
        config.output_path = args.output_path
    
    logger.info("=" * 60)
    logger.info("Flow-Factory Evaluation")
    logger.info(f"Config: {args.config}")
    logger.info(f"Samples: {config.samples_dir}")
    logger.info(f"Reference: {config.reference_dir}")
    logger.info(f"Metrics: {list(config.metrics.keys())}")
    logger.info("=" * 60)
    
    # Run evaluation
    pipeline = EvaluationPipeline(config)
    results = pipeline.run()
    pipeline.save_results(results)
    
    logger.info("Evaluation completed successfully")
    return results


if __name__ == "__main__":
    main()
