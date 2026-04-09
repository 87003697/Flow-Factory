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

# src/flow_factory/metrics/registry.py
"""
Metrics Registry System.
Centralized registry for evaluation metrics with dynamic loading.
"""
from typing import Type, Dict, Any
import importlib
import logging
import torch

logger = logging.getLogger(__name__)


# Metrics Registry Storage
_METRIC_REGISTRY: Dict[str, str] = {
    # Image quality metrics
    'fid': 'flow_factory.metrics.fid.FIDMetric',
    'lpips': 'flow_factory.metrics.lpips.LPIPSMetric',
    'inception_score': 'flow_factory.metrics.inception.InceptionScoreMetric',
    
    # Text-image alignment metrics (reuse reward models)
    'clip_score': 'flow_factory.metrics.clip_score.CLIPScoreMetric',
    
    # Legacy reward model wrappers
    'pickscore': 'flow_factory.metrics.reward_wrapper.PickScoreMetric',
}
_METRIC_REGISTRY = {k.lower(): v for k, v in _METRIC_REGISTRY.items()}


def register_metric(name: str):
    """
    Decorator for registering metrics.
    
    Usage:
        @register_metric('my_metric')
        class MyMetric(BaseMetric):
            ...
    """
    def decorator(cls):
        _METRIC_REGISTRY[name.lower()] = f"{cls.__module__}.{cls.__name__}"
        logger.info(f"Registered metric: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_metric_class(identifier: str) -> Type:
    """
    Resolve and import a metric class from registry or python path.
    
    Args:
        identifier: Metric name or fully qualified class path.
    
    Returns:
        Metric class.
    
    Raises:
        ImportError: If the metric cannot be loaded.
    """
    identifier_lower = identifier.lower()
    
    # Check registry first
    class_path = _METRIC_REGISTRY.get(identifier_lower, identifier)
    
    # Dynamic import
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        metric_class = getattr(module, class_name)
        
        logger.debug(f"Loaded metric: {identifier} -> {class_name}")
        return metric_class
        
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(
            f"Could not load metric '{identifier}'. "
            f"Ensure it is either:\n"
            f"  1. A registered metric: {list(_METRIC_REGISTRY.keys())}\n"
            f"  2. A valid python path (e.g., 'my_package.metrics.CustomMetric')\n"
            f"Error: {e}"
        ) from e


def load_metric(name: str, config: Dict[str, Any], device: torch.device = None):
    """
    Load and instantiate a metric.
    
    Args:
        name: Metric name or class path.
        config: Configuration dict for the metric.
        device: Device to run computations on.
    
    Returns:
        Instantiated metric object.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    metric_class = get_metric_class(name)
    return metric_class(config, device)


def list_registered_metrics() -> Dict[str, str]:
    """
    Get all registered metrics.
    
    Returns:
        Dictionary mapping metric names to their class paths.
    """
    return _METRIC_REGISTRY.copy()
