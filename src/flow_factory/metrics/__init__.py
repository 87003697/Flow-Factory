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

# src/flow_factory/metrics/__init__.py
"""
Metrics module for Flow-Factory.
Provides evaluation metrics for generated images/videos.
"""
from .registry import (
    load_metric,
    list_registered_metrics,
    register_metric,
)
from .abc import BaseMetric

__all__ = [
    'BaseMetric',
    'load_metric',
    'list_registered_metrics',
    'register_metric',
]
