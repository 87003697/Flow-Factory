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

# src/flow_factory/data_utils/image_3D_dataset.py
from PIL import Image

from .dataset import GeneralDataset


class Image3DDataset(GeneralDataset):
    """
    Dataset for image-to-3D tasks.

    Extends GeneralDataset with two key behaviours:

    1. **RGBA preservation** — images are loaded in RGBA mode instead of RGB.
       This keeps the alpha channel that image-to-3D models (e.g. Trellis2) use
       as a foreground mask, allowing them to skip expensive background-removal
       inference (REMBG) when a clean alpha is already present in the source file.

    2. **Camera metadata** (future) — support for per-sample camera parameters
       (intrinsics, extrinsics, etc.) can be added by overriding
       ``_build_metadata_for_batch``.
    """

    def _load_image(self, path: str) -> Image.Image:
        """Load image in RGBA mode, preserving the alpha channel."""
        return Image.open(path).convert("RGBA")
