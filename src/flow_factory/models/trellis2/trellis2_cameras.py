"""Shared camera construction for Trellis2 render pipeline."""

from typing import List, Tuple

import numpy as np
import torch


def get_render_cameras(
    num_frames: int = 24,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Reconstruct cameras matching ``render_mesh`` exactly.

    Returns:
        (extrinsics_list, intrinsics_list): Lists of (4,4) and (3,3) tensors.
    """
    from trellis2.utils import render_utils

    _FOV_DEG = 40.0
    _PITCH_DEG = 20.0
    _FILL_RATIO = 0.9
    _START_YAW = np.pi
    fov_rad = np.deg2rad(_FOV_DEG)
    r = 0.5 / (_FILL_RATIO * np.tan(fov_rad / 2))

    yaws = torch.linspace(_START_YAW, _START_YAW + 2 * np.pi, num_frames + 1)[:-1].tolist()
    pitchs = [np.deg2rad(_PITCH_DEG)] * num_frames
    return render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, _FOV_DEG)
