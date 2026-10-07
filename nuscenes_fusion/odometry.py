"""KISS-ICP lidar odometry, used to check drift against nuScenes' localisation poses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP


def kiss_icp_poses(
    scan_paths: Sequence[Path],
    load_points: Callable[[Path], np.ndarray],
    minimum_range: float,
    maximum_range: float,
    voxel_size: float = 0.5,
) -> list[np.ndarray]:
    """One 4x4 pose per scan, relative to the first scan (no deskewing: sweeps carry no point times)."""
    config = KISSConfig()
    config.data.min_range = minimum_range
    config.data.max_range = maximum_range
    config.data.deskew = False
    config.mapping.voxel_size = voxel_size
    odometry = KissICP(config)
    poses = []
    for scan_path in scan_paths:
        odometry.register_frame(load_points(scan_path), np.empty(0, dtype=np.float64))
        poses.append(np.array(odometry.last_pose, copy=True))
    return poses
