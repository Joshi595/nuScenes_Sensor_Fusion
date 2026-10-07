"""Coloured point clouds: voxel averaging and PLY output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class ColoredPointCloud:
    """World-frame point coordinates and normalized RGB colors."""

    points: np.ndarray
    colors: np.ndarray


def voxel_downsample_colored(
    point_cloud: ColoredPointCloud, voxel_size: float
) -> ColoredPointCloud:
    """Voxel-reduce XYZ and RGB together, averaging colors within each voxel."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than zero.")
    if len(point_cloud.points) != len(point_cloud.colors):
        raise ValueError("Point and color arrays must have the same length.")
    if len(point_cloud.points) == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return ColoredPointCloud(empty, empty.copy())

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud.points)
    cloud.colors = o3d.utility.Vector3dVector(point_cloud.colors)
    reduced = cloud.voxel_down_sample(voxel_size)
    return ColoredPointCloud(
        np.asarray(reduced.points).copy(), np.asarray(reduced.colors).copy()
    )


def save_point_cloud(
    points: np.ndarray, destination: Path, colors: np.ndarray | None = None
) -> None:
    """Write a binary PLY (with RGB when given) for Open3D, CloudCompare, or the viewer."""
    if len(points) == 0:
        raise ValueError("Cannot save an empty point cloud.")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(destination), cloud, write_ascii=False):
        raise OSError(f"Could not write point cloud: {destination}")
