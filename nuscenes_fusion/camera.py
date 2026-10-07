"""Pinhole camera projection of lidar points, with nearest-return-per-pixel occlusion handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MINIMUM_CAMERA_DEPTH = 0.1


@dataclass(frozen=True)
class CameraCalibration:
    """A 3x4 camera projection matrix and the 4x4 lidar-to-camera transform."""

    projection: np.ndarray
    velo_to_camera: np.ndarray


def project_points(
    points: np.ndarray, calibration: CameraCalibration, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project Velodyne XYZ into pixels.

    Returns integer pixel coordinates and camera depths for visible points, plus
    the boolean mask selecting those points from the input.
    """
    homogeneous = np.hstack((points[:, :3], np.ones((len(points), 1))))
    camera_points = homogeneous @ calibration.velo_to_camera.T
    depth = camera_points[:, 2]
    in_front = depth > MINIMUM_CAMERA_DEPTH

    image_points = camera_points[in_front] @ calibration.projection.T
    pixels = np.floor(image_points[:, :2] / image_points[:, 2:3]).astype(np.int64)
    in_image = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )

    visible = np.zeros(len(points), dtype=bool)
    visible[np.flatnonzero(in_front)[in_image]] = True
    return pixels[in_image], depth[visible], visible


def nearest_return_per_pixel(pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Return indices keeping only the closest point that lands on each pixel."""
    order = np.lexsort((depths, pixels[:, 1], pixels[:, 0]))
    _, first = np.unique(pixels[order], axis=0, return_index=True)
    return order[first]


def transform_points_to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a 4x4 sensor-to-world pose to Nx3 points."""
    return points @ pose[:3, :3].T + pose[:3, 3]
