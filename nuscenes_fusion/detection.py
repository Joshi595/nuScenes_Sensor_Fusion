"""Label-free lidar object detection: ground removal, BEV clustering, oriented box fitting.

Classical pipeline used before (and alongside) learned detectors:
1. Ground: per 2 m cell the lowest return, smoothed with a 3x3 minimum filter, is
   the local ground height; returns within `ground_clearance` of it are ground.
2. Clusters: remaining returns are rasterized into a bird's-eye-view occupancy grid
   and split into 8-connected components.
3. Boxes: each cluster gets the minimum-area rectangle (search over headings), then
   a size filter keeps vehicle-sized objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

GROUND_CELL = 2.0          # [m] ground-height grid
GROUND_CLEARANCE = 0.3     # [m] returns this close to the ground are ground
MAXIMUM_HEIGHT = 4.5       # [m] above ground; drops tree canopies and overhangs
CLUSTER_CELL = 0.3         # [m] BEV occupancy resolution for connected components
MINIMUM_POINTS = 10
MAXIMUM_BOTTOM = 0.8       # [m] a vehicle's lowest return sits near the ground (rejects canopies)
HEADING_STEPS = np.radians(np.arange(0.0, 90.0, 2.0))
PRIOR_LENGTH, PRIOR_WIDTH = 4.4, 1.8  # [m] typical passenger car, for completing partial views


@dataclass(frozen=True)
class Detection:
    """Oriented box in the frame of the input points; length >= width, yaw in [0, pi)."""

    center: np.ndarray       # [x, y, z]
    length: float
    width: float
    height: float
    yaw: float
    point_count: int


def height_above_ground(points: np.ndarray) -> np.ndarray:
    """Height of every return above the local ground estimate (points: Nx3, z up)."""
    cells = np.floor(points[:, :2] / GROUND_CELL).astype(np.int64)
    origin = cells.min(axis=0)
    cells -= origin
    shape = tuple(cells.max(axis=0) + 1)
    lowest = np.full(shape, np.inf)
    np.minimum.at(lowest, (cells[:, 0], cells[:, 1]), points[:, 2])
    ground = ndimage.minimum_filter(lowest, size=3, mode="nearest")[cells[:, 0], cells[:, 1]]
    return points[:, 2] - ground


def cluster_points(points: np.ndarray) -> np.ndarray:
    """Connected-component label per point (0..K-1) on a BEV occupancy grid."""
    cells = np.floor(points[:, :2] / CLUSTER_CELL).astype(np.int64)
    cells -= cells.min(axis=0)
    occupancy = np.zeros(tuple(cells.max(axis=0) + 1), dtype=bool)
    occupancy[cells[:, 0], cells[:, 1]] = True
    labels, _ = ndimage.label(occupancy, structure=np.ones((3, 3), dtype=bool))
    return labels[cells[:, 0], cells[:, 1]] - 1


def fit_box(points: np.ndarray) -> Detection:
    """Minimum-area oriented rectangle over candidate headings, plus vertical extent."""
    xy = points[:, :2]
    c_all, s_all = np.cos(HEADING_STEPS), np.sin(HEADING_STEPS)
    u_all = xy[:, :1] * c_all + xy[:, 1:2] * s_all     # N x headings
    v_all = -xy[:, :1] * s_all + xy[:, 1:2] * c_all
    best = int(np.argmin(np.ptp(u_all, axis=0) * np.ptp(v_all, axis=0)))
    angle, u, v = HEADING_STEPS[best], u_all[:, best], v_all[:, best]
    c, s = np.cos(angle), np.sin(angle)
    mid_u, mid_v = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2
    center_xy = np.array([c * mid_u - s * mid_v, s * mid_u + c * mid_v])
    extent_u, extent_v = np.ptp(u), np.ptp(v)
    if extent_v > extent_u:  # make length the longer side
        extent_u, extent_v, angle = extent_v, extent_u, angle + np.pi / 2
    z_low, z_high = points[:, 2].min(), points[:, 2].max()
    return Detection(
        center=np.array([*center_xy, (z_low + z_high) / 2]),
        length=float(extent_u),
        width=float(extent_v),
        height=float(z_high - z_low),
        yaw=float(angle % np.pi),
        point_count=len(points),
    )


def complete_box(box: Detection) -> Detection:
    """Lidar sees only the faces turned to it, so a partial view's fitted centre sits too close.

    Where an extent is shorter than the car prior, move the centre away from the sensor
    (at the origin) along that axis by half the missing extent, and report the prior size.
    """
    center = box.center.copy()
    length, width = box.length, box.width
    for axis_angle, extent, prior in ((box.yaw, length, PRIOR_LENGTH), (box.yaw + np.pi / 2, width, PRIOR_WIDTH)):
        if extent < prior:
            direction = np.array([np.cos(axis_angle), np.sin(axis_angle)])
            away = 1.0 if center[:2] @ direction >= 0 else -1.0
            center[:2] += away * direction * (prior - extent) / 2
    return Detection(center, max(length, PRIOR_LENGTH), max(width, PRIOR_WIDTH), box.height, box.yaw, box.point_count)


def is_vehicle_sized(box: Detection) -> bool:
    """Geometric vehicle test, chosen on nuScenes-mini TP/FP statistics.

    Height <= 3 m and >= 6 returns per m^2 of side area reject trees, walls, and poles
    (precision 0.23 -> 0.44 at 89% of true vehicles kept).
    # ponytail: geometric class filter drops buses/tall trucks (> 3 m); a learned detector replaces it.
    """
    density = box.point_count / max(box.length * box.height, 0.1)
    return (1.2 <= box.length <= 12.0 and box.width <= 3.5 and 0.5 <= box.height <= 3.0
            and density >= 6.0)


def detect_objects(points: np.ndarray, maximum_range: float = 50.0) -> tuple[list[Detection], list[np.ndarray]]:
    """Vehicle-sized boxes from one sweep (Nx3, z up, any frame) and each box's point indices."""
    in_range = np.flatnonzero(np.hypot(points[:, 0], points[:, 1]) <= maximum_range)
    if len(in_range) == 0:
        return [], []
    height = height_above_ground(points[in_range])
    keep = (height > GROUND_CLEARANCE) & (height < MAXIMUM_HEIGHT)
    candidates, candidate_height = in_range[keep], height[keep]
    if len(candidates) == 0:
        return [], []
    labels = cluster_points(points[candidates])
    order = np.argsort(labels, kind="stable")
    splits = np.flatnonzero(np.diff(labels[order])) + 1
    groups = np.split(candidates[order], splits)
    bottoms = [h.min() for h in np.split(candidate_height[order], splits)]

    detections, members = [], []
    for indices, bottom in zip(groups, bottoms):
        if len(indices) < MINIMUM_POINTS or bottom > MAXIMUM_BOTTOM:
            continue
        box = fit_box(points[indices])
        if is_vehicle_sized(box):
            detections.append(complete_box(box))
            members.append(indices)
    return detections, members
