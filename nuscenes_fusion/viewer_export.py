"""Export a point-cloud map as a static, tile-streaming Three.js viewer folder.

Layout written to the destination directory:

    index.html, vendor/          static viewer (copied from sensor_fusion/viewer)
    manifest.json                bounds, trajectory, tile index, replay index
    overview.bin                 spatially even subset of the whole map
    tiles/<i>.bin                every map point, split into XY tiles
    replay.bin                   sampled real scans for playback (optional)
    camera/<i>.jpg               downscaled camera frame per replayed scan (camera-colored replay)
    radar.bin                    float32 [x, y, z, vx, vy] radar returns per replay frame (optional)

Each .bin holds N uint16 XYZ triples quantized to its own bounds, followed by N
uint8 RGB triples (colored maps, camera-colored replay) or N uint8 intensities
(intensity-only replay). Browsers
block fetch() on file:// URLs, so serve the folder: python -m http.server -d <dir>
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .visualization import robust_range


VIEWER_SOURCE_DIRECTORY = Path(__file__).with_name("viewer")
UINT16_MAX = np.iinfo(np.uint16).max


@dataclass(frozen=True)
class LidarReplay:
    """Sensor-frame XYZI scans and the shared XYZ bounds used to quantize them.

    ``colors`` (uint8 RGB per point) and ``image_paths`` are set when the replay
    was built with a camera calibration.
    """

    source_frame_indices: np.ndarray
    scans: tuple[np.ndarray, ...]
    minimum: np.ndarray
    maximum: np.ndarray
    colors: tuple[np.ndarray, ...] | None = None
    image_paths: tuple[Path, ...] | None = None


def quantize(points: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    """Map XYZ into uint16 steps of (maximum - minimum) / 65535."""
    spans = np.maximum(maximum - minimum, 1e-9)
    return np.rint(np.clip((points - minimum) / spans, 0.0, 1.0) * UINT16_MAX).astype(np.uint16)


def write_points(
    root: Path, name: str, points: np.ndarray, colors: np.ndarray | None
) -> dict[str, object]:
    """Write one quantized block to root/name and return its manifest entry."""
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    payload = quantize(points, minimum, maximum).tobytes()
    if colors is not None:
        payload += np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8).tobytes()
    (root / name).write_bytes(payload)
    return {
        "file": name,
        "count": len(points),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
    }


def spatially_sample_indices(points: np.ndarray, maximum_points: int, cell_size: float) -> np.ndarray:
    """Pick up to maximum_points indices spread evenly over XY cells (a long route stays visible)."""
    if len(points) <= maximum_points:
        return np.arange(len(points))

    cells = np.floor(points[:, :2] / cell_size).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    per_cell = max(1, maximum_points // (int(inverse.max()) + 1))
    generator = np.random.default_rng(42)

    order = np.argsort(inverse, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1)
    sampled = np.concatenate(
        [g if len(g) <= per_cell else generator.choice(g, per_cell, replace=False) for g in groups]
    )
    if len(sampled) > maximum_points:
        sampled = generator.choice(sampled, maximum_points, replace=False)
    return np.sort(sampled)


def tile_groups(points: np.ndarray, tile_size: float) -> list[np.ndarray]:
    """Split point indices into square XY tiles."""
    tiles = np.floor((points[:, :2] - points[:, :2].min(axis=0)) / tile_size).astype(np.int64)
    _, inverse = np.unique(tiles, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1)


def write_replay(
    root: Path,
    replay: LidarReplay,
    poses: Sequence[np.ndarray],
    camera_width: int,
    rate_hz: float = 10.0,
    overlays: dict | None = None,
) -> dict[str, object]:
    """Write sampled scans, their camera frames (if colored), and each replayed pose."""
    scans = replay.scans
    xyz = np.concatenate([scan[:, :3] for scan in scans])
    payload = quantize(xyz, replay.minimum, replay.maximum).tobytes()
    if replay.colors is not None:
        payload += np.concatenate(replay.colors).astype(np.uint8).tobytes()
    else:
        intensity = np.concatenate([scan[:, 3] for scan in scans])
        payload += np.clip(np.rint(intensity * 255.0), 0, 255).astype(np.uint8).tobytes()
    (root / "replay.bin").write_bytes(payload)

    camera_files = None
    if replay.image_paths is not None and camera_width > 0:
        (root / "camera").mkdir()
        camera_files = []
        for number, image_path in enumerate(replay.image_paths):
            image = Image.open(image_path).convert("RGB")
            camera_width = min(camera_width, image.width)
            height = round(image.height * camera_width / image.width)
            name = f"camera/{number}.jpg"
            image.resize((camera_width, height), Image.Resampling.BILINEAR).save(root / name, quality=80)
            camera_files.append(name)

    extras: dict[str, object] = {}
    if overlays:
        # Per replay frame: tracks [id, x, y, z, yaw, length, width, height, vx, vy, moving],
        # labels [x, y, z, yaw, length, width, height], radar Nx5 world [x, y, z, vx, vy].
        extras = {"tracks": overlays.get("tracks"), "labels": overlays.get("labels")}
        if overlays.get("radar") is not None:
            radar = overlays["radar"]
            (root / "radar.bin").write_bytes(np.concatenate(radar).astype(np.float32).tobytes())
            extras["radarCounts"] = [len(frame) for frame in radar]

    return {
        **extras,
        "file": "replay.bin",
        "rateHz": rate_hz,
        "hasColor": replay.colors is not None,
        "cameraFiles": camera_files,
        "frames": replay.source_frame_indices.tolist(),
        "counts": [len(scan) for scan in scans],
        "min": replay.minimum.tolist(),
        "max": replay.maximum.tolist(),
        "poses": [np.round(poses[i][:3, :4].reshape(-1), 5).tolist() for i in replay.source_frame_indices],
    }


def write_viewer(
    destination: Path,
    points: np.ndarray,
    poses: Sequence[np.ndarray],
    title: str,
    colors: np.ndarray | None = None,
    replay: LidarReplay | None = None,
    point_size: float = 0.15,
    tile_size: float = 20.0,
    overview_points: int = 300_000,
    camera_width: int = 480,
    replay_rate_hz: float = 10.0,
    replay_overlays: dict | None = None,
) -> Path:
    """Write the viewer folder for a map and return the path of its index.html."""
    if len(points) == 0:
        raise ValueError("Cannot export an empty map.")
    if tile_size <= 0 or overview_points < 1:
        raise ValueError("tile_size must be positive and overview_points at least one.")

    # Clear stale data but keep the folder itself: on Windows a running
    # `http.server -d <destination>` locks it.
    shutil.rmtree(destination / "tiles", ignore_errors=True)
    shutil.rmtree(destination / "camera", ignore_errors=True)
    (destination / "replay.bin").unlink(missing_ok=True)
    (destination / "radar.bin").unlink(missing_ok=True)
    shutil.copytree(VIEWER_SOURCE_DIRECTORY, destination, dirs_exist_ok=True)
    (destination / "tiles").mkdir()

    overview_indices = spatially_sample_indices(points, overview_points, tile_size)
    overview = write_points(
        destination,
        "overview.bin",
        points[overview_indices],
        None if colors is None else colors[overview_indices],
    )
    tiles = [
        write_points(
            destination,
            f"tiles/{number}.bin",
            points[indices],
            None if colors is None else colors[indices],
        )
        for number, indices in enumerate(tile_groups(points, tile_size))
    ]

    manifest = {
        "title": title,
        "pointCount": len(points),
        "hasColor": colors is not None,
        "pointSize": point_size,
        "bounds": {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
        "heightRange": robust_range(points[overview_indices, 2]),
        "trajectory": np.round(np.asarray([pose[:3, 3] for pose in poses]), 3).tolist(),
        "overview": overview,
        "tiles": tiles,
        "replay": None if replay is None else write_replay(
            destination, replay, poses, camera_width, replay_rate_hz, replay_overlays
        ),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    return destination / "index.html"
