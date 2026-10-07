"""Minimal numpy-only nuScenes reader (no devkit dependency).

Everything is returned with full 4x4 sensor-to-world poses built from each
sample_data record's own ego pose, so data captured at different instants
(camera vs. lidar vs. radar) is combined with the vehicle's true motion
in between, exactly like the devkit's map_pointcloud_to_image.
"""

from __future__ import annotations

import json
import struct
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage

CAMERA_CHANNELS = (
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
)
RADAR_CHANNELS = (
    "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
)
LIDAR_CHANNEL = "LIDAR_TOP"
MAP_RESOLUTION = 0.1  # [m/pixel] nuScenes semantic-prior (drivable area) masks


def quaternion_to_rotation(q) -> np.ndarray:
    """nuScenes quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quaternion_yaw(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def transform(translation, rotation) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_rotation(rotation)
    matrix[:3, 3] = translation
    return matrix


def read_lidar(path: Path) -> np.ndarray:
    """LIDAR_TOP .pcd.bin -> Nx4 float32 [x, y, z, intensity in 0..1] in the sensor frame."""
    records = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4].copy()
    records[:, 3] /= 255.0
    return records


def read_radar(path: Path) -> np.ndarray:
    """Radar binary PCD -> structured array, with the devkit's default quality filters applied."""
    header, data = path.read_bytes().split(b"DATA binary\n", 1)
    meta = {line.split()[0]: line.split()[1:] for line in header.decode().splitlines() if line[:1] != "#"}
    kinds = {"F": "f", "I": "i", "U": "u"}
    dtype = np.dtype(
        [(name, f"<{kinds[kind]}{size}") for name, kind, size in zip(meta["FIELDS"], meta["TYPE"], meta["SIZE"])]
    )
    points = np.frombuffer(data, dtype=dtype, count=int(meta["POINTS"][0]))
    keep = (points["invalid_state"] == 0) & (points["dyn_prop"] <= 6) & (points["ambig_state"] == 3)
    return points[keep]


@dataclass(frozen=True)
class SensorFrame:
    """One capture: time [s], file, 4x4 sensor-to-world pose, camera intrinsics (cameras only)."""

    timestamp: float
    path: Path
    pose: np.ndarray
    intrinsic: np.ndarray | None = None


@dataclass(frozen=True)
class Box:
    """Ground-truth 3D box (global frame), used only for evaluation."""

    instance: str
    category: str
    center: np.ndarray
    size: np.ndarray  # [width, length, height]
    yaw: float
    num_lidar_points: int


@dataclass(frozen=True)
class DrivableArea:
    """Crop of the drivable-area mask: True where vehicles can be. Global-frame lookups."""

    mask: np.ndarray        # [rows, cols], row 0 = northern edge
    minimum_x: float
    maximum_y: float

    def contains(self, xy: np.ndarray) -> np.ndarray:
        cols = np.floor((xy[:, 0] - self.minimum_x) / MAP_RESOLUTION).astype(int)
        rows = np.floor((self.maximum_y - xy[:, 1]) / MAP_RESOLUTION).astype(int)
        inside = (rows >= 0) & (rows < self.mask.shape[0]) & (cols >= 0) & (cols < self.mask.shape[1])
        result = np.zeros(len(xy), dtype=bool)
        result[inside] = self.mask[rows[inside], cols[inside]]
        return result


def read_mask_window(path: Path, row_range: tuple[int, int], col_range: tuple[int, int]) -> np.ndarray:
    """Stream-decode only the needed rows of an 8-bit greyscale PNG whose rows all use the Sub filter.

    The nuScenes masks reach 32k x 37k pixels (1.2 GB decoded); this keeps memory to the window.
    """
    data = path.read_bytes()
    position, chunks, width = 8, [], 0
    while position < len(data):
        length, kind = struct.unpack(">I4s", data[position:position + 8])
        body = data[position + 8:position + 8 + length]
        if kind == b"IHDR":
            width, _, depth, color_type = struct.unpack(">IIBB", body[:10])
            if (depth, color_type) != (8, 0):
                raise ValueError(f"Expected an 8-bit greyscale PNG: {path}")
        elif kind == b"IDAT":
            chunks.append(body)
        position += 12 + length

    (row_start, row_stop), (col_start, col_stop) = row_range, col_range
    stride = width + 1
    window = np.zeros((row_stop - row_start, col_stop - col_start), dtype=bool)
    decompressor, buffer, row = zlib.decompressobj(), b"", 0
    for chunk in chunks:
        buffer += decompressor.decompress(chunk)
        while len(buffer) >= stride and row < row_stop:
            line, buffer = buffer[:stride], buffer[stride:]
            if row >= row_start:
                if line[0] != 1:
                    raise ValueError(f"Unsupported PNG row filter {line[0]} in {path}")
                raw = np.cumsum(np.frombuffer(line, np.uint8, offset=1), dtype=np.uint8)  # Sub filter
                window[row - row_start] = raw[col_start:col_stop] > 0
            row += 1
        if row >= row_stop:
            break
    return window


def load_drivable_area(
    dataroot: Path, scene_name: str, xy: np.ndarray, margin: float = 80.0,
    dilation: float = 1.0, version: str = "v1.0-mini",
) -> DrivableArea:
    """Drivable area around the given global positions, grown by `dilation` metres for kerb-side parking."""
    tables = load_tables(Path(dataroot), version)
    scene = next(s for s in tables["scene"].values() if s["name"] == scene_name)
    record = next(m for m in tables["map"].values() if scene["log_token"] in m["log_tokens"])
    path = Path(dataroot) / record["filename"]
    _, height = struct.unpack(">II", path.read_bytes()[16:24])
    minimum_x, maximum_x = xy[:, 0].min() - margin, xy[:, 0].max() + margin
    minimum_y, maximum_y = xy[:, 1].min() - margin, xy[:, 1].max() + margin
    cols = (max(int(minimum_x / MAP_RESOLUTION), 0), int(np.ceil(maximum_x / MAP_RESOLUTION)))
    rows = (max(int(height - maximum_y / MAP_RESOLUTION), 0), min(int(np.ceil(height - minimum_y / MAP_RESOLUTION)), height))
    mask = read_mask_window(path, rows, cols)
    radius = int(round(dilation / MAP_RESOLUTION))
    if radius:
        mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=radius)
    return DrivableArea(mask, cols[0] * MAP_RESOLUTION, (height - rows[0]) * MAP_RESOLUTION)


@dataclass
class NuScenesScene:
    name: str
    description: str
    lidar: list[SensorFrame]
    cameras: dict[str, list[SensorFrame]]
    radars: dict[str, list[SensorFrame]]
    keyframe_boxes: list[tuple[float, list[Box]]] = field(default_factory=list)

    def nearest(self, frames: list[SensorFrame], timestamp: float) -> SensorFrame:
        times = np.array([frame.timestamp for frame in frames])
        return frames[int(np.argmin(np.abs(times - timestamp)))]

    def boxes_at(self, timestamp: float) -> list[Box]:
        """Annotations linearly interpolated between the 2 Hz keyframes around `timestamp`."""
        times = np.array([t for t, _ in self.keyframe_boxes])
        after = int(np.searchsorted(times, timestamp))
        if after == 0 or after == len(times):
            index = min(max(after, 0), len(times) - 1)
            return self.keyframe_boxes[index][1] if abs(times[index] - timestamp) < 0.05 else []
        (t0, boxes0), (t1, boxes1) = self.keyframe_boxes[after - 1], self.keyframe_boxes[after]
        alpha = (timestamp - t0) / (t1 - t0)
        later = {box.instance: box for box in boxes1}
        interpolated = []
        for box in boxes0:
            other = later.get(box.instance)
            if other is None:
                continue
            yaw_step = np.angle(np.exp(1j * (other.yaw - box.yaw)))
            interpolated.append(
                Box(
                    box.instance, box.category,
                    box.center + alpha * (other.center - box.center),
                    box.size, box.yaw + alpha * yaw_step,
                    min(box.num_lidar_points, other.num_lidar_points),
                )
            )
        return interpolated


@lru_cache(maxsize=2)
def load_tables(dataroot: Path, version: str) -> dict[str, dict[str, dict]]:
    names = ("scene", "sample", "sample_data", "calibrated_sensor", "sensor", "ego_pose",
             "sample_annotation", "instance", "category", "map")
    return {
        name: {row["token"]: row for row in json.loads((dataroot / version / f"{name}.json").read_text())}
        for name in names
    }


def scene_names(dataroot: Path, version: str = "v1.0-mini") -> list[str]:
    return sorted(scene["name"] for scene in load_tables(Path(dataroot), version)["scene"].values())


def load_scene(dataroot: Path, scene_name: str, version: str = "v1.0-mini") -> NuScenesScene:
    """Collect every lidar sweep, camera image, and radar sweep of a scene, time-ordered."""
    dataroot = Path(dataroot)
    tables = load_tables(dataroot, version)
    scene = next((s for s in tables["scene"].values() if s["name"] == scene_name), None)
    if scene is None:
        raise ValueError(f"Unknown scene {scene_name!r}; available: {scene_names(dataroot, version)}")
    samples = {token: row for token, row in tables["sample"].items() if row["scene_token"] == scene["token"]}

    frames: dict[str, list[SensorFrame]] = defaultdict(list)
    for record in tables["sample_data"].values():
        if record["sample_token"] not in samples:
            continue
        calibration = tables["calibrated_sensor"][record["calibrated_sensor_token"]]
        channel = tables["sensor"][calibration["sensor_token"]]["channel"]
        ego = tables["ego_pose"][record["ego_pose_token"]]
        pose = transform(ego["translation"], ego["rotation"]) @ transform(
            calibration["translation"], calibration["rotation"]
        )
        intrinsic = np.array(calibration["camera_intrinsic"]) if calibration["camera_intrinsic"] else None
        frames[channel].append(
            SensorFrame(record["timestamp"] * 1e-6, dataroot / record["filename"], pose, intrinsic)
        )
    for channel_frames in frames.values():
        channel_frames.sort(key=lambda frame: frame.timestamp)

    by_sample: dict[str, list[Box]] = defaultdict(list)
    for annotation in tables["sample_annotation"].values():
        if annotation["sample_token"] not in samples:
            continue
        instance = tables["instance"][annotation["instance_token"]]
        by_sample[annotation["sample_token"]].append(
            Box(
                annotation["instance_token"],
                tables["category"][instance["category_token"]]["name"],
                np.array(annotation["translation"]),
                np.array(annotation["size"]),
                quaternion_yaw(annotation["rotation"]),
                annotation["num_lidar_pts"],
            )
        )
    keyframes = sorted(
        ((samples[token]["timestamp"] * 1e-6, by_sample.get(token, [])) for token in samples),
        key=lambda keyframe: keyframe[0],
    )

    return NuScenesScene(
        name=scene_name,
        description=scene["description"],
        lidar=frames[LIDAR_CHANNEL],
        cameras={channel: frames[channel] for channel in CAMERA_CHANNELS},
        radars={channel: frames[channel] for channel in RADAR_CHANNELS},
        keyframe_boxes=keyframes,
    )
