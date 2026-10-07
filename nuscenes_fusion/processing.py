"""nuScenes pipeline stages: tracking, tracking evaluation, 6-camera colouring, static map, replay."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from .camera import CameraCalibration, nearest_return_per_pixel, project_points
from .detection import Detection, detect_objects
from .mapping import ColoredPointCloud, voxel_downsample_colored
from .nuscenes import CAMERA_CHANNELS, DrivableArea, NuScenesScene, SensorFrame, read_lidar, read_radar
from .tracking import MultiObjectTracker, TrackState
from .viewer_export import LidarReplay

EGO_RADIUS = 2.5            # [m] returns closer than this hit the ego vehicle itself
MAXIMUM_RANGE = 50.0        # [m] detection and evaluation range (nuScenes car class range)
VEHICLE_PREFIXES = (
    "vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.trailer",
    "vehicle.construction", "vehicle.emergency",
)
MATCH_DISTANCE = 2.0        # [m] centre distance for a true positive (nuScenes' loosest threshold)


@dataclass(frozen=True)
class FrameResult:
    """Everything the tracker produced for one lidar sweep."""

    timestamp: float
    sensor_xyz: np.ndarray
    detections: list[Detection]   # centres relative to sensor_xyz, world-aligned axes
    tracks: list[TrackState]


def lidar_world_points(frame: SensorFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sensor-frame XYZI without ego-vehicle returns, and the same points in the world frame."""
    records = read_lidar(frame.path)
    records = records[np.hypot(records[:, 0], records[:, 1]) > EGO_RADIUS]
    world = records[:, :3] @ frame.pose[:3, :3].T + frame.pose[:3, 3]
    return records, world


def radar_detections(frame: SensorFrame) -> tuple[np.ndarray, np.ndarray]:
    """Global-frame [range, bearing, range_rate] per radar return, plus [x, y, z, vx, vy] for display."""
    radar = read_radar(frame.path)
    local = np.column_stack((radar["x"], radar["y"], radar["z"])).astype(float)
    local_velocity = np.column_stack((radar["vx_comp"], radar["vy_comp"], np.zeros(len(radar))))
    rotation = frame.pose[:3, :3]
    world = local @ rotation.T + frame.pose[:3, 3]
    velocity = local_velocity @ rotation.T
    offset = world[:, :2] - frame.pose[:2, 3]
    distance = np.maximum(np.hypot(offset[:, 0], offset[:, 1]), 1e-6)
    measurements = np.column_stack((
        distance,
        np.arctan2(offset[:, 1], offset[:, 0]),
        (offset[:, 0] * velocity[:, 0] + offset[:, 1] * velocity[:, 1]) / distance,
    ))
    return measurements, np.column_stack((world, velocity[:, :2]))


def run_tracking(
    scene: NuScenesScene, drivable: DrivableArea | None = None
) -> tuple[list[FrameResult], dict[str, int]]:
    """Feed lidar and all five radars to the tracker in timestamp order.

    With a drivable-area map (HD-map prior), lidar boxes off the road surface
    (vegetation, facades, street furniture) are discarded before association.
    """
    events = [(frame.timestamp, "lidar", frame) for frame in scene.lidar]
    events += [(frame.timestamp, "radar", frame) for frames in scene.radars.values() for frame in frames]
    events.sort(key=lambda event: event[0])

    tracker = MultiObjectTracker()
    results: list[FrameResult] = []
    radar_total = radar_accepted = 0
    for timestamp, kind, frame in events:
        if kind == "lidar":
            _, world = lidar_world_points(frame)
            sensor_xyz = frame.pose[:3, 3]
            detections, _ = detect_objects(world - sensor_xyz, MAXIMUM_RANGE)
            if drivable is not None and detections:
                on_road = drivable.contains(np.array([d.center[:2] + sensor_xyz[:2] for d in detections]))
                detections = [d for d, keep in zip(detections, on_road) if keep]
            tracker.update_lidar(timestamp, detections, sensor_xyz)
            results.append(FrameResult(timestamp, sensor_xyz, detections, tracker.states()))
        else:
            measurements, _ = radar_detections(frame)
            radar_total += len(measurements)
            radar_accepted += tracker.update_radar(timestamp, frame.pose[:2, 3], measurements)
    return results, {"radar_detections": radar_total, "radar_updates": radar_accepted}


def ground_truth_vehicles(scene: NuScenesScene, timestamp: float, ego_xy: np.ndarray) -> list[tuple]:
    """(instance, centre, size, yaw, velocity) of annotated vehicles with lidar support in range."""
    now = scene.boxes_at(timestamp)
    before = {b.instance: b for b in scene.boxes_at(timestamp - 0.25)}
    after = {b.instance: b for b in scene.boxes_at(timestamp + 0.25)}
    vehicles = []
    for box in now:
        if not box.category.startswith(VEHICLE_PREFIXES) or box.num_lidar_points < 5:
            continue
        if np.hypot(*(box.center[:2] - ego_xy)) > MAXIMUM_RANGE:
            continue
        a, b = before.get(box.instance, box), after.get(box.instance, box)
        span = 0.5 if box.instance in before and box.instance in after else 0.25
        velocity = (b.center[:2] - a.center[:2]) / span if (a is not box or b is not box) else np.zeros(2)
        vehicles.append((box.instance, box.center, box.size, box.yaw, velocity))
    return vehicles


def evaluate_tracking(scene: NuScenesScene, frames: list[FrameResult]) -> dict[str, dict[str, float]]:
    """CLEAR-MOT style scores against the labels for all vehicles and for moving vehicles.

    A track and an annotated vehicle match when their centres are within MATCH_DISTANCE
    (Hungarian assignment per sweep). 'moving' scores only moving vehicles (> 1 m/s) and
    tracks the filter considers moving; a moving track on a parked car is not a false positive.
    """
    scores = {}
    for subset in ("all_vehicles", "moving_vehicles"):
        tp = fp = fn = switches = 0
        position_errors, velocity_errors, last_match = [], [], {}
        for frame in frames:
            ego = frame.sensor_xyz[:2]
            vehicles = ground_truth_vehicles(scene, frame.timestamp, ego)
            tracks = [t for t in frame.tracks if np.hypot(*(t.center[:2] - ego)) <= MAXIMUM_RANGE]
            if subset == "moving_vehicles":
                any_vehicle = [v[1][:2] for v in vehicles]
                vehicles = [v for v in vehicles if np.hypot(*v[4]) > 1.0]
                tracks = [t for t in tracks if t.moving]
            cost = np.array([[np.linalg.norm(t.center[:2] - v[1][:2]) for v in vehicles] for t in tracks])
            matched_tracks, matched_vehicles = set(), set()
            if cost.size:
                for i, j in zip(*linear_sum_assignment(cost)):
                    if cost[i, j] <= MATCH_DISTANCE:
                        matched_tracks.add(i)
                        matched_vehicles.add(j)
                        instance, track_id = vehicles[j][0], tracks[i].track_id
                        if last_match.get(instance, track_id) != track_id:
                            switches += 1
                        last_match[instance] = track_id
                        position_errors.append(cost[i, j])
                        velocity_errors.append(np.linalg.norm(tracks[i].velocity - vehicles[j][4]))
            tp += len(matched_vehicles)
            fn += len(vehicles) - len(matched_vehicles)
            for i, track in enumerate(tracks):
                if i in matched_tracks:
                    continue
                if subset == "moving_vehicles" and any(
                    np.linalg.norm(track.center[:2] - xy) <= MATCH_DISTANCE for xy in any_vehicle
                ):
                    continue
                fp += 1
        ground_truth = tp + fn
        scores[subset] = {
            "ground_truth": ground_truth,
            "recall": tp / ground_truth if ground_truth else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "mota": 1 - (fn + fp + switches) / ground_truth if ground_truth else float("nan"),
            "id_switches": switches,
            "position_error_m": float(np.mean(position_errors)) if position_errors else float("nan"),
            "velocity_error_mps": float(np.mean(velocity_errors)) if velocity_errors else float("nan"),
        }
    return scores


@lru_cache(maxsize=16)
def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def camera_colors(
    scene: NuScenesScene, lidar_frame: SensorFrame, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """uint8 RGB per sensor-frame point from the 6 cameras, and a mask of points any camera saw.

    Each camera uses its own capture's pose, so the vehicle's motion between the lidar
    and camera timestamps is compensated. The nearest return per pixel wins (occlusion).
    """
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    seen = np.zeros(len(points), dtype=bool)
    for channel in CAMERA_CHANNELS:
        camera = scene.nearest(scene.cameras[channel], lidar_frame.timestamp)
        image = load_image(camera.path)
        projection = np.hstack((camera.intrinsic, np.zeros((3, 1))))
        calibration = CameraCalibration(projection, np.linalg.inv(camera.pose) @ lidar_frame.pose)
        pixels, depths, visible = project_points(points, calibration, image.shape[1], image.shape[0])
        keep = nearest_return_per_pixel(pixels, depths)
        indices = np.flatnonzero(visible)[keep]
        fresh = ~seen[indices]
        colors[indices[fresh]] = image[pixels[keep][fresh, 1], pixels[keep][fresh, 0]]
        seen[indices[fresh]] = True
    return colors, seen


def inside_boxes(points: np.ndarray, tracks: list[TrackState], margin: float = 0.4) -> np.ndarray:
    """Mask of world points inside any track's footprint (inflated by margin)."""
    mask = np.zeros(len(points), dtype=bool)
    for track in tracks:
        c, s = np.cos(track.yaw), np.sin(track.yaw)
        d = points[:, :2] - track.center[:2]
        along, across = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        length, width = track.size[0] / 2 + margin, track.size[1] / 2 + margin
        mask |= (np.abs(along) <= length) & (np.abs(across) <= width)
    return mask


def build_static_map(
    scene: NuScenesScene,
    frames: list[FrameResult],
    voxel_size: float,
    frame_stride: int = 1,
    chunk_frames: int = 25,
) -> tuple[ColoredPointCloud, int]:
    """Camera-coloured world map with returns on moving tracked objects removed (no smearing)."""
    chunks, pending_points, pending_colors = [], [], []
    removed = 0
    for position, (lidar_frame, result) in enumerate(zip(scene.lidar[::frame_stride], frames[::frame_stride]), 1):
        records, world = lidar_world_points(lidar_frame)
        in_range = np.hypot(records[:, 0], records[:, 1]) <= MAXIMUM_RANGE
        dynamic = inside_boxes(world, [t for t in result.tracks if t.moving])
        removed += int((dynamic & in_range).sum())
        keep = in_range & ~dynamic
        colors, seen = camera_colors(scene, lidar_frame, records[keep, :3])
        pending_points.append(world[keep][seen])
        pending_colors.append(colors[seen] / 255.0)
        if position % chunk_frames == 0:
            chunks.append(voxel_downsample_colored(
                ColoredPointCloud(np.concatenate(pending_points), np.concatenate(pending_colors)), voxel_size))
            pending_points.clear()
            pending_colors.clear()
            print(f"Coloured sweep {position * frame_stride:>4}/{len(scene.lidar)}")
    if pending_points:
        chunks.append(voxel_downsample_colored(
            ColoredPointCloud(np.concatenate(pending_points), np.concatenate(pending_colors)), voxel_size))
    merged = voxel_downsample_colored(
        ColoredPointCloud(np.concatenate([c.points for c in chunks]), np.concatenate([c.colors for c in chunks])),
        voxel_size,
    )
    return merged, removed


def camera_mosaic(scene: NuScenesScene, timestamp: float, tile_width: int) -> Image.Image:
    """3x2 grid: front-left, front, front-right over back-left, back, back-right."""
    tile_height = round(tile_width * 900 / 1600)
    mosaic = Image.new("RGB", (tile_width * 3, tile_height * 2))
    for index, channel in enumerate(CAMERA_CHANNELS):
        frame = scene.nearest(scene.cameras[channel], timestamp)
        tile = Image.fromarray(load_image(frame.path)).resize((tile_width, tile_height), Image.Resampling.BILINEAR)
        mosaic.paste(tile, ((index % 3) * tile_width, (index // 3) * tile_height))
    return mosaic


def build_replay(
    scene: NuScenesScene,
    frames: list[FrameResult],
    points_per_scan: int,
    frame_stride: int,
    mosaic_directory: Path | None,
    tile_width: int,
) -> tuple[LidarReplay, dict[str, object]]:
    """Camera-coloured sampled sweeps plus per-frame tracks, labels, and radar for the viewer."""
    generator = np.random.default_rng(42)
    scans, colors, images = [], [], []
    tracks, labels, radar = [], [], []
    indices = range(0, len(scene.lidar), frame_stride)
    for index in indices:
        lidar_frame, result = scene.lidar[index], frames[index]
        records, _ = lidar_world_points(lidar_frame)
        if len(records) > points_per_scan:
            records = records[np.sort(generator.choice(len(records), points_per_scan, replace=False))]
        rgb, seen = camera_colors(scene, lidar_frame, records[:, :3])
        grey = (40 + 110 * np.clip(records[:, 3] * 4, 0, 1)).astype(np.uint8)
        rgb[~seen] = grey[~seen, None]
        scans.append(records.astype(np.float32))
        colors.append(rgb)

        if mosaic_directory is not None:
            path = mosaic_directory / f"{index:04d}.jpg"
            camera_mosaic(scene, lidar_frame.timestamp, tile_width).save(path, quality=80)
            images.append(path)

        tracks.append([
            [t.track_id, *np.round(t.center, 2), round(t.yaw, 3), *np.round(t.size, 2),
             *np.round(t.velocity, 2), int(t.moving)]
            for t in result.tracks
        ])
        labels.append([
            [*np.round(center, 2), round(yaw, 3), round(size[1], 2), round(size[0], 2), round(size[2], 2)]
            for _, center, size, yaw, _ in ground_truth_vehicles(scene, lidar_frame.timestamp, lidar_frame.pose[:2, 3])
        ])
        window = [
            radar_detections(frame)[1]
            for frames_ in scene.radars.values()
            for frame in frames_
            if 0 <= lidar_frame.timestamp - frame.timestamp < 0.05 * frame_stride
        ]
        radar.append(np.concatenate(window) if window else np.zeros((0, 5)))

    stacked = np.concatenate([scan[:, :3] for scan in scans])
    replay = LidarReplay(
        source_frame_indices=np.array(list(indices), dtype=np.int32),
        scans=tuple(scans),
        minimum=stacked.min(axis=0),
        maximum=stacked.max(axis=0),
        colors=tuple(colors),
        image_paths=tuple(images) if images else None,
    )
    return replay, {"tracks": tracks, "labels": labels, "radar": radar}
