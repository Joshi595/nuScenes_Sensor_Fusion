"""nuScenes: lidar odometry check, lidar+radar EKF tracking, 6-camera coloured map, 3D viewer.

For each scene:
1. KISS-ICP lidar odometry, scored against nuScenes' localisation poses (ATE / RPE).
2. Label-free lidar detection (+ drivable-area prior) and a multi-object CTRV EKF
   fusing lidar boxes with Doppler radar from all five radars; scored against the labels.
3. A static map coloured by all six cameras, with moving tracked objects removed.
4. A streaming viewer: map, fused scan replay, tracked boxes, labels, radar, camera mosaic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nuscenes_fusion.mapping import save_point_cloud
from nuscenes_fusion.nuscenes import load_drivable_area, load_scene, read_lidar, scene_names
from nuscenes_fusion.odometry import kiss_icp_poses
from nuscenes_fusion.processing import (
    EGO_RADIUS,
    build_replay,
    build_static_map,
    evaluate_tracking,
    run_tracking,
)
from nuscenes_fusion.trajectory_eval import evaluate_trajectory
from nuscenes_fusion.viewer_export import write_viewer
from nuscenes_fusion.visualization import save_topdown_map

PROJECT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DATAROOT = PROJECT_DIRECTORY / "data" / "nuscenes"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "output"
LIDAR_RATE_HZ = 20.0


def odometry_check(scene) -> dict[str, float]:
    """KISS-ICP on LIDAR_TOP vs. the nuScenes lidar poses, both relative to the first sweep."""
    estimate = kiss_icp_poses(
        [frame.path for frame in scene.lidar],
        lambda path: read_lidar(path)[:, :3].astype(np.float64),
        minimum_range=EGO_RADIUS, maximum_range=70.0,
    )
    first = np.linalg.inv(scene.lidar[0].pose)
    reference = [first @ frame.pose for frame in scene.lidar]
    return evaluate_trajectory(estimate, reference, window=int(LIDAR_RATE_HZ))


def process_scene(arguments: argparse.Namespace, name: str) -> dict:
    output = arguments.output_directory
    prefix = f"nuscenes_{name.replace('-', '_')}"
    scene = load_scene(arguments.dataroot, name, arguments.version)
    print(f"\n=== {name}: {scene.description}")

    report: dict = {"scene": name, "description": scene.description, "lidar_sweeps": len(scene.lidar)}
    report["odometry"] = odometry_check(scene)
    print(f"KISS-ICP vs nuScenes poses: ATE {report['odometry']['ate_rmse_m']:.3f} m, "
          f"drift {report['odometry']['drift_percent']:.2f} %")

    ego_xy = np.array([frame.pose[:2, 3] for frame in scene.lidar])
    drivable = None if arguments.no_map_prior else load_drivable_area(
        arguments.dataroot, name, ego_xy, version=arguments.version
    )
    frames, radar_stats = run_tracking(scene, drivable)
    report["radar"] = radar_stats
    report["tracking"] = evaluate_tracking(scene, frames)
    moving = report["tracking"]["moving_vehicles"]
    print(f"Tracking moving vehicles: recall {moving['recall']:.2f}, precision {moving['precision']:.2f}, "
          f"position {moving['position_error_m']:.2f} m, velocity {moving['velocity_error_mps']:.2f} m/s")

    colored_map, removed = build_static_map(scene, frames, arguments.map_voxel_size, arguments.map_frame_stride)
    report["map"] = {"points": len(colored_map.points), "dynamic_returns_removed": removed}
    save_point_cloud(colored_map.points, output / f"{prefix}_map.ply", colored_map.colors)
    poses = [frame.pose for frame in scene.lidar]
    save_topdown_map(colored_map.points, poses, output / f"{prefix}_topdown.png", 0.1, colored_map.colors)

    if not arguments.skip_viewer:
        viewer = output / f"{prefix}_viewer"
        mosaic_directory = output / f"{prefix}_mosaics"
        mosaic_directory.mkdir(parents=True, exist_ok=True)
        replay, overlays = build_replay(
            scene, frames, arguments.replay_points_per_scan, arguments.replay_frame_stride,
            mosaic_directory, arguments.camera_tile_width,
        )
        write_viewer(
            viewer, colored_map.points, poses,
            title=f"nuScenes {name} · 6-camera colored map + lidar/radar EKF tracking",
            colors=colored_map.colors, replay=replay, point_size=arguments.map_voxel_size * 1.5,
            camera_width=arguments.camera_tile_width * 3,
            replay_rate_hz=LIDAR_RATE_HZ,
            replay_overlays=overlays,
        )
        for image in mosaic_directory.glob("*.jpg"):
            image.unlink()
        mosaic_directory.rmdir()
        print(f"Viewer: python -m http.server -d \"{viewer}\"")

    (output / f"{prefix}_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def summary_table(reports: list[dict]) -> str:
    lines = [
        "| scene | ATE [m] | drift [%] | moving recall | moving precision | moving pos err [m] | "
        "moving vel err [m/s] | all-vehicle recall | all-vehicle precision |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        o, m, a = r["odometry"], r["tracking"]["moving_vehicles"], r["tracking"]["all_vehicles"]
        lines.append(
            f"| {r['scene']} | {o['ate_rmse_m']:.2f} | {o['drift_percent']:.1f} | {m['recall']:.2f} | "
            f"{m['precision']:.2f} | {m['position_error_m']:.2f} | {m['velocity_error_mps']:.2f} | "
            f"{a['recall']:.2f} | {a['precision']:.2f} |"
        )
    return "\n".join(lines) + "\n"


SCENE_PICKER = """<!doctype html>
<meta charset="utf-8"><title>nuScenes Scenes</title>
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #ddd; font: 14px sans-serif; }
  body { display: flex; flex-direction: column; }
  header { padding: 6px 10px; }
  iframe { flex: 1; border: 0; width: 100%; }
</style>
<header>Scene <select id="scene">OPTIONS</select></header>
<iframe id="view"></iframe>
<script>
  const scene = document.getElementById("scene"), view = document.getElementById("view");
  const show = () => { view.src = scene.value + "/index.html"; location.hash = scene.value; };
  if (location.hash) scene.value = location.hash.slice(1);
  scene.onchange = show;
  show();
</script>
"""


def write_scene_picker(output: Path) -> None:
    """output/index.html: a dropdown over every scene viewer in the output directory."""
    viewers = sorted(path.name for path in output.glob("nuscenes_*_viewer") if (path / "index.html").exists())
    options = "".join(
        f'<option value="{name}">{name.removeprefix("nuscenes_").removesuffix("_viewer").replace("_", "-")}</option>'
        for name in viewers
    )
    (output / "index.html").write_text(SCENE_PICKER.replace("OPTIONS", options), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataroot", type=Path, default=DEFAULT_DATAROOT, help=f"default: {DEFAULT_DATAROOT}")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene", default="scene-0061", help="Scene name, or 'all'")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--map-voxel-size", type=float, default=0.1)
    parser.add_argument("--map-frame-stride", type=int, default=1)
    parser.add_argument("--replay-points-per-scan", type=int, default=6000)
    parser.add_argument("--replay-frame-stride", type=int, default=1)
    parser.add_argument("--camera-tile-width", type=int, default=320, help="width of each of the 6 mosaic tiles")
    parser.add_argument("--no-map-prior", action="store_true", help="Do not filter detections by drivable area.")
    parser.add_argument("--skip-viewer", action="store_true")
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    names = scene_names(arguments.dataroot, arguments.version) if arguments.scene == "all" else [arguments.scene]
    reports = [process_scene(arguments, name) for name in names]
    write_scene_picker(arguments.output_directory)
    table = summary_table(reports)
    print("\n" + table)
    if len(reports) > 1:
        (arguments.output_directory / "nuscenes_summary.md").write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
