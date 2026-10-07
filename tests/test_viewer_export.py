from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nuscenes_fusion.viewer_export import LidarReplay
from nuscenes_fusion.viewer_export import tile_groups, write_viewer


def decode_block(viewer: Path, entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of the decode() function in viewer/index.html."""
    raw = (viewer / entry["file"]).read_bytes()
    count = entry["count"]
    quantized = np.frombuffer(raw, dtype=np.uint16, count=count * 3).reshape(-1, 3)
    low, high = np.array(entry["min"]), np.array(entry["max"])
    points = low + quantized * (high - low) / 65535
    rgb = np.frombuffer(raw, dtype=np.uint8, offset=count * 6).reshape(-1, 3)
    return points, rgb


def test_write_viewer_tiles_round_trip_every_point(tmp_path: Path) -> None:
    generator = np.random.default_rng(0)
    points = generator.uniform((-100, -50, -2), (100, 50, 3), size=(5000, 3))
    colors = generator.uniform(0, 1, size=(5000, 3))
    poses = [np.eye(4), np.eye(4)]
    poses[1][:3, 3] = (5.0, 0.0, 0.0)
    replay = LidarReplay(
        source_frame_indices=np.array((1,), dtype=np.int32),
        scans=(np.array(((1.0, 2.0, 0.5, 0.25), (3.0, 1.0, 0.0, 1.0)), dtype=np.float32),),
        minimum=np.array((1.0, 1.0, 0.0), dtype=np.float32),
        maximum=np.array((3.0, 2.0, 0.5), dtype=np.float32),
    )

    index = write_viewer(
        tmp_path / "viewer", points, poses, "test", colors=colors, replay=replay,
        tile_size=50.0, overview_points=500,
    )

    viewer = index.parent
    manifest = json.loads((viewer / "manifest.json").read_text())
    assert (viewer / "vendor" / "three.min.js").is_file()
    assert manifest["pointCount"] == 5000 and manifest["hasColor"]
    assert manifest["overview"]["count"] <= 500
    assert sum(tile["count"] for tile in manifest["tiles"]) == 5000

    for tile, indices in zip(manifest["tiles"], tile_groups(points, 50.0)):
        restored, rgb = decode_block(viewer, tile)
        np.testing.assert_allclose(restored, points[indices], atol=0.01)
        np.testing.assert_allclose(rgb / 255, colors[indices], atol=1 / 255)

    assert manifest["replay"]["frames"] == [1]
    assert manifest["replay"]["counts"] == [2]
    assert manifest["replay"]["poses"][0][3] == 5.0
    assert (viewer / "replay.bin").stat().st_size == 2 * 6 + 2
