from __future__ import annotations

import numpy as np

from nuscenes_fusion.detection import Detection, complete_box, detect_objects
from nuscenes_fusion.trajectory_eval import evaluate_trajectory


def synthetic_sweep(car_center: tuple[float, float]) -> np.ndarray:
    """Flat ground at z = -1.8 m plus the surface of a 4.4 x 1.8 x 1.5 m car."""
    generator = np.random.default_rng(0)
    ground = np.column_stack((generator.uniform(-30, 30, (20000, 2)), np.full(20000, -1.8)))
    u, v, w = generator.uniform(-0.5, 0.5, (3, 3000))
    car = np.column_stack((car_center[0] + 4.4 * u, car_center[1] + 1.8 * v, -1.8 + 0.2 + 1.5 * (w + 0.5)))
    return np.vstack((ground, car))


def test_detect_objects_finds_the_car_and_ignores_ground() -> None:
    detections, members = detect_objects(synthetic_sweep((12.0, -4.0)))

    assert len(detections) == 1
    box = detections[0]
    np.testing.assert_allclose(box.center[:2], (12.0, -4.0), atol=0.3)
    assert 4.0 < box.length < 4.8 and 1.5 < box.width < 2.1
    assert len(members[0]) > 1000


def test_complete_box_moves_a_rear_face_away_from_the_sensor() -> None:
    rear_face_only = Detection(np.array([10.0, 0.0, 0.0]), 1.2, 1.8, 1.4, 0.0, 50)

    completed = complete_box(rear_face_only)

    np.testing.assert_allclose(completed.center[:2], (10.0 + (4.4 - 1.2) / 2, 0.0))
    assert completed.length == 4.4


def test_trajectory_metrics_ignore_a_rigid_offset_and_measure_drift() -> None:
    reference = []
    for i in range(50):
        pose = np.eye(4)
        pose[:3, 3] = (i * 1.0, 0.0, 0.0)
        reference.append(pose)
    offset = np.eye(4)
    offset[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    offset[:3, 3] = (5, 5, 0)

    rigid = evaluate_trajectory([offset @ p for p in reference], reference, window=10)
    assert rigid["ate_rmse_m"] < 1e-9 and rigid["rpe_translation_rmse_m"] < 1e-9

    stretched = [p.copy() for p in reference]
    for p in stretched:
        p[0, 3] *= 1.02  # 2 % scale drift
    drift = evaluate_trajectory(stretched, reference, window=10)
    assert abs(drift["drift_percent"] - 2.0) < 1e-6
