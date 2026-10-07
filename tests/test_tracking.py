from __future__ import annotations

import numpy as np

from nuscenes_fusion.detection import Detection
from nuscenes_fusion.tracking import MultiObjectTracker, ctrv, radar_model


def numeric_jacobian(function, x: np.ndarray, step: float = 1e-6) -> np.ndarray:
    base = function(x)
    columns = []
    for i in range(len(x)):
        shifted = x.copy()
        shifted[i] += step
        columns.append((function(shifted) - base) / step)
    return np.column_stack(columns)


def test_ctrv_jacobian_matches_finite_differences_turning_and_straight() -> None:
    # At zero yaw rate the straight-line branch ignores yaw rate, so the finite-difference step
    # must be large enough to leave it; the analytic Jacobian is the turning model's limit.
    for x, step, tolerance in (
        (np.array([1.0, 2.0, 8.0, 0.4, 0.3]), 1e-6, 1e-4),
        (np.array([1.0, 2.0, 8.0, 0.4, 0.0]), 1e-3, 1e-3),
    ):
        _, analytic = ctrv(x, 0.1)
        numeric = numeric_jacobian(lambda s: ctrv(s, 0.1)[0], x, step)
        np.testing.assert_allclose(analytic, numeric, atol=tolerance)


def test_radar_jacobian_matches_finite_differences() -> None:
    x, sensor = np.array([12.0, 5.0, 7.0, 0.3, 0.1]), np.array([1.0, -1.0])
    _, analytic = radar_model(x, sensor)
    np.testing.assert_allclose(analytic, numeric_jacobian(lambda s: radar_model(s, sensor)[0], x), atol=1e-4)


def box_at(x: float, y: float) -> Detection:
    return Detection(np.array([x, y, 0.0]), 4.4, 1.8, 1.5, 0.0, 200)


def test_tracker_confirms_moving_object_and_estimates_velocity() -> None:
    generator = np.random.default_rng(0)
    tracker = MultiObjectTracker()
    for step in range(60):  # 3 s at 20 Hz, 6 m/s along +x, 0.2 m measurement noise
        time = step * 0.05
        noisy = np.array([10.0 + 6.0 * time, 3.0]) + generator.normal(0, 0.2, 2)
        tracker.update_lidar(time, [box_at(*noisy)], np.zeros(3))

    (state,) = tracker.states()
    assert state.moving
    np.testing.assert_allclose(state.velocity, (6.0, 0.0), atol=0.6)
    np.testing.assert_allclose(state.center[:2], (10.0 + 6.0 * 2.95, 3.0), atol=0.6)


def test_static_clutter_does_not_become_moving() -> None:
    generator = np.random.default_rng(1)
    tracker = MultiObjectTracker()
    for step in range(60):
        tracker.update_lidar(step * 0.05, [box_at(*(np.array([20.0, -4.0]) + generator.normal(0, 0.3, 2)))], np.zeros(3))
    (state,) = tracker.states()
    assert not state.moving


def test_radar_doppler_updates_a_confirmed_track() -> None:
    tracker = MultiObjectTracker()
    for step in range(5):
        tracker.update_lidar(step * 0.05, [box_at(20.0 + 0.5 * step, 0.0)], np.zeros(3))
    track = next(t for t in tracker.tracks if t.confirmed)
    before = track.ekf.P[2, 2]
    # Radar at the origin sees the target straight ahead closing away at 10 m/s.
    position = track.ekf.x[:2]
    z = np.array([[np.hypot(*position) - 0.9, np.arctan2(position[1], position[0]), 10.0]])
    assert tracker.update_radar(0.25, np.zeros(2), z) == 1
    assert track.ekf.P[2, 2] < before
