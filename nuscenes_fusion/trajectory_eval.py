"""Standard odometry metrics: ATE after SE(3) alignment, and RPE over a fixed time window."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def align_se3(estimate: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Umeyama/Horn rigid alignment (no scale): R, t minimizing ||R @ estimate + t - reference||."""
    mean_estimate, mean_reference = estimate.mean(axis=0), reference.mean(axis=0)
    covariance = (reference - mean_reference).T @ (estimate - mean_estimate)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    sign[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ sign @ vt
    return rotation, mean_reference - rotation @ mean_estimate


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))))


def evaluate_trajectory(
    estimate: Sequence[np.ndarray], reference: Sequence[np.ndarray], window: int
) -> dict[str, float]:
    """Compare two equally long lists of 4x4 poses.

    ate_rmse_m: translation RMSE after rigid alignment (global consistency).
    rpe_*: error of the relative motion over `window` frames (local drift).
    drift_percent: RPE translation RMSE as a share of the distance driven per window.
    """
    if len(estimate) != len(reference) or len(estimate) <= window:
        raise ValueError("Trajectories must have equal length greater than the RPE window.")
    est_xyz = np.array([pose[:3, 3] for pose in estimate])
    ref_xyz = np.array([pose[:3, 3] for pose in reference])
    rotation, translation = align_se3(est_xyz, ref_xyz)
    ate = np.linalg.norm(est_xyz @ rotation.T + translation - ref_xyz, axis=1)

    translation_errors, rotation_errors, distances = [], [], []
    for i in range(len(estimate) - window):
        ref_step = np.linalg.inv(reference[i]) @ reference[i + window]
        est_step = np.linalg.inv(estimate[i]) @ estimate[i + window]
        error = np.linalg.inv(ref_step) @ est_step
        translation_errors.append(np.linalg.norm(error[:3, 3]))
        rotation_errors.append(rotation_angle_degrees(error[:3, :3]))
        distances.append(np.linalg.norm(ref_step[:3, 3]))
    rpe_translation = float(np.sqrt(np.mean(np.square(translation_errors))))
    mean_distance = float(np.mean(distances))
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(ate**2))),
        "ate_max_m": float(ate.max()),
        "rpe_translation_rmse_m": rpe_translation,
        "rpe_rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rotation_errors)))),
        "drift_percent": 100.0 * rpe_translation / mean_distance if mean_distance > 0.5 else float("nan"),
        "path_length_m": float(np.linalg.norm(np.diff(ref_xyz, axis=0), axis=1).sum()),
    }
