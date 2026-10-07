"""Multi-object tracking: CTRV Extended Kalman Filter per object, lidar + radar fusion.

The EKF, CTRV model, and radar measurement model come from the ADAS lead-vehicle
tracker (D:/Learn/KalmanFilter), tuned there with NIS on nuScenes. On top of it:

* Association: lidar boxes -> tracks by Hungarian assignment (global nearest
  neighbour) on the Mahalanobis distance, chi-square gated at 99%.
* Radar: each Doppler detection updates the best gated track (range, bearing,
  absolute range rate from nuScenes' ego-motion-compensated velocities).
* Track management: tentative tracks need M=3 lidar hits with no gap above
  0.25 s before they are confirmed and get an EKF initialised from a
  least-squares velocity fit; confirmed tracks coast for up to 1 s without a
  lidar hit. A track counts as moving only once its speed is significant
  (> 2 sigma of the EKF velocity uncertainty) after MOVING_MIN_HITS hits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from .detection import Detection

# Tuning (process noise and radar R from the KalmanFilter project's NIS study).
SIGMA_ACCELERATION = 2.0        # [m/s^2]
SIGMA_YAW_ACCELERATION = 0.5    # [rad/s^2]
LIDAR_SIGMA = 0.5               # [m] box centre (partial views bias the fitted centre)
RADAR_SIGMA = np.array([1.5, np.radians(6.0), 0.5])  # range [m], bearing [rad], range rate [m/s]
RADAR_SURFACE_OFFSET = 0.9      # [m] radar returns come from the near surface, not the centre
GATE = 0.99
CONFIRM_HITS = 3
TENTATIVE_MAX_GAP = 0.25        # [s]
TENTATIVE_GATE = 2.5            # [m]
PRE_GATE = 5.0                  # [m] cheap Euclidean pre-gate before the chi-square test
SPAWN_EXCLUSION = 3.0           # [m] no new track this close to a confirmed one
CONFIRMED_MAX_COAST = 1.0       # [s]
MOVING_SPEED = 1.0              # [m/s]
MOVING_MIN_HITS = 10            # lidar hits before a track may be called moving
EPS_YAW_RATE = 1e-4


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def ctrv(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant Turn Rate and Velocity propagation of x = [px, py, v, yaw, yaw_rate]. Returns (f(x), df/dx)."""
    px, py, v, yaw, w = x
    s0, c0 = np.sin(yaw), np.cos(yaw)
    F = np.eye(5)
    if abs(w) > EPS_YAW_RATE:
        s1, c1 = np.sin(yaw + w * dt), np.cos(yaw + w * dt)
        dx, dy = v / w * (s1 - s0), v / w * (c0 - c1)
        F[0, 2:] = [(s1 - s0) / w, v / w * (c1 - c0), v * dt * c1 / w - dx / w]
        F[1, 2:] = [(c0 - c1) / w, v / w * (s1 - s0), v * dt * s1 / w - dy / w]
    else:  # straight-line limit
        dx, dy = v * c0 * dt, v * s0 * dt
        F[0, 2:] = [c0 * dt, -v * s0 * dt, -0.5 * v * dt**2 * s0]
        F[1, 2:] = [s0 * dt, v * c0 * dt, 0.5 * v * dt**2 * c0]
    F[3, 4] = dt
    return np.array([px + dx, py + dy, v, wrap(yaw + w * dt), w]), F


def ctrv_process_noise(x: np.ndarray, dt: float) -> np.ndarray:
    """Q from white longitudinal acceleration and yaw acceleration."""
    yaw = x[3]
    G = np.array([
        [0.5 * dt**2 * np.cos(yaw), 0.0],
        [0.5 * dt**2 * np.sin(yaw), 0.0],
        [dt, 0.0],
        [0.0, 0.5 * dt**2],
        [0.0, dt],
    ])
    return G @ np.diag([SIGMA_ACCELERATION**2, SIGMA_YAW_ACCELERATION**2]) @ G.T


def radar_model(x: np.ndarray, sensor_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """h(x) = [range - offset, bearing, range_rate] seen from sensor_xy, and its Jacobian."""
    px, py, v, yaw, _ = x
    dx, dy = px - sensor_xy[0], py - sensor_xy[1]
    r2 = dx**2 + dy**2
    r = np.sqrt(r2)
    vx, vy = v * np.cos(yaw), v * np.sin(yaw)
    rr = (dx * vx + dy * vy) / r
    H = np.array([
        [dx / r, dy / r, 0, 0, 0],
        [-dy / r2, dx / r2, 0, 0, 0],
        [(vx - rr * dx / r) / r, (vy - rr * dy / r) / r, (dx * np.cos(yaw) + dy * np.sin(yaw)) / r,
         (-dx * vy + dy * vx) / r, 0],
    ])
    return np.array([r - RADAR_SURFACE_OFFSET, np.arctan2(dy, dx), rr]), H


LIDAR_H = np.hstack((np.eye(2), np.zeros((2, 3))))
LIDAR_R = np.eye(2) * LIDAR_SIGMA**2
RADAR_R = np.diag(RADAR_SIGMA**2)


class EKF:
    """Predict / update algebra; models supply f, h and their Jacobians."""

    def __init__(self, x: np.ndarray, P: np.ndarray):
        self.x = np.asarray(x, dtype=float).copy()
        self.P = np.asarray(P, dtype=float).copy()

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        Q = ctrv_process_noise(self.x, dt)
        self.x, F = ctrv(self.x, dt)
        self.P = F @ self.P @ F.T + Q

    def nis(self, y: np.ndarray, H: np.ndarray, R: np.ndarray) -> float:
        S = H @ self.P @ H.T + R
        return float(y @ np.linalg.solve(S, y))

    def update(self, y: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        self.x[3] = wrap(self.x[3])
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form


@dataclass
class Track:
    track_id: int
    last_time: float
    history: list[tuple[float, np.ndarray]] = field(default_factory=list)  # (t, xy) lidar hits
    ekf: EKF | None = None
    z: float = 0.0
    size: np.ndarray = field(default_factory=lambda: np.zeros(3))  # length, width, height
    box_yaw: float = 0.0
    last_update: float = 0.0
    hits: int = 1

    def is_moving(self) -> bool:
        if self.ekf is None or self.hits < MOVING_MIN_HITS:
            return False
        speed = abs(self.ekf.x[2])
        return speed > MOVING_SPEED and speed > 2.0 * np.sqrt(self.ekf.P[2, 2])

    @property
    def confirmed(self) -> bool:
        return self.ekf is not None

    def position(self) -> np.ndarray:
        return self.ekf.x[:2] if self.ekf is not None else self.history[-1][1]

    def velocity(self) -> np.ndarray:
        if self.ekf is None:
            return np.zeros(2)
        v, yaw = self.ekf.x[2], self.ekf.x[3]
        return v * np.array([np.cos(yaw), np.sin(yaw)])

    def absorb_box(self, box: Detection, world_center: np.ndarray) -> None:
        """Smooth the displayed box size and height (partial views make single boxes noisy)."""
        measured = np.array([box.length, box.width, box.height])
        self.size = measured if not self.size.any() else 0.8 * self.size + 0.2 * np.maximum(measured, 0.8 * self.size)
        self.z = world_center[2] if self.z == 0.0 else 0.8 * self.z + 0.2 * world_center[2]
        self.box_yaw = box.yaw


@dataclass(frozen=True)
class TrackState:
    """Output for one confirmed track at one instant (global frame)."""

    track_id: int
    center: np.ndarray
    size: np.ndarray
    yaw: float
    velocity: np.ndarray
    moving: bool

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.velocity))


def initial_state(history: list[tuple[float, np.ndarray]], box_yaw: float) -> EKF:
    """Least-squares constant-velocity fit through the tentative hits -> CTRV state and covariance."""
    times = np.array([t for t, _ in history])
    positions = np.array([xy for _, xy in history])
    design = np.column_stack((np.ones_like(times), times - times[-1]))
    (position, velocity), *_ = np.linalg.lstsq(design, positions, rcond=None)
    speed = float(np.hypot(*velocity))
    if speed > MOVING_SPEED:
        yaw, yaw_sigma = float(np.arctan2(velocity[1], velocity[0])), 0.3
    else:
        speed, yaw, yaw_sigma = 0.0, box_yaw, np.pi / 2
    return EKF(
        np.array([*position, speed, yaw, 0.0]),
        np.diag([LIDAR_SIGMA**2, LIDAR_SIGMA**2, 2.0**2, yaw_sigma**2, 0.3**2]),
    )


class MultiObjectTracker:
    def __init__(self) -> None:
        self.tracks: list[Track] = []
        self.next_id = 0
        self.lidar_gate = chi2.ppf(GATE, 2)
        self.radar_gate = chi2.ppf(GATE, 3)

    def predict(self, time: float) -> None:
        for track in self.tracks:
            if track.ekf is not None:
                track.ekf.predict(time - track.last_time)
            track.last_time = time

    def update_lidar(self, time: float, boxes: list[Detection], sensor_xyz: np.ndarray) -> None:
        """Associate one sweep's boxes (centres relative to the sensor, world-aligned axes)."""
        self.predict(time)
        centers = [box.center + sensor_xyz for box in boxes]
        unmatched = set(range(len(boxes)))

        confirmed = [t for t in self.tracks if t.confirmed]
        if confirmed and boxes:
            cost = np.full((len(confirmed), len(boxes)), 1e6)
            positions = np.array([t.ekf.x[:2] for t in confirmed])
            box_xy = np.array([c[:2] for c in centers])
            near = np.linalg.norm(positions[:, None] - box_xy[None], axis=2) <= PRE_GATE
            for i, j in zip(*np.nonzero(near)):
                distance = confirmed[i].ekf.nis(centers[j][:2] - confirmed[i].ekf.x[:2], LIDAR_H, LIDAR_R)
                if distance <= self.lidar_gate:
                    cost[i, j] = distance
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e6:
                    track = confirmed[i]
                    track.ekf.update(centers[j][:2] - track.ekf.x[:2], LIDAR_H, LIDAR_R)
                    track.absorb_box(boxes[j], centers[j])
                    track.last_update = time
                    track.hits += 1
                    unmatched.discard(j)

        tentative = [t for t in self.tracks if not t.confirmed]
        remaining = sorted(unmatched)
        if tentative and remaining:
            cost = np.array([
                [np.linalg.norm(centers[j][:2] - track.history[-1][1]) for j in remaining] for track in tentative
            ])
            for i, k in zip(*linear_sum_assignment(cost)):
                if cost[i, k] <= TENTATIVE_GATE:
                    track, j = tentative[i], remaining[k]
                    track.history.append((time, centers[j][:2]))
                    track.absorb_box(boxes[j], centers[j])
                    track.last_update = time
                    track.hits += 1
                    unmatched.discard(j)
                    if len(track.history) >= CONFIRM_HITS:
                        track.ekf = initial_state(track.history, track.box_yaw)

        occupied = [t.position() for t in self.tracks if t.confirmed]
        for j in sorted(unmatched):
            if any(np.linalg.norm(centers[j][:2] - xy) < SPAWN_EXCLUSION for xy in occupied):
                continue  # a fragment of an existing object, not a new one
            track = Track(self.next_id, time, [(time, centers[j][:2])], last_update=time)
            track.absorb_box(boxes[j], centers[j])
            self.tracks.append(track)
            self.next_id += 1

        self.tracks = [
            t for t in self.tracks
            if time - t.last_update <= (CONFIRMED_MAX_COAST if t.confirmed else TENTATIVE_MAX_GAP)
        ]

    def update_radar(self, time: float, sensor_xy: np.ndarray, detections: np.ndarray) -> int:
        """detections: Kx3 [range, bearing, range_rate] in the global frame. Returns accepted count."""
        self.predict(time)
        confirmed = [t for t in self.tracks if t.confirmed]
        if not confirmed or len(detections) == 0:
            return 0
        positions = np.array([t.ekf.x[:2] for t in confirmed])
        points = sensor_xy + detections[:, :1] * np.column_stack((np.cos(detections[:, 1]), np.sin(detections[:, 1])))
        near = np.linalg.norm(points[:, None] - positions[None], axis=2) <= PRE_GATE
        accepted = 0
        for z, candidates in zip(detections, near):
            best, best_nis, best_model = None, self.radar_gate, None
            for track in (confirmed[i] for i in np.flatnonzero(candidates)):
                z_pred, H = radar_model(track.ekf.x, sensor_xy)
                y = z - z_pred
                y[1] = wrap(y[1])
                nis = track.ekf.nis(y, H, RADAR_R)
                if nis < best_nis:
                    best, best_nis, best_model = track, nis, (y, H)
            if best is not None:
                # Radar refines the state but does not keep a track alive: static radar
                # clutter would otherwise sustain lidar false positives indefinitely.
                best.ekf.update(*best_model, RADAR_R)
                accepted += 1
        return accepted

    def states(self) -> list[TrackState]:
        out = []
        for track in self.tracks:
            if not track.confirmed:
                continue
            moving = track.is_moving()
            velocity = track.velocity() if moving else np.zeros(2)
            yaw = float(np.arctan2(velocity[1], velocity[0])) if moving else track.box_yaw
            out.append(TrackState(
                track.track_id, np.array([*track.position(), track.z]), track.size.copy(), yaw, velocity, moving
            ))
        return out
