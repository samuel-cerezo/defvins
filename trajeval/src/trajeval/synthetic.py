"""Synthetic trajectories with known geometry, for tests and examples."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .geometry import rot_z
from .trajectory import Trajectory


def _planar(x, y, yaw, dt: float, name: str) -> Trajectory:
    t = np.arange(len(x)) * dt
    return Trajectory(np.column_stack([x, y, np.zeros(len(x))]), rot_z(yaw), t,
                      name=name, planar=True)


def straight_line(length: float = 20.0, n: int = 201, dt: float = 0.1,
                  name: str = "line") -> Trajectory:
    """Straight segment along +x."""
    s = np.linspace(0.0, length, n)
    return _planar(s, np.zeros(n), np.zeros(n), dt, name)


def arc(radius: float = 10.0, angle: float = np.pi / 2, n: int = 201, dt: float = 0.1,
        name: str = "arc") -> Trajectory:
    """Counter-clockwise circular arc starting at the origin heading along +x."""
    a = np.linspace(0.0, angle, n)
    x = radius * np.sin(a)
    y = radius * (1.0 - np.cos(a))
    return _planar(x, y, a, dt, name)


def s_curve(length: float = 30.0, amplitude: float = 3.0, n: int = 301, dt: float = 0.1,
            name: str = "s_curve") -> Trajectory:
    """One period of ``y = A sin(2 pi x / L)``; heading is the analytic tangent."""
    x = np.linspace(0.0, length, n)
    k = 2.0 * np.pi / length
    y = amplitude * np.sin(k * x)
    yaw = np.arctan(amplitude * k * np.cos(k * x))
    return _planar(x, y, yaw, dt, name)


def lateral_offset(traj: Trajectory, offset: float) -> Trajectory:
    """Shift every pose by ``offset`` metres along the left normal of its heading."""
    yaw = traj.yaw()
    out = traj.copy()
    out.positions = traj.positions + offset * np.column_stack([-np.sin(yaw), np.cos(yaw), np.zeros(len(yaw))])
    out.name = f"{traj.name}_offset"
    return out


def add_noise(traj: Trajectory, sigma_xy: float = 0.0, sigma_yaw: float = 0.0,
              sigma_z: float = 0.0, seed: Optional[int] = 0) -> Trajectory:
    """Add i.i.d. Gaussian noise to x, y (and z) and yaw."""
    rng = np.random.default_rng(seed)
    out = traj.copy()
    n = len(traj)
    noise = np.column_stack([rng.normal(0, sigma_xy, n), rng.normal(0, sigma_xy, n),
                             rng.normal(0, sigma_z, n) if sigma_z > 0 else np.zeros(n)])
    out.positions = traj.positions + noise
    if sigma_yaw > 0:
        dyaw = rng.normal(0, sigma_yaw, n)
        out.rotations = np.einsum("nij,njk->nik", traj.rotations_or_tangent(), rot_z(dyaw))
    return out


def random_rotation(seed: Optional[int] = 0) -> np.ndarray:
    """A random proper 3D rotation (QR of a Gaussian matrix)."""
    rng = np.random.default_rng(seed)
    Q, Rm = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(Rm)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q
