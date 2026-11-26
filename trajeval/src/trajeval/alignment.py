"""Rigid / similarity alignment of an estimate onto a reference.

Methods
-------
``none``    no alignment.
``origin``  the first matched estimate pose is moved onto the first matched
            reference pose (SE(3)); useful to measure drift.
``se2``     planar rotation about z + xy translation (Umeyama in 2D).
``sim2``    ``se2`` + scale.
``se3``     full 3D rotation + translation (Umeyama / Horn).
``sim3``    ``se3`` + scale (Umeyama with scale), e.g. monocular VO.

All methods return ``(R, t, s)`` such that ``p_aligned = s * R @ p_est + t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import make_se3, rotation_angle, se3_inv, yaw_from_rot
from .trajectory import Trajectory

ALIGN_METHODS = ("none", "origin", "se2", "sim2", "se3", "sim3")


@dataclass
class Alignment:
    method: str
    R: np.ndarray
    t: np.ndarray
    s: float = 1.0
    n_used: int = 0

    def apply(self, traj: Trajectory) -> Trajectory:
        return traj.transformed(self.R, self.t, self.s)

    def inverse(self) -> "Alignment":
        Ri = self.R.T
        return Alignment(self.method, Ri, -Ri @ self.t / self.s, 1.0 / self.s, self.n_used)

    @property
    def yaw(self) -> float:
        return float(yaw_from_rot(self.R))

    @property
    def angle(self) -> float:
        return float(rotation_angle(self.R))

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "rotation": self.R.tolist(),
            "translation": self.t.tolist(),
            "scale": float(self.s),
            "yaw_deg": float(np.degrees(self.yaw)),
            "rotation_angle_deg": float(np.degrees(self.angle)),
            "n_used": int(self.n_used),
        }


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False):
    """Least-squares ``dst ≈ s R src + t`` (Umeyama, 1991).

    ``src`` and ``dst`` are ``(N, d)`` arrays of corresponding points. Returns
    ``(R, t, s)`` with ``R`` a proper rotation (``det(R) = +1``).
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2:
        raise ValueError("src and dst must be (N, d) arrays of the same shape")
    n, d = src.shape
    if n < 2:
        raise ValueError("need at least two point pairs")
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (xs ** 2).sum() / n
        if var_s <= 0:
            raise ValueError("degenerate source points (zero variance)")
        s = float(np.trace(np.diag(D) @ S) / var_s)
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def align(ref: Trajectory, est: Trajectory, method: str = "se2",
          n_align: Optional[int] = None) -> Alignment:
    """Estimate the transform bringing ``est`` onto ``ref``.

    ``ref`` and ``est`` must already be in one-to-one correspondence (same
    length). ``n_align`` restricts the fit to the first ``n_align`` pairs.
    """
    if method not in ALIGN_METHODS:
        raise ValueError(f"unknown alignment {method!r}; expected one of {ALIGN_METHODS}")
    if len(ref) != len(est):
        raise ValueError("ref and est must be associated (same length) before alignment")
    n = len(ref) if n_align is None or n_align <= 0 else min(n_align, len(ref))
    P, Q = est.positions[:n], ref.positions[:n]
    if method == "none":
        return Alignment("none", np.eye(3), np.zeros(3), 1.0, 0)
    if method == "origin":
        Tr = make_se3(ref.rotations_or_tangent()[0], ref.positions[0])
        Te = make_se3(est.rotations_or_tangent()[0], est.positions[0])
        T = Tr @ se3_inv(Te)
        return Alignment("origin", T[:3, :3], T[:3, 3], 1.0, 1)
    if method in ("se2", "sim2"):
        R2, t2, s = umeyama(P[:, :2], Q[:, :2], with_scale=(method == "sim2"))
        R = np.eye(3)
        R[:2, :2] = R2
        t = np.array([t2[0], t2[1], 0.0])
        return Alignment(method, R, t, s, n)
    R, t, s = umeyama(P, Q, with_scale=(method == "sim3"))
    return Alignment(method, R, t, s, n)
