"""Trajectory container."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from .geometry import rot_z, wrap_angle, yaw_from_rot


@dataclass
class Trajectory:
    """A sequence of poses.

    Attributes
    ----------
    positions:
        ``(N, 3)`` array of positions in metres. 2D inputs get ``z = 0``.
    rotations:
        Optional ``(N, 3, 3)`` array of body-to-world rotations. ``None`` when
        the source file has no orientation; the heading is then derived from
        the path tangent where needed.
    timestamps:
        Optional ``(N,)`` array of timestamps in seconds.
    name:
        Free-form label (usually the file name).
    planar:
        ``True`` when the source only had planar data (x, y[, yaw]).
    """

    positions: np.ndarray
    rotations: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    name: str = ""
    planar: bool = False
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        p = np.asarray(self.positions, dtype=float)
        if p.ndim != 2 or p.shape[1] not in (2, 3):
            raise ValueError(f"positions must be (N, 2) or (N, 3), got {p.shape}")
        if p.shape[1] == 2:
            p = np.column_stack([p, np.zeros(len(p))])
            self.planar = True
        self.positions = p
        if self.rotations is not None:
            R = np.asarray(self.rotations, dtype=float)
            if R.shape != (len(p), 3, 3):
                raise ValueError(f"rotations must be (N, 3, 3), got {R.shape}")
            self.rotations = R
        if self.timestamps is not None:
            t = np.asarray(self.timestamps, dtype=float).reshape(-1)
            if len(t) != len(p):
                raise ValueError("timestamps and positions differ in length")
            self.timestamps = t

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.positions)

    @classmethod
    def from_xy_yaw(cls, x, y, yaw=None, timestamps=None, name: str = "") -> "Trajectory":
        """Build a planar trajectory from x, y and optional yaw (radians)."""
        pos = np.column_stack([np.asarray(x, float), np.asarray(y, float), np.zeros(len(x))])
        R = None if yaw is None else rot_z(np.asarray(yaw, float))
        return cls(pos, R, timestamps, name=name, planar=True)

    @property
    def has_orientation(self) -> bool:
        return self.rotations is not None

    @property
    def has_timestamps(self) -> bool:
        return self.timestamps is not None

    @property
    def xy(self) -> np.ndarray:
        return self.positions[:, :2]

    def arc_length(self) -> np.ndarray:
        """Cumulative travelled distance at each pose (starts at 0)."""
        if len(self) == 0:
            return np.zeros(0)
        steps = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(steps)])

    @property
    def length(self) -> float:
        s = self.arc_length()
        return float(s[-1]) if len(s) else 0.0

    def tangent_yaw(self) -> np.ndarray:
        """Heading of the path tangent in the xy plane (radians)."""
        n = len(self)
        if n < 2:
            return np.zeros(n)
        d = np.gradient(self.xy, axis=0)
        norm = np.linalg.norm(d, axis=1)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        # Carry the last valid heading over stationary samples.
        valid = norm > 1e-12
        if not valid.any():
            return np.zeros(n)
        idx = np.where(valid, np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        first = int(np.argmax(valid))
        idx[:first] = first
        return yaw[idx]

    def yaw(self) -> np.ndarray:
        """Heading in radians: from orientations if present, else path tangent."""
        if self.rotations is not None:
            return yaw_from_rot(self.rotations)
        return self.tangent_yaw()

    def rotations_or_tangent(self) -> np.ndarray:
        """Orientations, falling back to yaw-only rotations from the tangent."""
        if self.rotations is not None:
            return self.rotations
        return rot_z(self.tangent_yaw())

    # --------------------------------------------------------------- transforms
    def copy(self) -> "Trajectory":
        return replace(
            self,
            positions=self.positions.copy(),
            rotations=None if self.rotations is None else self.rotations.copy(),
            timestamps=None if self.timestamps is None else self.timestamps.copy(),
            meta=dict(self.meta),
        )

    def subset(self, idx) -> "Trajectory":
        idx = np.asarray(idx)
        return replace(
            self,
            positions=self.positions[idx],
            rotations=None if self.rotations is None else self.rotations[idx],
            timestamps=None if self.timestamps is None else self.timestamps[idx],
            meta=dict(self.meta),
        )

    def transformed(self, R: np.ndarray, t: np.ndarray, s: float = 1.0) -> "Trajectory":
        """Apply ``p' = s R p + t`` and ``R_i' = R R_i``."""
        R = np.asarray(R, float)
        t = np.asarray(t, float).reshape(3)
        out = self.copy()
        out.positions = s * self.positions @ R.T + t
        if self.rotations is not None:
            out.rotations = np.einsum("ij,njk->nik", R, self.rotations)
        return out

    def with_yaw_offset(self, dyaw: float) -> "Trajectory":
        """Rotate every orientation by ``dyaw`` about its own z axis."""
        out = self.copy()
        out.rotations = np.einsum("nij,jk->nik", self.rotations_or_tangent(), rot_z(dyaw))
        return out


def heading_difference(yaw_a: np.ndarray, yaw_b: np.ndarray) -> np.ndarray:
    """Wrapped difference ``yaw_a - yaw_b``."""
    return wrap_angle(np.asarray(yaw_a) - np.asarray(yaw_b))
