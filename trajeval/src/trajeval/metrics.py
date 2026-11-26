"""Trajectory error metrics.

All functions take a reference and an (already aligned) estimate in one-to-one
correspondence.

Per-sample errors (``compute_errors``)
    * ``translation``: Euclidean position error ``||p_est - p_ref||`` (ATE).
    * ``lateral`` / ``longitudinal``: the position error expressed in the
      frame of the reference path: longitudinal along the path tangent
      (positive = ahead), lateral along its left normal (positive = left of
      the path). Vertical error is reported separately.
    * ``heading``: wrapped yaw difference ``yaw_est - yaw_ref`` (radians).
      When a trajectory has no orientation, its path-tangent heading is used.

Relative pose error (``rpe``)
    Errors of relative motions over a fixed distance (metres travelled along
    the reference) or a fixed number of frames, Sturm et al. 2012 style:
    ``E_ij = (T_ref_i^-1 T_ref_j)^-1 (T_est_i^-1 T_est_j)``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .geometry import make_se3, rotation_angle, se3_inv, wrap_angle
from .trajectory import Trajectory


def summary_stats(values: np.ndarray) -> Dict[str, float]:
    """RMSE, mean, median, std, min, max and max |.| of a 1D array."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {k: float("nan") for k in ("rmse", "mean", "median", "std", "min", "max", "max_abs")}
    return {
        "rmse": float(np.sqrt(np.mean(v ** 2))),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "max_abs": float(np.max(np.abs(v))),
    }


def compute_errors(ref: Trajectory, est: Trajectory,
                   distance: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Per-sample errors between associated poses."""
    if len(ref) != len(est):
        raise ValueError("ref and est must have the same length")
    e = est.positions - ref.positions
    tangent = ref.tangent_yaw()
    c, s = np.cos(tangent), np.sin(tangent)
    longitudinal = e[:, 0] * c + e[:, 1] * s
    lateral = -e[:, 0] * s + e[:, 1] * c
    heading = wrap_angle(est.yaw() - ref.yaw())
    if distance is None:
        distance = ref.arc_length()
    out = {
        "distance": np.asarray(distance, float),
        "translation": np.linalg.norm(e, axis=1),
        "translation_xy": np.linalg.norm(e[:, :2], axis=1),
        "lateral": lateral,
        "longitudinal": longitudinal,
        "vertical": e[:, 2],
        "heading": heading,
    }
    if ref.has_timestamps:
        out["t"] = ref.timestamps
    return out


def rpe(ref: Trajectory, est: Trajectory, delta: float = 1.0, unit: str = "m",
        distance: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Relative pose error over ``delta`` metres (``unit='m'``) or frames (``'frames'``).

    Returns per-pair translational error (m), rotational error (rad), yaw
    error (rad) and the reference distance at the start of each pair.
    """
    n = len(ref)
    if distance is None:
        distance = ref.arc_length()
    distance = np.asarray(distance, float)
    i = np.arange(n)
    if unit in ("m", "meters", "metres", "distance"):
        if delta <= 0:
            raise ValueError("RPE delta must be > 0")
        j = np.searchsorted(distance, distance + delta - 1e-9, side="left")
    elif unit in ("frames", "f", "frame"):
        step = int(round(delta))
        if step < 1:
            raise ValueError("RPE delta in frames must be >= 1")
        j = i + step
    else:
        raise ValueError(f"unknown RPE unit {unit!r}")
    ok = j < n
    i, j = i[ok], j[ok]
    empty = {"i": i, "j": j, "translation": np.zeros(0), "rotation": np.zeros(0),
             "yaw": np.zeros(0), "distance": np.zeros(0)}
    if len(i) == 0:
        return empty
    Tr = make_se3(ref.rotations_or_tangent(), ref.positions)
    Te = make_se3(est.rotations_or_tangent(), est.positions)
    dr = se3_inv(Tr[i]) @ Tr[j]
    de = se3_inv(Te[i]) @ Te[j]
    E = se3_inv(dr) @ de
    yaw = wrap_angle(np.arctan2(E[:, 1, 0], E[:, 0, 0]))
    return {
        "i": i,
        "j": j,
        "translation": np.linalg.norm(E[:, :3, 3], axis=1),
        "rotation": rotation_angle(E[:, :3, :3]),
        "yaw": yaw,
        "distance": distance[i],
    }


def compute_metrics(ref: Trajectory, est: Trajectory, *, distance: Optional[np.ndarray] = None,
                    rpe_delta: float = 1.0, rpe_unit: str = "m",
                    errors: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, object]:
    """All scalar metrics for one (ref, aligned est) pair, as a JSON-ready dict."""
    if errors is None:
        errors = compute_errors(ref, est, distance)
    d = errors["distance"]
    path_length = float(d[-1] - d[0]) if len(d) else 0.0
    final = float(errors["translation"][-1]) if len(d) else float("nan")
    r = rpe(ref, est, rpe_delta, rpe_unit, d)
    heading_deg = np.degrees(errors["heading"])
    metrics: Dict[str, object] = {
        "n_samples": int(len(ref)),
        "path_length_m": path_length,
        "ate": summary_stats(errors["translation"]),
        "ate_xy": summary_stats(errors["translation_xy"]),
        "lateral": summary_stats(errors["lateral"]),
        "longitudinal": summary_stats(errors["longitudinal"]),
        "vertical": summary_stats(errors["vertical"]),
        "heading_deg": summary_stats(heading_deg),
        "rpe": {
            "delta": float(rpe_delta),
            "unit": rpe_unit,
            "n_pairs": int(len(r["translation"])),
            "translation": summary_stats(r["translation"]),
            "rotation_deg": summary_stats(np.degrees(r["rotation"])),
        },
        "final_error_m": final,
        "drift_pct": float(100.0 * final / path_length) if path_length > 0 else float("nan"),
    }
    if rpe_unit in ("m", "meters", "metres", "distance") and len(r["translation"]):
        metrics["rpe"]["translation_pct"] = summary_stats(100.0 * r["translation"] / rpe_delta)
    return metrics
