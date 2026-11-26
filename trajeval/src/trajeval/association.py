"""Putting a reference and an estimate in one-to-one correspondence.

Three strategies are available:

``time``
    Nearest-neighbour association of timestamps within ``max_dt`` (each
    reference sample is used at most once).
``interp``
    The reference is linearly interpolated (SLERP for orientations) at every
    estimate timestamp that falls inside the reference time span. Best when
    the reference is a dense ground truth.
``distance``
    For trajectories without timestamps: both trajectories are resampled on
    a common grid of travelled distance. By default the estimate's arc length
    is normalised to the reference length (``distance_mode="normalized"``),
    which is robust to scale errors and to the arc-length inflation caused by
    noise; ``"absolute"`` uses metres travelled as-is.

``auto`` picks ``time`` when both trajectories carry timestamps, ``distance``
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .geometry import slerp_rot
from .trajectory import Trajectory

SYNC_METHODS = ("auto", "time", "interp", "distance")


@dataclass
class SyncResult:
    ref: Trajectory
    est: Trajectory
    method: str
    info: dict = field(default_factory=dict)
    #: distance travelled along the (full) reference at each matched sample
    distance: Optional[np.ndarray] = None


def associate_timestamps(t_ref: np.ndarray, t_est: np.ndarray, max_dt: float,
                         offset: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one nearest-neighbour association of two sorted time axes.

    Returns index arrays ``(i_ref, i_est)`` with ``|t_ref - (t_est + offset)| <= max_dt``.
    """
    t_ref = np.asarray(t_ref, float)
    t_est = np.asarray(t_est, float) + offset
    if len(t_ref) == 0 or len(t_est) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    j = np.searchsorted(t_ref, t_est)
    j0 = np.clip(j - 1, 0, len(t_ref) - 1)
    j1 = np.clip(j, 0, len(t_ref) - 1)
    pick = np.where(np.abs(t_ref[j0] - t_est) <= np.abs(t_ref[j1] - t_est), j0, j1)
    dt = np.abs(t_ref[pick] - t_est)
    ok = dt <= max_dt
    i_est = np.where(ok)[0]
    i_ref = pick[ok]
    dt = dt[ok]
    # Enforce one-to-one: for duplicate reference indices keep the closest estimate.
    order = np.lexsort((dt, i_ref))
    i_ref, i_est = i_ref[order], i_est[order]
    first = np.concatenate([[True], np.diff(i_ref) != 0]) if len(i_ref) else np.zeros(0, bool)
    i_ref, i_est = i_ref[first], i_est[first]
    order = np.argsort(i_est)
    return i_ref[order], i_est[order]


def interpolate_at(traj: Trajectory, key: np.ndarray, query: np.ndarray) -> Trajectory:
    """Interpolate ``traj`` at ``query`` values of a monotonic per-sample ``key``.

    Positions are interpolated linearly and orientations by SLERP. ``query``
    must lie inside ``[key[0], key[-1]]``.
    """
    key = np.asarray(key, float)
    query = np.asarray(query, float)
    pos = np.column_stack([np.interp(query, key, traj.positions[:, k]) for k in range(3)])
    R = None
    if traj.rotations is not None:
        i1 = np.clip(np.searchsorted(key, query, side="right"), 1, len(key) - 1)
        i0 = i1 - 1
        span = key[i1] - key[i0]
        alpha = np.where(span > 0, (query - key[i0]) / np.where(span > 0, span, 1.0), 0.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        R = np.stack([slerp_rot(traj.rotations[a], traj.rotations[b], w)
                      for a, b, w in zip(i0, i1, alpha)]) if len(query) else np.zeros((0, 3, 3))
    t = None
    if traj.timestamps is not None:
        t = np.interp(query, key, traj.timestamps)
    out = Trajectory(pos, R, t, name=traj.name, planar=traj.planar, meta=dict(traj.meta))
    return out


def _strictly_increasing(key: np.ndarray) -> np.ndarray:
    """Indices that make ``key`` strictly increasing (drops repeated values)."""
    keep = np.concatenate([[True], np.diff(key) > 1e-12])
    return np.where(keep)[0]


def sync_time(ref: Trajectory, est: Trajectory, max_dt: float = 0.02,
              offset: float = 0.0) -> SyncResult:
    i_ref, i_est = associate_timestamps(ref.timestamps, est.timestamps, max_dt, offset)
    r = ref.subset(i_ref)
    e = est.subset(i_est)
    return SyncResult(r, e, "time", {"max_dt": max_dt, "time_offset": offset,
                                     "n_ref": len(ref), "n_est": len(est),
                                     "n_matched": len(i_ref)},
                      distance=ref.arc_length()[i_ref])


def sync_interp(ref: Trajectory, est: Trajectory, offset: float = 0.0) -> SyncResult:
    t_est = est.timestamps + offset
    inside = np.where((t_est >= ref.timestamps[0]) & (t_est <= ref.timestamps[-1]))[0]
    e = est.subset(inside)
    r = interpolate_at(ref, ref.timestamps, t_est[inside])
    dist = np.interp(t_est[inside], ref.timestamps, ref.arc_length())
    return SyncResult(r, e, "interp", {"time_offset": offset, "n_ref": len(ref),
                                       "n_est": len(est), "n_matched": len(inside)},
                      distance=dist)


def robust_arc_length(positions: np.ndarray, min_step: float) -> np.ndarray:
    """Arc length measured between points at least ``min_step`` apart.

    Measuring distance between every consecutive noisy sample inflates the
    travelled distance (noise adds a zig-zag). Accumulating only over points
    at least ``min_step`` apart, and interpolating in between, removes most of
    that bias while keeping the result monotonic.
    """
    n = len(positions)
    if n == 0:
        return np.zeros(0)
    if min_step <= 0:
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(steps)])
    kept = [0]
    last = positions[0]
    for i in range(1, n):
        if np.linalg.norm(positions[i] - last) >= min_step:
            kept.append(i)
            last = positions[i]
    if kept[-1] != n - 1:
        kept.append(n - 1)
    kept = np.asarray(kept)
    chords = np.diff(positions[kept], axis=0)
    lengths = np.linalg.norm(chords, axis=1)
    s_k = np.concatenate([[0.0], np.cumsum(lengths)])
    # Samples between two kept points: project onto the chord joining them.
    seg = np.clip(np.searchsorted(kept, np.arange(n), side="right") - 1, 0, len(kept) - 2)
    u = chords[seg] / np.where(lengths[seg] > 0, lengths[seg], 1.0)[:, None]
    along = np.einsum("ij,ij->i", positions - positions[kept[seg]], u)
    s = s_k[seg] + np.clip(along, 0.0, lengths[seg])
    return np.maximum.accumulate(s)


def sync_distance(ref: Trajectory, est: Trajectory, step: Optional[float] = None,
                  mode: str = "normalized") -> SyncResult:
    """Resample both trajectories on a common travelled-distance grid."""
    if mode not in ("normalized", "absolute"):
        raise ValueError("distance mode must be 'normalized' or 'absolute'")
    s_ref = ref.arc_length()
    steps = np.diff(s_ref)
    steps = steps[steps > 0]
    # Compare like with like: measure the estimate at the reference resolution.
    # 0.75: keep every sample of an estimate as dense as the reference.
    min_step = 0.75 * float(np.median(steps)) if len(steps) else 0.0
    if mode == "normalized" and s_ref[-1] > 0:
        min_step *= est.length / s_ref[-1]  # estimate may be in another scale
    s_est = robust_arc_length(est.positions, min_step)
    kr = _strictly_increasing(s_ref)
    ke = _strictly_increasing(s_est)
    ref_u, est_u = ref.subset(kr), est.subset(ke)
    s_ref, s_est = s_ref[kr], s_est[ke]
    L_ref, L_est = float(s_ref[-1]), float(s_est[-1])
    if L_ref <= 0 or L_est <= 0:
        raise ValueError("cannot resample by distance: a trajectory has zero length")
    if mode == "normalized":
        s_est_m = s_est * (L_ref / L_est)
        L = L_ref
    else:
        s_est_m = s_est
        L = min(L_ref, L_est)
    if step is None or step <= 0:
        n = max(min(len(ref_u), len(est_u)), 2)
        grid = np.linspace(0.0, L, n)
    else:
        grid = np.arange(0.0, L + 1e-9, step)
    r = interpolate_at(ref_u, s_ref, grid)
    e = interpolate_at(est_u, s_est_m, grid)
    return SyncResult(r, e, "distance", {"mode": mode, "step": float(grid[1] - grid[0]),
                                         "n_ref": len(ref), "n_est": len(est),
                                         "n_matched": len(grid), "ref_length": L_ref,
                                         "est_length": L_est},
                      distance=grid)


def synchronize(ref: Trajectory, est: Trajectory, method: str = "auto", *,
                max_dt: float = 0.02, time_offset: float = 0.0,
                step: Optional[float] = None, distance_mode: str = "normalized") -> SyncResult:
    """Return reference and estimate samples in one-to-one correspondence."""
    if method not in SYNC_METHODS:
        raise ValueError(f"unknown sync method {method!r}; expected one of {SYNC_METHODS}")
    if method == "auto":
        method = "time" if (ref.has_timestamps and est.has_timestamps) else "distance"
    if method in ("time", "interp") and not (ref.has_timestamps and est.has_timestamps):
        raise ValueError(f"sync '{method}' needs timestamps in both trajectories")
    if method == "time":
        res = sync_time(ref, est, max_dt, time_offset)
    elif method == "interp":
        res = sync_interp(ref, est, time_offset)
    else:
        res = sync_distance(ref, est, step, distance_mode)
    if len(res.ref) < 3:
        raise ValueError(
            f"only {len(res.ref)} matched samples between {ref.name!r} and {est.name!r} "
            f"(sync={res.method}); check timestamps, --max-dt or --time-offset"
        )
    return res
