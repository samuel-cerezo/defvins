"""High-level evaluation pipeline: sync -> align -> errors -> metrics -> aggregate."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .alignment import Alignment, align
from .association import synchronize
from .metrics import compute_errors, compute_metrics
from .trajectory import Trajectory


@dataclass
class EvalConfig:
    align: str = "se2"
    sync: str = "auto"
    max_dt: float = 0.02
    time_offset: float = 0.0
    step: Optional[float] = None
    distance_mode: str = "normalized"
    rpe_delta: float = 1.0
    rpe_unit: str = "m"
    n_align: Optional[int] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class EvalResult:
    """Result of comparing one estimate against a reference."""

    label: str
    repetition: int
    ref: Trajectory               # matched reference samples
    est: Trajectory               # matched estimate samples, aligned
    est_full: Trajectory          # the whole estimate, aligned
    alignment: Alignment
    sync_method: str
    sync_info: dict
    errors: Dict[str, np.ndarray]
    metrics: Dict[str, object]
    source: str = ""
    ref_source: str = ""

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "repetition": self.repetition,
            "source": self.source,
            "reference": self.ref_source,
            "sync": {"method": self.sync_method, **_jsonable(self.sync_info)},
            "alignment": self.alignment.to_dict(),
            "metrics": self.metrics,
        }


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


def evaluate(ref: Trajectory, est: Trajectory, config: Optional[EvalConfig] = None, *,
             label: Optional[str] = None, repetition: int = 0) -> EvalResult:
    """Compare one estimate with the reference."""
    cfg = config or EvalConfig()
    sync = synchronize(ref, est, cfg.sync, max_dt=cfg.max_dt, time_offset=cfg.time_offset,
                       step=cfg.step, distance_mode=cfg.distance_mode)
    al = align(sync.ref, sync.est, cfg.align, cfg.n_align)
    est_aligned = al.apply(sync.est)
    errors = compute_errors(sync.ref, est_aligned, sync.distance)
    metrics = compute_metrics(sync.ref, est_aligned, rpe_delta=cfg.rpe_delta,
                              rpe_unit=cfg.rpe_unit, errors=errors)
    return EvalResult(
        label=label or est.name,
        repetition=repetition,
        ref=sync.ref,
        est=est_aligned,
        est_full=al.apply(est),
        alignment=al,
        sync_method=sync.method,
        sync_info=sync.info,
        errors=errors,
        metrics=metrics,
        source=str(est.meta.get("path", est.name)),
        ref_source=str(ref.meta.get("path", ref.name)),
    )


def evaluate_many(refs: Sequence[Trajectory], estimates: Sequence[tuple],
                  config: Optional[EvalConfig] = None) -> List[EvalResult]:
    """Evaluate ``(label, trajectory)`` pairs.

    Entries sharing a label are treated as repetitions of the same mission
    (repetition index = order of appearance). With a single reference every
    estimate is compared with it; with several references the i-th
    repetition of each label is compared with the i-th reference.
    """
    if not refs:
        raise ValueError("at least one reference is required")
    counts: Dict[str, int] = {}
    results = []
    for label, traj in estimates:
        rep = counts.get(label, 0)
        counts[label] = rep + 1
        if len(refs) == 1:
            ref = refs[0]
        elif rep < len(refs):
            ref = refs[rep]
        else:
            raise ValueError(
                f"label {label!r} has more repetitions ({rep + 1}) than references ({len(refs)})")
        results.append(evaluate(ref, traj, config, label=label, repetition=rep))
    return results


# ------------------------------------------------------------------ aggregation
def flatten_metrics(m: dict, prefix: str = "") -> Dict[str, float]:
    """``{"ate": {"rmse": 1}}`` -> ``{"ate.rmse": 1}`` (numeric leaves only)."""
    out: Dict[str, float] = {}
    for k, v in m.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_metrics(v, key + "."))
        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def aggregate_values(values: Iterable[float]) -> Dict[str, float]:
    """Mean, sample std (ddof=1; 0 for a single value), min, max and count."""
    v = np.asarray(list(values), float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "n": 0}
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(len(v)),
    }


_SKIP_AGG = {"rpe.delta"}


def aggregate(results: Sequence[EvalResult]) -> "OrderedDict[str, dict]":
    """Statistics of every scalar metric across the repetitions of each label."""
    groups: "OrderedDict[str, List[EvalResult]]" = OrderedDict()
    for r in results:
        groups.setdefault(r.label, []).append(r)
    out: "OrderedDict[str, dict]" = OrderedDict()
    for label, rs in groups.items():
        flats = [flatten_metrics(r.metrics) for r in rs]
        keys = [k for k in flats[0] if k not in _SKIP_AGG]
        out[label] = {
            "n_repetitions": len(rs),
            "metrics": {k: aggregate_values(f.get(k, np.nan) for f in flats) for k in keys},
        }
    return out


def mean_error_curve(results: Sequence[EvalResult], key: str = "translation",
                     n: int = 200, absolute: bool = True):
    """Mean and std of a per-sample error versus distance across repetitions.

    Curves are interpolated on a common distance grid limited to the shortest
    repetition. Returns ``(grid, mean, std)``.
    """
    curves = [(r.errors["distance"], r.errors[key]) for r in results if len(r.errors["distance"]) > 1]
    if not curves:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    lo = max(d[0] for d, _ in curves)
    hi = min(d[-1] for d, _ in curves)
    if hi <= lo:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    grid = np.linspace(lo, hi, n)
    Y = []
    for d, y in curves:
        y = np.abs(y) if absolute else y
        if key == "heading":
            y = np.degrees(y)
        Y.append(np.interp(grid, d, y))
    Y = np.array(Y)
    std = Y.std(axis=0, ddof=1) if len(Y) > 1 else np.zeros(len(grid))
    return grid, Y.mean(axis=0), std
