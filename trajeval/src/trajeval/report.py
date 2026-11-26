"""Writing results: metrics JSON, canonical aligned CSV, summary table, figures."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import __version__
from .evaluation import EvalConfig, EvalResult, aggregate
from .geometry import rot_to_quat, yaw_from_rot
from .io import _write_csv, write_trajectory_csv
from .trajectory import Trajectory

ALIGNED_COLUMNS = [
    "label", "repetition", "index", "t", "distance",
    "ref_x", "ref_y", "ref_z", "ref_qx", "ref_qy", "ref_qz", "ref_qw", "ref_yaw",
    "est_x", "est_y", "est_z", "est_qx", "est_qy", "est_qz", "est_qw", "est_yaw",
    "err_translation", "err_lateral", "err_longitudinal", "err_vertical", "err_heading_deg",
]


def _clean(obj):
    """Make numpy / NaN values JSON friendly (NaN -> null)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else None
    return obj


def metrics_document(results: Sequence[EvalResult], refs: Sequence[Trajectory],
                     config: EvalConfig) -> dict:
    return _clean({
        "trajeval_version": __version__,
        "config": config.to_dict(),
        "references": [
            {"name": r.name, "path": r.meta.get("path", ""), "n_poses": len(r),
             "length_m": r.length, "has_timestamps": r.has_timestamps,
             "has_orientation": r.has_orientation}
            for r in refs
        ],
        "estimates": [r.to_dict() for r in results],
        "aggregate": aggregate(results),
    })


def write_metrics_json(results, refs, config, path: str) -> str:
    with open(path, "w") as fh:
        json.dump(metrics_document(results, refs, config), fh, indent=2)
    return path


def aligned_rows(r: EvalResult):
    n = len(r.ref)
    Rr, Re = r.ref.rotations_or_tangent(), r.est.rotations_or_tangent()
    qr, qe = rot_to_quat(Rr), rot_to_quat(Re)
    yr, ye = yaw_from_rot(Rr), yaw_from_rot(Re)
    t = r.ref.timestamps if r.ref.timestamps is not None else np.full(n, np.nan)
    e = r.errors
    for i in range(n):
        yield [r.label, r.repetition, i, t[i], e["distance"][i],
               *r.ref.positions[i], *qr[i], yr[i],
               *r.est.positions[i], *qe[i], ye[i],
               e["translation"][i], e["lateral"][i], e["longitudinal"][i], e["vertical"][i],
               np.degrees(e["heading"][i])]


def write_aligned_csv(results: Sequence[EvalResult], path: str) -> str:
    """One canonical long-format CSV with every matched (ref, aligned est) pair."""
    rows = (row for r in results for row in aligned_rows(r))
    _write_csv(path, ALIGNED_COLUMNS, rows)
    return path


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "traj"


SUMMARY_COLUMNS = [
    ("ATE rmse [m]", "ate.rmse"),
    ("lat rmse [m]", "lateral.rmse"),
    ("lon rmse [m]", "longitudinal.rmse"),
    ("head rmse [deg]", "heading_deg.rmse"),
    ("RPE t rmse [m]", "rpe.translation.rmse"),
    ("RPE r rmse [deg]", "rpe.rotation_deg.rmse"),
    ("final [m]", "final_error_m"),
    ("drift [%]", "drift_pct"),
]


def _get(m: dict, dotted: str):
    cur = m
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return float("nan")
        cur = cur[part]
    return cur


def summary_table(results: Sequence[EvalResult]) -> str:
    """Markdown table: one row per estimate, then mean ± std per label."""
    head = ["estimate", "rep", "n"] + [c for c, _ in SUMMARY_COLUMNS]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in results:
        vals = [f"{_get(r.metrics, k):.4f}" for _, k in SUMMARY_COLUMNS]
        lines.append("| " + " | ".join([r.label, str(r.repetition), str(r.metrics["n_samples"])] + vals) + " |")
    agg = aggregate(results)
    if any(a["n_repetitions"] > 1 for a in agg.values()):
        lines += ["", "Across repetitions (mean ± std):", "",
                  "| " + " | ".join(["estimate", "reps"] + [c for c, _ in SUMMARY_COLUMNS]) + " |",
                  "|" + "|".join("---" for _ in range(2 + len(SUMMARY_COLUMNS))) + "|"]
        for label, a in agg.items():
            vals = []
            for _, k in SUMMARY_COLUMNS:
                s = a["metrics"].get(k)
                vals.append("nan" if s is None else f"{s['mean']:.4f} ± {s['std']:.4f}")
            lines.append("| " + " | ".join([label, str(a["n_repetitions"])] + vals) + " |")
    return "\n".join(lines)


def write_report(results: Sequence[EvalResult], refs: Sequence[Trajectory], config: EvalConfig,
                 out_dir: str, plots: bool = True) -> Dict[str, object]:
    """Write every output to ``out_dir`` and return the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, object] = {}
    written["metrics"] = write_metrics_json(results, refs, config, os.path.join(out_dir, "metrics.json"))
    written["aligned"] = write_aligned_csv(results, os.path.join(out_dir, "aligned.csv"))
    traj_dir = os.path.join(out_dir, "aligned_trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    files: List[str] = []
    for r in results:
        p = os.path.join(traj_dir, f"{_slug(r.label)}_rep{r.repetition}.csv")
        write_trajectory_csv(r.est_full, p)
        files.append(p)
    written["aligned_trajectories"] = files
    table = summary_table(results)
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write("# trajeval summary\n\n" + table + "\n")
    written["summary"] = os.path.join(out_dir, "summary.md")
    if plots:
        from . import plotting

        figs = [plotting.plot_trajectories(results, refs, os.path.join(out_dir, "trajectories.png")),
                plotting.plot_error_vs_distance(results, os.path.join(out_dir, "error_vs_distance.png"))]
        agg_fig = plotting.plot_aggregate_error(results, os.path.join(out_dir, "error_vs_distance_repetitions.png"))
        if agg_fig:
            figs.append(agg_fig)
        figs.append(plotting.plot_metric_summary(aggregate(results), os.path.join(out_dir, "metrics_summary.png")))
        written["figures"] = figs
    return written
