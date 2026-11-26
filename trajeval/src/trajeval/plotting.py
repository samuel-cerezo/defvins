"""Matplotlib figures (PNG) for evaluation results."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .evaluation import EvalResult, mean_error_curve  # noqa: E402
from .trajectory import Trajectory  # noqa: E402

# Okabe-Ito colour-blind safe palette (black reserved for the reference).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
REF_COLOR = "#222222"


def _groups(results: Sequence[EvalResult]) -> "OrderedDict[str, List[EvalResult]]":
    g: "OrderedDict[str, List[EvalResult]]" = OrderedDict()
    for r in results:
        g.setdefault(r.label, []).append(r)
    return g


def label_colors(results: Sequence[EvalResult]) -> Dict[str, str]:
    return {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(_groups(results))}


def _style_axes(ax) -> None:
    ax.grid(True, color="#dddddd", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_trajectories(results: Sequence[EvalResult], refs: Sequence[Trajectory], path: str,
                      title: Optional[str] = None) -> str:
    """Top view (x-y) of the reference(s) and every aligned estimate."""
    colors = label_colors(results)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for k, ref in enumerate(refs):
        ax.plot(ref.positions[:, 0], ref.positions[:, 1], color=REF_COLOR, lw=2.0,
                ls="--", label="reference" if k == 0 else None, zorder=3)
        ax.plot(*ref.positions[0, :2], marker="o", color=REF_COLOR, ms=6, zorder=4)
    for label, rs in _groups(results).items():
        multi = len(rs) > 1
        for r in rs:
            p = r.est_full.positions
            ax.plot(p[:, 0], p[:, 1], color=colors[label], lw=1.1 if multi else 1.6,
                    alpha=0.6 if multi else 0.95,
                    label=f"{label} (n={len(rs)})" if r.repetition == 0 and multi
                    else (label if r.repetition == 0 else None))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    align = results[0].alignment.method if results else ""
    ax.set_title(title or f"Trajectories (alignment: {align})")
    _style_axes(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_error_vs_distance(results: Sequence[EvalResult], path: str) -> str:
    """Translation, lateral and heading error against distance travelled."""
    colors = label_colors(results)
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.0), sharex=True)
    specs = [("translation", "translation error [m]", 1.0),
             ("lateral", "lateral error [m]", 1.0),
             ("heading", "heading error [deg]", 180.0 / np.pi)]
    for ax, (key, ylabel, k) in zip(axes, specs):
        for label, rs in _groups(results).items():
            multi = len(rs) > 1
            for r in rs:
                ax.plot(r.errors["distance"], k * r.errors[key], color=colors[label],
                        lw=0.9 if multi else 1.3, alpha=0.55 if multi else 0.95,
                        label=label if r.repetition == 0 else None)
        if key != "translation":
            ax.axhline(0.0, color="#888888", lw=0.7)
        ax.set_ylabel(ylabel)
        _style_axes(ax)
    axes[-1].set_xlabel("distance travelled along reference [m]")
    axes[0].legend(frameon=False, loc="upper left", ncol=min(4, len(colors)))
    axes[0].set_title("Error vs distance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_aggregate_error(results: Sequence[EvalResult], path: str) -> Optional[str]:
    """Mean ± std (across repetitions) of |error| vs distance, one band per label."""
    groups = _groups(results)
    if not any(len(rs) > 1 for rs in groups.values()):
        return None
    colors = label_colors(results)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
    for ax, key, ylabel in ((axes[0], "translation", "|translation error| [m]"),
                            (axes[1], "heading", "|heading error| [deg]")):
        for label, rs in groups.items():
            grid, mean, std = mean_error_curve(rs, key)
            if not len(grid):
                continue
            ax.plot(grid, mean, color=colors[label], lw=1.6, label=f"{label} (n={len(rs)})")
            ax.fill_between(grid, mean - std, mean + std, color=colors[label], alpha=0.2, lw=0)
        ax.set_ylabel(ylabel)
        _style_axes(ax)
    axes[-1].set_xlabel("distance travelled along reference [m]")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_title("Error vs distance across repetitions (mean ± 1 std)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_summary(aggregate: dict, path: str) -> str:
    """Bar chart of ATE RMSE, lateral RMSE and heading RMSE (mean ± std)."""
    labels = list(aggregate)
    keys = [("ate.rmse", "ATE RMSE [m]"), ("lateral.rmse", "lateral RMSE [m]"),
            ("heading_deg.rmse", "heading RMSE [deg]"), ("rpe.translation.rmse", "RPE trans. RMSE [m]")]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.6))
    for ax, (key, title) in zip(axes, keys):
        means = [aggregate[lab]["metrics"].get(key, {}).get("mean", np.nan) for lab in labels]
        stds = [aggregate[lab]["metrics"].get(key, {}).get("std", 0.0) for lab in labels]
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=stds, color=[PALETTE[i % len(PALETTE)] for i in xs],
               capsize=4, width=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title, fontsize=10)
        _style_axes(ax)
        ax.grid(False, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
