"""Command line interface: ``trajeval compare`` and ``trajeval info``."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional, Sequence, Tuple

from . import __version__
from .alignment import ALIGN_METHODS
from .association import SYNC_METHODS
from .evaluation import EvalConfig, evaluate_many
from .io import FORMATS, read_trajectory
from .report import summary_table, write_report
from .trajectory import Trajectory


def _split_label(spec: str) -> Tuple[Optional[str], str]:
    """``"lidar=runs/a.csv"`` -> ``("lidar", "runs/a.csv")``; plain paths get no label."""
    if "=" in spec and not os.path.exists(spec):
        label, path = spec.split("=", 1)
        if label:
            return label, path
    return None, spec


def _load(path: str, fmt: str, columns: Optional[str], name: Optional[str] = None) -> Trajectory:
    traj = read_trajectory(path, fmt=fmt, columns=columns, name=name)
    traj.meta["path"] = path
    return traj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trajeval",
        description="Evaluate mobile-robot / SLAM trajectories against a reference.")
    p.add_argument("--version", action="version", version=f"trajeval {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser(
        "compare", help="compare one or more estimates with a reference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Compare estimated trajectories with a reference. Estimates may be "
            "given as LABEL=PATH; estimates sharing a label are repetitions of "
            "the same mission and get aggregated statistics."))
    c.add_argument("--ref", nargs="+", required=True, metavar="PATH",
                   help="reference trajectory; with several files the i-th repetition of "
                        "each label is compared with the i-th reference")
    c.add_argument("--est", nargs="+", required=True, metavar="[LABEL=]PATH",
                   help="estimated trajectories")
    c.add_argument("--labels", nargs="+", help="labels for --est (alternative to LABEL=PATH)")
    c.add_argument("--out", default="results", help="output directory")
    c.add_argument("--align", choices=ALIGN_METHODS, default="se2", help="alignment method")
    c.add_argument("--n-align", type=int, default=None,
                   help="fit the alignment on the first N matched poses only")
    c.add_argument("--sync", choices=SYNC_METHODS, default="auto",
                   help="association: timestamps (time/interp) or travelled distance")
    c.add_argument("--max-dt", type=float, default=0.02,
                   help="max timestamp difference [s] for --sync time")
    c.add_argument("--time-offset", type=float, default=0.0,
                   help="offset [s] added to estimate timestamps")
    c.add_argument("--step", type=float, default=None,
                   help="resampling step [m] for --sync distance (default: keep sample count)")
    c.add_argument("--distance-mode", choices=("normalized", "absolute"), default="normalized",
                   help="how estimate arc length is mapped onto the reference for --sync distance")
    c.add_argument("--rpe-delta", type=float, default=1.0, help="RPE interval")
    c.add_argument("--rpe-unit", choices=("m", "frames"), default="m", help="RPE interval unit")
    c.add_argument("--format", choices=FORMATS, default="auto", help="input format (all files)")
    c.add_argument("--ref-format", choices=FORMATS, default=None, help="override format for --ref")
    c.add_argument("--columns", default=None,
                   help="explicit column names for all inputs, e.g. 't,x,y,yaw'")
    c.add_argument("--ref-columns", default=None, help="explicit column names for --ref")
    c.add_argument("--no-plots", action="store_true", help="skip PNG figures")
    c.add_argument("--quiet", action="store_true", help="do not print the summary table")

    i = sub.add_parser("info", help="show how a trajectory file is parsed")
    i.add_argument("paths", nargs="+")
    i.add_argument("--format", choices=FORMATS, default="auto")
    i.add_argument("--columns", default=None)
    return p


def cmd_compare(a: argparse.Namespace) -> int:
    ref_fmt = a.ref_format or a.format
    ref_cols = a.ref_columns or a.columns
    refs = [_load(p, ref_fmt, ref_cols) for p in a.ref]
    if a.labels and len(a.labels) != len(a.est):
        raise SystemExit("--labels must have as many entries as --est")
    ests = []
    for k, spec in enumerate(a.est):
        label, path = _split_label(spec)
        if a.labels:
            label = a.labels[k]
        # LABEL=pattern is not expanded by the shell; expand it here.
        paths = sorted(glob.glob(path)) if glob.has_magic(path) and not os.path.exists(path) else [path]
        if not paths:
            raise ValueError(f"no file matches {path!r}")
        for pth in paths:
            traj = _load(pth, a.format, a.columns)
            ests.append((label or traj.name, traj))
    cfg = EvalConfig(align=a.align, sync=a.sync, max_dt=a.max_dt, time_offset=a.time_offset,
                     step=a.step, distance_mode=a.distance_mode, rpe_delta=a.rpe_delta,
                     rpe_unit=a.rpe_unit, n_align=a.n_align)
    results = evaluate_many(refs, ests, cfg)
    written = write_report(results, refs, cfg, a.out, plots=not a.no_plots)
    if not a.quiet:
        print(summary_table(results))
        print(f"\nWrote results to {os.path.abspath(a.out)}")
        for key in ("metrics", "aligned", "summary"):
            print(f"  {written[key]}")
        for f in written.get("figures", []):
            print(f"  {f}")
    return 0


def cmd_info(a: argparse.Namespace) -> int:
    for path in a.paths:
        t = read_trajectory(path, fmt=a.format, columns=a.columns)
        print(f"{path}")
        print(f"  format       : {t.meta.get('format', '?')}")
        if "header" in t.meta:
            print(f"  header       : {t.meta['header']}")
        if "columns" in t.meta:
            print(f"  columns      : {t.meta['columns']}")
        print(f"  poses        : {len(t)}")
        print(f"  length       : {t.length:.3f} m")
        print(f"  planar       : {t.planar}")
        print(f"  orientation  : {t.has_orientation}")
        if t.has_timestamps:
            dur = t.timestamps[-1] - t.timestamps[0]
            print(f"  timestamps   : {t.timestamps[0]:.6f} .. {t.timestamps[-1]:.6f} ({dur:.3f} s)")
        else:
            print("  timestamps   : none")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "compare":
            return cmd_compare(args)
        if args.command == "info":
            return cmd_info(args)
    except (ValueError, OSError) as exc:
        print(f"trajeval: error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
