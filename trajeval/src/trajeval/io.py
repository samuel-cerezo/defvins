"""Reading and writing trajectories.

Supported inputs
----------------
* **CSV / whitespace tables with a header.** Columns are detected by name
  (``t``/``time``/``timestamp``/``stamp``…, ``x``/``pos_x``/``position.x``…,
  ``qx``/``orientation.x``…, ``yaw``/``theta``/``heading``…). ROS ``rostopic
  echo -p`` exports (``field.pose.pose.position.x``) are recognised too.
* **TUM** (``timestamp tx ty tz qx qy qz qw``, whitespace separated, optional
  ``#`` header).
* **KITTI** (12 numbers per line: the row-major 3x4 pose matrix, no time).
  A 13-column variant with a leading timestamp is accepted as well.
* **Headerless numeric tables** with 2 (``x y``), 3 (``t x y``) or 4
  (``t x y yaw``) columns.

An explicit column mapping (``columns="t,x,y,yaw"``) overrides detection.
"""

from __future__ import annotations

import csv
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import quat_to_rot, rot_to_quat, rot_z, yaw_from_rot
from .trajectory import Trajectory

FORMATS = ("auto", "csv", "tum", "kitti")

# Canonical field -> exact aliases (after normalisation).
_ALIASES: Dict[str, Tuple[str, ...]] = {
    "t": ("t", "time", "times", "timestamp", "timestamps", "stamp", "ts", "sec", "secs",
          "seconds", "time_s", "timestamp_s", "t_s", "header_stamp", "field_header_stamp",
          "%time", "unix_time", "epoch"),
    "x": ("x", "px", "tx", "pos_x", "p_x", "position_x", "x_m", "pose_x", "trans_x"),
    "y": ("y", "py", "ty", "pos_y", "p_y", "position_y", "y_m", "pose_y", "trans_y"),
    "z": ("z", "pz", "tz", "pos_z", "p_z", "position_z", "z_m", "pose_z", "trans_z"),
    "qx": ("qx", "q_x", "quat_x", "orientation_x", "quaternion_x", "q_i"),
    "qy": ("qy", "q_y", "quat_y", "orientation_y", "quaternion_y", "q_j"),
    "qz": ("qz", "q_z", "quat_z", "orientation_z", "quaternion_z", "q_k"),
    "qw": ("qw", "q_w", "quat_w", "orientation_w", "quaternion_w", "q_r"),
    "yaw": ("yaw", "theta", "heading", "psi", "yaw_rad", "theta_rad", "heading_rad", "th",
            "pose_theta", "pose_yaw"),
}
_DEG_YAW = ("yaw_deg", "theta_deg", "heading_deg", "psi_deg")
_NS_TIME = ("t_ns", "time_ns", "timestamp_ns", "stamp_ns", "ns", "nsec", "nanoseconds")

_RX = {
    "x": re.compile(r"(?:^|_)(?:position|pos|p|trans|translation)_?x$"),
    "y": re.compile(r"(?:^|_)(?:position|pos|p|trans|translation)_?y$"),
    "z": re.compile(r"(?:^|_)(?:position|pos|p|trans|translation)_?z$"),
    "qx": re.compile(r"(?:^|_)(?:orientation|quat|quaternion|q)_?x$"),
    "qy": re.compile(r"(?:^|_)(?:orientation|quat|quaternion|q)_?y$"),
    "qz": re.compile(r"(?:^|_)(?:orientation|quat|quaternion|q)_?z$"),
    "qw": re.compile(r"(?:^|_)(?:orientation|quat|quaternion|q)_?w$"),
    "t": re.compile(r"(?:^|_)(?:time|timestamp|stamp)(?:_s)?$"),
}
# EuRoC / ASL style names: p_RS_R_x, q_RS_w, ...
_RX_FRAME = {
    "x": re.compile(r"^p(?:_[a-z0-9]+)+_x$"),
    "y": re.compile(r"^p(?:_[a-z0-9]+)+_y$"),
    "z": re.compile(r"^p(?:_[a-z0-9]+)+_z$"),
    "qx": re.compile(r"^q(?:_[a-z0-9]+)+_x$"),
    "qy": re.compile(r"^q(?:_[a-z0-9]+)+_y$"),
    "qz": re.compile(r"^q(?:_[a-z0-9]+)+_z$"),
    "qw": re.compile(r"^q(?:_[a-z0-9]+)+_w$"),
}


def normalize_name(name: str) -> str:
    n = name.strip().strip('"').strip("'").lower().lstrip("#").strip()
    n = re.sub(r"\[.*?\]|\(.*?\)", "", n)  # drop units like x[m] / x(m)
    n = re.sub(r"[\s.\-/:]+", "_", n).strip("_")
    return n


def detect_columns(header: Sequence[str]) -> Dict[str, int]:
    """Map canonical fields (t, x, y, z, qx..qw, yaw, yaw_deg, t_ns) to indices."""
    names = [normalize_name(h) for h in header]
    found: Dict[str, int] = {}
    used = set()

    def take(field: str, idx: int) -> None:
        found[field] = idx
        used.add(idx)

    # 1) exact aliases
    for field, aliases in _ALIASES.items():
        for i, n in enumerate(names):
            if i not in used and n in aliases:
                take(field, i)
                break
    for i, n in enumerate(names):
        if "yaw" not in found and "yaw_deg" not in found and i not in used and n in _DEG_YAW:
            take("yaw_deg", i)
        if "t" not in found and "t_ns" not in found and i not in used and n in _NS_TIME:
            take("t_ns", i)
    # 2) regex fallbacks (e.g. field_pose_pose_position_x)
    for table in (_RX, _RX_FRAME):
        for field, rx in table.items():
            if field in found or (field == "t" and "t_ns" in found):
                continue
            for i, n in enumerate(names):
                if i not in used and rx.search(n):
                    take(field, i)
                    break
    return found


def _split(line: str, delim: Optional[str]) -> List[str]:
    if delim is None:
        return line.split()
    return [c for c in next(csv.reader([line], delimiter=delim))]


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _guess_delimiter(line: str) -> Optional[str]:
    for d in (",", ";", "\t"):
        if d in line:
            return d
    return None


def _seconds(t: np.ndarray) -> np.ndarray:
    """Heuristically convert ns / us / ms epoch timestamps to seconds."""
    if len(t) == 0:
        return t
    m = float(np.nanmedian(np.abs(t)))
    if m > 1e17:
        return t * 1e-9
    if m > 1e14:
        return t * 1e-6
    if m > 1e11:
        return t * 1e-3
    return t


def _parse_columns_spec(spec: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, name in enumerate(s.strip() for s in spec.split(",")):
        if name in ("", "_", "-", "skip"):
            continue
        out[name] = i
    return out


def read_table(path: str) -> Tuple[Optional[List[str]], np.ndarray]:
    """Read a delimited numeric table; returns (header or None, data)."""
    with open(path, "r", encoding="utf-8-sig") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]
    header: Optional[List[str]] = None
    delim: Optional[str] = None
    rows: List[List[float]] = []
    pending_comment: Optional[str] = None
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("%") and not rows and header is None and "time" not in s.lower():
            continue
        if s.startswith("#") or s.startswith("//"):
            if not rows and header is None:
                pending_comment = s.lstrip("#/").strip()
            continue
        if delim is None and not rows:
            delim = _guess_delimiter(s)
        toks = [tok.strip() for tok in _split(s, delim)]
        if not rows and header is None and not all(_is_number(tok) for tok in toks if tok):
            header = toks
            continue
        try:
            rows.append([float(tok) if tok else np.nan for tok in toks])
        except ValueError:
            continue  # skip malformed / repeated header lines
    if header is None and pending_comment:
        cand = _split(pending_comment, delim)
        if rows and len(cand) == len(rows[0]) and detect_columns(cand):
            header = [c.strip() for c in cand]
    if not rows:
        raise ValueError(f"{path}: no numeric rows found")
    width = max(len(r) for r in rows)
    data = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        data[i, : len(r)] = r
    return header, data


def _from_mapping(data: np.ndarray, cols: Dict[str, int], name: str) -> Trajectory:
    if "x" not in cols or "y" not in cols:
        raise ValueError(f"{name}: could not find x/y columns (found {sorted(cols)})")
    x = data[:, cols["x"]]
    y = data[:, cols["y"]]
    planar = "z" not in cols
    z = data[:, cols["z"]] if not planar else np.zeros(len(x))
    t = None
    if "t" in cols:
        t = _seconds(data[:, cols["t"]])
    elif "t_ns" in cols:
        t = data[:, cols["t_ns"]] * 1e-9
    R = None
    if all(k in cols for k in ("qx", "qy", "qz", "qw")):
        q = data[:, [cols["qx"], cols["qy"], cols["qz"], cols["qw"]]]
        R = quat_to_rot(q)
    elif "yaw" in cols:
        R = rot_z(data[:, cols["yaw"]])
    elif "yaw_deg" in cols:
        R = rot_z(np.deg2rad(data[:, cols["yaw_deg"]]))
    pos = np.column_stack([x, y, z])
    keep = np.isfinite(pos).all(axis=1)
    if t is not None:
        keep &= np.isfinite(t)
    traj = Trajectory(pos, R, t, name=name, planar=planar)
    traj = traj.subset(np.where(keep)[0])
    traj.meta["columns"] = {k: int(v) for k, v in cols.items()}
    return _sorted_unique_time(traj)


def _sorted_unique_time(traj: Trajectory) -> Trajectory:
    if traj.timestamps is None:
        return traj
    order = np.argsort(traj.timestamps, kind="stable")
    t = traj.timestamps[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    return traj.subset(order[keep])


def _from_kitti(data: np.ndarray, name: str, with_time: bool) -> Trajectory:
    off = 1 if with_time else 0
    M = data[:, off : off + 12].reshape(-1, 3, 4)
    traj = Trajectory(M[:, :, 3].copy(), M[:, :, :3].copy(),
                      data[:, 0] if with_time else None, name=name)
    traj.meta["format"] = "kitti"
    return traj


def read_trajectory(path: str, fmt: str = "auto", columns: Optional[str] = None,
                    name: Optional[str] = None) -> Trajectory:
    """Read a trajectory file.

    Parameters
    ----------
    path: file to read.
    fmt: ``auto`` (default), ``csv``, ``tum`` or ``kitti``.
    columns: optional comma separated column names overriding detection, e.g.
        ``"t,x,y,yaw"`` or ``"t,_,x,y"`` (``_`` skips a column).
    name: label; defaults to the file stem.
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    name = name or os.path.splitext(os.path.basename(path))[0]
    header, data = read_table(path)
    ncol = data.shape[1]
    if columns:
        return _from_mapping(data, _parse_columns_spec(columns), name)
    if fmt == "kitti":
        return _from_kitti(data, name, with_time=(ncol == 13))
    if fmt == "tum":
        cols = {"t": 0, "x": 1, "y": 2, "z": 3, "qx": 4, "qy": 5, "qz": 6, "qw": 7}
        traj = _from_mapping(data, cols, name)
        traj.meta["format"] = "tum"
        return traj
    if header is not None:
        cols = detect_columns(header)
        if "x" in cols and "y" in cols:
            traj = _from_mapping(data, cols, name)
            traj.meta["format"] = "csv"
            traj.meta["header"] = header
            return traj
        if fmt == "csv":
            raise ValueError(f"{path}: could not detect x/y columns in header {header}")
    # headerless: infer from column count
    if ncol == 12:
        return _from_kitti(data, name, with_time=False)
    if ncol == 13:
        return _from_kitti(data, name, with_time=True)
    if ncol == 8:
        return read_trajectory(path, fmt="tum", name=name)
    positional = {2: "x,y", 3: "t,x,y", 4: "t,x,y,yaw", 7: "t,x,y,z,_,_,yaw"}
    if ncol in positional:
        traj = _from_mapping(data, _parse_columns_spec(positional[ncol]), name)
        traj.meta["format"] = f"headerless:{positional[ncol]}"
        return traj
    raise ValueError(
        f"{path}: cannot infer layout of a headerless file with {ncol} columns; "
        "add a header or pass columns='t,x,y,...'"
    )


# ----------------------------------------------------------------------- write
CANONICAL_COLUMNS = ["t", "x", "y", "z", "qx", "qy", "qz", "qw", "yaw"]


def trajectory_table(traj: Trajectory) -> np.ndarray:
    n = len(traj)
    t = traj.timestamps if traj.timestamps is not None else np.full(n, np.nan)
    R = traj.rotations_or_tangent()
    q = rot_to_quat(R) if n else np.zeros((0, 4))
    return np.column_stack([t, traj.positions, q, yaw_from_rot(R)])


def write_trajectory_csv(traj: Trajectory, path: str) -> None:
    """Write the canonical CSV (``t,x,y,z,qx,qy,qz,qw,yaw``), readable back."""
    _write_csv(path, CANONICAL_COLUMNS, trajectory_table(traj))


def write_tum(traj: Trajectory, path: str) -> None:
    """Write a TUM file (requires timestamps; uses indices otherwise)."""
    tab = trajectory_table(traj)
    if traj.timestamps is None:
        tab[:, 0] = np.arange(len(traj), dtype=float)
    with open(path, "w") as fh:
        fh.write("# timestamp tx ty tz qx qy qz qw\n")
        for r in tab:
            fh.write(" ".join(f"{v:.9f}" for v in r[:8]) + "\n")


def write_kitti(traj: Trajectory, path: str) -> None:
    R = traj.rotations_or_tangent()
    with open(path, "w") as fh:
        for Ri, pi in zip(R, traj.positions):
            M = np.column_stack([Ri, pi]).reshape(-1)
            fh.write(" ".join(f"{v:.9e}" for v in M) + "\n")


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.9g}"


def _write_csv(path: str, header: Sequence[str], rows) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(v) for v in r])
