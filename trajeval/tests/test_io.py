import numpy as np
import pytest

from trajeval import read_trajectory, write_kitti, write_trajectory_csv, write_tum
from trajeval.io import detect_columns
from trajeval.synthetic import add_noise, s_curve


@pytest.fixture
def traj():
    return add_noise(s_curve(n=101), sigma_xy=0.01, sigma_yaw=0.01, seed=1)


def _same(a, b, time=True):
    assert np.allclose(a.positions, b.positions, atol=1e-6)
    assert np.allclose(a.rotations, b.rotations, atol=1e-6)
    if time:
        assert np.allclose(a.timestamps, b.timestamps, atol=1e-6)


def test_tum_roundtrip(tmp_path, traj):
    p = tmp_path / "a.txt"
    write_tum(traj, p)
    b = read_trajectory(str(p))
    assert b.meta["format"] in ("tum", "csv")
    _same(traj, b)
    _same(traj, read_trajectory(str(p), fmt="tum"))


def test_kitti_roundtrip(tmp_path, traj):
    p = tmp_path / "a.txt"
    write_kitti(traj, p)
    b = read_trajectory(str(p))
    assert b.meta["format"] == "kitti" and b.timestamps is None
    _same(traj, b, time=False)


def test_canonical_csv_roundtrip(tmp_path, traj):
    p = tmp_path / "a.csv"
    write_trajectory_csv(traj, p)
    _same(traj, read_trajectory(str(p)))


@pytest.mark.parametrize("header,expect", [
    ("time,x,y,theta", {"t": 0, "x": 1, "y": 2, "yaw": 3}),
    ("timestamp;pos_x;pos_y;heading_deg", {"t": 0, "x": 1, "y": 2, "yaw_deg": 3}),
    ("%time,field.header.seq,field.pose.pose.position.x,field.pose.pose.position.y,"
     "field.pose.pose.position.z,field.pose.pose.orientation.x,field.pose.pose.orientation.y,"
     "field.pose.pose.orientation.z,field.pose.pose.orientation.w",
     {"t": 0, "x": 2, "y": 3, "z": 4, "qx": 5, "qy": 6, "qz": 7, "qw": 8}),
    ("#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []",
     {"x": 1, "y": 2, "z": 3, "qw": 4, "qx": 5, "qy": 6, "qz": 7}),
    ("Stamp X Y Yaw", {"t": 0, "x": 1, "y": 2, "yaw": 3}),
])
def test_detect_columns(header, expect):
    delim = ";" if ";" in header else ("," if "," in header else None)
    cols = detect_columns(header.split(delim) if delim else header.split())
    for k, v in expect.items():
        assert cols[k] == v, (k, cols)


def test_csv_with_degrees_and_ms_timestamps(tmp_path):
    p = tmp_path / "odom.csv"
    t_ms = 1_700_000_000_000 + np.arange(5) * 100
    with open(p, "w") as fh:
        fh.write("stamp_ms_unused,timestamp,pos_x,pos_y,yaw_deg\n")
        for i, t in enumerate(t_ms):
            fh.write(f"0,{t},{i},{2 * i},{10 * i}\n")
    tr = read_trajectory(str(p))
    assert np.allclose(np.diff(tr.timestamps), 0.1)
    assert tr.planar and np.allclose(tr.yaw(), np.radians(10 * np.arange(5)))


def test_euroc_style_ns_header(tmp_path):
    p = tmp_path / "gt.csv"
    with open(p, "w") as fh:
        fh.write("#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n")
        for i in range(4):
            fh.write(f"{1403636579758555392 + i * 5_000_000},{i},0,0,1,0,0,0\n")
    tr = read_trajectory(str(p))
    assert np.allclose(np.diff(tr.timestamps), 0.005, atol=1e-6)
    assert np.allclose(tr.rotations, np.eye(3))


def test_headerless_layouts(tmp_path):
    p = tmp_path / "xy.txt"
    np.savetxt(p, np.column_stack([np.arange(5), np.zeros(5)]))
    assert read_trajectory(str(p)).timestamps is None
    p = tmp_path / "txyyaw.csv"
    np.savetxt(p, np.column_stack([np.arange(5) * 0.1, np.arange(5), np.zeros(5), np.zeros(5)]), delimiter=",")
    tr = read_trajectory(str(p))
    assert tr.has_orientation and np.allclose(tr.timestamps, np.arange(5) * 0.1)


def test_explicit_columns(tmp_path):
    p = tmp_path / "weird.csv"
    np.savetxt(p, np.column_stack([np.arange(5), np.arange(5) * 0.1, np.arange(5) * 2.0, np.ones(5)]), delimiter=",")
    tr = read_trajectory(str(p), columns="_,t,x,y")
    assert np.allclose(tr.positions[:, 0], np.arange(5) * 2.0)
    assert np.allclose(tr.timestamps, np.arange(5) * 0.1)


def test_unsorted_and_duplicate_timestamps(tmp_path):
    p = tmp_path / "u.csv"
    with open(p, "w") as fh:
        fh.write("t,x,y\n0.2,2,0\n0.0,0,0\n0.1,1,0\n0.1,1,0\n")
    tr = read_trajectory(str(p))
    assert np.allclose(tr.timestamps, [0.0, 0.1, 0.2])


def test_bad_inputs(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b,c,d,e\n1,2,3,4,5\n")
    with pytest.raises(ValueError):
        read_trajectory(str(p))
    with pytest.raises(ValueError):
        read_trajectory(str(p), fmt="nope")
