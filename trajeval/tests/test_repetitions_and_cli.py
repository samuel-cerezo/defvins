import json
import os

import numpy as np
import pytest

from trajeval import EvalConfig, aggregate, evaluate_many, write_trajectory_csv
from trajeval.cli import main
from trajeval.evaluation import aggregate_values, mean_error_curve
from trajeval.synthetic import add_noise, lateral_offset, s_curve


def test_aggregate_values():
    a = aggregate_values([1.0, 2.0, 3.0, 4.0])
    assert a["mean"] == 2.5 and a["n"] == 4
    assert a["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert aggregate_values([5.0])["std"] == 0.0


def test_repetitions_aggregate_matches_offsets():
    ref = s_curve(n=601)
    offsets = [0.1, 0.2, 0.3]
    ests = [("lidar", lateral_offset(ref, d)) for d in offsets]
    ests.append(("odom", lateral_offset(ref, 0.5)))
    res = evaluate_many([ref], ests, EvalConfig(align="none"))
    agg = aggregate(res)
    assert agg["lidar"]["n_repetitions"] == 3 and agg["odom"]["n_repetitions"] == 1
    ate = agg["lidar"]["metrics"]["ate.rmse"]
    assert ate["mean"] == pytest.approx(0.2)
    assert ate["std"] == pytest.approx(0.1)
    assert agg["lidar"]["metrics"]["lateral.mean"]["mean"] == pytest.approx(0.2, rel=5e-3)
    grid, mean, std = mean_error_curve([r for r in res if r.label == "lidar"])
    assert np.allclose(mean, 0.2) and np.allclose(std, 0.1)


def test_per_repetition_references():
    refs = [s_curve(n=301), s_curve(n=301, amplitude=4.0)]
    ests = [("a", refs[0].copy()), ("a", refs[1].copy())]
    res = evaluate_many(refs, ests, EvalConfig(align="none"))
    assert all(r.metrics["ate"]["rmse"] < 1e-12 for r in res)
    with pytest.raises(ValueError):
        evaluate_many(refs, ests + [("a", refs[0])], EvalConfig())


@pytest.fixture
def files(tmp_path):
    ref = s_curve(n=601)
    p_ref = tmp_path / "ref.csv"
    write_trajectory_csv(ref, p_ref)
    paths = []
    for k in range(3):
        e = add_noise(lateral_offset(ref, 0.1 * (k + 1)), sigma_xy=0.01, seed=k)
        from trajeval.geometry import rot_z
        e = e.transformed(rot_z(0.3), np.array([1.0, 2.0, 0.0]))
        p = tmp_path / f"lidar_{k}.csv"
        write_trajectory_csv(e, p)
        paths.append(str(p))
    p_odom = tmp_path / "odom.csv"
    write_trajectory_csv(add_noise(ref, sigma_xy=0.05, seed=9), p_odom)
    return str(p_ref), paths, str(p_odom)


def test_cli_compare(tmp_path, files, capsys):
    p_ref, lidar, odom = files
    out = tmp_path / "out"
    est = [f"lidar={p}" for p in lidar] + [odom]
    rc = main(["compare", "--ref", p_ref, "--est", *est, "--align", "se2", "--out", str(out)])
    assert rc == 0
    for f in ("metrics.json", "aligned.csv", "summary.md", "trajectories.png",
              "error_vs_distance.png", "error_vs_distance_repetitions.png", "metrics_summary.png"):
        assert (out / f).is_file(), f
    doc = json.loads((out / "metrics.json").read_text())
    assert len(doc["estimates"]) == 4
    assert doc["aggregate"]["lidar"]["n_repetitions"] == 3
    assert doc["aggregate"]["odom"]["n_repetitions"] == 1
    for e in doc["estimates"]:
        assert e["alignment"]["yaw_deg"] == pytest.approx(-np.degrees(0.3), abs=1.0) or e["label"] == "odom"
    header = (out / "aligned.csv").read_text().splitlines()[0]
    assert header.startswith("label,repetition,index,t,distance,ref_x")
    assert len(os.listdir(out / "aligned_trajectories")) == 4
    assert "lidar" in capsys.readouterr().out


def test_cli_labels_and_errors(tmp_path, files, capsys):
    p_ref, lidar, _ = files
    out = tmp_path / "o"
    assert main(["compare", "--ref", p_ref, "--est", lidar[0], "--labels", "x",
                 "--no-plots", "--quiet", "--out", str(out), "--sync", "distance",
                 "--rpe-unit", "frames", "--rpe-delta", "10"]) == 0
    doc = json.loads((out / "metrics.json").read_text())
    assert doc["estimates"][0]["label"] == "x"
    assert doc["estimates"][0]["sync"]["method"] == "distance"
    assert main(["compare", "--ref", str(tmp_path / "missing.csv"), "--est", lidar[0],
                 "--out", str(out)]) == 2


def test_cli_info(files, capsys):
    p_ref, _, _ = files
    assert main(["info", p_ref]) == 0
    assert "poses" in capsys.readouterr().out


def test_cli_label_with_glob(tmp_path, files):
    p_ref, lidar, _ = files
    out = tmp_path / "g"
    pattern = os.path.join(os.path.dirname(lidar[0]), "lidar_*.csv")
    assert main(["compare", "--ref", p_ref, "--est", f"lid={pattern}", "--no-plots",
                 "--quiet", "--out", str(out)]) == 0
    doc = json.loads((out / "metrics.json").read_text())
    assert doc["aggregate"]["lid"]["n_repetitions"] == 3
