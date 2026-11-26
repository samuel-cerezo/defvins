import numpy as np
import pytest

from trajeval import EvalConfig, Trajectory, evaluate, synchronize
from trajeval.association import associate_timestamps
from trajeval.geometry import rot_z
from trajeval.synthetic import add_noise, arc, s_curve


def test_associate_timestamps_nearest_one_to_one():
    t_ref = np.arange(0, 10, 0.1)
    t_est = np.arange(0, 9.9, 0.3) + 0.004
    i_ref, i_est = associate_timestamps(t_ref, t_est, max_dt=0.01)
    assert len(i_est) == len(t_est)
    assert np.all(np.abs(t_ref[i_ref] - t_est[i_est]) <= 0.01)
    assert len(np.unique(i_ref)) == len(i_ref)


def test_associate_respects_max_dt_and_offset():
    t_ref = np.arange(0, 5, 0.1)
    t_est = t_ref + 0.5
    i_ref, _ = associate_timestamps(t_ref, t_est, max_dt=0.01)
    assert np.all(t_ref[i_ref] >= 0.5 - 1e-9)  # only the overlapping part
    i_ref, i_est = associate_timestamps(t_ref, t_est, max_dt=0.01, offset=-0.5)
    assert len(i_ref) == len(t_ref) and np.all(i_ref == i_est)


def test_time_sync_different_rates():
    ref = s_curve(n=3001, dt=0.01)          # 100 Hz ground truth
    est = ref.subset(np.arange(0, 3001, 7))  # ~14 Hz estimate
    res = evaluate(ref, est, EvalConfig(align="none", sync="time", max_dt=0.001))
    assert res.metrics["n_samples"] == len(est)
    assert res.metrics["ate"]["rmse"] == pytest.approx(0.0, abs=1e-12)


def test_interp_sync_handles_unaligned_timestamps():
    ref = arc(n=2001, dt=0.01)
    est = ref.subset(np.arange(0, 2001, 10))
    est.timestamps = est.timestamps + 0.003  # off-grid, but use the true poses at those times
    sub = arc(n=2001, dt=0.01)
    t = est.timestamps
    est.positions = np.column_stack([np.interp(t, sub.timestamps, sub.positions[:, k]) for k in range(3)])
    res = evaluate(ref, est, EvalConfig(align="none", sync="interp"))
    assert res.metrics["ate"]["max"] < 1e-12
    with pytest.raises(ValueError):
        evaluate(ref, est, EvalConfig(align="none", sync="time", max_dt=0.001))


def test_distance_resampling_without_timestamps(shape):
    # Estimate sampled at a different, non-uniform density, rigidly moved, no time.
    rng = np.random.default_rng(0)
    idx = np.unique(np.concatenate([[0, len(shape) - 1], rng.choice(len(shape), 150, replace=False)]))
    est = shape.subset(idx).transformed(rot_z(0.6), np.array([3.0, -2.0, 0.0]))
    est.timestamps = None
    ref = shape.copy()
    ref.timestamps = None
    res = evaluate(ref, est, EvalConfig(align="se2", sync="auto"))
    assert res.sync_method == "distance"
    assert res.alignment.yaw == pytest.approx(-0.6, abs=2e-3)
    # only linear-interpolation (chord) error remains
    assert res.metrics["ate"]["rmse"] < 1e-2


def test_distance_normalized_handles_scale():
    ref = s_curve(n=601)
    est = ref.transformed(rot_z(0.2), np.zeros(3), 0.5)
    ref.timestamps = est.timestamps = None
    res = evaluate(ref, est, EvalConfig(align="sim2", sync="distance"))
    assert res.alignment.s == pytest.approx(2.0, rel=1e-6)
    assert res.metrics["ate"]["rmse"] < 1e-6
    with pytest.raises(ValueError):
        synchronize(ref, est, "time")


def test_distance_step():
    ref = s_curve(n=601)
    ref.timestamps = None
    res = synchronize(ref, ref.copy(), "distance", step=0.5)
    assert np.allclose(np.diff(res.distance), 0.5)


def test_noisy_estimate_without_timestamps():
    # Spacing (~0.2 m) well above the noise: resampled ATE stays at noise level.
    sigma = 0.01
    ref = s_curve(n=151)
    est = add_noise(ref, sigma_xy=sigma, seed=3)
    ref.timestamps = est.timestamps = None
    res = evaluate(ref, est, EvalConfig(align="se2"))
    assert res.metrics["ate"]["rmse"] < 1.5 * sigma * np.sqrt(2)


def test_too_few_matches_raises():
    ref = s_curve(n=50, dt=0.1)
    est = ref.copy()
    est.timestamps = est.timestamps + 100.0
    with pytest.raises(ValueError, match="matched"):
        evaluate(ref, est)


def test_position_only_inputs_use_tangent_heading():
    ref = s_curve(n=601)
    est = Trajectory(ref.positions.copy(), None, ref.timestamps)
    res = evaluate(ref, est, EvalConfig(align="none"))
    assert res.metrics["heading_deg"]["max_abs"] < 0.5


def test_dense_noisy_estimate_arc_length_is_not_inflated():
    # Estimate 4x denser than the reference with noise comparable to its spacing:
    # naive arc length would be inflated and shift samples along the path.
    sigma = 0.02
    ref = s_curve(n=151)
    est = add_noise(s_curve(n=601), sigma_xy=sigma, seed=3)
    ref.timestamps = est.timestamps = None
    res = evaluate(ref, est, EvalConfig(align="se2"))
    assert res.metrics["ate"]["rmse"] < 2 * sigma * np.sqrt(2)
