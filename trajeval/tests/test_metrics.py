"""Metrics against closed-form values on synthetic line / curve / S paths."""

import numpy as np
import pytest

from trajeval import EvalConfig, compute_errors, compute_metrics, evaluate, rpe, summary_stats
from trajeval.geometry import rot_z
from trajeval.synthetic import add_noise, lateral_offset, straight_line


def test_identical_trajectories_have_zero_error(shape):
    m = compute_metrics(shape, shape.copy())
    assert m["ate"]["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert m["rpe"]["translation"]["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert m["heading_deg"]["max_abs"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("d", [0.25, -0.4])
def test_constant_lateral_offset(shape, d):
    est = lateral_offset(shape, d)
    e = compute_errors(shape, est)
    # Lateral error equals the offset everywhere, longitudinal error is ~0.
    assert np.allclose(e["lateral"], d, atol=2e-3 * abs(d) + 1e-4)
    assert np.allclose(e["longitudinal"], 0.0, atol=2e-3 * abs(d) + 1e-4)
    m = compute_metrics(shape, est)
    assert m["ate"]["rmse"] == pytest.approx(abs(d), rel=1e-9)
    assert m["lateral"]["mean"] == pytest.approx(d, rel=5e-3)
    assert m["final_error_m"] == pytest.approx(abs(d))


def test_constant_longitudinal_offset_on_line():
    ref = straight_line(length=10, n=101)
    est = ref.transformed(np.eye(3), np.array([0.3, 0.0, 0.0]))
    e = compute_errors(ref, est)
    assert np.allclose(e["longitudinal"], 0.3)
    assert np.allclose(e["lateral"], 0.0)


@pytest.mark.parametrize("bias_deg", [2.0, -7.5])
def test_constant_heading_bias(shape, bias_deg):
    est = shape.with_yaw_offset(np.radians(bias_deg))
    m = compute_metrics(shape, est)
    assert m["heading_deg"]["mean"] == pytest.approx(bias_deg, abs=1e-9)
    assert m["heading_deg"]["std"] == pytest.approx(0.0, abs=1e-9)
    assert m["ate"]["rmse"] == pytest.approx(0.0, abs=1e-12)


def test_heading_error_wraps():
    ref = straight_line(n=11)
    est = ref.with_yaw_offset(np.radians(350.0))
    assert compute_metrics(ref, est)["heading_deg"]["mean"] == pytest.approx(-10.0)


def test_rpe_scale_drift_on_line():
    # est travels (1 + k) metres per reference metre: RPE over d metres = k * d.
    k, d = 0.05, 2.0
    ref = straight_line(length=20.0, n=201)  # 0.1 m spacing
    est = ref.copy()
    est.positions[:, 0] *= 1 + k
    r = rpe(ref, est, delta=d, unit="m")
    assert np.allclose(r["translation"], k * d, atol=1e-9)
    assert np.allclose(r["rotation"], 0.0)
    assert len(r["translation"]) == 201 - 20
    m = compute_metrics(ref, est, rpe_delta=d)
    assert m["rpe"]["translation_pct"]["mean"] == pytest.approx(100 * k)
    # ATE grows linearly: e(s) = k s -> final error k L, drift k * 100 %.
    assert m["final_error_m"] == pytest.approx(k * 20.0)
    assert m["drift_pct"] == pytest.approx(100 * k)
    assert m["ate"]["rmse"] == pytest.approx(k * np.sqrt(np.mean(ref.positions[:, 0] ** 2)))


def test_rpe_frames_heading_drift():
    # est yaw drifts by w rad per frame: RPE rotation over n frames = n w.
    ref = straight_line(n=101)
    w = np.radians(0.1)
    est = ref.copy()
    est.rotations = rot_z(w * np.arange(len(ref)))
    r = rpe(ref, est, delta=5, unit="frames")
    assert np.allclose(r["rotation"], 5 * w)
    # Relative translation Δp (1 m) is seen from a frame rotated by w*i:
    # |E_t| = |Δp| * 2 sin(w i / 2).
    dp = 5 * 20.0 / 100
    assert np.allclose(r["translation"], dp * 2 * np.sin(w * r["i"] / 2), atol=1e-12)


def test_rpe_invariant_to_rigid_transform(shape):
    est = add_noise(shape, sigma_xy=0.02, sigma_yaw=0.01, seed=2)
    moved = est.transformed(rot_z(1.0), np.array([5.0, -3.0, 0.0]))
    a = rpe(shape, est, 1.0)
    b = rpe(shape, moved, 1.0)
    assert np.allclose(a["translation"], b["translation"], atol=1e-9)
    assert np.allclose(a["rotation"], b["rotation"], atol=1e-9)


def test_gaussian_noise_ate_matches_theory():
    # i.i.d. N(0, s^2) noise in x and y: E[|e|^2] = 2 s^2 -> ATE RMSE ~ s sqrt(2).
    sigma = 0.05
    ref = straight_line(length=100.0, n=20001)
    est = add_noise(ref, sigma_xy=sigma, seed=11)
    m = compute_metrics(ref, est)
    assert m["ate"]["rmse"] == pytest.approx(sigma * np.sqrt(2), rel=0.02)
    assert m["lateral"]["std"] == pytest.approx(sigma, rel=0.03)
    assert m["longitudinal"]["std"] == pytest.approx(sigma, rel=0.03)
    # Rayleigh mean: s sqrt(pi/2)
    assert m["ate"]["mean"] == pytest.approx(sigma * np.sqrt(np.pi / 2), rel=0.02)


def test_end_to_end_transform_offset_noise(shape):
    """Transformed + offset + noisy estimate: SE(2) alignment then analytic errors."""
    sigma, d = 0.01, 0.2
    est = add_noise(lateral_offset(shape, d), sigma_xy=sigma, seed=5)
    est = est.transformed(rot_z(-0.8), np.array([12.0, 3.0, 0.0]))
    # No alignment would be terrible; SE(2) brings it back.
    res = evaluate(shape, est, EvalConfig(align="se2", sync="time"))
    assert abs(res.alignment.yaw - 0.8) < 0.02
    # After alignment the error is dominated by noise + residual offset <= d.
    expected = np.sqrt(d ** 2 + 2 * sigma ** 2)
    assert res.metrics["ate"]["rmse"] <= expected * 1.05
    # Without alignment (est only offset + noise), metrics match theory.
    res0 = evaluate(shape, add_noise(lateral_offset(shape, d), sigma_xy=sigma, seed=5),
                    EvalConfig(align="none"))
    assert res0.metrics["lateral"]["mean"] == pytest.approx(d, abs=3 * sigma / np.sqrt(len(shape)) + 1e-3)
    assert res0.metrics["ate"]["rmse"] == pytest.approx(expected, rel=0.05)


def test_summary_stats():
    s = summary_stats(np.array([3.0, -4.0]))
    assert s["rmse"] == pytest.approx(np.sqrt(12.5))
    assert s["mean"] == -0.5 and s["max_abs"] == 4.0 and s["min"] == -4.0
