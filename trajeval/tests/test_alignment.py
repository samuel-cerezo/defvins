import numpy as np
import pytest

from trajeval import align, umeyama
from trajeval.geometry import rot_z, yaw_from_rot
from trajeval.synthetic import add_noise, random_rotation, s_curve


def test_umeyama_recovers_sim3():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(100, 3)) * 5
    R = random_rotation(3)
    t = np.array([1.0, -2.0, 0.5])
    s = 2.5
    Q = s * P @ R.T + t
    R_, t_, s_ = umeyama(P, Q, with_scale=True)
    assert np.allclose(R_, R, atol=1e-10)
    assert np.allclose(t_, t, atol=1e-9)
    assert np.isclose(s_, s)
    R_, t_, s_ = umeyama(P, s * 0 + P @ R.T + t)
    assert s_ == 1.0 and np.allclose(R_, R, atol=1e-10)


def test_umeyama_proper_rotation_on_reflection():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(50, 3))
    Q = P * np.array([1, 1, -1])  # mirror image: best proper rotation, det = +1
    R, _, _ = umeyama(P, Q)
    assert np.isclose(np.linalg.det(R), 1.0)


@pytest.mark.parametrize("yaw,t", [(0.7, (3.0, -4.0)), (-2.5, (-10.0, 2.0)), (np.pi - 0.01, (0.0, 0.0))])
def test_se2_recovers_planar_transform(shape, yaw, t):
    # est = T^-1(ref): aligning est onto ref must give back T.
    R = rot_z(yaw)
    t3 = np.array([t[0], t[1], 0.0])
    est = shape.transformed(R.T, -R.T @ t3)
    al = align(shape, est, "se2")
    assert np.isclose(al.yaw, yaw, atol=1e-9)
    assert np.allclose(al.t, t3, atol=1e-9)
    aligned = al.apply(est)
    assert np.allclose(aligned.positions, shape.positions, atol=1e-9)
    assert np.allclose(yaw_from_rot(aligned.rotations), yaw_from_rot(shape.rotations), atol=1e-9)


def test_sim2_recovers_scale(shape):
    s = 0.5
    R = rot_z(0.4)
    est = shape.transformed(R, np.array([1.0, 2.0, 0.0]), s)
    al = align(shape, est, "sim2")
    assert np.isclose(al.s, 1 / s)
    assert np.isclose(al.yaw, -0.4)
    assert np.allclose(al.apply(est).positions, shape.positions, atol=1e-9)


def test_se3_and_sim3_recover_3d_transform():
    ref = s_curve()
    # give the path some height so the 3D problem is well-posed
    ref.positions[:, 2] = 0.5 * np.sin(ref.positions[:, 0] / 5.0)
    R, t, s = random_rotation(7), np.array([5.0, -1.0, 2.0]), 1.7
    est = ref.transformed(R, t, s)
    al = align(ref, est, "sim3")
    assert np.allclose(al.R, R.T, atol=1e-9)
    assert np.isclose(al.s, 1 / s)
    assert np.allclose(al.apply(est).positions, ref.positions, atol=1e-9)
    est = ref.transformed(R, t)
    al = align(ref, est, "se3")
    assert np.allclose(al.R, R.T, atol=1e-9)
    assert np.allclose(al.apply(est).positions, ref.positions, atol=1e-9)


def test_origin_alignment_matches_first_pose(shape):
    est = shape.transformed(rot_z(1.2), np.array([4.0, 4.0, 0.0]))
    al = align(shape, est, "origin")
    a = al.apply(est)
    assert np.allclose(a.positions, shape.positions, atol=1e-9)


def test_se2_with_noise_is_close():
    ref = s_curve(n=2001)
    R = rot_z(0.3)
    est = add_noise(ref.transformed(R, np.array([2.0, 1.0, 0.0])), sigma_xy=0.05, seed=4)
    al = align(ref, est, "se2")
    assert abs(al.yaw + 0.3) < 2e-3


def test_n_align_uses_prefix_only():
    ref = s_curve(n=301)
    est = ref.copy()
    est.positions[150:, 1] += 5.0  # a jump after the first half
    al = align(ref, est, "se2", n_align=100)
    assert np.allclose(al.t, 0, atol=1e-9) and np.isclose(al.yaw, 0, atol=1e-12)
    assert al.n_used == 100


def test_unknown_method():
    ref = s_curve(n=10)
    with pytest.raises(ValueError):
        align(ref, ref, "bogus")
