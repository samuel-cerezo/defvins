import numpy as np

from trajeval.geometry import (quat_to_rot, rot_to_quat, rot_z, rotation_angle, slerp_rot,
                               so3_exp, so3_log, wrap_angle, yaw_from_rot)
from trajeval.synthetic import random_rotation


def test_wrap_angle():
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_angle(-3 * np.pi / 2), np.pi / 2)
    assert np.isclose(wrap_angle(0.3), 0.3)


def test_quaternion_roundtrip():
    for seed in range(20):
        R = random_rotation(seed)
        q = rot_to_quat(R)
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert np.allclose(quat_to_rot(q), R, atol=1e-12)


def test_yaw_roundtrip():
    yaw = np.linspace(-3, 3, 13)
    assert np.allclose(yaw_from_rot(rot_z(yaw)), yaw)


def test_so3_log_exp():
    for seed in range(10):
        R = random_rotation(seed)
        assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-10)
    R = rot_z(np.pi)  # edge case
    assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-6)


def test_slerp_midpoint():
    R = slerp_rot(rot_z(0.0), rot_z(1.0), 0.5)
    assert np.isclose(yaw_from_rot(R), 0.5)
    assert np.isclose(rotation_angle(R), 0.5)
