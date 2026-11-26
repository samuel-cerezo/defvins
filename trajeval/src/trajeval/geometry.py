"""Small rotation / rigid-body helpers (numpy only)."""

from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def rot_z(yaw) -> np.ndarray:
    """Rotation matrix (or stack of them) about z for yaw angle(s) in radians."""
    yaw = np.asarray(yaw, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.zeros(yaw.shape + (3, 3))
    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c
    R[..., 2, 2] = 1.0
    return R


def yaw_from_rot(R: np.ndarray) -> np.ndarray:
    """Yaw (heading about z) of rotation matrix / matrices."""
    R = np.asarray(R)
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion(s) in (qx, qy, qz, qw) order to rotation matrix / matrices."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix / matrices to quaternion(s) (qx, qy, qz, qw), qw >= 0."""
    R = np.asarray(R, dtype=float)
    single = R.ndim == 2
    R = R.reshape(-1, 3, 3)
    q = np.empty((R.shape[0], 4))
    for k, M in enumerate(R):
        tr = np.trace(M)
        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            w = 0.25 * s
            x = (M[2, 1] - M[1, 2]) / s
            y = (M[0, 2] - M[2, 0]) / s
            z = (M[1, 0] - M[0, 1]) / s
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = 2.0 * np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2])
            w = (M[2, 1] - M[1, 2]) / s
            x = 0.25 * s
            y = (M[0, 1] + M[1, 0]) / s
            z = (M[0, 2] + M[2, 0]) / s
        elif M[1, 1] > M[2, 2]:
            s = 2.0 * np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2])
            w = (M[0, 2] - M[2, 0]) / s
            x = (M[0, 1] + M[1, 0]) / s
            y = 0.25 * s
            z = (M[1, 2] + M[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1])
            w = (M[1, 0] - M[0, 1]) / s
            x = (M[0, 2] + M[2, 0]) / s
            y = (M[1, 2] + M[2, 1]) / s
            z = 0.25 * s
        qk = np.array([x, y, z, w])
        if qk[3] < 0:
            qk = -qk
        q[k] = qk / np.linalg.norm(qk)
    return q[0] if single else q


def rotation_angle(R: np.ndarray) -> np.ndarray:
    """Rotation angle (radians) of rotation matrix / matrices."""
    R = np.asarray(R)
    c = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(c, -1.0, 1.0))


def so3_log(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a single rotation matrix."""
    theta = float(rotation_angle(R))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-10:
        return 0.5 * w
    if np.pi - theta < 1e-6:
        # Near pi: use the symmetric part to recover the axis.
        B = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(B), 0.0, None))
        k = int(np.argmax(axis))
        axis = B[:, k] / np.sqrt(B[k, k])
        return theta * axis / np.linalg.norm(axis)
    return theta / (2.0 * np.sin(theta)) * w


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rotation matrix from an axis-angle vector (Rodrigues)."""
    w = np.asarray(w, dtype=float)
    theta = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if theta < 1e-10:
        return np.eye(3) + K
    K = K / theta
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def slerp_rot(R0: np.ndarray, R1: np.ndarray, alpha: float) -> np.ndarray:
    """Geodesic interpolation between two rotations."""
    return R0 @ so3_exp(alpha * so3_log(R0.T @ R1))


def make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Stack(s) of 4x4 homogeneous transforms from rotations and translations."""
    R = np.asarray(R)
    t = np.asarray(t)
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    """Inverse of homogeneous transform(s)."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    return make_se3(Rt, -np.einsum("...ij,...j->...i", Rt, t))
