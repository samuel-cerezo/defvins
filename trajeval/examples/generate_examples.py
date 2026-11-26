"""Generate the synthetic example dataset in ``examples/data``.

Mission: a ground robot drives ~70 m (straight, left turn, S-bend, right turn).

* ``ground_truth.csv``   100 Hz reference, ``time,x,y,theta`` header.
* ``planned_path.csv``   the theoretical path, ``x,y`` only (no time), 0.5 m spacing.
* ``lidar_run{1,2,3}.txt``  10 Hz lidar SLAM in TUM format, expressed in a map frame
  rotated by 30 deg and shifted, with small noise.
* ``odom_run{1,2,3}.csv``   20 Hz wheel odometry (``stamp,pos_x,pos_y,yaw_deg``)
  re-integrated from corrupted increments: scale error + gyro bias -> drift.

Run:  python examples/generate_examples.py
"""

from __future__ import annotations

import os

import numpy as np

from trajeval import Trajectory, write_trajectory_csv, write_tum
from trajeval.geometry import rot_z, wrap_angle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")


def mission(dt: float = 0.01, speed: float = 1.0) -> Trajectory:
    """Integrate a piecewise curvature profile at constant speed."""
    # (length [m], curvature [1/m])
    segments = [(15.0, 0.0), (np.pi / 2 * 6.0, 1 / 6.0), (10.0, 0.0),
                (np.pi / 2 * 4.0, -1 / 4.0), (np.pi / 2 * 4.0, 1 / 4.0),
                (10.0, 0.0), (np.pi / 2 * 5.0, -1 / 5.0), (8.0, 0.0)]
    ds = speed * dt
    kappa = np.concatenate([np.full(int(round(L / ds)), k) for L, k in segments])
    yaw = np.concatenate([[0.0], np.cumsum(kappa * ds)])
    mid = yaw[:-1] + 0.5 * kappa * ds
    x = np.concatenate([[0.0], np.cumsum(ds * np.cos(mid))])
    y = np.concatenate([[0.0], np.cumsum(ds * np.sin(mid))])
    t = np.arange(len(x)) * dt
    return Trajectory.from_xy_yaw(x, y, yaw, t, name="ground_truth")


def odometry(gt: Trajectory, rate_div: int, scale: float, gyro_bias: float,
             sigma: float, rng) -> Trajectory:
    """Re-integrate body-frame increments with a scale error and a yaw-rate bias."""
    sub = gt.subset(np.arange(0, len(gt), rate_div))
    yaw = sub.yaw()
    dp = np.diff(sub.positions[:, :2], axis=0)
    c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
    fwd = c * dp[:, 0] + s * dp[:, 1]
    lat = -s * dp[:, 0] + c * dp[:, 1]
    dyaw = wrap_angle(np.diff(yaw))
    dt = np.diff(sub.timestamps)
    fwd = fwd * scale * (1 + rng.normal(0, sigma, len(fwd)))
    dyaw = dyaw + gyro_bias * dt + rng.normal(0, sigma * 0.01, len(dyaw))
    x, y, th = [0.0], [0.0], [0.0]
    for f, l, w in zip(fwd, lat, dyaw):
        x.append(x[-1] + f * np.cos(th[-1]) - l * np.sin(th[-1]))
        y.append(y[-1] + f * np.sin(th[-1]) + l * np.cos(th[-1]))
        th.append(th[-1] + w)
    return Trajectory.from_xy_yaw(x, y, th, sub.timestamps)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2026)
    gt = mission()

    with open(os.path.join(OUT, "ground_truth.csv"), "w") as fh:
        fh.write("time,x,y,theta\n")
        for t, (x, y, _), th in zip(gt.timestamps, gt.positions, gt.yaw()):
            fh.write(f"{t:.3f},{x:.5f},{y:.5f},{th:.6f}\n")

    planned = gt.subset(np.arange(0, len(gt), 50))
    with open(os.path.join(OUT, "planned_path.csv"), "w") as fh:
        fh.write("x,y\n")
        for x, y, _ in planned.positions:
            fh.write(f"{x:.4f},{y:.4f}\n")

    R_map = rot_z(np.radians(30.0))
    t_map = np.array([5.0, -3.0, 0.0])
    for k in range(1, 4):
        sub = gt.subset(np.arange(0, len(gt), 10))
        noisy = sub.copy()
        noisy.positions = sub.positions + np.column_stack(
            [rng.normal(0, 0.03, len(sub)), rng.normal(0, 0.03, len(sub)), np.zeros(len(sub))])
        noisy.rotations = rot_z(sub.yaw() + rng.normal(0, np.radians(0.5), len(sub)))
        noisy.timestamps = sub.timestamps + rng.normal(0, 0.001, len(sub))
        # lidar map frame: est = T_map^-1 * gt
        lidar = noisy.transformed(R_map.T, -R_map.T @ t_map)
        write_tum(lidar, os.path.join(OUT, f"lidar_run{k}.txt"))

        odom = odometry(gt, rate_div=5, scale=1.0 + 0.01 * k,
                        gyro_bias=np.radians(0.05 + 0.03 * rng.standard_normal()),
                        sigma=0.01, rng=rng)
        with open(os.path.join(OUT, f"odom_run{k}.csv"), "w") as fh:
            fh.write("stamp,pos_x,pos_y,yaw_deg\n")
            for t, (x, y, _), th in zip(odom.timestamps, odom.positions, odom.yaw()):
                fh.write(f"{t:.3f},{x:.5f},{y:.5f},{np.degrees(th):.4f}\n")

    print(f"wrote example data to {OUT}")


if __name__ == "__main__":
    main()
