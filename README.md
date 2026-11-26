<p align="center">
  <h1 align="center">DefVINS: Visual–Inertial Odometry for Deformable Scenes</h1>
  <p align="center">
    <a><strong>Samuel Cerezo</strong></a> ·
    <a><strong>Javier Civera</strong></a>
  </p>
  <h3 align="center">2026</h3>
  <p align="center">
    <a href="https://www.arxiv.org/abs/2601.00702">Paper (PDF)</a> |
    <a>Video</a> |
    <a href="https://samuel-cerezo.github.io/defvins">Project Page</a>
  </p>
</p>

---

This repository provides a C++ implementation accompanying the paper **“DefVINS: Visual–Inertial Odometry for Deformable Scenes”**.

DefVINS addresses deformable environments by explicitly separating:
- a **rigid, IMU-anchored** visual–inertial state (pose, velocity, gravity direction, IMU biases), and
- a **non-rigid deformation model** represented with an embedded deformation graph built from long-term feature tracks.

In addition, the paper includes an **observability analysis** of the visual–inertial deformable estimation problem, characterizing how inertial measurements improve conditioning and stabilize the rigid trajectory in the presence of deformation.

## 📏 Trajectory evaluation (`trajeval`)

The [`trajeval/`](trajeval/) folder contains a standalone, pip-installable Python tool that
evaluates robot / SLAM trajectories against a reference: ATE, RPE, and lateral,
longitudinal and heading errors, with SE(2)/SE(3)/Sim(3) alignment and aggregation across
repeated runs. See [`trajeval/README.md`](trajeval/README.md).

## ✨ Highlights

- Visual–Inertial Odometry designed for **deformable scenes**
- Rigid (IMU-anchored) + non-rigid (deformation graph) formulation
- Deformation graph regularization:
  - elastic term
  - viscous (temporal) term
  - photometric term
- Experimental evaluation on synthetic and real deformable datasets
- Explicit **observability / conditioning analysis** (paper)

## 🧱 Requirements

The project is built with **CMake** and requires the following software:

- **C++17 compiler**
- **CMake ≥ 3.10**
- **Eigen3** (`find_package(Eigen3 REQUIRED)`)
- **OpenCV (opencv4)** (`find_package(OpenCV REQUIRED)`)
- **yaml-cpp**
- **Ceres Solver** (`find_package(Ceres REQUIRED)`)
- **Pangolin** (`find_package(Pangolin REQUIRED)`)
- **Python 3.8** development framework + **pybind11 headers**
  - The CMake configuration expects the Python framework installation layout (macOS-style)
  - The code links against `libpython3.8.dylib`

> ⚠️ Note (macOS): your `CMakeLists.txt` currently points to Homebrew paths for OpenCV and yaml-cpp, and to a Python **framework** installation (e.g., `/Library/Frameworks/Python.framework/...`).

## 📁 Repository Layout (high level)

