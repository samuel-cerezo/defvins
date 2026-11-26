# syntax=docker/dockerfile:1
#
# Reproducible build environment for DefVINS (Ubuntu 22.04 / jammy).
#
# Dependency matrix (all mutually compatible, C++17, GCC 11):
#   - Ubuntu 22.04 base image pinned by digest
#   - From apt (jammy):        Eigen 3.4.0, OpenCV 4.5.4, yaml-cpp 0.7.0,
#                              pybind11 2.9.1, Python 3.10, CMake 3.22.1,
#                              glog 0.4.0, gflags 2.2.2, SuiteSparse 5.10.1
#                              (OpenCV, protobuf, etc. all come from the same
#                              distro release -> no protobuf ABI mismatch)
#   - From source (pinned tag + commit check):
#                              Ceres Solver 2.2.0  (>= 2.1 needed for ceres::Manifold)
#                              Pangolin v0.9.1     (installed to /usr/local)
#   - From PyPI (pinned):      numpy 1.26.4, nonmin-pose 0.0.1 (C2P solver used
#                              through the embedded Python interpreter)
#
# NOTE: nonmin-pose only ships x86_64 Linux wheels, so on Apple Silicon build
# and run this image with --platform linux/amd64 (docker-compose.yml does it).
#
# Build:   docker build -f docker/Dockerfile -t defvins:latest .

ARG BASE_IMAGE=ubuntu:22.04@sha256:b000e69a50ff940390233025e7a337fb8469b509e2334fa4d018ac30f3e18cdb

###############################################################################
# Stage 1: third-party dependencies
###############################################################################
FROM ${BASE_IMAGE} AS deps

ARG DEBIAN_FRONTEND=noninteractive
ARG CERES_VERSION=2.2.0
ARG CERES_COMMIT=85331393dc0dff09f6fb9903ab0c4bfa3e134b01
ARG PANGOLIN_VERSION=v0.9.1
ARG PANGOLIN_COMMIT=7364b59a58045d1c429b146e44f6f088fd1cc518
ARG NUMPY_VERSION=1.26.4
ARG NONMIN_POSE_VERSION=0.0.1
# Optional: freeze apt to an Ubuntu snapshot, e.g. 20251101T000000Z
# (see https://snapshot.ubuntu.com). Empty = current jammy archive.
ARG UBUNTU_SNAPSHOT=

RUN if [ -n "${UBUNTU_SNAPSHOT}" ]; then \
        sed -i -E "s#http://(archive|security|ports).ubuntu.com/(ubuntu|ubuntu-ports)/?#http://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/#" /etc/apt/sources.list; \
    fi \
 && printf 'Acquire::Retries "5";\nAcquire::http::Timeout "60";\n' > /etc/apt/apt.conf.d/80-retries \
 && apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates pkg-config \
        libeigen3-dev \
        libopencv-dev \
        libyaml-cpp-dev \
        python3 python3-dev python3-pip pybind11-dev \
        libgoogle-glog-dev libgflags-dev libsuitesparse-dev \
        libopenblas-dev liblapack-dev \
        libgl1-mesa-dev libegl1-mesa-dev libglew-dev libepoxy-dev \
        libwayland-dev libxkbcommon-dev wayland-protocols \
        xvfb xauth \
 && rm -rf /var/lib/apt/lists/*

# ---- Ceres Solver -----------------------------------------------------------
RUN git clone --depth 1 --branch ${CERES_VERSION} https://github.com/ceres-solver/ceres-solver.git /tmp/ceres \
 && test "$(git -C /tmp/ceres rev-parse HEAD)" = "${CERES_COMMIT}" \
 && cmake -S /tmp/ceres -B /tmp/ceres/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
        -DBUILD_DOCUMENTATION=OFF -DUSE_CUDA=OFF \
 && cmake --build /tmp/ceres/build \
 && cmake --install /tmp/ceres/build \
 && rm -rf /tmp/ceres

# ---- Pangolin (compiled AND installed) --------------------------------------
RUN git clone --depth 1 --branch ${PANGOLIN_VERSION} https://github.com/stevenlovegrove/Pangolin.git /tmp/pangolin \
 && test "$(git -C /tmp/pangolin rev-parse HEAD)" = "${PANGOLIN_COMMIT}" \
 && cmake -S /tmp/pangolin -B /tmp/pangolin/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_TESTS=OFF \
        -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_PANGOLIN_FFMPEG=OFF \
        -DBUILD_PANGOLIN_REALSENSE=OFF -DBUILD_PANGOLIN_REALSENSE2=OFF \
        -DBUILD_PANGOLIN_OPENNI=OFF -DBUILD_PANGOLIN_OPENNI2=OFF \
        -DBUILD_PANGOLIN_LIBDC1394=OFF -DBUILD_PANGOLIN_V4L=OFF \
        -DBUILD_PANGOLIN_DEPTHSENSE=OFF -DBUILD_PANGOLIN_TELICAM=OFF \
        -DBUILD_PANGOLIN_PLEORA=OFF -DBUILD_PANGOLIN_UVC_MEDIAFOUNDATION=OFF \
 && cmake --build /tmp/pangolin/build \
 && cmake --install /tmp/pangolin/build \
 && ldconfig \
 && rm -rf /tmp/pangolin

# ---- Python side: C2P relative-pose solver ----------------------------------
RUN python3 -m pip install --no-cache-dir \
        "numpy==${NUMPY_VERSION}" "nonmin-pose==${NONMIN_POSE_VERSION}" \
 && python3 -c "from nonmin_pose import C2PFast; print('nonmin_pose OK')"

###############################################################################
# Stage 2: DefVINS
###############################################################################
FROM deps AS build

WORKDIR /opt/defvins
COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src
COPY scripts/nonmin_pose_wrapper.py ./scripts/nonmin_pose_wrapper.py
COPY docker/run_headless.sh /usr/local/bin/defvins-run

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build \
 && ln -s /opt/defvins/build/defvins /usr/local/bin/defvins \
 && chmod +x /usr/local/bin/defvins-run

# The C2P wrapper is imported by the embedded interpreter: PYTHONPATH is added
# to sys.path, so the hard-coded macOS path in src/c2p_wrapper.cpp is harmless.
ENV PYTHONPATH=/opt/defvins/scripts \
    DEFVINS_DATA=/workspace/data \
    DEFVINS_OUTPUT=/workspace/results

VOLUME ["/workspace/data", "/workspace/results"]
WORKDIR /workspace/results
ENTRYPOINT ["defvins-run"]
CMD ["--help"]
