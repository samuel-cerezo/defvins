# DefVINS in Docker

Reproducible Ubuntu 22.04 build of DefVINS with every dependency pinned
(see the header of [`Dockerfile`](Dockerfile) for the full matrix):

| Dependency | Version | Source |
|---|---|---|
| Ubuntu | 22.04 (image pinned by digest) | Docker Hub |
| GCC / CMake | 11.2 / 3.22.1 | apt |
| Eigen | 3.4.0 | apt |
| OpenCV | 4.5.4 (with its own distro protobuf, no Homebrew mismatch) | apt |
| yaml-cpp | 0.7.0 | apt |
| Python / pybind11 | 3.10 / 2.9.1 | apt |
| Ceres Solver | 2.2.0 (commit checked) | source |
| Pangolin | v0.9.1 (commit checked), built **and installed** in `/usr/local` | source |
| numpy / nonmin-pose (C2P) | 1.26.4 / 0.0.1 | PyPI |

Optionally freeze the apt packages too with an Ubuntu snapshot:
`--build-arg UBUNTU_SNAPSHOT=20251101T000000Z`.

## Build

```bash
docker compose build
```

or without compose:

```bash
docker build -f docker/Dockerfile -t defvins:latest .
```

On Apple Silicon the image is `linux/amd64` (the `nonmin-pose` wheel only
exists for x86_64 Linux); compose sets this already, with plain `docker build`
add `--platform linux/amd64`. The first build compiles Ceres and Pangolin and
takes a while; later builds only recompile DefVINS.

## Run (headless)

Put (or symlink) the sequences in `./data` — each sequence folder in the
Mandala layout described in [`ARCHITECTURE.md`](../ARCHITECTURE.md#10-input-data-layout) — then:

```bash
docker compose run --rm defvins mandala4 mode_full
```

```bash
DEFVINS_DATA_DIR=/path/to/datasets DEFVINS_RESULTS_DIR=$PWD/out docker compose run --rm defvins mandala0 mode_no_imu
```

Without compose, `scripts/run_docker.sh <data_dir> <sequence> [mode]` does the
same.

Results appear in `./results_docker` (or `$DEFVINS_RESULTS_DIR`):
`<seq>_full.csv` / `_noNR.csv` / `_noIMU.csv`, `<seq>_gt.csv` and the init
log `full_<seq>_<date>.csv`. Formats: [`ARCHITECTURE.md` §11](../ARCHITECTURE.md#11-output-files).

The current `main()` does not open any window. If no `DISPLAY` is set, the
entrypoint still wraps the binary in `xvfb-run`, so enabling the Pangolin or
OpenCV viewers later will not crash a headless run.

## Changes made to build on Linux

- `CMakeLists.txt`: the Homebrew OpenCV/yaml-cpp paths and the python.org 3.8
  framework are still used when they exist (macOS); otherwise CMake uses
  `find_package(OpenCV)`, `find_package(yaml-cpp)`, `FindPython3` and
  `pybind11` (CMake package or pip). Ceres is required to be ≥ 2.1 because the
  code uses `ceres::Manifold`.
- `CMakeLists.txt`: source `src/alignment_utils` → `src/alignment_utils.cpp`
  (extension-less source names stop working with policy CMP0115 / CMake ≥ 3.20).
- `src/c2p_wrapper.cpp` keeps its hard-coded macOS `sys.path` entry; the image
  sets `PYTHONPATH=/opt/defvins/scripts`, which the embedded interpreter also
  searches.
