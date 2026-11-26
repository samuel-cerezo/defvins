#!/usr/bin/env bash
# Run DefVINS headless in Docker without docker compose.
#   scripts/run_docker.sh <data_dir> <sequence> [mode_full|mode_no_nr|mode_no_imu] [results_dir]
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "usage: $0 <data_dir> <sequence> [mode_full|mode_no_nr|mode_no_imu] [results_dir]" >&2
    exit 2
fi
DATA_DIR="$(cd "$1" && pwd)"
SEQ="$2"
MODE="${3:-mode_full}"
RESULTS_DIR="${4:-$PWD/results_docker}"
IMAGE="${DEFVINS_IMAGE:-defvins:latest}"
mkdir -p "${RESULTS_DIR}"
RESULTS_DIR="$(cd "${RESULTS_DIR}" && pwd)"

exec docker run --rm --platform linux/amd64 \
    -v "${DATA_DIR}:/workspace/data:ro" \
    -v "${RESULTS_DIR}:/workspace/results" \
    "${IMAGE}" "${SEQ}" "${MODE}"
