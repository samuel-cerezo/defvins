#!/usr/bin/env bash
# Run DefVINS inside the container.
#
#   defvins-run <sequence> [mode_full|mode_no_nr|mode_no_imu]
#
# <sequence> is either an absolute path or a folder name under $DEFVINS_DATA
# (default /workspace/data). The sequence folder must follow the Mandala layout
# (imu_data.txt, associations.txt, flange_poses.txt, params/{camera,imu,flange}.yaml).
#
# Outputs (see ARCHITECTURE.md, "Output files"):
#   $DEFVINS_OUTPUT/<seq>_{full,noNR,noIMU}.csv   estimated KF trajectory
#   $DEFVINS_OUTPUT/<seq>_gt.csv                  ground truth (camera wrt world)
#   <seq>/../../results/full_<seq>_<date>.csv      init-logger CSV; with the default
#                                                  layout this is $DEFVINS_OUTPUT too
#
# Headless: the current main() never opens a Pangolin/OpenCV window, but if no
# X display is available the binary is wrapped in xvfb-run so that any viewer
# code (Pangolin / cv::imshow) enabled in the future cannot abort the run.
set -euo pipefail

DATA_ROOT="${DEFVINS_DATA:-/workspace/data}"
OUT_DIR="${DEFVINS_OUTPUT:-/workspace/results}"

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    echo
    echo "Available sequences in ${DATA_ROOT}:"
    ls -1 "${DATA_ROOT}" 2>/dev/null | sed 's/^/  /' || echo "  (none: mount a data folder)"
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

SEQ="$1"
MODE="${2:-mode_full}"
case "${MODE}" in
    mode_full|mode_no_nr|mode_no_imu) ;;
    *) echo "Unknown mode '${MODE}' (use mode_full | mode_no_nr | mode_no_imu)" >&2; exit 2 ;;
esac

if [[ "${SEQ}" != /* ]]; then
    SEQ="${DATA_ROOT}/${SEQ}"
fi
SEQ="${SEQ%/}"
if [[ ! -d "${SEQ}" ]]; then
    echo "Sequence folder not found: ${SEQ}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"   # trajectory CSVs are written to the current directory

CMD=(defvins "${SEQ}" "${MODE}")
if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null 2>&1; then
    echo "[defvins-run] No DISPLAY: running headless under xvfb-run"
    exec xvfb-run -a -s "-screen 0 1280x720x24" "${CMD[@]}"
fi
exec "${CMD[@]}"
