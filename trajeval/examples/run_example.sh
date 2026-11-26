#!/usr/bin/env bash
# Regenerate the synthetic data and evaluate it (run from the trajeval/ folder).
set -euo pipefail
cd "$(dirname "$0")"
python generate_examples.py
# 1) lidar and odometry (3 repetitions each) vs the 100 Hz ground truth
trajeval compare --ref data/ground_truth.csv \
  --est "lidar=data/lidar_run*.txt" "odom=data/odom_run*.csv" \
  --align se2 --out results/gt
# 2) same estimates vs the planned path (no timestamps -> distance resampling)
trajeval compare --ref data/planned_path.csv \
  --est "lidar=data/lidar_run*.txt" "odom=data/odom_run*.csv" \
  --align se2 --step 0.5 --out results/planned
# 3) drift from the start pose (no global fit)
trajeval compare --ref data/ground_truth.csv --est "odom=data/odom_run*.csv" \
  --align origin --out results/odom_drift
