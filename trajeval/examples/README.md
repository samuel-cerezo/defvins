# Examples

Synthetic data for a ground robot driving about 73 m: a straight, a left turn, an S-bend,
a right turn and a final straight. `generate_examples.py` recreates `data/`
deterministically.

| file | content |
|---|---|
| `data/ground_truth.csv` | 100 Hz reference, header `time,x,y,theta` |
| `data/planned_path.csv` | the theoretical path, `x,y` only (no timestamps), 0.5 m spacing |
| `data/lidar_run{1,2,3}.txt` | 10 Hz lidar SLAM, TUM format, in a map frame rotated 30° and shifted (5, −3) m; σ = 3 cm, 0.5° |
| `data/odom_run{1,2,3}.csv` | 20 Hz wheel odometry, header `stamp,pos_x,pos_y,yaw_deg`; scale error 1–3 % and a gyro bias, so it drifts |

Run all three scenarios:

```bash
./run_example.sh
```

The script runs these steps:

1. **Lidar and odometry vs ground truth**, timestamp association, SE(2) alignment:
   ```bash
   trajeval compare --ref data/ground_truth.csv \
       --est "lidar=data/lidar_run*.txt" "odom=data/odom_run*.csv" --align se2 --out results/gt
   ```
2. **Against the planned path.** The reference has no timestamps, so both sides are
   resampled every 0.5 m of travelled distance:
   ```bash
   trajeval compare --ref data/planned_path.csv \
       --est "lidar=data/lidar_run*.txt" "odom=data/odom_run*.csv" --align se2 --step 0.5 --out results/planned
   ```
3. **Odometry drift.** The estimate is aligned on the first pose only:
   ```bash
   trajeval compare --ref data/ground_truth.csv --est "odom=data/odom_run*.csv" --align origin --out results/odom_drift
   ```
