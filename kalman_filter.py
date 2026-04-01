import numpy as np
import pandas as pd
from pathlib import Path
import shutil


class PhysiologicalKalmanFilter:
    def __init__(self, initial_state):
        # x = [HRV, RHR, Temp, Temp_Rate]
        self.x = np.array(initial_state).reshape(4, 1)
        self.P = np.diag([10.0, 10.0, 1.0, 0.1])

        # State Transition Matrix (Ad) - Constant Velocity for Temperature
        self.Ad = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],  # Temp = Temp + Temp_Rate
            [0, 0, 0, 1]   # Temp_Rate = Temp_Rate
        ])

        # Control Matrix (Bd)
        # s1-s6 map distance and intensity to physiological costs
        s1, s2, s3, s4, s5, s6 = 0.0001, 0.0002, 0.0, 0.0, 0.005, 0.005
        self.Bd = np.array([
            [-s1, -s2],  # HRV decrease
            [s5,  s6],   # RHR increase
            [s3,  s4],   # Temp rise
            [0,   0]     # Rate not directly influenced
        ])

        self.Q = np.diag([0.5, 0.5, 0.05, 0.01])  # Process noise
        self.R = np.diag([5, 0.01, 5.0])         # Measurement noise [HRV, RHR, Temp]
        self.C = np.array([                          # Observation matrix
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])

    def predict(self, u):
        u = np.array(u).reshape(2, 1)
        self.x = self.Ad @ self.x + self.Bd @ u
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q

    def update(self, z):
        z = np.array(z).flatten()
        mask = ~np.isnan(z)
        if not any(mask):
            return

        z_masked = z[mask].reshape(-1, 1)
        C_masked = self.C[mask]
        R_masked = self.R[np.ix_(mask, mask)]

        # Kalman Gain
        S = C_masked @ self.P @ C_masked.T + R_masked
        K = self.P @ C_masked.T @ np.linalg.inv(S)

        # Posterior estimate
        y = z_masked - (C_masked @ self.x)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ C_masked) @ self.P


def clip_measurement(z):
    """Clip physiologically implausible values to NaN before filtering."""
    hrv, rhr, temp = z
    if not np.isnan(hrv) and (hrv < 5 or hrv > 200):
        hrv = np.nan
    if not np.isnan(rhr) and (rhr < 30 or rhr > 120):
        rhr = np.nan
    if not np.isnan(temp) and abs(temp) > 3.0:
        temp = np.nan
    return [hrv, rhr, temp]


DATA_DIR = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model/per_id_daily_features")
OUTPUT_DIR = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model/kalman_filtered")

# Clear output directory to avoid stale files from previous runs
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)

# Use interval files where they exist; fall back to base files for IDs with no intervals
all_files = list(DATA_DIR.glob("id_*_study_interval_*_daily_features.csv"))
interval_ids = {f.name.split("_")[1] for f in all_files}
all_files += [
    f for f in DATA_DIR.glob("id_*_daily_features.csv")
    if f.name.split("_")[1] not in interval_ids
]

print(f"Starting processing for {len(all_files)} files...")

for file_path in all_files:
    raw_df = pd.read_csv(file_path)
    df = raw_df.copy()

    filtered_data = []
    days_since_update = 0
    max_gap = 5  # Stop predicting if no data for more than 5 consecutive days
    kf = None    # Filter is not created until first valid measurement

    for _, row in df.iterrows():
        u = [row['daily_distance'] / 1000.0, row['activity_intensity']]
        z = clip_measurement([row['rmssd_avg'], row['resting_heart_rate'], row['temp_dev_norm']])

        if kf is None:
            if np.isnan(z).all():
                # No valid data yet — don't start the filter
                filtered_data.append([np.nan, np.nan, np.nan, np.nan])
                continue
            else:
                # First valid measurement — initialize filter from it
                hrv_init = z[0] if not np.isnan(z[0]) else 50.0
                rhr_init = z[1] if not np.isnan(z[1]) else 65.0
                kf = PhysiologicalKalmanFilter([hrv_init, rhr_init, 0, 0])
                kf.predict(u)
                kf.update(z)

        elif not np.isnan(z).all():
            # Valid data — full predict + update
            days_since_update = 0
            kf.predict(u)
            kf.update(z)

        elif days_since_update < max_gap:
            # Gap within threshold — predict only with zero control input to avoid drift
            if days_since_update == 0:
                kf.x[3] = 0  # Reset Temp_Rate at start of gap to prevent compounding
            kf.predict([0, 0])
            days_since_update += 1

        # If gap exceeds max_gap, state is frozen (no append needed, falls through to below)

        filtered_data.append(kf.x.flatten())

    # Concatenate filtered states with original data and save
    cols = ['f_HRV', 'f_RHR', 'f_Temp', 'f_Temp_Rate']
    filtered_df = pd.DataFrame(filtered_data, columns=cols)
    final_df = pd.concat([df, filtered_df], axis=1)

    out_file = OUTPUT_DIR / file_path.name
    final_df.to_csv(out_file, index=False)
    print(f"Successfully processed {file_path.name}")

print(f"\nDone! Check: {OUTPUT_DIR}")