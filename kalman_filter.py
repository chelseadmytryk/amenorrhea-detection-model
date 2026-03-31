import numpy as np
import pandas as pd
from pathlib import Path

class PhysiologicalKalmanFilter:
    def __init__(self, initial_state):
        # x = [HRV, RHR, Temp, Temp_Rate]
        self.x = np.array(initial_state).reshape(4, 1)
        self.P = np.eye(4) * 10.0  # Initial uncertainty
        
        # State Transition Matrix (Ad) - Constant Velocity for Temperature [cite: 105, 106]
        self.Ad = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1], # Temp = Temp + Temp_Rate [cite: 106]
            [0, 0, 0, 1]  # Temp_Rate = Temp_Rate [cite: 106]
        ])
        
        # Control Matrix (Bd) [cite: 111]
        # s1-s6 map distance and intensity to physiological costs [cite: 112, 113, 114]
        s1, s2, s3, s4, s5, s6 = 0.001, 0.002, 0.05, 0.1, 0.5, 1.0 # Tune these
        self.Bd = np.array([
            [-s1, -s2], # HRV decrease [cite: 112]
            [s5, s6],   # RHR increase [cite: 113]
            [s3, s4],   # Temp rise [cite: 114]
            [0, 0]      # Rate not directly influenced [cite: 115]
        ])
        
        self.Q = np.eye(4) * 0.01  # Process noise [cite: 24]
        self.R = np.eye(3) * 0.1   # Measurement noise [cite: 31]
        self.C = np.array([        # Observation matrix [cite: 30]
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])

    def predict(self, u):
        # Predict step [cite: 19]
        u = np.array(u).reshape(2, 1)
        self.x = self.Ad @ self.x + self.Bd @ u
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q

    def update(self, z):
        # Update step (with support for missing data) [cite: 27, 118]
        mask = ~np.isnan(z).flatten() # Omit missing elements [cite: 118, 119]
        if not any(mask): return
        
        z_masked = z[mask].reshape(-1, 1)
        C_masked = self.C[mask]
        R_masked = self.R[np.ix_(mask, mask)]
        
        # Kalman Gain
        S = C_masked @ self.P @ C_masked.T + R_masked
        K = self.P @ C_masked.T @ np.linalg.inv(S)
        
        # Posterior estimate [cite: 32]
        y = z_masked - (C_masked @ self.x)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ C_masked) @ self.P

DATA_DIR = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model/per_id_daily_features")
OUTPUT_DIR = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model/kalman_filtered")
OUTPUT_DIR.mkdir(exist_ok=True)

all_files = list(DATA_DIR.glob("id_*_daily_features.csv"))
print(f"Starting processing for {len(all_files)} files...")

for file_path in all_files:
    raw_df = pd.read_csv(file_path)
    
    # Pre-clean the RHR 0s and ensure we have a valid start
    df = raw_df.copy()
    
    # Initialize using first available data (not necessarily row 0 if row 0 is NaN)
    valid_start = df.dropna(subset=['rmssd_avg', 'resting_heart_rate', 'temp_dev_norm']).iloc[0]
    
    kf = PhysiologicalKalmanFilter([
        valid_start['rmssd_avg'], 
        valid_start['resting_heart_rate'], 
        valid_start['temp_dev_norm'], 
        0 
    ])
    
    filtered_data = []
    for _, row in df.iterrows():
        # Predict [cite: 19]
        u = [row['daily_distance'] / 1000.0, row['activity_intensity']] #[cite: 87, 108]
        kf.predict(u)
        
        # Update [cite: 27]
        z = np.array([row['rmssd_avg'], row['resting_heart_rate'], row['temp_dev_norm']]).reshape(3, 1)
        if not np.isnan(z).all():
            kf.update(z)
        
        filtered_data.append(kf.x.flatten())
    
    # Concatenate and Save
    cols = ['f_HRV', 'f_RHR', 'f_Temp', 'f_Temp_Rate']
    filtered_df = pd.DataFrame(filtered_data, columns=cols)
    final_df = pd.concat([df, filtered_df], axis=1)
    
    out_file = OUTPUT_DIR / file_path.name
    final_df.to_csv(out_file, index=False)
    print(f"Successfully processed {file_path.name}")

print(f"Done! Check: {OUTPUT_DIR}")