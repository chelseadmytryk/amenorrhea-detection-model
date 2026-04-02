import pandas as pd
import numpy as np
import os
from pathlib import Path

def generate_amenorrhea_data(input_csv, output_csv, new_id):
    # Load healthy baseline
    df = pd.read_csv(input_csv)
    df['id'] = new_id # Assign new synthetic ID
    
    # 1. HRV Decay (alpha) and Temp Drop (beta)
    alpha = 0.008  # 0.8% decay per day
    beta = 0.015   # 0.015°C drop per day
    
    # 2. Suppress Luteal Shift (Anovulation Simulation)
    # Note: Using 'temp_dev_norm' and 'rmssd_avg' to match your mcPHASES column names
    if 'temp_dev_norm' in df.columns:
        # Flatten any potential rise to simulate the lack of progesterone shift
        follicular_mean = df.iloc[0:14]['temp_dev_norm'].mean()
        df.loc[14:, 'temp_dev_norm'] = follicular_mean + np.random.normal(0, 0.05, len(df)-14)

    # 3. Apply Cumulative Metabolic Decay (The Trend for T_dot)
    df['days'] = np.arange(len(df))
    df['temp_dev_norm'] = df['temp_dev_norm'] - (beta * df['days'])
    
    # 4. Suppress HRV Elasticity
    if 'rmssd_avg' in df.columns:
        # Cumulative autonomic fatigue
        df['rmssd_avg'] = df['rmssd_avg'] * (1 - (alpha * df['days']))
        # Prevent recovery spikes (Ceiling)
        hrv_ceiling = df['rmssd_avg'].mean() * 1.1
        df['rmssd_avg'] = df['rmssd_avg'].clip(upper=hrv_ceiling)

    # 5. Maintain Sensor Noise (R)
    df['temp_dev_norm'] += np.random.normal(0, 0.03, len(df))
    df['rmssd_avg'] += np.random.normal(0, 1.5, len(df))

    # Cleanup and Save
    df = df.drop(columns=['days'])
    df.to_csv(output_csv, index=False)

# --- EXECUTION LOGIC ---
DATA_DIR = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model/per_id_daily_features")
AUG_DIR = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model/augmented_data")
AUG_DIR.mkdir(exist_ok=True)

all_files = list(DATA_DIR.glob("id_*_daily_features.csv"))

print(f"Augmenting {len(all_files)} files to create a balanced dataset...")

for file_path in all_files:
    # Generate a unique ID for the synthetic person (e.g., original ID + 100)
    original_id = int(file_path.name.split('_')[1])
    new_id = original_id + 200 # Using 200-series for Synthetic
    
    output_name = f"id_{new_id}_daily_features.csv"
    generate_amenorrhea_data(file_path, AUG_DIR / output_name, new_id)

print(f"Done! Created {len(all_files)} high-risk synthetic participants in {AUG_DIR}")