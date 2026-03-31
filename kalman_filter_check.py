import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Use ONLY the filtered directory since it contains both raw and filtered columns
BASE_PATH = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project")
FILTERED_DIR = BASE_PATH / "amenorrhea-detection-model" / "kalman_filtered"

# Load ID 6
df_verify = pd.read_csv(FILTERED_DIR / "id_6_daily_features.csv")

plt.figure(figsize=(12, 6))
df_verify = df_verify.dropna(subset=['rmssd_avg'])
# Plotting using the column names defined in your previous scripts
plt.scatter(df_verify['day_in_study'], df_verify['rmssd_avg'], 
            color='gray', alpha=0.4, label='Raw RMSSD (zk)') #[cite: 72, 73]
plt.plot(df_verify['day_in_study'], df_verify['f_HRV'], 
         color='blue', linewidth=2, label='Filtered HRV (xk)') #[cite: 103, 104]

plt.title("Kalman Filter Verification: Participant ID 6")
plt.xlabel("Day in Study")
plt.ylabel("HRV (ms)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()