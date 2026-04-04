import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to your successfully filtered ID 6 file
# FILTERED_FILE = Path("C:/Users/lfearn/Documents/546Proj/amenorrhea-detection-model/kalman_filtered/id_48_study_interval_2022_daily_features.csv")
FILTERED_FILE = Path("C:/Users/lfearn/Documents/546Proj/amenorrhea-detection-model/kalman_filtered_Risky/id_250_daily_features.csv")




#40 2022

df = pd.read_csv(FILTERED_FILE)

fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# 1. HRV Verification
ax[0].scatter(df['day_in_study'], df['rmssd_avg'], color='gray', alpha=0.3, label='Raw HRV (zk)')
ax[0].plot(df['day_in_study'], df['f_HRV'], color='blue', linewidth=2, label='Filtered HRV (xk)')
ax[0].set_ylabel('HRV (ms)')
ax[0].legend()

# 2. RHR Verification
ax[1].scatter(df['day_in_study'], df['resting_heart_rate'], color='gray', alpha=0.3, label='Raw RHR (zk)')
ax[1].plot(df['day_in_study'], df['f_RHR'], color='red', linewidth=2, label='Filtered RHR (xk)')
ax[1].set_ylabel('RHR (bpm)')
ax[1].legend()

# 3. Temperature Verification
ax[2].scatter(df['day_in_study'], df['temp_dev_norm'], color='gray', alpha=0.3, label='Raw Temp Dev (zk)')
ax[2].plot(df['day_in_study'], df['f_Temp'], color='orange', linewidth=2, label='Filtered Temp (xk)')
ax[2].set_ylabel('Temp Dev (°C)')
ax[2].set_xlabel('Day in Study')
ax[2].legend()

plt.suptitle('Kalman Filter State Verification: Participant ID 250')
plt.tight_layout()
plt.show()