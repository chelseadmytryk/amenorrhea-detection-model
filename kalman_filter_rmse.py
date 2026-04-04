import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Path to your successfully filtered path
FILTERED_PATH = Path("C:/Users/lfearn/Documents/546Proj/amenorrhea-detection-model/kalman_filtered/")
AUG_PATH = Path("C:/Users/lfearn/Documents/546Proj/amenorrhea-detection-model/kalman_filtered_Risky/")

# all_files = list(FILTERED_PATH.glob("*"))
# print(f'{len(all_files)}')
files = [f for f in FILTERED_PATH.iterdir() if f.is_file()]
aug_files = [f for f in AUG_PATH.iterdir() if f.is_file()]

file_names = []
rmse_hrv_vals = []
rmse_hr_vals = []
rmse_temp_vals = []

aug_rmse_hrv_vals = []
aug_rmse_hr_vals = []
aug_rmse_temp_vals = []

for f in files:
    df = pd.read_csv(f)
    # print(df)
    # Calculate RMSE for this file
    rmse_hrv = np.sqrt(np.mean((df['rmssd_avg'] - df['f_HRV'])**2))
    rmse_hr = np.sqrt(np.mean((df['resting_heart_rate'] - df['f_RHR'])**2))
    rmse_temp = np.sqrt(np.mean((df['temp_dev_norm'] - df['f_Temp'])**2))

    rmse_hrv_vals.append(rmse_hrv)
    rmse_hr_vals.append(rmse_hr)
    rmse_temp_vals.append(rmse_temp)

    # Split the string at every underscore
    parts = f.name.split('_')

    # Index 1 is the '4', Index 4 is the '2022'
    subject_id = parts[1]
    year = parts[4]
    filename = "ID: " + subject_id + ", Year: " + year
    # print(f"ID: {subject_id}, Year: {year}")
    file_names.append(filename)

for f in aug_files:
    df = pd.read_csv(f)
    # print(df)
    # Calculate RMSE for this file
    rmse_hrv = np.sqrt(np.mean((df['rmssd_avg'] - df['f_HRV'])**2))
    rmse_hr = np.sqrt(np.mean((df['resting_heart_rate'] - df['f_RHR'])**2))
    rmse_temp = np.sqrt(np.mean((df['temp_dev_norm'] - df['f_Temp'])**2))

    aug_rmse_hrv_vals.append(rmse_hrv)
    aug_rmse_hr_vals.append(rmse_hr)
    aug_rmse_temp_vals.append(rmse_temp)

    # Split the string at every underscore
    parts = f.name.split('_')

    # Index 1 is the '4', Index 4 is the '2022'
    subject_id = parts[1]
    filename = "ID: " + subject_id 
    # print(f"ID: {subject_id}, Year: {year}")
    file_names.append(filename)


# #scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(rmse_hrv_vals, file_names, color='firebrick', s=100) 
# plt.scatter(rmse_temp_vals, file_names, color='tab:blue', s=100) 
# plt.scatter(rmse_hr_vals, file_names, color='tab:green', s=100) 
# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.xlabel('RMSE Value')
# plt.ylabel('File Name')
# plt.title('RMSE HRV per File (Dot Plot)')
# plt.tight_layout()

# Create a new list excluding any NaNs
group = [rmse_hrv_vals, rmse_hr_vals, rmse_temp_vals, aug_rmse_hrv_vals,aug_rmse_hr_vals, aug_rmse_temp_vals]
 
cleaned_group = [[x for x in g if np.isfinite(x)] for g in group]

# plt.boxplot([rmse_hrv_vals, rmse_hr_vals, rmse_temp_vals], labels = ['HRV', 'RHR', 'Temp'])
plt.boxplot(cleaned_group, labels = ['HRV', 'RHR', 'Temp', 'Aug HRV', 'Aug RHR', 'Aug Temp'])

for i, data in enumerate(cleaned_group):
    x = np.random.normal(i + 1, 0.04, size=len(data))
    plt.scatter(x, data, alpha=0.2)



all_data = [
    ([cleaned_group[0], cleaned_group[3]], ['HRV', 'Aug HRV'], 'HRV RMSE'),
    ([cleaned_group[1], cleaned_group[4]], ['RHR', 'Aug RHR'], 'RHR RMSE'),
    ([cleaned_group[2], cleaned_group[5]], ['Temp', 'Aug Temp'], 'Temp RMSE')
]

plt.figure(figsize=(18, 6)) 

for idx, (data_group, labels, title) in enumerate(all_data):
    plt.subplot(1, 3, idx + 1) 
    
    plt.boxplot(data_group, labels=labels)
    
    for i, data in enumerate(data_group):
        x = np.random.normal(i + 1, 0.04, size=len(data))
        plt.scatter(x, data, alpha=0.2)
    
    plt.ylabel('RMSE')
    plt.title(title)

plt.tight_layout() #
plt.show()