import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path

# --- CONFIGURATION ---
BASE_PATH = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model")
ORIGINAL_FILTERED = BASE_PATH / "kalman_filtered"
AUGMENTED_FILTERED = BASE_PATH / "kalman_filtered_Risky"
NOTES_FILE = BASE_PATH / "participant_risk_notes.csv"
OUTPUT_DIR = BASE_PATH / "bayesian_risk_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. LOAD LABELS
notes_df = pd.read_csv(NOTES_FILE)
low_risk_ids = notes_df[notes_df['risk_level'] == 'low']['participant_id'].tolist()

# 2. AGGREGATE DATA FOR TRAINING
def get_training_pool():
    all_rows = []
    # Load Real Low Risk
    for pid in low_risk_ids:
        # Match your naming convention (ID and Interval)
        files = list(ORIGINAL_FILTERED.glob(f"id_{pid}_*_daily_features.csv"))
        for f in files:
            df = pd.read_csv(f)
            df['label'] = 0
            all_rows.append(df)

    # Load Augmented High Risk (All files in the augmented folder)
    for f in AUGMENTED_FILTERED.glob("id_*.csv"):
        df = pd.read_csv(f)
        df['label'] = 1
        all_rows.append(df)
        
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

train_df = get_training_pool()

if train_df.empty:
    print("ERROR: Training pool is empty. Check your file paths and IDs.")
else:
    # 3. CALCULATE PARAMETERS WITH EPSILON (Option B)
    FEATURES = ['f_HRV', 'f_RHR', 'f_Temp']
    params = {}
    EPSILON = 1e-4 # Prevents division by zero if std is 0

    print("\n--- Model Training Results ---")
    for feat in FEATURES:
        r_group = train_df[train_df['label'] == 1][feat].dropna()
        n_group = train_df[train_df['label'] == 0][feat].dropna()
        
        params[feat] = {
            'mean_r': r_group.mean(), 'std_r': max(r_group.std(), EPSILON),
            'mean_n': n_group.mean(), 'std_n': max(n_group.std(), EPSILON)
        }
        print(f"{feat}: Normal Mean={params[feat]['mean_n']:.2f}, Risk Mean={params[feat]['mean_r']:.2f}")

    # 4. LOG-LIKELIHOOD INFERENCE (Stabilized)
    def compute_risk_prob_stable(row, p_params, prior=0.3):
        # Using Log-Likelihoods to prevent underflow
        log_p_risk = np.log(prior)
        log_p_norm = np.log(1.0 - prior)
        
        for feat in FEATURES:
            val = row[feat]
            if np.isnan(val): continue
            
            # Add log of the Gaussian PDF
            log_p_risk += norm.logpdf(val, p_params[feat]['mean_r'], p_params[feat]['std_r'])
            log_p_norm += norm.logpdf(val, p_params[feat]['mean_n'], p_params[feat]['std_n'])
        
        # Convert back to probability: P = 1 / (1 + exp(log_norm - log_risk))
        try:
            prob = 1.0 / (1.0 + np.exp(log_p_norm - log_p_risk))
            return prob
        except OverflowError:
            return 0.0 if log_p_norm > log_p_risk else 1.0

    # 5. RUN ON ALL FILES
    all_files = list(ORIGINAL_FILTERED.glob("id_*.csv")) + list(AUGMENTED_FILTERED.glob("id_*.csv"))
    for file_path in all_files:
        df = pd.read_csv(file_path)
        df['risk_prob'] = df.apply(lambda r: compute_risk_prob_stable(r, params), axis=1)
        
        # Handle classification with categorical safety
        df['risk_level'] = 'Green'
        df.loc[df['risk_prob'] >= 0.3, 'risk_level'] = 'Yellow'
        df.loc[df['risk_prob'] >= 0.7, 'risk_level'] = 'Red'
        
        df.to_csv(OUTPUT_DIR / file_path.name, index=False)

    print(f"\nSuccessfully processed {len(all_files)} files into {OUTPUT_DIR}")