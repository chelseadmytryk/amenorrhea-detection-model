#!/usr/bin/env python3
"""
Bayesian Amenorrhea Risk Model

Loads pre-computed per-interval ground truth labels from interval_risk_labels.csv,
fits per-class Gaussian likelihoods from the filtered Kalman features,
then applies Bayes' theorem to produce daily risk probabilities and
classifies each day into Green / Yellow / Red.

Label mapping:
    low    -> 0
    medium -> 1
    high   -> 1
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm


# =========================
# Configuration
# =========================
KALMAN_DIR      = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model/kalman_filtered")
INTERVAL_LABELS = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model/interval_risk_labels.csv")
OUTPUT_DIR      = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model/bayesian_risk")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = ['f_HRV', 'f_RHR', 'f_Temp']

# Risk thresholds (from report)
THRESH_GREEN  = 0.3
THRESH_YELLOW = 0.7


# =========================
# Step 1: Load per-interval ground truth labels
# =========================
def load_interval_labels(labels_path: Path) -> pd.DataFrame:
    """
    Load the pre-computed interval_risk_labels.csv.
    Expected columns: id, study_interval, interval_risk, label, reason_note
    """
    df = pd.read_csv(labels_path)

    # Ensure binary label column exists (0=low, 1=medium/high)
    if 'label' not in df.columns:
        df['label'] = (df['interval_risk'] != 'low').astype(int)

    print(f"Loaded {len(df)} interval labels")
    print(f"  At-risk (medium/high): {df['label'].sum()}")
    print(f"  Low-risk:              {(df['label'] == 0).sum()}")
    print(f"\nInterval risk distribution:")
    print(df['interval_risk'].value_counts().to_string())

    return df


def get_label_for_file(filename: str, labels_df: pd.DataFrame):
    """
    Match a Kalman output filename to its ground truth label.
    Filenames: id_{pid}_study_interval_{year}_daily_features.csv
    The dataset uses study_interval=1 for 2022, study_interval=2 for 2024.
    """
    match = re.search(r'id_(\d+)_study_interval_(\d+)_', filename)
    if not match:
        return None

    pid           = int(match.group(1))
    interval_year = int(match.group(2))  # 2022 or 2024

    row = labels_df[
        (labels_df['id'] == pid) &
        (labels_df['study_interval'] == interval_year)
    ]

    if row.empty:
        return None

    return int(row.iloc[0]['label'])


# =========================
# Step 2: Load all filtered data and attach labels
# =========================
def load_labeled_data(kalman_dir: Path, labels_df: pd.DataFrame):
    """Load all Kalman-filtered CSVs and attach binary ground truth labels."""
    all_dfs = []
    skipped = []

    for f in sorted(kalman_dir.glob("id_*_daily_features.csv")):
        label = get_label_for_file(f.name, labels_df)
        if label is None:
            skipped.append(f.name)
            continue

        df = pd.read_csv(f)
        df['label']       = label
        df['source_file'] = f.name
        all_dfs.append(df)

    if skipped:
        print(f"\n  Warning: no label found for {len(skipped)} file(s), skipping:")
        for s in skipped:
            print(f"    {s}")

    if not all_dfs:
        raise ValueError("No labeled data found. Check file paths and interval_risk_labels.csv.")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nLoaded {len(all_dfs)} files — {len(combined)} total daily rows")
    print(f"Label distribution: {combined['label'].value_counts().to_dict()}")
    return combined


# =========================
# Step 3: Estimate Gaussian likelihood parameters from labeled data
# =========================
def estimate_likelihood_params(combined_df: pd.DataFrame, features: list) -> dict:
    """
    Fit mean and std of each feature under risk=0 and risk=1
    using only rows where the filtered state is available.
    """
    params    = {}
    risk_df   = combined_df[combined_df['label'] == 1]
    normal_df = combined_df[combined_df['label'] == 0]

    print(f"\n{'Feature':<10} {'mean_risk':>10} {'std_risk':>10} {'mean_normal':>12} {'std_normal':>11}")
    print("-" * 55)

    for feat in features:
        r_vals = risk_df[feat].dropna()
        n_vals = normal_df[feat].dropna()

        mean_r, std_r = r_vals.mean(), r_vals.std()
        mean_n, std_n = n_vals.mean(), n_vals.std()

        # Guard against zero/tiny std (e.g. very few samples)
        std_r = max(std_r, 0.5)
        std_n = max(std_n, 0.5)

        params[feat] = {
            'mean_r': mean_r, 'std_r': std_r,
            'mean_n': mean_n, 'std_n': std_n
        }
        print(f"{feat:<10} {mean_r:>10.3f} {std_r:>10.3f} {mean_n:>12.3f} {std_n:>11.3f}")

    return params


# =========================
# Step 4: Bayesian inference
# =========================
def compute_risk_probability(row, params: dict, prior_risk: float) -> float:
    """
    Compute P(Risk=1 | x_k) using Bayes' theorem with Gaussian likelihoods.
    Assumes conditional independence across features.
    """
    p_x_given_risk   = 1.0
    p_x_given_normal = 1.0

    for feat, p in params.items():
        val = row[feat]
        if pd.isna(val):
            continue  # skip missing filtered states

        p_x_given_risk   *= norm.pdf(val, loc=p['mean_r'], scale=p['std_r'])
        p_x_given_normal *= norm.pdf(val, loc=p['mean_n'], scale=p['std_n'])

    numerator   = p_x_given_risk * prior_risk
    denominator = numerator + p_x_given_normal * (1 - prior_risk)

    if denominator == 0:
        return np.nan

    return numerator / denominator


def classify_risk(prob: float) -> str:
    if pd.isna(prob):
        return 'Unknown'
    elif prob < THRESH_GREEN:
        return 'Green'
    elif prob < THRESH_YELLOW:
        return 'Yellow'
    else:
        return 'Red'


# =========================
# Step 5: Apply model and save outputs
# =========================
def apply_bayesian_model(kalman_dir: Path, output_dir: Path,
                         params: dict, prior_risk: float,
                         labels_df: pd.DataFrame):
    """Apply the fitted Bayesian model to each participant interval file."""
    summary_rows = []

    for f in sorted(kalman_dir.glob("id_*_daily_features.csv")):
        label = get_label_for_file(f.name, labels_df)
        df    = pd.read_csv(f)

        df['risk_prob']  = df.apply(
            lambda row: compute_risk_probability(row, params, prior_risk), axis=1
        )

        df['risk_prob_pct'] = (df.apply(
            lambda row: compute_risk_probability(row, params, prior_risk), axis=1
        ) * 100).round(1)

        # df['risk_level'] = df['risk_prob'].apply(classify_risk)

        mean_prob      = df['risk_prob_pct'].mean()
        # dominant_level = df['risk_level'].value_counts().idxmax()
        # counts         = df['risk_level'].value_counts().to_dict()

        summary_rows.append({
            'file':           f.name,
            'true_label':     label,
            'mean_risk_prob': round(mean_prob , 1),  # as percentage
        })

        # summary_rows.append({
        #     'file':           f.name,
        #     'true_label':     label,
        #     'mean_risk_prob': round(mean_prob, 3),
        #     'dominant_level': dominant_level,
        #     'Green_days':     counts.get('Green',   0),
        #     'Yellow_days':    counts.get('Yellow',  0),
        #     'Red_days':       counts.get('Red',     0),
        #     'Unknown_days':   counts.get('Unknown', 0),
        # })

        df.to_csv(output_dir / f.name, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "risk_summary.csv", index=False)

    print("\n=== Per-Interval Risk Summary ===")
    print(summary_df.to_string(index=False))

    return summary_df


# =========================
# Main
# =========================
if __name__ == "__main__":
    print("=== Bayesian Amenorrhea Risk Model ===\n")

    # Load pre-computed interval labels
    print("Step 1: Loading interval labels...")
    labels_df  = load_interval_labels(INTERVAL_LABELS)
    prior_risk = labels_df['label'].mean()
    print(f"\n  Prior P(Risk): {prior_risk:.3f}")

    # Load labeled Kalman data
    print("\nStep 2: Loading Kalman-filtered data...")
    combined_df = load_labeled_data(KALMAN_DIR, labels_df)

    # Estimate likelihood parameters
    print("\nStep 3: Estimating likelihood parameters...")
    params = estimate_likelihood_params(combined_df, FEATURES)

    # Apply model and save
    print("\nStep 4: Applying Bayesian model...")
    summary = apply_bayesian_model(
        KALMAN_DIR, OUTPUT_DIR, params, prior_risk, labels_df
    )

    print(f"\nDone! Outputs saved to: {OUTPUT_DIR}")
    print(f"Risk summary: {OUTPUT_DIR / 'risk_summary.csv'}")