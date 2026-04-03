#!/usr/bin/env python3
"""
Bayesian Amenorrhea Risk Model — Three-Class Output with Train/Test Splits

Changes from previous version:
- Three-class output: Green (low), Yellow (medium), Red (high)
- Ground truth uses interval_risk_labels.csv with low/medium/high per interval
- Proper train/test splitting using evaluation_splits_readable_v3.txt
- Likelihood parameters estimated from training set only (no data leakage)
- Uniform prior of 0.5 as stated in report
- Evaluation reported per split and averaged across all splits
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Configuration
# =========================
BASE_PATH          = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model")
ORIGINAL_FILTERED  = BASE_PATH / "kalman_filtered"
AUGMENTED_FILTERED = BASE_PATH / "kalman_filtered_Risky"
INTERVAL_LABELS    = BASE_PATH / "interval_risk_labels.csv"
SPLITS_FILE        = BASE_PATH / "evaluation_splits_readable_v3.txt"
OUTPUT_DIR         = BASE_PATH / "bayesian_risk_results"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = ['f_HRV', 'f_RHR', 'f_Temp', 'daily_distance', 'activity_intensity']
PRIOR    = 0.5   # uniform prior as stated in report
EPSILON  = 1e-4  # prevents zero std

# Map ground truth and predictions to consistent class names
RISK_TO_CLASS = {'low': 'Low', 'high': 'High'}
CLASS_ORDER   = ['Low', 'High']


# =========================
# Step 1: Load ground truth labels
# =========================
def load_labels(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    df['risk_class'] = df['interval_risk'].map(RISK_TO_CLASS)
    return df


def get_label_for_file(filename: str, labels_df: pd.DataFrame):
    """Return (binary_label, three_class_label) for a given filename."""
    m = re.search(r'id_(\d+)_study_interval_(\d+)_', filename)
    if not m:
        return None, None
    pid  = int(m.group(1))
    year = int(m.group(2))  # 2022 or 2024

    row = labels_df[
        (labels_df['id'] == pid) &
        (labels_df['study_interval'] == year)
    ]
    if row.empty:
        return None, None

    return int(row.iloc[0]['label']), row.iloc[0]['risk_class']


# =========================
# Step 2: Parse evaluation splits
# =========================
def parse_splits(splits_path: Path) -> list:
    splits = []
    with open(splits_path) as f:
        content = f.read()

    blocks = re.split(r'=== SPLIT \d+ ===', content)[1:]
    for block in blocks:
        train_match = re.search(r'TRAIN:.*?ids:\s*([\d,\s]+)', block, re.DOTALL)
        test_match  = re.search(r'TEST:.*?ids:\s*([\d,\s]+)',  block, re.DOTALL)
        if train_match and test_match:
            splits.append({
                'train': [int(x.strip()) for x in train_match.group(1).split(',') if x.strip()],
                'test':  [int(x.strip()) for x in test_match.group(1).split(',')  if x.strip()]
            })
    return splits


# =========================
# Step 3: Build training data pool for a given split
# =========================
def build_train_pool(train_ids: list, labels_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for pid in train_ids:
        # Check both original and augmented filtered folders
        for folder in [ORIGINAL_FILTERED, AUGMENTED_FILTERED]:
            for f in folder.glob(f"id_{pid}_study_interval_*_daily_features.csv"):
                binary_label, _ = get_label_for_file(f.name, labels_df)
                if binary_label is None:
                    continue
                df = pd.read_csv(f)
                df['label'] = binary_label
                rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

# =========================
# Step 4: Estimate likelihood parameters from training data
# =========================
def estimate_params(train_df: pd.DataFrame) -> dict:
    params = {}
    for feat in FEATURES:
        r_vals = train_df[train_df['label'] == 1][feat].dropna()
        n_vals = train_df[train_df['label'] == 0][feat].dropna()

        params[feat] = {
            'mean_r': r_vals.mean(), 'std_r': max(r_vals.std(), EPSILON),
            'mean_n': n_vals.mean(), 'std_n': max(n_vals.std(), EPSILON)
        }
    return params


# =========================
# Step 5: Bayesian inference (log-likelihood for numerical stability)
# =========================
def compute_risk_prob(row, params: dict, prior: float = PRIOR) -> float:
    log_p_risk = np.log(prior)
    log_p_norm = np.log(1.0 - prior)

    for feat in FEATURES:
        val = row[feat]
        if pd.isna(val):
            continue
        log_p_risk += norm.logpdf(val, params[feat]['mean_r'], params[feat]['std_r'])
        log_p_norm += norm.logpdf(val, params[feat]['mean_n'], params[feat]['std_n'])

    try:
        return 1.0 / (1.0 + np.exp(log_p_norm - log_p_risk))
    except OverflowError:
        return 0.0 if log_p_norm > log_p_risk else 1.0


def classify_two_way(prob: float) -> str:
    return 'High' if prob >= 0.5 else 'Low'

# =========================
# Step 6: Evaluate one split
# =========================
def evaluate_split(split: dict, labels_df: pd.DataFrame, split_num: int) -> dict:
    print(f"\n{'='*55}")
    print(f"SPLIT {split_num}  |  train={len(split['train'])} participants  "
          f"test={len(split['test'])} participants")
    print(f"{'='*55}")

    # Estimate parameters from training data only
    train_df = build_train_pool(split['train'], labels_df)
    if train_df.empty:
        print("  WARNING: empty training pool, skipping split")
        return {}

    params = estimate_params(train_df)
    print("\nLikelihood parameters (from training set only):")
    print(f"  {'Feature':<10} {'mean_risk':>10} {'std_risk':>9} "
          f"{'mean_normal':>12} {'std_normal':>11}")
    for feat, p in params.items():
        print(f"  {feat:<10} {p['mean_r']:>10.3f} {p['std_r']:>9.3f} "
              f"{p['mean_n']:>12.3f} {p['std_n']:>11.3f}")

    # Apply model to test participants only
    y_true_class  = []
    y_pred_class  = []
    y_true_binary = []
    y_pred_binary = []
    y_prob        = []
    file_results  = []

    for pid in split['test']:
        for folder in [ORIGINAL_FILTERED, AUGMENTED_FILTERED]:
            for f in folder.glob(f"id_{pid}_study_interval_*_daily_features.csv"):
                binary_label, true_class = get_label_for_file(f.name, labels_df)
                if binary_label is None:
                    continue

                df = pd.read_csv(f)
                df['risk_prob']  = df.apply(lambda r: compute_risk_prob(r, params), axis=1)
                df['risk_level'] = df['risk_prob'].apply(classify_two_way)

                mean_prob   = df['risk_prob'].mean()
                pred_class  = classify_two_way(mean_prob)
                pred_binary = 0 if pred_class == 'Green' else 1

                y_true_class.append(true_class)
                y_pred_class.append(pred_class)
                y_true_binary.append(binary_label)
                y_pred_binary.append(pred_binary)
                y_prob.append(mean_prob)
                file_results.append((f.name, true_class, pred_class, mean_prob))

                # Save output file organised by split
                split_output = OUTPUT_DIR / f"split{split_num}"
                split_output.mkdir(exist_ok=True)
                df.to_csv(split_output / f.name, index=False)

    if not y_true_class:
        print("  WARNING: no test results, skipping")
        return {}

    # Print per-interval results table
    print("\nPer-interval predictions:")
    print(f"  {'':4} {'File':<48} {'True':>8} {'Pred':>8} {'Prob':>6}")
    for fname, tc, pc, prob in file_results:
        match = 'OK  ' if tc == pc else 'MISS'
        print(f"  {match} {fname:<48} {tc:>8} {pc:>8} {prob:>6.3f}")

    # Three-class classification report
    print("\nClassification report (3-class):")
    print(classification_report(
        y_true_class, y_pred_class,
        labels=CLASS_ORDER,
        target_names=CLASS_ORDER,
        zero_division=0
    ))

    # Binary classification report
    print("Classification report (binary — low vs at-risk):")
    print(classification_report(
        y_true_binary, y_pred_binary,
        labels=[0, 1],
        target_names=['low', 'at-risk'],
        zero_division=0
    ))

    return {
        'y_true_class':  y_true_class,
        'y_pred_class':  y_pred_class,
        'y_true_binary': y_true_binary,
        'y_pred_binary': y_pred_binary,
        'y_prob':        y_prob,
        'file_results':  file_results
    }


# =========================
# Main
# =========================
if __name__ == "__main__":
    print("=== Bayesian Amenorrhea Risk Model — Three-Class Evaluation ===")

    labels_df = load_labels(INTERVAL_LABELS)
    print(f"\nLoaded {len(labels_df)} interval labels")
    print(f"Distribution: {labels_df['interval_risk'].value_counts().to_dict()}")

    splits = parse_splits(SPLITS_FILE)
    print(f"Loaded {len(splits)} evaluation splits")

    all_results = []
    for i, split in enumerate(splits, 1):
        result = evaluate_split(split, labels_df, i)
        if result:
            all_results.append(result)

    # Aggregate across all splits
    if all_results:
        print(f"\n{'='*55}")
        print("AGGREGATED RESULTS ACROSS ALL SPLITS")
        print(f"{'='*55}")

        all_true_class  = [y for r in all_results for y in r['y_true_class']]
        all_pred_class  = [y for r in all_results for y in r['y_pred_class']]
        all_true_binary = [y for r in all_results for y in r['y_true_binary']]
        all_pred_binary = [y for r in all_results for y in r['y_pred_binary']]

        print("\nAggregated classification report (3-class):")
        print(classification_report(
            all_true_class, all_pred_class,
            labels=CLASS_ORDER,
            target_names=CLASS_ORDER,
            zero_division=0
        ))

        print("Aggregated classification report (binary):")
        print(classification_report(
            all_true_binary, all_pred_binary,
            labels=[0, 1],
            target_names=['low', 'at-risk'],
            zero_division=0
        ))

        print("\nAggregated confusion matrix (3-class):")
        cm = confusion_matrix(all_true_class, all_pred_class, labels=CLASS_ORDER)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{c}"  for c in CLASS_ORDER],
            columns=[f"pred_{c}" for c in CLASS_ORDER]
        )
        print(cm_df.to_string())

        # Save aggregated summary CSV
        summary_rows = []
        for r in all_results:
            for tc, pc, tb, pb, prob in zip(
                r['y_true_class'], r['y_pred_class'],
                r['y_true_binary'], r['y_pred_binary'],
                r['y_prob']
            ):
                summary_rows.append({
                    'true_class':    tc,
                    'pred_class':    pc,
                    'true_binary':   tb,
                    'pred_binary':   pb,
                    'mean_risk_prob': round(prob, 3),
                    'correct':       tc == pc
                })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(OUTPUT_DIR / "evaluation_summary.csv", index=False)
        print(f"\nEvaluation summary saved to: {OUTPUT_DIR / 'evaluation_summary.csv'}")

    print(f"\nDone! All outputs saved to: {OUTPUT_DIR}")