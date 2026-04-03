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
def get_ground_truth(filename: str):
    """
    Directly determines label based on Participant ID.
    IDs >= 200: High Risk (1)
    IDs < 200: Low Risk (0)
    """
    m = re.search(r'id_(\d+)_', filename)
    if not m:
        return None, None
    
    pid = int(m.group(1))
    
    if pid >= 200:
        return 1, 'High'
    else:
        return 0, 'Low'

# =========================
# Step 2: Parse evaluation splits
# =========================
def parse_splits(splits_path: Path) -> list:
    splits = []
    with open(splits_path) as f:
        content = f.read()

    blocks = re.split(r'=== SPLIT \d+ ===', content)[1:]
    for block in blocks:
        train_match = re.search(r'TRAIN:.*?all_ids_with_augmented:\s*([\d,\s]+)', block, re.DOTALL)
        test_match  = re.search(r'TEST:.*?all_ids_with_augmented:\s*([\d,\s]+)',  block, re.DOTALL)
        if train_match and test_match:
            splits.append({
                'train': [int(x.strip()) for x in train_match.group(1).split(',') if x.strip()],
                'test':  [int(x.strip()) for x in test_match.group(1).split(',')  if x.strip()]
            })
    return splits


# =========================
# Step 3: Build training data pool for a given split
# =========================
def build_train_pool(train_ids: list) -> pd.DataFrame:
    rows = []
    for pid in train_ids:
        # Search both original and augmented folders
        for folder in [ORIGINAL_FILTERED, AUGMENTED_FILTERED]:
            for f in folder.glob(f"id_{pid}_*.csv"):
                # LOGIC: Include if it's augmented (>=200) OR if it's the 2022 interval
                if pid >= 200 or "study_interval_2022" in f.name:
                    binary_label, _ = get_ground_truth(f.name)
                    df = pd.read_csv(f)
                    df['label'] = binary_label
                    rows.append(df)
                    
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

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
        # mean -> is mu; and 
        log_p_risk += norm.logpdf(val, params[feat]['mean_r'], params[feat]['std_r']) # Calculating Gaussian Likelihood for each feature, uses log-probabilities to avoid num overflow
        log_p_norm += norm.logpdf(val, params[feat]['mean_n'], params[feat]['std_n'])

    try:
        # This is the BAYES Theorem: Posterior P(Risk|x_k)
        return 1.0 / (1.0 + np.exp(log_p_norm - log_p_risk))
    except OverflowError:
        return 0.0 if log_p_norm > log_p_risk else 1.0


def classify_two_way(prob: float) -> str:
    return 'High' if prob >= 0.5 else 'Low'

# =========================
# Step 6: Evaluate one split
# =========================
def evaluate_split(split: dict, split_num: int) -> dict:
    print(f"\n{'='*55}")
    print(f"SPLIT {split_num}  |  train={len(split['train'])} participants  "
          f"test={len(split['test'])} participants")
    print(f"{'='*55}")

    # Estimate parameters from training data only
    train_df = build_train_pool(split['train'])
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
            for f in folder.glob(f"id_{pid}_*.csv"):
                
                # THE 2022 FILTER:
                # Skip real participants' 2024 data
                if pid < 200 and "study_interval_2024" in f.name:
                    continue

                binary_label, true_class = get_ground_truth(f.name)
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
    print("=== Bayesian Amenorrhea Risk Model — ID-Based Evaluation ===")
    print("Logic: IDs >= 200 are 'High Risk' (Augmented), IDs < 200 are 'Low Risk' (Original)")

    # 1. Load your evaluation splits (which contain the ID lists)
    splits = parse_splits(SPLITS_FILE)
    print(f"Loaded {len(splits)} evaluation splits from {SPLITS_FILE.name}")

    all_results = []
    
    # 2. Iterate through splits
    for i, split in enumerate(splits, 1):
        # We no longer pass labels_df here!
        result = evaluate_split(split, i) 
        if result:
            all_results.append(result)

    # 3. Aggregate results (keep the rest of your aggregation logic the same)
    if all_results:
        print(f"\n{'='*55}")
        print("FINAL AGGREGATED RESULTS (ID-BASED GROUND TRUTH)")
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