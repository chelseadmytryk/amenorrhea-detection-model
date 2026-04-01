import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

# =========================
# Likelihood parameters
# These should ideally be estimated from your labeled training data
# =========================
PARAMS = {
    'f_HRV':  {'mean_r': 30,   'std_r': 10,  'mean_n': 55,  'std_n': 15},
    'f_RHR':  {'mean_r': 75,   'std_r': 8,   'mean_n': 60,  'std_n': 8},
    'f_Temp': {'mean_r': -0.3, 'std_r': 0.4, 'mean_n': 0.0, 'std_n': 0.3},
}

# Prior probability of being at risk (base rate in your dataset)
PRIOR_RISK = 0.3  # adjust based on proportion of high-risk participants

def gaussian_likelihood(x, mean, std):
    return norm.pdf(x, loc=mean, scale=std)

def compute_risk_probability(row, prior=PRIOR_RISK):
    """Compute P(Risk | x_k) for a single day's filtered state."""
    
    p_x_given_risk   = 1.0
    p_x_given_normal = 1.0
    
    for feature, p in PARAMS.items():
        val = row[feature]
        if np.isnan(val):
            continue  # skip missing filtered states
        
        p_x_given_risk   *= gaussian_likelihood(val, p['mean_r'], p['std_r'])
        p_x_given_normal *= gaussian_likelihood(val, p['mean_n'], p['std_n'])
    
    # Bayes: P(R|x) = P(x|R)*P(R) / [P(x|R)*P(R) + P(x|~R)*P(~R)]
    numerator   = p_x_given_risk * prior
    denominator = numerator + p_x_given_normal * (1 - prior)
    
    if denominator == 0:
        return np.nan
    
    return numerator / denominator

def classify_risk(prob):
    if np.isnan(prob):
        return 'Unknown'
    elif prob < 0.3:
        return 'Green'
    elif prob < 0.7:
        return 'Yellow'
    else:
        return 'Red'

# =========================
# Run over all filtered files
# =========================
INPUT_DIR  = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model/kalman_filtered")
OUTPUT_DIR = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model/bayesian_risk")
OUTPUT_DIR.mkdir(exist_ok=True)

for file_path in INPUT_DIR.glob("id_*_daily_features.csv"):
    df = pd.read_csv(file_path)
    
    df['risk_prob']  = df.apply(compute_risk_probability, axis=1)
    df['risk_level'] = df['risk_prob'].apply(classify_risk)
    
    # Summarize per-interval risk
    print(f"\n{file_path.name}")
    print(df['risk_level'].value_counts())
    
    df.to_csv(OUTPUT_DIR / file_path.name, index=False)