import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
RESULTS_DIR = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model/bayesian_risk_results/split2")
# Pick a participant to visualize (e.g., an augmented one to see the trend)
PARTICIPANT_ID = "id_10_study_interval_2022_daily_features.csv" 

def plot_participant_risk(file_path):
    df = pd.read_csv(file_path)
    
    # Ensure days are sorted
    df = df.sort_values('day_in_study')
    
    plt.figure(figsize=(12, 6))
    
    # 1. Plot the Risk Probability Line
    plt.plot(df['day_in_study'], df['risk_prob'], color='black', linewidth=2, label='Risk Probability $P(Risk|x_k)$')
    
    # 2. Add the Bayesian Threshold Zones
    plt.axhspan(0, 0.5, color='green', alpha=0.2, label='Green (Low Risk)')
    plt.axhspan(0.5, 1.0, color='red', alpha=0.2, label='Red (High Risk / FHA)')
    
    # 3. Add Threshold Lines
    plt.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    
    # Formatting
    plt.title(f"Physiological Risk Progression: Participant {file_path.name.split('_')[1]}", fontsize=14)
    plt.xlabel("Day in Study", fontsize=12)
    plt.ylabel("Probability of Amenorrhea Risk", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(loc='upper left')
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"risk_trend_{file_path.name.split('_')[1]}.png")
    plt.show()

# Run the plot
plot_participant_risk(RESULTS_DIR / PARTICIPANT_ID)