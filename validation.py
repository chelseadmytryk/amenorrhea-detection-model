import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("bayesian_risk_results")
GT_FILE = Path("participant_risk_notes.csv")

# Load Ground Truth
gt = pd.read_csv(GT_FILE)

validation_data = []

for file in RESULTS_DIR.glob("id_*.csv"):
    df = pd.read_csv(file)
    pid = int(file.name.split('_')[1])
    
    # Calculate percentages
    total_days = len(df)
    red_pct = (df['risk_level'] == 'Red').sum() / total_days
    
    # Model's Final Call: If >20% of days are Red, call it High Risk
    pred_label = 'high' if red_pct > 0.20 else 'low'
    
    # Get actual label from GT
    if pid >= 200:
        actual_label = 'high' # All augmented data is high risk by design
    else:
        # Look up the real clinical label
        actual_label_match = gt.loc[gt['participant_id'] == pid, 'risk_level'].values
        actual_label = actual_label_match[0] if len(actual_label_match) > 0 else 'unknown'
    
    validation_data.append({
        'ID': pid,
        'Actual': actual_label,
        'Predicted': pred_label,
        'Red_Days_Pct': red_pct
    })

val_df = pd.DataFrame(validation_data)
BASE_PATH = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model")
VALIDATION_OUTPUT = BASE_PATH / "model_performance_summary.csv"
val_df.to_csv(VALIDATION_OUTPUT, index=False)

print(f"Validation summary saved to: {VALIDATION_OUTPUT}")

# Print a quick snapshot of the accuracy to the console
accuracy = (val_df['Actual'] == val_df['Predicted']).mean()
print(f"Overall Model Accuracy: {accuracy:.2%}")