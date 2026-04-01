#!/usr/bin/env python3
"""
Create one daily feature CSV per participant ID and study interval from mcPHASES tables.

Outputs one CSV per (ID, study_interval) with one row per day containing:
1. RMSSD averaged across available HRV entries for that day
2. Resting heart rate (RHR)
3. Normalized nightly temperature deviation:
      baseline_relative_sample_sum / temperature_samples
4. Daily total distance
5. Daily activity intensity:
      zone1 + 2*zone2 + 3*zone3

Assumptions:
- Input CSVs are in the same folder.
- Files used:
    - heart_rate_variability_details.csv
    - resting_heart_rate.csv
    - computed_temperature.csv
    - distance.csv
    - time_in_heart_rate_zones.csv
- Daily sleep-derived features are assigned to the day the sleep ENDS
  when available. For HRV details, the provided day_in_study is used.
- Resting heart rate values of 0 are treated as missing.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# Configuration
# =========================
INPUT_DIR = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/mcphases-a-dataset-of-physiological-hormonal-and-self-reported-events-and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0")
OUTPUT_DIR = Path("/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/amenorrhea-detection-model/per_id_daily_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HRV_FILE = INPUT_DIR / "heart_rate_variability_details.csv"
RHR_FILE = INPUT_DIR / "resting_heart_rate.csv"
TEMP_FILE = INPUT_DIR / "computed_temperature.csv"
DIST_FILE = INPUT_DIR / "distance.csv"
HRZ_FILE = INPUT_DIR / "time_in_heart_rate_zones.csv"


# =========================
# Helpers
# =========================
def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =========================
# Load data
# =========================
hrv = pd.read_csv(HRV_FILE)
rhr = pd.read_csv(RHR_FILE)
temp = pd.read_csv(TEMP_FILE)
dist = pd.read_csv(DIST_FILE)
hrz = pd.read_csv(HRZ_FILE)

# =========================
# 1) HRV: RMSSD averaged per day
# =========================
require_columns(
    hrv,
    ["id", "study_interval", "day_in_study", "rmssd"],
    "heart_rate_variability_details.csv"
)
hrv = safe_numeric(hrv, ["id", "study_interval", "day_in_study", "rmssd"])
hrv_daily = (
    hrv.groupby(["id", "study_interval", "day_in_study"], as_index=False)["rmssd"]
    .mean()
    .rename(columns={"rmssd": "rmssd_avg"})
)

# =========================
# 2) RHR: daily resting heart rate
# =========================
require_columns(
    rhr,
    ["id", "study_interval", "day_in_study", "value"],
    "resting_heart_rate.csv"
)
rhr = safe_numeric(rhr, ["id", "study_interval", "day_in_study", "value", "error"])
rhr.loc[rhr["value"] == 0, "value"] = np.nan
rhr_daily = (
    rhr.groupby(["id", "study_interval", "day_in_study"], as_index=False)["value"]
    .mean()
    .rename(columns={"value": "resting_heart_rate"})
)

# =========================
# 3) Temperature: baseline_relative_sample_sum / temperature_samples
#    assign to sleep_end_day_in_study when available
# =========================
require_columns(
    temp,
    [
        "id",
        "study_interval",
        "temperature_samples",
        "baseline_relative_sample_sum",
    ],
    "computed_temperature.csv"
)

temp_day_col = (
    "sleep_end_day_in_study"
    if "sleep_end_day_in_study" in temp.columns
    else "sleep_start_day_in_study"
)
if temp_day_col not in temp.columns:
    raise ValueError(
        "computed_temperature.csv must contain either "
        "'sleep_end_day_in_study' or 'sleep_start_day_in_study'."
    )

temp = safe_numeric(
    temp,
    [
        "id",
        "study_interval",
        temp_day_col,
        "temperature_samples",
        "baseline_relative_sample_sum",
    ],
)
temp["temp_dev_norm"] = (
    temp["baseline_relative_sample_sum"] / temp["temperature_samples"]
)
temp["temp_dev_norm"] = temp["temp_dev_norm"].where(
    abs(temp["temp_dev_norm"]) <= 3.0, np.nan
)
temp_daily = (
    temp.groupby(["id", "study_interval", temp_day_col], as_index=False)["temp_dev_norm"]
    .mean()
    .rename(columns={temp_day_col: "day_in_study"})
)

# =========================
# 4) Distance: total daily distance
# =========================
require_columns(
    dist,
    ["id", "study_interval", "day_in_study", "distance"],
    "distance.csv"
)
dist = safe_numeric(dist, ["id", "study_interval", "day_in_study", "distance"])
dist_daily = (
    dist.groupby(["id", "study_interval", "day_in_study"], as_index=False)["distance"]
    .sum()
    .rename(columns={"distance": "daily_distance"})
)

# =========================
# 5) Activity intensity: Z1 + 2*Z2 + 3*Z3
# =========================
require_columns(
    hrz,
    [
        "id",
        "study_interval",
        "day_in_study",
        "in_default_zone_1",
        "in_default_zone_2",
        "in_default_zone_3",
    ],
    "time_in_heart_rate_zones.csv"
)
hrz = safe_numeric(
    hrz,
    [
        "id",
        "study_interval",
        "day_in_study",
        "in_default_zone_1",
        "in_default_zone_2",
        "in_default_zone_3",
    ],
)
hrz["activity_intensity"] = (
    hrz["in_default_zone_1"]
    + 2 * hrz["in_default_zone_2"]
    + 3 * hrz["in_default_zone_3"]
)
hrz_daily = hrz[
    ["id", "study_interval", "day_in_study", "activity_intensity"]
].copy()

# =========================
# Build daily master index
# =========================
daily_index_parts = [
    hrv_daily[["id", "study_interval", "day_in_study"]],
    rhr_daily[["id", "study_interval", "day_in_study"]],
    temp_daily[["id", "study_interval", "day_in_study"]],
    dist_daily[["id", "study_interval", "day_in_study"]],
    hrz_daily[["id", "study_interval", "day_in_study"]],
]

master = (
    pd.concat(daily_index_parts, ignore_index=True)
    .drop_duplicates()
    .sort_values(["id", "study_interval", "day_in_study"])
    .reset_index(drop=True)
)

# =========================
# Merge all features
# =========================
merged = master.merge(
    hrv_daily, on=["id", "study_interval", "day_in_study"], how="left"
)
merged = merged.merge(
    rhr_daily, on=["id", "study_interval", "day_in_study"], how="left"
)
merged = merged.merge(
    temp_daily, on=["id", "study_interval", "day_in_study"], how="left"
)
merged = merged.merge(
    dist_daily, on=["id", "study_interval", "day_in_study"], how="left"
)
merged = merged.merge(
    hrz_daily, on=["id", "study_interval", "day_in_study"], how="left"
)

merged = merged.sort_values(["id", "study_interval", "day_in_study"]).reset_index(drop=True)

# =========================
# Identify (ID, study_interval) groups with missing entire features
# =========================
features = [
    "rmssd_avg",
    "resting_heart_rate",
    "temp_dev_norm",
    "daily_distance",
    "activity_intensity",
]

groups_to_ignore = set()

for (participant_id, study_interval), df_group in merged.groupby(["id", "study_interval"]):
    for feature in features:
        if df_group[feature].isna().all():
            print(
                f"ID {int(participant_id)}, study_interval {int(study_interval)} "
                f"has NO data for {feature}"
            )
            groups_to_ignore.add((participant_id, study_interval))
            break

# =========================
# Write one CSV per ID per study_interval
# =========================
feature_cols = [
    "study_interval",
    "day_in_study",
    "rmssd_avg",
    "resting_heart_rate",
    "temp_dev_norm",
    "daily_distance",
    "activity_intensity",
]

written_count = 0

for (participant_id, study_interval), df_group in merged.groupby(["id", "study_interval"]):
    if (participant_id, study_interval) in groups_to_ignore:
        continue

    out_path = OUTPUT_DIR / f"id_{int(participant_id)}_study_interval_{int(study_interval)}_daily_features.csv"
    df_out = df_group[["id"] + feature_cols].copy()
    df_out.to_csv(out_path, index=False)
    written_count += 1

print(
    "\nID/study_interval groups ignored: "
    f"{[(int(i), int(s)) for i, s in sorted(groups_to_ignore)]}"
)
print(f"Done. Wrote {written_count} files to: {OUTPUT_DIR.resolve()}")