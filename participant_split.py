import pandas as pd
import numpy as np
import random

INPUT_CSV = "/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/mcphases-a-dataset-of-physiological-hormonal-and-self-reported-events-and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0/hormones_and_selfreport.csv"
OUTPUT_NOTES_CSV = "participant_risk_notes.csv"
OUTPUT_INTERVAL_CSV = "interval_risk_labels.csv"        # NEW: per-interval labels
OUTPUT_SPLITS_TXT = "evaluation_splits_readable_v3.txt"

BLEEDING_MAP = {
    "Not at all": 0,
    "Spotting / Very Light": 1,
    "Light": 1,
    "Somewhat Light": 1,
    "Moderate": 1,
    "Somewhat Heavy": 1,
    "Heavy": 1,
    "Very Heavy": 1,
}

NORMAL_CYCLE_MIN = 21
NORMAL_CYCLE_MAX = 45
MERGE_START_GAP_DAYS = 3
MIN_PDG_SAMPLES_FOR_HORMONE_CALL = 3
PDG_HIGH_THRESHOLD = 5.0
PDG_SUPPORT_THRESHOLD = 3.0

N_SPLITS = 3
TEST_FRACTION = 0.30
RANDOM_SEED = 42


def find_bleed_episodes(sub, max_gap_days=2):
    bleed_days = sub.loc[sub["bleeding"] == 1, "day_in_study"].sort_values().to_numpy()
    if len(bleed_days) == 0:
        return []

    episodes = [[int(bleed_days[0]), int(bleed_days[0])]]
    for day in bleed_days[1:]:
        day = int(day)
        if day - episodes[-1][1] <= max_gap_days + 1:
            episodes[-1][1] = day
        else:
            episodes.append([day, day])
    return episodes


def merge_starts(days, max_gap_days=3):
    days = sorted(int(d) for d in days)
    if not days:
        return []

    merged = [days[0]]
    for day in days[1:]:
        if day - merged[-1] > max_gap_days:
            merged.append(day)
    return merged


def classify_hormone_pattern(sub):
    pdg = sub["pdg"].dropna()
    pdg_count = int(len(pdg))

    if pdg_count < MIN_PDG_SAMPLES_FOR_HORMONE_CALL:
        return "unknown", pdg_count

    if (pdg >= PDG_HIGH_THRESHOLD).sum() >= 1 or (pdg >= PDG_SUPPORT_THRESHOLD).sum() >= 2:
        return "ovulatory", pdg_count

    if pdg.max() < PDG_SUPPORT_THRESHOLD:
        return "suppressed", pdg_count

    return "unclear", pdg_count


def classify_interval(sub):
    sub = sub.sort_values("day_in_study").copy()

    sub["bleeding"] = sub["flow_volume"].map(BLEEDING_MAP)
    sub.loc[(sub["phase"] == "Menstrual") & (sub["bleeding"].isna()), "bleeding"] = 1
    sub["bleeding"] = sub["bleeding"].fillna(0).astype(int)

    observed_days = int(sub["day_in_study"].max() - sub["day_in_study"].min() + 1)

    bleed_episode_starts = [e[0] for e in find_bleed_episodes(sub)]
    menstrual_phase_starts = sub.loc[
        (sub["phase"] == "Menstrual") &
        (sub["phase"].shift(fill_value="") != "Menstrual"),
        "day_in_study"
    ].tolist()

    episode_starts = merge_starts(
        bleed_episode_starts + menstrual_phase_starts,
        max_gap_days=MERGE_START_GAP_DAYS
    )

    cycle_lengths = np.diff(episode_starts) if len(episode_starts) >= 2 else np.array([])

    phases_present = set(sub["phase"].dropna())
    has_full_phase_progression = all(
        p in phases_present for p in ["Menstrual", "Follicular", "Fertility", "Luteal"]
    )
    has_ovulatory_phase_pattern = all(
        p in phases_present for p in ["Fertility", "Luteal"]
    )

    hormone_support, pdg_count = classify_hormone_pattern(sub)

    high_points = 0
    low_points = 0
    short_reasons = []

    if observed_days >= 60:
        if len(episode_starts) <= 1:
            high_points += 2
            short_reasons.append("very few cycles")
        elif len(episode_starts) == 2:
            high_points += 1
            short_reasons.append("2 cycles")
        else:
            low_points += 1
            short_reasons.append("regular bleeding count")

    if len(cycle_lengths) >= 1:
        in_range_fraction = np.mean(
            (cycle_lengths >= NORMAL_CYCLE_MIN) &
            (cycle_lengths <= NORMAL_CYCLE_MAX)
        )
        if in_range_fraction >= 0.67:
            low_points += 2
            short_reasons.append("cycle lengths normal")
        elif (cycle_lengths > NORMAL_CYCLE_MAX).any():
            high_points += 1
            short_reasons.append("long cycle")
    else:
        if observed_days >= 45:
            high_points += 1
            short_reasons.append("unclear cycle pattern")

    if has_full_phase_progression:
        low_points += 1
        short_reasons.append("full phase progression")
    elif not has_ovulatory_phase_pattern:
        high_points += 1
        short_reasons.append("missing ovulatory phases")

    if hormone_support == "ovulatory":
        low_points += 1
        short_reasons.append("ovulatory PdG")
    elif hormone_support == "suppressed":
        high_points += 1
        short_reasons.append("low PdG")
    elif hormone_support == "unknown":
        short_reasons.append("PdG unavailable")

    if high_points >= 3 and high_points > low_points:
        risk = "high"
    elif low_points >= 3 and high_points <= 1:
        risk = "low"
    else:
        risk = "medium"

    return {
        "id": sub["id"].iloc[0],
        "study_interval": sub["study_interval"].iloc[0],
        "interval_risk": risk,
        "label": 0 if risk == "low" else 1,   # binary: low=0, medium/high=1
        "reason_note": ", ".join(short_reasons),
    }


def aggregate_subject_notes(interval_df):
    rows = []
    for participant_id, sub in interval_df.groupby("id", sort=True):
        interval_risks = sub["interval_risk"].tolist()

        if "high" in interval_risks:
            final_risk = "high"
        elif all(r == "low" for r in interval_risks):
            final_risk = "low"
        else:
            final_risk = "medium"

        interval_notes = [
            f"{int(row.study_interval)}: {row.interval_risk} ({row.reason_note})"
            for row in sub.itertuples(index=False)
        ]

        rows.append({
            "participant_id": participant_id,
            "risk_level": final_risk,
            "reason_note": " | ".join(interval_notes)
        })

    return pd.DataFrame(rows).sort_values("participant_id")


def count_risk_levels(df_subset, risk_col="risk_level"):
    return {
        "low":    int((df_subset[risk_col] == "low").sum()),
        "medium": int((df_subset[risk_col] == "medium").sum()),
        "high":   int((df_subset[risk_col] == "high").sum()),
    }


def write_readable_splits(subject_df, interval_df, filename):
    """
    Splits are defined at the participant level (to avoid data leakage),
    but the output now also shows how many intervals fall in train/test.
    """
    rng = random.Random(RANDOM_SEED)

    low_ids     = sorted(subject_df.loc[subject_df["risk_level"] == "low", "participant_id"].tolist())
    at_risk_ids = sorted(
        subject_df.loc[subject_df["risk_level"].isin(["medium", "high"]), "participant_id"].tolist()
    )
    all_ids = sorted(subject_df["participant_id"].tolist())

    with open(filename, "w") as f:
        for split in range(1, N_SPLITS + 1):
            test_ids = []

            for group_ids in [low_ids, at_risk_ids]:
                ids_copy = group_ids[:]
                rng.shuffle(ids_copy)

                if len(ids_copy) <= 1:
                    n_test = len(ids_copy)
                else:
                    n_test = max(1, int(len(ids_copy) * TEST_FRACTION))

                test_ids.extend(ids_copy[:n_test])

            test_ids  = sorted(set(test_ids))
            train_ids = sorted([x for x in all_ids if x not in test_ids])

            # Subject-level counts
            train_subj = subject_df[subject_df["participant_id"].isin(train_ids)]
            test_subj  = subject_df[subject_df["participant_id"].isin(test_ids)]

            # Interval-level counts
            train_intervals = interval_df[interval_df["id"].isin(train_ids)]
            test_intervals  = interval_df[interval_df["id"].isin(test_ids)]

            f.write(f"=== SPLIT {split} ===\n")

            f.write("TRAIN:\n")
            f.write(f"  n_participants: {len(train_ids)}\n")
            f.write(f"  n_intervals:    {len(train_intervals)}\n")
            subj_c = count_risk_levels(train_subj)
            intv_c = count_risk_levels(train_intervals, risk_col="interval_risk")
            f.write(f"  participant distribution: low={subj_c['low']}, medium={subj_c['medium']}, high={subj_c['high']}\n")
            f.write(f"  interval distribution:    low={intv_c['low']}, medium={intv_c['medium']}, high={intv_c['high']}\n")
            f.write(f"  ids: {', '.join(map(str, train_ids))}\n\n")

            f.write("TEST:\n")
            f.write(f"  n_participants: {len(test_ids)}\n")
            f.write(f"  n_intervals:    {len(test_intervals)}\n")
            subj_c = count_risk_levels(test_subj)
            intv_c = count_risk_levels(test_intervals, risk_col="interval_risk")
            f.write(f"  participant distribution: low={subj_c['low']}, medium={subj_c['medium']}, high={subj_c['high']}\n")
            f.write(f"  interval distribution:    low={intv_c['low']}, medium={intv_c['medium']}, high={intv_c['high']}\n")
            f.write(f"  ids: {', '.join(map(str, test_ids))}\n\n")


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df.sort_values(["id", "study_interval", "day_in_study"])

    # Classify each (participant, interval) pair
    interval_results = []
    for (_, _), sub in df.groupby(["id", "study_interval"]):
        interval_results.append(classify_interval(sub))

    interval_df     = pd.DataFrame(interval_results).sort_values(["id", "study_interval"])
    subject_notes_df = aggregate_subject_notes(interval_df)

    # Save per-participant notes (unchanged format)
    subject_notes_df.to_csv(OUTPUT_NOTES_CSV, index=False)

    # Save per-interval labels (NEW)
    interval_df.to_csv(OUTPUT_INTERVAL_CSV, index=False)

    # Save splits (now includes interval counts)
    write_readable_splits(subject_notes_df, interval_df, OUTPUT_SPLITS_TXT)

    print("Saved:")
    print(f"  {OUTPUT_NOTES_CSV}  — per-participant summary (unchanged)")
    print(f"  {OUTPUT_INTERVAL_CSV}  — per-interval labels for Bayesian model")
    print(f"  {OUTPUT_SPLITS_TXT}  — evaluation splits with interval counts")

    print("\nInterval risk distribution:")
    print(interval_df["interval_risk"].value_counts().to_string())


if __name__ == "__main__":
    main()