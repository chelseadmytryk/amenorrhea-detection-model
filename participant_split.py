import pandas as pd
import numpy as np
import random
from pathlib import Path

INPUT_CSV = "/Users/natalietsang/Documents/DocumentsLocal/4B_ExtraFiles/MTE546/project/mcphases-a-dataset-of-physiological-hormonal-and-self-reported-events-and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0/hormones_and_selfreport.csv"
OUTPUT_NOTES_CSV = "participant_risk_notes.csv"
OUTPUT_SPLITS_TXT = "evaluation_splits_readable_v3.txt"
OUTPUT_SPLIT_MEMBERSHIP_CSV = "split_membership.csv"

# folder with files like id_201_daily_features.csv, id_202_daily_features.csv, etc.
AUGMENTED_DATA_DIR = Path("augmented_data")
SYNTHETIC_OFFSET = 200

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

    # only low or medium
    if low_points >= 3 and high_points <= 1:
        risk = "low"
    else:
        risk = "medium"

    return {
        "id": int(sub["id"].iloc[0]),
        "study_interval": int(sub["study_interval"].iloc[0]),
        "interval_risk": risk,
        "reason_note": ", ".join(short_reasons),
    }


def aggregate_subject_notes(interval_df):
    rows = []
    for participant_id, sub in interval_df.groupby("id", sort=True):
        interval_risks = sub["interval_risk"].tolist()

        if all(r == "low" for r in interval_risks):
            final_risk = "low"
        else:
            final_risk = "medium"

        interval_notes = [
            f"{int(row.study_interval)}: {row.interval_risk} ({row.reason_note})"
            for row in sub.itertuples(index=False)
        ]

        rows.append({
            "participant_id": int(participant_id),
            "risk_level": final_risk,
            "reason_note": " | ".join(interval_notes)
        })

    return pd.DataFrame(rows).sort_values("participant_id")


def count_risk_levels(df_subset):
    return {
        "low": int((df_subset["risk_level"] == "low").sum()),
        "medium": int((df_subset["risk_level"] == "medium").sum()),
    }


def filter_to_2022(df):
    return df[df["study_interval"] == 2022].copy()


def get_augmented_id_map(aug_dir, synthetic_offset=200):
    """
    Returns:
      real_to_synthetic: {real_id: [synthetic_ids]}
      synthetic_ids_found: sorted list of all synthetic ids found
    """
    real_to_synthetic = {}

    if not aug_dir.exists():
        print(f"Warning: augmented data directory not found: {aug_dir.resolve()}")
        return real_to_synthetic, []

    for file_path in aug_dir.glob("id_*_daily_features.csv"):
        try:
            synthetic_id = int(file_path.name.split("_")[1])
            real_id = synthetic_id - synthetic_offset

            if real_id <= 0:
                continue

            real_to_synthetic.setdefault(real_id, []).append(synthetic_id)
        except (IndexError, ValueError):
            print(f"Skipping unrecognized augmented filename: {file_path.name}")

    for real_id in real_to_synthetic:
        real_to_synthetic[real_id] = sorted(real_to_synthetic[real_id])

    synthetic_ids_found = sorted(
        synthetic_id
        for synthetic_list in real_to_synthetic.values()
        for synthetic_id in synthetic_list
    )

    return real_to_synthetic, synthetic_ids_found


def expand_split_with_augmented_ids(real_ids, real_to_synthetic):
    expanded_ids = []
    for real_id in sorted(real_ids):
        expanded_ids.append(real_id)
        expanded_ids.extend(real_to_synthetic.get(real_id, []))
    return sorted(expanded_ids)


def write_readable_splits(subject_df, filename, aug_dir=AUGMENTED_DATA_DIR):
    rng = random.Random(RANDOM_SEED)

    low_ids = sorted(subject_df.loc[subject_df["risk_level"] == "low", "participant_id"].tolist())
    medium_ids = sorted(subject_df.loc[subject_df["risk_level"] == "medium", "participant_id"].tolist())
    all_real_ids = sorted(subject_df["participant_id"].tolist())

    real_to_synthetic, synthetic_ids_found = get_augmented_id_map(aug_dir, SYNTHETIC_OFFSET)

    with open(filename, "w") as f:
        for split in range(1, N_SPLITS + 1):
            test_real_ids = []

            for group_ids in [low_ids, medium_ids]:
                ids_copy = group_ids[:]
                rng.shuffle(ids_copy)

                if len(ids_copy) <= 1:
                    n_test = len(ids_copy)
                else:
                    n_test = max(1, int(len(ids_copy) * TEST_FRACTION))

                test_real_ids.extend(ids_copy[:n_test])

            test_real_ids = sorted(set(test_real_ids))
            train_real_ids = sorted([x for x in all_real_ids if x not in test_real_ids])

            train_df = subject_df[subject_df["participant_id"].isin(train_real_ids)]
            test_df = subject_df[subject_df["participant_id"].isin(test_real_ids)]

            train_counts = count_risk_levels(train_df)
            test_counts = count_risk_levels(test_df)

            train_all_ids = expand_split_with_augmented_ids(train_real_ids, real_to_synthetic)
            test_all_ids = expand_split_with_augmented_ids(test_real_ids, real_to_synthetic)

            train_synth_count = sum(len(real_to_synthetic.get(rid, [])) for rid in train_real_ids)
            test_synth_count = sum(len(real_to_synthetic.get(rid, [])) for rid in test_real_ids)

            f.write(f"=== SPLIT {split} ===\n")

            f.write("TRAIN:\n")
            f.write(f"  n_real_participants: {len(train_real_ids)}\n")
            f.write(f"  n_augmented_participants: {train_synth_count}\n")
            f.write(f"  n_total_with_augmented: {len(train_all_ids)}\n")
            f.write(
                f"  real_distribution: low={train_counts['low']}, "
                f"medium={train_counts['medium']}\n"
            )
            f.write(f"  real_ids: {', '.join(map(str, train_real_ids))}\n")
            f.write(f"  all_ids_with_augmented: {', '.join(map(str, train_all_ids))}\n\n")

            f.write("TEST:\n")
            f.write(f"  n_real_participants: {len(test_real_ids)}\n")
            f.write(f"  n_augmented_participants: {test_synth_count}\n")
            f.write(f"  n_total_with_augmented: {len(test_all_ids)}\n")
            f.write(
                f"  real_distribution: low={test_counts['low']}, "
                f"medium={test_counts['medium']}\n"
            )
            f.write(f"  real_ids: {', '.join(map(str, test_real_ids))}\n")
            f.write(f"  all_ids_with_augmented: {', '.join(map(str, test_all_ids))}\n\n")

        f.write("=== AUGMENTED SUMMARY ===\n")
        f.write(f"augmented_dir: {aug_dir.resolve()}\n")
        f.write(f"synthetic_ids_found: {len(synthetic_ids_found)}\n")
        if synthetic_ids_found:
            f.write(f"synthetic_id_list: {', '.join(map(str, synthetic_ids_found))}\n")
        else:
            f.write("synthetic_id_list: none\n")


def write_split_membership_csv(subject_df, output_csv, aug_dir=AUGMENTED_DATA_DIR):
    """
    Writes one row per participant per split.
    Includes both real and augmented participants.
    """
    rng = random.Random(RANDOM_SEED)

    low_ids = sorted(subject_df.loc[subject_df["risk_level"] == "low", "participant_id"].tolist())
    medium_ids = sorted(subject_df.loc[subject_df["risk_level"] == "medium", "participant_id"].tolist())
    all_real_ids = sorted(subject_df["participant_id"].tolist())

    risk_map = dict(zip(subject_df["participant_id"], subject_df["risk_level"]))
    real_to_synthetic, _ = get_augmented_id_map(aug_dir, SYNTHETIC_OFFSET)

    rows = []

    for split in range(1, N_SPLITS + 1):
        test_real_ids = []

        for group_ids in [low_ids, medium_ids]:
            ids_copy = group_ids[:]
            rng.shuffle(ids_copy)

            if len(ids_copy) <= 1:
                n_test = len(ids_copy)
            else:
                n_test = max(1, int(len(ids_copy) * TEST_FRACTION))

            test_real_ids.extend(ids_copy[:n_test])

        test_real_ids = sorted(set(test_real_ids))
        train_real_ids = sorted([x for x in all_real_ids if x not in test_real_ids])

        for real_id in train_real_ids:
            rows.append({
                "split": split,
                "participant_id": real_id,
                "source_id": real_id,
                "is_augmented": False,
                "set": "train",
                "risk_level": risk_map[real_id],
            })
            for synthetic_id in real_to_synthetic.get(real_id, []):
                rows.append({
                    "split": split,
                    "participant_id": synthetic_id,
                    "source_id": real_id,
                    "is_augmented": True,
                    "set": "train",
                    "risk_level": risk_map[real_id],
                })

        for real_id in test_real_ids:
            rows.append({
                "split": split,
                "participant_id": real_id,
                "source_id": real_id,
                "is_augmented": False,
                "set": "test",
                "risk_level": risk_map[real_id],
            })
            for synthetic_id in real_to_synthetic.get(real_id, []):
                rows.append({
                    "split": split,
                    "participant_id": synthetic_id,
                    "source_id": real_id,
                    "is_augmented": True,
                    "set": "test",
                    "risk_level": risk_map[real_id],
                })

    pd.DataFrame(rows).sort_values(
        ["split", "set", "source_id", "is_augmented", "participant_id"]
    ).to_csv(output_csv, index=False)


def main():
    df = pd.read_csv(INPUT_CSV)

    # use only 2022 data
    df = filter_to_2022(df)
    df = df.sort_values(["id", "study_interval", "day_in_study"])

    interval_results = []
    for (_, _), sub in df.groupby(["id", "study_interval"]):
        interval_results.append(classify_interval(sub))

    interval_df = pd.DataFrame(interval_results).sort_values(["id", "study_interval"])
    subject_notes_df = aggregate_subject_notes(interval_df)

    subject_notes_df.to_csv(OUTPUT_NOTES_CSV, index=False)
    write_readable_splits(subject_notes_df, OUTPUT_SPLITS_TXT)
    write_split_membership_csv(subject_notes_df, OUTPUT_SPLIT_MEMBERSHIP_CSV)

    print("Saved:")
    print(OUTPUT_NOTES_CSV)
    print(OUTPUT_SPLITS_TXT)
    print(OUTPUT_SPLIT_MEMBERSHIP_CSV)


if __name__ == "__main__":
    main()
