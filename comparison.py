import pandas as pd
import re

from utils import (
    normalize_text,
    normalize_name,
    normalize_date,
    fuzzy_match_name,
    is_missing,
    clean_employer,
    calculate_accuracy,
    print_accuracy_report,
)

def clean_employer(value):
    """Remove 'PAYSLIP' or 'payslip' text from employer field."""
    if pd.isna(value) or value == "":
        return value
    value = re.sub(r'\b[Pp][Aa][Yy][Ss][Ll][Ii][Pp]\b', '', str(value)).strip()
    value = re.sub(r'\s+', ' ', value).strip()
    return value if value else None


def compare_csvs(submit_path="submit.csv", truth_path="customer_table.csv"):
    submit_df = pd.read_csv(submit_path)
    truth_df = pd.read_csv(truth_path)

    text_columns = ["address", "occupation", "employer"]

    for col in text_columns:
        if col in submit_df.columns:
            submit_df[col] = submit_df[col].apply(normalize_text)
        if col in truth_df.columns:
            truth_df[col] = truth_df[col].apply(normalize_text)

    submit_df["name"] = submit_df["name"].apply(normalize_name)
    truth_df["name"] = truth_df["name"].apply(normalize_name)

    submit_df["date_of_birth"] = submit_df["date_of_birth"].apply(normalize_date)
    truth_df["date_of_birth"] = truth_df["date_of_birth"].apply(normalize_date)

    truth_names = truth_df["name"].dropna().unique().tolist()

    submit_df["matched_name"] = submit_df["name"].apply(
        lambda n: fuzzy_match_name(n, truth_names)
    )

    unmatched = submit_df[submit_df["matched_name"].isna()]
    matched = submit_df.dropna(subset=["matched_name"])

    merged = matched.merge(
        truth_df,
        left_on="matched_name",
        right_on="name",
        suffixes=("_submit", "_truth"),
        how="inner"
    )

    fields_to_check = {
        "name": 1,
        "date_of_birth": 1,
        "address": 2,
        "occupation": 2,
        "employer": 3,
    }

    results = []
    total_fields_checked = 0
    total_mismatches = 0

    # Per-field tracking
    field_stats = {field: {"checked": 0, "mismatches": 0} for field in fields_to_check}

    for _, row in merged.iterrows():
        for field, category in fields_to_check.items():
            submit_val = row.get(f"{field}_submit")
            truth_val = row.get(f"{field}_truth")

            if field == "employer" and is_missing(submit_val):
                continue

            if is_missing(submit_val) and is_missing(truth_val):
                continue

            total_fields_checked += 1
            field_stats[field]["checked"] += 1

            if submit_val == truth_val:
                continue

            total_mismatches += 1
            field_stats[field]["mismatches"] += 1
            results.append({
                "filename": row.get("filename"),
                "name": row.get("name"),
                "date_of_birth": row.get("date_of_birth"),
                "category": category,
                "field": field,
                "submit_value": submit_val,
                "truth_value": truth_val,
            })

    # Count unmatched records (each counts as a mismatch on the name field)
    total_fields_checked += len(unmatched)
    total_mismatches += len(unmatched)
    field_stats["name"]["checked"] += len(unmatched)
    field_stats["name"]["mismatches"] += len(unmatched)

    for _, row in unmatched.iterrows():
        results.append({
            "filename": row.get("filename"),
            "name": row.get("name"),
            "date_of_birth": row.get("date_of_birth"),
            "category": "N/A",
            "field": "name",
            "submit_value": row.get("name"),
            "truth_value": "No matching name found",
        })

    overall_accuracy = calculate_accuracy(total_fields_checked, total_mismatches)

    # Calculate per-field accuracy
    per_field_accuracy = {}
    for field, stats in field_stats.items():
        per_field_accuracy[field] = calculate_accuracy(stats["checked"], stats["mismatches"])

    return pd.DataFrame(results), total_fields_checked, total_mismatches, overall_accuracy, per_field_accuracy


def run_comparison():
    mismatches_df, total_fields, total_mismatches, accuracy, per_field_accuracy = compare_csvs()
    mismatches_df["sort_key"] = mismatches_df["category"].apply(lambda x: 999 if x == "N/A" else x)
    mismatches_df = mismatches_df.sort_values(by=["sort_key", "name"]).drop(columns="sort_key").reset_index(drop=True)
    print(mismatches_df)
    mismatches_df.to_csv("mismatches.csv", index=False)

    print_accuracy_report(total_fields, total_mismatches, accuracy, per_field_accuracy)
