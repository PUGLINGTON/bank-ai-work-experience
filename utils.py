import csv
import io
import json
import os
import re
import base64

import pandas as pd
from rapidfuzz import fuzz, process

# ── Shared regex patterns ──────────────────────────────────────────────────────

TITLE_PATTERN = re.compile(
    r'\b(mr|mrs|ms|miss|dr|prof|account holder)\b\.?:?'
)
POSTCODE_PATTERN = re.compile(r'([A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2})')


# ── Value helpers ──────────────────────────────────────────────────────────────

def is_missing(value):
    """Check if a value is NaN/None or an empty string."""
    return pd.isna(value) or value == ""


# ── Text normalisation ─────────────────────────────────────────────────────────

def normalize_text(value):
    """Lowercase, strip whitespace, and remove common titles/prefixes."""
    if pd.isna(value):
        return value
    value = str(value).lower().strip()
    value = TITLE_PATTERN.sub('', value)
    value = re.sub(r'\s+', ' ', value).strip()
    return value


def normalize_name(value):
    """normalize_text + reduce to first-name last-name."""
    value = normalize_text(value)
    if pd.isna(value):
        return value
    parts = value.split()
    if len(parts) > 2:
        value = f"{parts[0]} {parts[-1]}"
    return value


def normalize_date(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return pd.to_datetime(value, format=fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return None


# ── Fuzzy matching ─────────────────────────────────────────────────────────────

def fuzzy_match_name(name, choices, threshold=85):
    if pd.isna(name) or not choices:
        return None
    match = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= threshold:
        return match[0]
    return None


# ── Postcode helpers ───────────────────────────────────────────────────────────

def extract_postcode(address):
    if pd.isna(address):
        return None
    match = POSTCODE_PATTERN.search(address)
    return match.group(1) if match else None


def remove_postcode(address):
    if pd.isna(address):
        return address
    return POSTCODE_PATTERN.sub('', address).rstrip(', ').strip()


# ── Image / API helpers ───────────────────────────────────────────────────────

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_llm_json(raw):
    """Strip markdown fences from an LLM response and parse as JSON."""
    clean = raw.replace("```json", "").replace("```", "").strip()
    if not clean:
        print("WARNING: empty response")
        return None
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        print("WARNING: could not parse JSON")
        print(f"Raw response: {raw}")
        return None


# ── Accuracy helpers ───────────────────────────────────────────────────────────

def calculate_accuracy(total_fields, total_mismatches):
    """Return accuracy percentage given total fields compared and mismatches."""
    if total_fields > 0:
        return (total_fields - total_mismatches) / total_fields * 100
    return 0.0


def print_accuracy_report(total_fields, total_mismatches, accuracy):
    """Print a formatted accuracy report to stdout."""
    print(f"\n--- Accuracy Report ---")
    print(f"Total fields compared: {total_fields}")
    print(f"Mismatches found:      {total_mismatches}")
    print(f"Matches:               {total_fields - total_mismatches}")
    print(f"Accuracy:              {accuracy:.2f}%")


# ── CSV helpers ────────────────────────────────────────────────────────────────

def ensure_csv(path, fieldnames):
    """Create *path* with a header row if it does not already exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
