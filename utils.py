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

def normalize_address(address):
    """Normalize an address for comparison: remove commas, pipes, collapse whitespace, strip postcode spaces."""
    if pd.isna(address):
        return address
    address = str(address)
    # Replace pipe separators
    address = address.replace('|', ' ')
    # Remove all commas
    address = address.replace(',', '')
    # Lowercase
    address = address.lower()
    # Strip postcode internal spaces
    def _strip_postcode_space(m):
        return m.group(0).replace(' ', '')
    address = re.sub(r'[a-z]{1,2}\d{1,2}[a-z]?\s?\d[a-z]{2}', _strip_postcode_space, address)
    # Collapse whitespace
    address = re.sub(r'\s+', ' ', address).strip()
    return address


def normalize_postcode(address):
    """Normalize postcode within an address by removing spaces and normalizing separators."""
    return normalize_address(address)


def clean_employer(value):
    """Remove 'PAYSLIP', 'payslip', and 'payslip-' prefix from employer field."""
    if pd.isna(value) or value == "":
        return value
    value = re.sub(r'(?i)\bpayslip[-\u2013\u2014]?\s*', '', str(value)).strip()
    value = re.sub(r'\s+', ' ', value).strip()
    return value if value else None


def extract_customer_id(filename):
    """Extract customer ID (e.g. CUST-1001) from a filename."""
    match = re.search(r'(CUST-\d+)', str(filename))
    return match.group(1) if match else None


def extract_dates_from_text(text):
    """Extract date-like patterns from OCR text (document-only, no customer table)."""
    patterns = [
        r'\b(\d{4}[-/]\d{2}[-/]\d{2})\b',
        r'\b(\d{2}[-/]\d{2}[-/]\d{4})\b',
        r'\b(\d{2}[-/]\d{2}[-/]\d{2})\b',
    ]
    dates = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text))
    return dates


def extract_document_type(filename):
    """Extract document type from filename prefix."""
    fname = str(filename).lower()
    if fname.startswith('bank_statement'):
        return 'Bank Statement'
    elif fname.startswith('payslip'):
        return 'Payslip'
    elif fname.startswith('utility_bill'):
        return 'Utility Bill'
    return 'Unknown'


def fuzzy_match_name(name, choices, threshold=70):
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


# Postcode inside an already-normalized (lowercased, space-stripped) address.
NORMALIZED_POSTCODE_PATTERN = re.compile(r'[a-z]{1,2}\d{1,2}[a-z]?\d[a-z]{2}')


def split_address(address):
    """Normalize an address and split it into (street, postcode).

    Runs normalize_address first (lowercase, drop commas/pipes, strip postcode
    spaces), then separates the postcode from the rest so the two can be
    compared independently. Returns (street, postcode); postcode is None when
    no postcode is present.
    """
    if pd.isna(address):
        return None, None
    normalized = str(normalize_address(address))
    match = NORMALIZED_POSTCODE_PATTERN.search(normalized)
    if not match:
        return normalized, None
    postcode = match.group(0)
    street = (normalized[:match.start()] + normalized[match.end():])
    street = re.sub(r'\s+', ' ', street).strip()
    return street, postcode


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


def print_accuracy_report(total_fields, total_mismatches, accuracy, per_field_accuracy=None):
    """Print a formatted accuracy report to stdout, with optional per-field breakdown."""
    print(f"\n--- Accuracy Report ---")
    print(f"Total fields compared: {total_fields}")
    print(f"Mismatches found:      {total_mismatches}")
    print(f"Matches:               {total_fields - total_mismatches}")
    print(f"Overall accuracy:      {accuracy:.2f}%")

    if per_field_accuracy:
        print(f"\n--- Per-Field Accuracy ---")
        for field, field_accuracy in per_field_accuracy.items():
            print(f"  {field:<15} {field_accuracy:.2f}%")


# ── CSV helpers ────────────────────────────────────────────────────────────────

def ensure_csv(path, fieldnames):
    """Create *path* with a header row if it does not already exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
