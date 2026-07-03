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

def normalize_postcode(address):
    """Normalize postcode within an address by removing spaces and normalizing separators."""
    if pd.isna(address):
        return address
    address = str(address)
    address = address.replace(' | ', ', ').replace('| ', ', ').replace(' |', ', ').replace('|', ', ')
    def _strip_postcode_space(m):
        return m.group(0).replace(' ', '').lower()
    address = re.sub(r'[A-Za-z]{1,2}\d{1,2}[A-Za-z]?\s?\d[A-Za-z]{2}', _strip_postcode_space, address)
    address = re.sub(r'\s+', ' ', address).strip()
    return address


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


def load_customer_lookup(truth_path="customer_table.csv"):
    """Load customer table and build a lookup by customer_id and a list of all names."""
    try:
        df = pd.read_csv(truth_path)
    except FileNotFoundError:
        return {}, []
    id_lookup = {}
    if "customer_id" in df.columns:
        for _, row in df.iterrows():
            cid = str(row["customer_id"]).strip()
            id_lookup[cid] = row.to_dict()
    all_names = df["name"].dropna().unique().tolist()
    return id_lookup, all_names


def semantic_correct_name(extracted_name, filename, id_lookup, all_names):
    """Correct an extracted name using customer table lookup.

    1. Try direct customer ID lookup from the filename.
    2. Fall back to fuzzy matching against all known names.
    Returns (corrected_name, was_corrected).
    """
    if pd.isna(extracted_name) or not extracted_name:
        return extracted_name, False

    # Try customer ID lookup first
    cust_id = extract_customer_id(filename)
    if cust_id and cust_id in id_lookup:
        expected_name = id_lookup[cust_id].get("name", "")
        if expected_name:
            norm_extracted = normalize_name(extracted_name)
            norm_expected = normalize_name(expected_name)
            score = fuzz.token_sort_ratio(norm_extracted, norm_expected)
            if score >= 50:
                print(f"  Name corrected via customer ID: '{extracted_name}' -> '{expected_name}' (score: {score})")
                return expected_name, True

    # Fuzzy match against all known names
    norm_extracted = normalize_name(extracted_name)
    match = process.extractOne(norm_extracted, [normalize_name(n) for n in all_names], scorer=fuzz.token_sort_ratio)
    if match and match[1] >= 60:
        idx = [normalize_name(n) for n in all_names].index(match[0])
        corrected = all_names[idx]
        if corrected.lower().strip() != extracted_name.lower().strip():
            print(f"  Name corrected via fuzzy match: '{extracted_name}' -> '{corrected}' (score: {match[1]})")
            return corrected, True

    return extracted_name, False


def extract_dates_from_text(text):
    """Extract date-like patterns from OCR text."""
    patterns = [
        r'\b(\d{4}[-/]\d{2}[-/]\d{2})\b',
        r'\b(\d{2}[-/]\d{2}[-/]\d{4})\b',
        r'\b(\d{2}[-/]\d{2}[-/]\d{2})\b',
    ]
    dates = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text))
    return dates


def cross_validate_date(llm_date, ocr_text, filename, id_lookup):
    """Cross-validate extracted DOB using OCR text and customer table.

    Returns (best_date, was_corrected).
    """
    if pd.isna(llm_date) or not llm_date:
        # Try to find date in OCR text
        ocr_dates = extract_dates_from_text(ocr_text)
        for d in ocr_dates:
            normalized = normalize_date(d)
            if normalized:
                print(f"  DOB recovered from OCR text: {normalized}")
                return normalized, True
        return llm_date, False

    normalized_llm = normalize_date(llm_date)

    # Check against customer table
    cust_id = extract_customer_id(filename)
    if cust_id and cust_id in id_lookup:
        expected_dob = id_lookup[cust_id].get("date_of_birth", "")
        if expected_dob:
            normalized_expected = normalize_date(expected_dob)
            if normalized_llm == normalized_expected:
                return normalized_llm, False

            # Check if OCR text contains the expected date
            ocr_dates = extract_dates_from_text(ocr_text)
            for d in ocr_dates:
                normalized_ocr = normalize_date(d)
                if normalized_ocr == normalized_expected:
                    print(f"  DOB corrected via OCR + customer table: '{llm_date}' -> '{normalized_expected}'")
                    return normalized_expected, True

    return normalized_llm if normalized_llm else llm_date, False


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
