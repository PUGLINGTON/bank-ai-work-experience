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


# ── Prefix stripping ────────────────────────────────────────────────────────
#
# Documents print a field's value next to a label, and headers/watermarks leak
# into the extracted value (e.g. the "BILL" on a *utility bill* getting read as
# part of a name → "Billie Lily Cooper"). We (1) strip the field's own printed
# label when it is prepended, and (2) strip a leading document-type header word.

# Header keywords implied by the document type (from the filename).
DOC_TYPE_KEYWORDS = {
    'Bank Statement': ['bank statement', 'statement', 'bank'],
    'Payslip': ['payslip', 'pay slip', 'payslip-'],
    'Utility Bill': ['utility bill', 'utility', 'bill'],
}

# Label text that can precede a field's value on the page.
FIELD_LABEL_PREFIXES = {
    'name': ['account holder', 'account name', 'customer name', 'employee name',
             'full name', 'name', 'mr', 'mrs', 'ms', 'miss', 'dr', 'prof'],
    'date_of_birth': ['date of birth', 'd.o.b', 'd o b', 'dob', 'born', 'birth date'],
    'address': ['billing address', 'home address', 'address'],
    'occupation': ['job title', 'occupation', 'position', 'role'],
    'employer': ['employer name', 'company name', 'employer', 'company'],
}


def _strip_leading_phrases(text, phrases):
    """Remove any leading label phrase (longest first) plus trailing separators."""
    changed = True
    while changed:
        changed = False
        for phrase in sorted(phrases, key=len, reverse=True):
            # phrase must be followed by a word boundary / separator, not glued
            # into a longer word (so "name" won't eat "nathan").
            pattern = r'^' + re.escape(phrase) + r'\b[\s:\-|]*'
            new = re.sub(pattern, '', text, flags=re.IGNORECASE)
            if new != text:
                text = new
                changed = True
                break
    return text.strip()


def strip_field_prefixes(value, field, filename=None):
    """Remove field labels and document-type header words from an extracted value.

    Strips the field's own printed label (e.g. "Date of Birth: 1984-…" → "1984-…")
    and, for the name field, a leading header token that fuzzy-matches the
    document type implied by the filename (e.g. a utility *bill* leaking
    "Billie" onto the front of the name). Only strips a name header token when a
    real name still remains afterwards, so genuine names are preserved.
    """
    if pd.isna(value) or value == "":
        return value
    text = str(value).strip()

    # 1) Strip the field's own label if it was prepended.
    text = _strip_leading_phrases(text, FIELD_LABEL_PREFIXES.get(field, []))

    # 2) For names, drop a leading header word (bill/payslip/statement/…) that
    #    the OCR/model glued on, but only if two or more name tokens remain.
    if field == 'name' and filename is not None:
        doc_type = extract_document_type(filename)
        keywords = DOC_TYPE_KEYWORDS.get(doc_type, [])
        tokens = text.split()
        if len(tokens) >= 3 and keywords:
            first = tokens[0].lower().strip(".,")
            if any(fuzz.ratio(first, kw) >= 80 for kw in keywords):
                text = " ".join(tokens[1:]).strip()

    return text if text else None


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


# Trailing UK-postcode pattern, tolerant to OCR noise: allows an optional
# internal space and lets the inward characters be a letter *or* a digit (OCR
# often swaps o/0, i/1, etc.). Anchored to the end of the string so it only
# ever grabs the postcode, never part of the street.
TRAILING_POSTCODE_PATTERN = re.compile(
    r'[a-z]{1,2}\d{1,2}[a-z0-9]?\s?\d[a-z0-9]{2}\s*$'
)


def split_address(address):
    """Normalize an address and split it into (street, postcode).

    Runs normalize_address first (lowercase, drop commas/pipes), then peels the
    trailing postcode off the end so the two can be compared independently. The
    postcode match is tolerant to OCR noise (an internal space, or a character
    read as the wrong type) so a slightly-garbled postcode is still separated
    from the street instead of being glued onto it. Returns (street, postcode);
    postcode is None when no postcode is present.
    """
    if pd.isna(address):
        return None, None
    normalized = str(normalize_address(address))
    match = TRAILING_POSTCODE_PATTERN.search(normalized)
    if not match:
        return normalized, None
    postcode = re.sub(r'\s+', '', match.group(0))
    street = re.sub(r'\s+', ' ', normalized[:match.start()]).strip()
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
