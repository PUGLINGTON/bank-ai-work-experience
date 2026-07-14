"""Streamlit auditor dashboard for the document extraction pipeline.

Two tabs:

  1. "Process Document" — upload a document image or point at a file on disk,
     run the bank-compliant extraction (document + OCR only, with a targeted
     re-read of any flagged field), then compare the result against the customer
     table. Address is shown and compared split into street + postcode.

  2. "Issues to Review" — a running list of everything a human should check:
     field mismatches, OCR warnings, and people not found in the customer table.
     Can also be reloaded from the saved audit CSVs.

The customer table is only read in the comparison/audit step, never during
extraction, so the extraction stage stays bank-compliant.

Run with:  streamlit run app.py
"""

import os
import tempfile

import pandas as pd
import streamlit as st

from final import process_file, compare_record
from utils import extract_customer_id, extract_document_type

TRUTH_PATH = "customer_table.csv"

st.set_page_config(
    page_title="Document Extraction Auditor",
    page_icon="🗂️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1400px; }
      /* Readable, roomier tables */
      table { font-size: 0.95rem; }
      thead th { background: #262730; color: #fafafa !important; }
      /* Tab labels a touch larger */
      button[data-baseweb="tab"] p { font-size: 1.02rem; font-weight: 600; }
      /* Wrap long values instead of truncating */
      td { white-space: normal !important; word-break: break-word; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "issues" not in st.session_state:
    st.session_state.issues = []


def add_issue(filename, field, extracted, note):
    st.session_state.issues.append({
        "customer_id": extract_customer_id(filename) or "-",
        "document_type": extract_document_type(filename),
        "filename": filename,
        "field": field,
        "extracted": extracted,
        "note": note,
    })


def resolve_input_path(uploaded, typed_path):
    """Return a filesystem path for the chosen document, or None."""
    if uploaded is not None:
        # Preserve the original name so CUST-#### can be parsed from it.
        tmp_dir = tempfile.mkdtemp(prefix="audit_")
        path = os.path.join(tmp_dir, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        return path
    typed_path = (typed_path or "").strip()
    if typed_path:
        return typed_path
    return None


def render_process_tab():
    st.subheader("Process a document")
    col_u, col_p = st.columns(2)
    with col_u:
        uploaded = st.file_uploader(
            "Upload a document image", type=["png", "jpg", "jpeg", "tif", "tiff"]
        )
    with col_p:
        typed_path = st.text_input("…or enter a file path on disk")

    if not st.button("Run extraction & compare", type="primary"):
        return

    path = resolve_input_path(uploaded, typed_path)
    if not path:
        st.warning("Upload a file or enter a path first.")
        return
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        return

    filename = os.path.basename(path)
    with st.spinner("Rotating, running OCR, extracting, and re-reading flagged fields…"):
        try:
            data, ocr_text, warnings, refinements = process_file(path)
        except Exception as exc:
            st.error(f"Extraction failed: {exc}")
            return

    if data is None:
        st.error("No data could be extracted from this document.")
        add_issue(filename, "-", "extraction failed", "No JSON returned by the model")
        return

    left, right = st.columns([1, 2], gap="large")
    with left:
        try:
            st.image(path, caption=filename, use_container_width=True)
        except Exception:
            st.caption("(no preview available)")

    with right:
        st.markdown("#### Extracted values")
        st.caption("Read from the document + OCR only — no customer data used.")

        street, postcode = _split_for_display(data.get("address"))
        ext_df = pd.DataFrame(
            [
                ("Name", data.get("name")),
                ("Date of birth", data.get("date_of_birth")),
                ("Address (street)", street),
                ("Postcode", postcode),
                ("Occupation", data.get("occupation")),
                ("Employer", data.get("employer")),
            ],
            columns=["Field", "Value"],
        )
        ext_df["Value"] = ext_df["Value"].apply(lambda v: "—" if not v else str(v))
        st.table(ext_df)

        if refinements:
            st.markdown("**Targeted re-reads applied**")
            for r in refinements:
                st.info(f"{r['field']}: '{r['old']}' → '{r['new']}'")

    st.divider()

    # Comparison vs customer table (audit stage)
    st.markdown("### Comparison vs customer table")
    if not os.path.exists(TRUTH_PATH):
        st.warning(f"{TRUTH_PATH} not found — comparison skipped.")
    else:
        try:
            rows, matched_name = compare_record(data, TRUTH_PATH)
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")
            rows, matched_name = [], None

        if matched_name is None:
            st.warning("Not found in customer table — logged as an unknown record.")
            add_issue(filename, "name", data.get("name", ""), "Not found in customer table")
        else:
            cmp_df = pd.DataFrame(rows)[["field", "extracted", "expected", "status"]]
            cmp_df = cmp_df.rename(columns={
                "field": "Field",
                "extracted": "Extracted",
                "expected": "Expected",
                "status": "Status",
            })
            st.dataframe(
                cmp_df.style.apply(_highlight_status, axis=1),
                use_container_width=True,
                hide_index=True,
            )
            mismatches = [r for r in rows if r["status"] == "mismatch"]
            checked = len(rows)
            acc = 100.0 * (checked - len(mismatches)) / checked if checked else 100.0
            summary = (
                f"Matched **{matched_name}** — {len(mismatches)} mismatch(es) "
                f"of {checked} fields ({acc:.0f}% match)."
            )
            if mismatches:
                st.warning(summary)
            else:
                st.success(summary)
            for r in mismatches:
                add_issue(filename, r["field"], r["extracted"], f"expected: {r['expected']}")

    for w in warnings:
        add_issue(filename, "OCR check", "", w["message"])

    st.caption("Issues from this document were added to the 'Issues to Review' tab.")


def _split_for_display(address):
    from utils import split_address
    if address is None:
        return None, None
    return split_address(address)


def _highlight_status(row):
    status = row.get("Status", row.get("status"))
    bg = "#f8d7da" if status == "mismatch" else "#d4edda"
    return [f"background-color: {bg}; color: #1a1a1a; font-weight: 500"] * len(row)


def render_issues_tab():
    st.subheader("Issues to review")
    top = st.columns([1, 1, 4])
    if top[0].button("Reload from audit CSVs"):
        _load_from_csvs()
    if top[1].button("Clear"):
        st.session_state.issues = []

    issues = st.session_state.issues
    st.caption(f"{len(issues)} issue{'s' if len(issues) != 1 else ''}")
    if issues:
        st.dataframe(pd.DataFrame(issues), use_container_width=True, hide_index=True)
    else:
        st.info("No issues yet. Process some documents, or reload from the audit CSVs.")


def _load_from_csvs():
    loaded = 0
    if os.path.exists("known_mismatches.csv"):
        df = pd.read_csv("known_mismatches.csv")
        for _, r in df.iterrows():
            st.session_state.issues.append({
                "customer_id": r.get("customer_id", "-"),
                "document_type": r.get("document_type", ""),
                "filename": r.get("filename", ""),
                "field": r.get("field", ""),
                "extracted": r.get("extracted_value", ""),
                "note": f"expected: {r.get('expected_value', '')}",
            })
            loaded += 1
    if os.path.exists("unknown_customers.csv"):
        df = pd.read_csv("unknown_customers.csv")
        for _, r in df.iterrows():
            st.session_state.issues.append({
                "customer_id": r.get("customer_id", "-"),
                "document_type": r.get("document_type", ""),
                "filename": r.get("filename", ""),
                "field": "name",
                "extracted": r.get("extracted_name", ""),
                "note": r.get("reason", "Not found in customer table"),
            })
            loaded += 1
    if loaded == 0:
        st.warning("No audit CSVs found yet. Run final.py for a full batch first.")


st.title("🗂️ Document Extraction Auditor")
st.caption(
    "Extract identity fields from a document, then audit them against the "
    "customer table. Address is split into street and postcode for both display "
    "and comparison."
)
tab_process, tab_issues = st.tabs(["Process Document", "Issues to Review"])
with tab_process:
    render_process_tab()
with tab_issues:
    render_issues_tab()
