"""Streamlit auditor dashboard for the document extraction pipeline.

Tabs:

  1. "Process Document" — upload a document image or point at a file on disk,
     run the bank-compliant extraction (document + OCR only, with a targeted
     re-read of any flagged field), then compare against the customer table.
     Address is shown and compared split into street + postcode.

  2. "Scan Folder" — point at a folder and process every image in it in one go,
     with a running results summary.

  3. "Flagged Files" — just the files that need a human look (mismatches, OCR
     warnings, unknown customers, extraction errors), for a quick triage.

  4. "Issues to Review" — the field-level list of everything to check; can also
     be reloaded from the saved audit CSVs.

The customer table is only read in the comparison/audit step, never during
extraction, so the extraction stage stays bank-compliant.

Run with:  streamlit run app.py
"""

import os
import tempfile

import pandas as pd
import streamlit as st

from final import process_file, compare_record
from utils import extract_customer_id, extract_document_type, split_address

TRUTH_PATH = "customer_table.csv"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

st.set_page_config(
    page_title="Document Extraction Auditor",
    page_icon="🗂️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1400px; }
      table { font-size: 0.95rem; }
      thead th { background: #262730; color: #fafafa !important; }
      button[data-baseweb="tab"] p { font-size: 1.02rem; font-weight: 600; }
      td { white-space: normal !important; word-break: break-word; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "issues" not in st.session_state:
    st.session_state.issues = []
if "results" not in st.session_state:
    st.session_state.results = []


# ── Helpers ─────────────────────────────────────────────────────────────────

def add_issue(filename, field, extracted, note):
    st.session_state.issues.append({
        "customer_id": extract_customer_id(filename) or "-",
        "document_type": extract_document_type(filename),
        "filename": filename,
        "field": field,
        "extracted": extracted,
        "note": note,
    })


def _split_for_display(address):
    if not address:
        return None, None
    return split_address(address)


def _highlight_status(row):
    status = row.get("Status", row.get("status"))
    bg = "#f8d7da" if status == "mismatch" else "#d4edda"
    return [f"background-color: {bg}; color: #1a1a1a; font-weight: 500"] * len(row)


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


def run_one(path):
    """Extract + compare a single document and return a structured result dict.

    Also appends the resulting issues to the shared 'Issues to Review' list.
    Never raises — extraction/comparison errors are captured in the result.
    """
    filename = os.path.basename(path)
    result = {
        "filename": filename,
        "path": path,
        "customer_id": extract_customer_id(filename) or "-",
        "document_type": extract_document_type(filename),
        "status": "ok",
        "error": None,
        "data": None,
        "warnings": [],
        "refinements": [],
        "matched_name": None,
        "rows": [],
        "mismatches": [],
    }

    try:
        data, ocr_text, warnings, refinements = process_file(path)
    except Exception as exc:  # extraction failed (e.g. API/key/network)
        result["status"] = "error"
        result["error"] = str(exc)
        add_issue(filename, "-", "extraction failed", str(exc))
        return result

    result["warnings"] = warnings
    result["refinements"] = refinements

    if data is None:
        result["status"] = "error"
        result["error"] = "No JSON returned by the model"
        add_issue(filename, "-", "extraction failed", result["error"])
        return result

    result["data"] = data

    if os.path.exists(TRUTH_PATH):
        try:
            rows, matched_name = compare_record(data, TRUTH_PATH)
        except Exception as exc:
            result["status"] = "error"
            result["error"] = f"comparison failed: {exc}"
            return result

        result["rows"] = rows
        result["matched_name"] = matched_name
        if matched_name is None:
            result["status"] = "unknown"
            add_issue(filename, "name", data.get("name", ""),
                      "Not found in customer table")
        else:
            mismatches = [r for r in rows if r["status"] == "mismatch"]
            result["mismatches"] = mismatches
            if mismatches:
                result["status"] = "flagged"
            for r in mismatches:
                add_issue(filename, r["field"], r["extracted"],
                          f"expected: {r['expected']}")

    for w in warnings:
        add_issue(filename, "OCR check", "", w["message"])
        if result["status"] == "ok":
            result["status"] = "flagged"

    return result


def _flagged_fields(result):
    fields = [m["field"] for m in result.get("mismatches", [])]
    fields += [w["field"] for w in result.get("warnings", [])]
    return ", ".join(sorted(set(fields))) if fields else ""


# ── Rendering ───────────────────────────────────────────────────────────────

def render_result_detail(result):
    data = result.get("data")
    if data is None:
        st.error(f"Extraction failed: {result.get('error')}")
        return

    left, right = st.columns([1, 2], gap="large")
    with left:
        try:
            st.image(result["path"], caption=result["filename"],
                     use_container_width=True)
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

        if result["refinements"]:
            st.markdown("**Targeted re-reads applied**")
            for r in result["refinements"]:
                st.info(f"{r['field']}: '{r['old']}' → '{r['new']}'")

    st.divider()
    st.markdown("### Comparison vs customer table")
    if not os.path.exists(TRUTH_PATH):
        st.warning(f"{TRUTH_PATH} not found — comparison skipped.")
        return
    if result["matched_name"] is None:
        st.warning("Not found in customer table — logged as an unknown record.")
        return

    cmp_df = pd.DataFrame(result["rows"])[["field", "extracted", "expected", "status"]]
    cmp_df = cmp_df.rename(columns={
        "field": "Field", "extracted": "Extracted",
        "expected": "Expected", "status": "Status",
    })
    st.dataframe(
        cmp_df.style.apply(_highlight_status, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    checked = len(result["rows"])
    n_mis = len(result["mismatches"])
    acc = 100.0 * (checked - n_mis) / checked if checked else 100.0
    summary = (f"Matched **{result['matched_name']}** — {n_mis} mismatch(es) "
               f"of {checked} fields ({acc:.0f}% match).")
    (st.warning if n_mis else st.success)(summary)


def render_process_tab():
    st.subheader("Process a document")
    col_u, col_p = st.columns(2)
    with col_u:
        uploaded = st.file_uploader(
            "Upload a document image", type=list(e.strip(".") for e in IMAGE_EXTS)
        )
    with col_p:
        typed_path = st.text_input("…or enter a file path on disk")

    if st.button("Run extraction & compare", type="primary"):
        path = resolve_input_path(uploaded, typed_path)
        if not path:
            st.warning("Upload a file or enter a path first.")
        elif not os.path.exists(path):
            st.error(f"File not found: {path}")
        else:
            with st.spinner("Rotating, running OCR, extracting, and re-reading flagged fields…"):
                result = run_one(path)
            st.session_state.results.append(result)
            st.session_state["last_result"] = result

    last = st.session_state.get("last_result")
    if last is not None:
        render_result_detail(last)
        st.caption("Issues from this document were added to the other tabs.")


def render_folder_tab():
    st.subheader("Scan a folder")
    st.caption("Processes every image in the folder and adds each to the results.")
    folder = st.text_input("Folder path", value="data")

    if st.button("Scan folder", type="primary"):
        if not os.path.isdir(folder):
            st.error(f"Not a folder: {folder}")
            return
        files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTS)
        )
        if not files:
            st.warning("No image files found in that folder.")
            return

        progress = st.progress(0.0, text="Starting…")
        summary = []
        for i, name in enumerate(files, start=1):
            progress.progress(i / len(files), text=f"Processing {name} ({i}/{len(files)})")
            result = run_one(os.path.join(folder, name))
            st.session_state.results.append(result)
            summary.append({
                "filename": name,
                "customer_id": result["customer_id"],
                "document_type": result["document_type"],
                "status": result["status"],
                "mismatches": len(result["mismatches"]),
                "flagged_fields": _flagged_fields(result),
            })
        progress.empty()

        df = pd.DataFrame(summary)
        n_flag = (df["status"] != "ok").sum()
        st.success(f"Scanned {len(files)} file(s) — {n_flag} need review "
                   "(see the Flagged Files tab).")
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_flagged_tab():
    st.subheader("Flagged files")
    st.caption("Files with mismatches, OCR warnings, unknown customers, or errors.")
    if st.button("Clear results"):
        st.session_state.results = []

    flagged = [r for r in st.session_state.results if r["status"] != "ok"]
    total = len(st.session_state.results)
    st.caption(f"{len(flagged)} flagged of {total} processed")

    if not flagged:
        st.info("Nothing flagged yet. Process a document or scan a folder.")
        return

    table = pd.DataFrame([{
        "filename": r["filename"],
        "customer_id": r["customer_id"],
        "document_type": r["document_type"],
        "status": r["status"],
        "mismatches": len(r["mismatches"]),
        "flagged_fields": _flagged_fields(r),
        "error": r["error"] or "",
    } for r in flagged])
    st.dataframe(table, use_container_width=True, hide_index=True)

    names = [r["filename"] for r in flagged]
    chosen = st.selectbox("Inspect a flagged file", ["—"] + names)
    if chosen != "—":
        result = next(r for r in flagged if r["filename"] == chosen)
        st.divider()
        render_result_detail(result)


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
tab_process, tab_folder, tab_flagged, tab_issues = st.tabs(
    ["Process Document", "Scan Folder", "Flagged Files", "Issues to Review"]
)
with tab_process:
    render_process_tab()
with tab_folder:
    render_folder_tab()
with tab_flagged:
    render_flagged_tab()
with tab_issues:
    render_issues_tab()
