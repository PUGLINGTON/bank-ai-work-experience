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

import io
import os
import tempfile
import zipfile

import pandas as pd
import streamlit as st

from final import (
    process_file, compare_record, annotate_read_regions,
    correct_rotation, get_token_usage, reset_token_usage,
)
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
if "approved" not in st.session_state:
    st.session_state.approved = {}   # filename -> path
if "rejected" not in st.session_state:
    st.session_state.rejected = {}   # filename -> reason


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


@st.cache_data(show_spinner=False)
def _oriented_image_cached(path, _mtime):
    return correct_rotation(path)


def _oriented_image(path):
    """Rotation-corrected image for display (cached per file), or None."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    try:
        return _oriented_image_cached(path, mtime)
    except Exception:
        return None


def _render_token_usage(prefix=""):
    """Show Mistral token usage: average per API call and session total."""
    stats = get_token_usage()
    if not stats["calls"]:
        return
    st.markdown(f"**{prefix}Mistral token usage**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("API calls", stats["calls"])
    c2.metric("Avg tokens / call", f"{stats['avg_total_per_call']:.0f}")
    c3.metric("Total tokens", f"{stats['total']:,}")
    c4.metric("Prompt / completion",
              f"{stats['prompt']:,} / {stats['completion']:,}")


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
        "relation": None,
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
    clean_match = False  # matched a customer with zero field mismatches

    if data is None:
        result["status"] = "error"
        result["error"] = "No JSON returned by the model"
        add_issue(filename, "-", "extraction failed", result["error"])
        return result

    result["data"] = data

    if os.path.exists(TRUTH_PATH):
        try:
            rows, matched_name, relation = compare_record(data, TRUTH_PATH)
        except Exception as exc:
            result["status"] = "error"
            result["error"] = f"comparison failed: {exc}"
            return result

        result["rows"] = rows
        result["matched_name"] = matched_name
        result["relation"] = relation
        if matched_name is None:
            if relation == "none":
                result["status"] = "unrelated"
                add_issue(filename, "name", data.get("name", ""),
                          "Suggested: no relation to the customer table (name, DOB "
                          "and postcode all unmatched) — flagged for manual review")
            else:
                result["status"] = "unknown"
                add_issue(filename, "name", data.get("name", ""),
                          "Name not found, but DOB/postcode relates to a customer — possible misread")
        else:
            mismatches = [r for r in rows if r["status"] == "mismatch"]
            result["mismatches"] = mismatches
            if mismatches:
                result["status"] = "flagged"
            else:
                clean_match = True
            for r in mismatches:
                add_issue(filename, r["field"], r["extracted"],
                          f"expected: {r['expected']}")

    # OCR warnings are informational — they don't hold back a document whose
    # fields all matched the customer table.
    for w in warnings:
        add_issue(filename, "OCR check", "", w["message"])
        if result["status"] == "ok" and not clean_match:
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
        img = _oriented_image(result["path"])
        if img is not None:
            st.image(img, caption=result["filename"], use_container_width=True)
        else:
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
        if result.get("relation") == "none":
            st.warning("Suggested: no relation to the customer table — name, DOB, "
                       "and postcode all match no customer. Flagged for manual "
                       "review; you decide whether to approve or reject.")
        else:
            st.warning("Name not found in customer table, but the DOB or postcode "
                       "relates to a customer — possible misread. Logged for review.")
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
            reset_token_usage()
            with st.spinner("Rotating, running OCR, extracting, and re-reading flagged fields…"):
                result = run_one(path)
            st.session_state.results.append(result)
            st.session_state["last_result"] = result
            st.session_state["last_usage"] = get_token_usage()

    last = st.session_state.get("last_result")
    if last is not None:
        render_result_detail(last)
        st.caption("Issues from this document were added to the other tabs.")
        if st.session_state.get("last_usage", {}).get("calls"):
            st.divider()
            _render_token_usage("This run — ")


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

        reset_token_usage()
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
        st.session_state["last_usage"] = get_token_usage()
        st.divider()
        _render_token_usage("Scan — ")


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


def compute_stats(results):
    """Aggregate per-field match counts and document-status counts across results."""
    field_order = ["name", "date_of_birth", "address", "postcode",
                   "occupation", "employer"]
    per_field = {f: {"match": 0, "mismatch": 0} for f in field_order}
    status_counts = {"ok": 0, "flagged": 0, "unknown": 0, "unrelated": 0, "error": 0}

    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        # Only known (matched) customers contribute field-level accuracy.
        if r.get("matched_name") is None:
            continue
        for row in r.get("rows", []):
            bucket = per_field.setdefault(row["field"], {"match": 0, "mismatch": 0})
            bucket[row["status"]] = bucket.get(row["status"], 0) + 1

    rows = []
    for f in field_order:
        m, mm = per_field[f]["match"], per_field[f]["mismatch"]
        total = m + mm
        if total:
            rows.append({"field": f, "match_rate_%": round(100 * m / total, 1),
                         "checked": total, "mismatches": mm})
    return rows, status_counts


def render_metrics_tab():
    st.subheader("Match success rate")
    st.caption("How often each extracted field matches the customer table "
               "(known customers only).")

    results = st.session_state.results
    if not results:
        st.info("No results yet. Process a document or scan a folder first.")
        return

    field_rows, status_counts = compute_stats(results)

    matched = sum(1 for r in results if r.get("matched_name"))
    total_checked = sum(r["checked"] for r in field_rows)
    total_match = sum(int(round(r["match_rate_%"] / 100 * r["checked"]))
                      for r in field_rows)
    overall = 100 * total_match / total_checked if total_checked else 0.0

    unrelated = status_counts.get("unrelated", 0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall field match", f"{overall:.0f}%")
    c2.metric("Fields checked", total_checked)
    c3.metric("Matched customers", matched)
    c4.metric("Not in customer table", unrelated,
              help="No relation on name, DOB, or postcode — excluded from match figures.")

    if field_rows:
        st.markdown("**Match rate by field**")
        chart_df = pd.DataFrame(field_rows).set_index("field")[["match_rate_%"]]
        st.bar_chart(chart_df, y="match_rate_%", height=300)
        st.dataframe(pd.DataFrame(field_rows), use_container_width=True,
                     hide_index=True)
    else:
        st.info("No known customers matched yet, so no field accuracy to show.")

    st.markdown("**Documents by status**")
    status_df = pd.DataFrame(
        [{"status": k, "count": v} for k, v in status_counts.items() if v]
    )
    if not status_df.empty:
        st.bar_chart(status_df.set_index("status"), height=260)


def _render_db_panel(result):
    """Right-hand panel: what the customer table says vs what we extracted."""
    data = result.get("data") or {}
    if result.get("matched_name"):
        st.success(f"Matched customer: **{result['matched_name']}**")
        cmp_df = pd.DataFrame(result["rows"])[["field", "extracted", "expected", "status"]]
        cmp_df = cmp_df.rename(columns={
            "field": "Field", "extracted": "Extracted",
            "expected": "Expected", "status": "Status",
        })
        st.dataframe(
            cmp_df.style.apply(_highlight_status, axis=1),
            use_container_width=True, hide_index=True,
        )
        return

    if result.get("relation") == "none":
        st.warning("Suggested: no relation to the customer table (name, DOB, and "
                   "postcode all unmatched). Flagged for manual review — approve "
                   "or reject below.")
    elif result.get("relation") == "possible":
        st.warning("Name not matched, but DOB/postcode relates to a customer — "
                   "possible misread.")
    else:
        st.info("Not compared against the customer table.")

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
        columns=["Field", "Extracted value"],
    )
    ext_df["Extracted value"] = ext_df["Extracted value"].apply(
        lambda v: "—" if not v else str(v))
    st.table(ext_df)


def _build_zip(approved, rejected):
    """Zip up the approved document files plus approved/rejected manifests."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fname, path in approved.items():
            if path and os.path.exists(path):
                z.write(path, arcname=os.path.join("approved", fname))
        z.writestr("approved_documents.txt",
                   "\n".join(sorted(approved)) or "(none)")
        z.writestr("rejected_documents.txt",
                   "\n".join(f"{f}\t{reason}" for f, reason in sorted(rejected.items()))
                   or "(none)")
    buf.seek(0)
    return buf.getvalue()


def render_review_tab():
    st.subheader("Final review")
    st.caption("Manually approve or reject each document. Approved documents go "
               "into a downloadable pack; rejected ones are listed separately.")

    results = st.session_state.results
    if not results:
        st.info("No results yet. Process a document or scan a folder first.")
        return

    approved = st.session_state.approved
    rejected = st.session_state.rejected

    # Documents that passed fully (status "ok") are added straight to the pack
    # and are not surfaced for manual review.
    for r in results:
        if r["status"] == "ok" and r["filename"] not in rejected:
            approved.setdefault(r["filename"], r["path"])

    # Review queue: only documents that need a human look and aren't decided yet.
    pending = [
        r for r in results
        if r["status"] != "ok"
        and r["filename"] not in approved
        and r["filename"] not in rejected
    ]

    auto_ok = sum(1 for r in results if r["status"] == "ok")
    st.caption(
        f"{len(pending)} awaiting review · {len(approved)} approved "
        f"({auto_ok} auto-added as fully passed) · {len(rejected)} rejected"
    )

    if not pending:
        st.success(
            "Nothing left to review. Fully-passed documents were added to the "
            "pack automatically — download it below."
        )
    else:
        idx = max(0, min(st.session_state.get("review_idx", 0), len(pending) - 1))
        labels = [f"{r['filename']}  ·  {r['status']}" for r in pending]
        picked = st.selectbox("Awaiting review", labels, index=idx)
        idx = labels.index(picked)
        st.session_state.review_idx = idx
        result = pending[idx]

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Scanned document** — read regions highlighted")
            try:
                with st.spinner("Locating read regions…"):
                    img = annotate_read_regions(result["path"])
            except Exception:
                img = _oriented_image(result["path"])
            if img is not None:
                st.image(img, use_container_width=True)
            else:
                st.caption("(no preview available)")
        with right:
            st.markdown("**Database info**")
            _render_db_panel(result)

        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("✅ Approve → next", key="review_approve"):
            approved[result["filename"]] = result["path"]
            rejected.pop(result["filename"], None)
            st.rerun()
        if b2.button("❌ Reject → next", key="review_reject"):
            rejected[result["filename"]] = result.get("status", "rejected")
            approved.pop(result["filename"], None)
            st.rerun()

    st.divider()
    col_a, col_r = st.columns(2)
    with col_a:
        st.markdown(f"**Approved — in the pack ({len(approved)})**")
        if approved:
            st.dataframe(pd.DataFrame({"document": sorted(approved)}),
                         use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download approved pack (.zip)",
                data=_build_zip(approved, rejected),
                file_name="approved_documents.zip",
                mime="application/zip",
            )
        else:
            st.caption("None approved yet.")
    with col_r:
        st.markdown(f"**Rejected / not added ({len(rejected)})**")
        if rejected:
            st.dataframe(
                pd.DataFrame([{"document": f, "reason": reason}
                              for f, reason in sorted(rejected.items())]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("None rejected yet.")


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
(tab_process, tab_folder, tab_flagged, tab_metrics,
 tab_review, tab_issues) = st.tabs(
    ["Process Document", "Scan Folder", "Flagged Files",
     "Accuracy", "Final Review", "Issues to Review"]
)
with tab_process:
    render_process_tab()
with tab_folder:
    render_folder_tab()
with tab_flagged:
    render_flagged_tab()
with tab_metrics:
    render_metrics_tab()
with tab_review:
    render_review_tab()
with tab_issues:
    render_issues_tab()
