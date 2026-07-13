"""Auditor dashboard for the document extraction pipeline.

A small Tkinter UI with two tabs:

  1. "Process Document" — pick or type the path to a document image, run the
     bank-compliant extraction (document + OCR only), then compare the result
     against the customer table and show a per-field match/mismatch breakdown.

  2. "Issues to Review" — a running list of everything that needs a human eye:
     field mismatches, OCR self-consistency warnings, and people not found in
     the customer table. Can also be reloaded from the saved audit CSVs.

The customer table is only ever read in the comparison/audit step, never during
extraction, so the extraction stage stays bank-compliant.
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

from final import (
    process_file,
    compare_record,
    extract_customer_id,
    extract_document_type,
)

FIELDS = ["name", "date_of_birth", "address", "occupation", "employer"]
TRUTH_PATH = "customer_table.csv"


class Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Extraction Auditor")
        self.root.geometry("980x680")

        self._preview_img = None  # keep a reference so the image isn't GC'd

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.process_tab = ttk.Frame(notebook)
        self.issues_tab = ttk.Frame(notebook)
        notebook.add(self.process_tab, text="Process Document")
        notebook.add(self.issues_tab, text="Issues to Review")

        self._build_process_tab()
        self._build_issues_tab()

    # ── Process tab ────────────────────────────────────────────────────────
    def _build_process_tab(self):
        top = ttk.Frame(self.process_tab)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(top, text="Document:").pack(side="left")
        self.path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.path_var).pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(top, text="Browse…", command=self._browse).pack(side="left")
        self.run_btn = ttk.Button(top, text="Run", command=self._run_clicked)
        self.run_btn.pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="Select a document to begin.")
        ttk.Label(self.process_tab, textvariable=self.status_var,
                  foreground="#555").pack(anchor="w", padx=10)

        body = ttk.Frame(self.process_tab)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Left: image preview
        left = ttk.LabelFrame(body, text="Preview")
        left.pack(side="left", fill="both", expand=False)
        self.preview_label = ttk.Label(left)
        self.preview_label.pack(padx=6, pady=6)

        # Right: extracted values + comparison
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        extracted_box = ttk.LabelFrame(right, text="Extracted values (document only)")
        extracted_box.pack(fill="x")
        self.extracted_vars = {}
        for i, field in enumerate(FIELDS):
            ttk.Label(extracted_box, text=field).grid(
                row=i, column=0, sticky="w", padx=6, pady=2
            )
            var = tk.StringVar()
            ttk.Entry(extracted_box, textvariable=var, state="readonly",
                      width=48).grid(row=i, column=1, sticky="w", padx=6, pady=2)
            self.extracted_vars[field] = var

        cmp_box = ttk.LabelFrame(right, text="Comparison vs customer table (audit)")
        cmp_box.pack(fill="both", expand=True, pady=(10, 0))
        cols = ("field", "extracted", "expected", "status")
        self.cmp_tree = ttk.Treeview(cmp_box, columns=cols, show="headings", height=6)
        for c, w in zip(cols, (110, 260, 260, 90)):
            self.cmp_tree.heading(c, text=c.capitalize())
            self.cmp_tree.column(c, width=w, anchor="w")
        self.cmp_tree.tag_configure("mismatch", background="#ffe0e0")
        self.cmp_tree.tag_configure("match", background="#e2f6e2")
        self.cmp_tree.pack(fill="both", expand=True, padx=6, pady=6)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Choose a document image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.path_var.set(path)

    def _run_clicked(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Choose a document first.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Not found", f"File does not exist:\n{path}")
            return

        self.run_btn.config(state="disabled")
        self.status_var.set("Processing… (rotation, OCR, extraction)")
        self._show_preview(path)
        threading.Thread(target=self._run_pipeline, args=(path,), daemon=True).start()

    def _run_pipeline(self, path):
        try:
            data, ocr_text, warnings = process_file(path)
        except Exception as exc:  # surface any API/OCR error in the UI
            msg = str(exc)
            self.root.after(0, lambda: self._pipeline_failed(msg))
            return

        matched_name, cmp_rows = None, []
        cmp_error = None
        if data is not None:
            if os.path.exists(TRUTH_PATH):
                try:
                    cmp_rows, matched_name = compare_record(data, TRUTH_PATH)
                except Exception as exc:
                    cmp_error = str(exc)
            else:
                cmp_error = f"{TRUTH_PATH} not found — comparison skipped."

        self.root.after(
            0,
            lambda: self._pipeline_done(
                path, data, warnings, cmp_rows, matched_name, cmp_error
            ),
        )

    def _pipeline_failed(self, msg):
        self.run_btn.config(state="normal")
        self.status_var.set("Failed.")
        messagebox.showerror("Extraction failed", msg)

    def _pipeline_done(self, path, data, warnings, cmp_rows, matched_name, cmp_error):
        self.run_btn.config(state="normal")
        filename = os.path.basename(path)

        if data is None:
            self.status_var.set("No data extracted from this document.")
            self._add_issue(filename, "-", "extraction failed",
                            "No JSON returned by the model")
            return

        for field in FIELDS:
            value = data.get(field)
            self.extracted_vars[field].set("" if value is None else str(value))

        for item in self.cmp_tree.get_children():
            self.cmp_tree.delete(item)

        mismatch_count = 0
        if matched_name is None:
            self.status_var.set(
                f"Extracted {filename}. Not found in customer table — logged as unknown."
            )
            self._add_issue(filename, "name", data.get("name", ""),
                            "Not found in customer table")
        else:
            for row in cmp_rows:
                self.cmp_tree.insert(
                    "", "end",
                    values=(row["field"], row["extracted"], row["expected"], row["status"]),
                    tags=(row["status"],),
                )
                if row["status"] == "mismatch":
                    mismatch_count += 1
                    self._add_issue(filename, row["field"], row["extracted"],
                                    f"expected: {row['expected']}")
            checked = len(cmp_rows)
            acc = 100.0 * (checked - mismatch_count) / checked if checked else 100.0
            status = f"Matched '{matched_name}'. {mismatch_count} mismatch(es) of {checked} fields ({acc:.0f}% match)."
            if cmp_error:
                status += f"  [{cmp_error}]"
            self.status_var.set(status)

        for w in warnings:
            self._add_issue(filename, "OCR check", "", w.strip())

    def _show_preview(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((360, 480))
            self._preview_img = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self._preview_img)
        except Exception:
            self.preview_label.config(image="", text="(no preview)")

    # ── Issues tab ─────────────────────────────────────────────────────────
    def _build_issues_tab(self):
        bar = ttk.Frame(self.issues_tab)
        bar.pack(fill="x", padx=10, pady=8)
        ttk.Button(bar, text="Reload from audit CSVs",
                   command=self._load_from_csvs).pack(side="left")
        ttk.Button(bar, text="Clear", command=self._clear_issues).pack(side="left", padx=6)
        self.issue_count_var = tk.StringVar(value="0 issues")
        ttk.Label(bar, textvariable=self.issue_count_var).pack(side="right")

        cols = ("customer_id", "document_type", "filename", "field", "extracted", "note")
        self.issues_tree = ttk.Treeview(self.issues_tab, columns=cols, show="headings")
        widths = (90, 110, 200, 110, 180, 220)
        for c, w in zip(cols, widths):
            self.issues_tree.heading(c, text=c.replace("_", " ").capitalize())
            self.issues_tree.column(c, width=w, anchor="w")
        self.issues_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _add_issue(self, filename, field, extracted, note):
        self.issues_tree.insert(
            "", "end",
            values=(
                extract_customer_id(filename) or "-",
                extract_document_type(filename),
                filename,
                field,
                extracted,
                note,
            ),
        )
        self._update_issue_count()

    def _clear_issues(self):
        for item in self.issues_tree.get_children():
            self.issues_tree.delete(item)
        self._update_issue_count()

    def _update_issue_count(self):
        n = len(self.issues_tree.get_children())
        self.issue_count_var.set(f"{n} issue{'s' if n != 1 else ''}")

    def _load_from_csvs(self):
        import pandas as pd

        self._clear_issues()
        loaded = False
        if os.path.exists("known_mismatches.csv"):
            try:
                df = pd.read_csv("known_mismatches.csv")
                for _, r in df.iterrows():
                    self.issues_tree.insert("", "end", values=(
                        r.get("customer_id", "-"),
                        r.get("document_type", ""),
                        r.get("filename", ""),
                        r.get("field", ""),
                        r.get("extracted_value", ""),
                        f"expected: {r.get('expected_value', '')}",
                    ))
                loaded = True
            except Exception as exc:
                messagebox.showerror("Load error", f"known_mismatches.csv: {exc}")
        if os.path.exists("unknown_customers.csv"):
            try:
                df = pd.read_csv("unknown_customers.csv")
                for _, r in df.iterrows():
                    self.issues_tree.insert("", "end", values=(
                        r.get("customer_id", "-"),
                        r.get("document_type", ""),
                        r.get("filename", ""),
                        "name",
                        r.get("extracted_name", ""),
                        r.get("reason", "Not found in customer table"),
                    ))
                loaded = True
            except Exception as exc:
                messagebox.showerror("Load error", f"unknown_customers.csv: {exc}")

        self._update_issue_count()
        if not loaded:
            messagebox.showinfo(
                "Nothing to load",
                "No audit CSVs found yet. Process some documents first, "
                "or run final.py for a full batch.",
            )


def main():
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
