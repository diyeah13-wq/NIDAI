"""Stage 1: Inspect the existing CIC-IDS2017 raw dataset (memory-efficient)."""
import os
import sys
import glob
import time
import pandas as pd
import numpy as np

DATA_RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

def find_dataset_root():
    start = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for root, dirs, files in os.walk(start):
        for d in dirs:
            if "IDS2017" in d:
                return os.path.join(root, d)
    return None

def main():
    t0 = time.time()
    print("=" * 78)
    print("STAGE 1 - RAW DATASET INSPECTION")
    print("=" * 78)

    root = find_dataset_root()
    if root is None:
        print("\n[WARNING] No CIC-IDS2017 directory found under project. Scanning home too...")
        return
    print(f"\nDataset directory: {root}")

    csv_files = sorted(glob.glob(os.path.join(root, "*.csv")))
    print(f"CSV files found: {len(csv_files)}")
    for f in csv_files:
        print("  -", os.path.basename(f))

    if not csv_files:
        return

    # ---- Per-file basic scan (header + row count) ----
    print("\n" + "-" * 78)
    print("PER-FILE SCAN  (header + row count only)")
    print("-" * 78)

    file_info = {}
    for f in csv_files:
        name = os.path.basename(f)
        fn = name.split(".")[0]
        try:
            with open(f, "r", encoding="latin-1") as fh:
                header = fh.readline().strip().split(",")
                ncols = len(header)
            # count rows without loading into df
            nlines = sum(1 for _ in open(f, "r", encoding="latin-1", errors="ignore"))
            nrows = nlines - 1  # minus header
        except Exception as e:
            print(f"  {name}: ERROR {e}")
            continue
        missing_na = [c for c in header if c.upper().strip() in ("", "N/A", "NAN")]
        # detect spaces / empty names
        spaced = [c for c in header if c != c.strip() or not c]
        file_info[fn] = {"path": f, "rows": nrows, "cols": ncols, "header": header}
        print(f"\n  {name}")
        print(f"    rows        : {nrows:,}")
        print(f"    columns     : {ncols}")
        print(f"    col['Flow ID'] present    : {'Flow ID' in header}")
        print(f"    col[' Label'] present     : {' Label' in header or 'Label' in header}")
        print(f"    cols w/ whitespace/empty  : {len(spaced)}")
        if len(spaced):
            print(f"      -> {spaced[:8]}{' ...' if len(spaced) > 8 else ''}")
        label_col = " Label" if " Label" in header else ("Label" if "Label" in header else None)
        if label_col:
            # quick label tally by sampling the CSV text (cheap-ish)
            lc = label_col.replace(",", "")
            # find its index from header
            idx = header.index(label_col)
            counts = {}
            with open(f, "r", encoding="latin-1", errors="ignore") as fh:
                next(fh)
                for i, line in enumerate(fh):
                    try:
                        parts = line.rstrip("\n").rsplit(",", 1)  # label is last col in ISCX
                        lbl = parts[1] if len(parts) == 2 else "?"
                    except Exception:
                        lbl = "?"
                    counts[lbl] = counts.get(lbl, 0) + 1
            file_info[fn]["labels"] = counts
            kl = sorted(counts.keys())
            tot = sum(counts.values())
            print("    label column: " + label_col)
            print("    unique labels:", len(kl))
            for k in kl:
                pct = 100.0 * counts[k] / tot if tot else 0
                print(f"      {k:<18} {counts[k]:>10,}  ({pct:5.2f}%)")

    print("\nTotal time (scan incl. label tallies): %.1f s" % (time.time() - t0))

if __name__ == "__main__":
    main()