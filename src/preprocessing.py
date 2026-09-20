"""Preprocessing pipeline: clean, normalize, log every removal, save splits."""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (DATA_PROCESSED, PROCESSED_CLEAN, PROCESSED_TRAIN,
                    PROCESSED_TEST, PROCESSED_DEMO, FEATURE_ORDER, CLASS_ORDER,
                    LABEL_GROUP_MAP, RANDOM_STATE, TEST_SIZE)
from data_loader import load_raw_file, list_raw_files, find_dataset_directory

LOG_STAGES = ["loaded_rows", "nan_cells_replaced", "inf_cells_winsorized",
              "duplicate_rows_removed", "final_rows"]
per_file_log = []


_DASH_VARIANTS = "\u2013\u2014\u2015\u2212\uFF0D\u2010\u2043"  # en/em/figure/minus/fullwidth/hyphen/hyphenbullet


def _clean_str_label(s):
    s = str(s).strip()
    s = s.replace("\ufffd", "-")
    for ch in _DASH_VARIANTS:
        s = s.replace(ch, "-")
    s = s.replace("\u00a0", " ")
    s = " ".join(s.split())
    return s


def _fix_non_finite(df):
    num = df.select_dtypes(include=[np.number]).columns
    nan_cells = 0
    inf_cells = 0
    for c in num:
        n_inf = int(np.isinf(df[c]).sum())
        n_nan = int(df[c].isna().sum())
        if n_inf > 0:
            mx = df.loc[np.isfinite(df[c]), c]
            mx = mx.max() if len(mx) else 0.0
            mask = np.isinf(df[c])
            df.loc[mask, c] = float(mx)
            inf_cells += n_inf
        if df[c].isna().any():
            df[c] = df[c].fillna(0.0)
            nan_cells += int(df[c].isna().sum())
    return nan_cells, inf_cells


def _downcast(df):
    for c in df.select_dtypes(include="number").columns:
        if str(df[c].dtype) == "int64":
            m = df[c].abs().max(skipna=True)
            if pd.notna(m) and np.isfinite(m) and m <= np.iinfo(np.int32).max:
                df[c] = df[c].astype(np.int32)
        elif str(df[c].dtype) == "float64":
            df[c] = df[c].astype(np.float32)
    return df


def clean_file(path):
    name = os.path.basename(path).split(".")[0]
    df = load_raw_file(path)
    n0 = len(df)

    label_col = "Label" if "Label" in df.columns else None
    missing_labels = label_col is None
    if missing_labels:
        df["Label"] = "UNKNOWN"

    df = df[[c for c in FEATURE_ORDER if c in df.columns] + [label_col]]
    for c in FEATURE_ORDER:
        if c not in df.columns:
            df[c] = 0.0
    df = df[FEATURE_ORDER + ["Label"]]
    df[FEATURE_ORDER] = df[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce")

    nan_cells, inf_cells = _fix_non_finite(df)
    df = _downcast(df)

    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    df["Label"] = df["Label"].map(_clean_str_label)
    df["LabelGroup"] = df["Label"].map(LABEL_GROUP_MAP).fillna("UNKNOWN")
    n_unk = int((df["LabelGroup"] == "UNKNOWN").sum())
    if n_unk:
        df.loc[df["LabelGroup"] == "UNKNOWN", "LabelGroup"] = "UNKNOWN"

    per_file_log.append({
        "file": name,
        "loaded_rows": n0,
        "nan_cells_replaced": nan_cells,
        "inf_cells_winsorized": inf_cells,
        "duplicate_rows_removed": n_dup,
        "final_rows": len(df),
        "unknown_labels_after_map": n_unk,
    })
    return df


def run_preprocessing():
    root = find_dataset_directory()
    print("Dataset directory: %s" % root)
    files = list_raw_files(root)
    print("Raw CSV files: %d" % len(files))
    per_file_log.clear()

    combined = None
    for f in files:
        df = clean_file(f)
        print("[%s] loaded=%d -> final=%d" % (
            per_file_log[-1]["file"], per_file_log[-1]["loaded_rows"],
            per_file_log[-1]["final_rows"]))
        combined = df if combined is None else pd.concat([combined, df], ignore_index=True)
        del df

    combined = combined.reset_index(drop=True)

    print("\n=== STAGE-BY-STAGE ROW LOG (per file) ===")
    print("%-52s %10s %12s %12s %12s %12s" % (
        "file", "loaded", "NaNfix(c)", "Inffix(c)", "dups", "final"))
    for r in per_file_log:
        print("%-52s %10d %12d %12d %12d %12d" % (
            r["file"][:52], r["loaded_rows"], r["nan_cells_replaced"],
            r["inf_cells_winsorized"], r["duplicate_rows_removed"], r["final_rows"]))

    print("\n=== COMBINED ===")
    print("Cleaned rows: %d  (%d removed = %.2f%%)" % (
        len(combined), sum(r["loaded_rows"] for r in per_file_log) - len(combined),
        100.0 * (sum(r["loaded_rows"] for r in per_file_log) - len(combined)) / sum(r["loaded_rows"] for r in per_file_log)))
    print("Features: %d" % len(FEATURE_ORDER))
    print("Class distribution:")
    cd = combined["LabelGroup"].value_counts()
    for c in CLASS_ORDER:
        if c in cd.index:
            print("  %-12s %10d  (%5.2f%%)" % (c, int(cd[c]), 100.0 * cd[c] / len(combined)))

    combined.to_csv(PROCESSED_CLEAN, index=False, compression="gzip")

    X = combined[FEATURE_ORDER]
    y = combined["LabelGroup"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    tr = pd.concat([Xtr, ytr.to_frame("LabelGroup")], axis=1)
    te = pd.concat([Xte, yte.to_frame("LabelGroup")], axis=1)
    tr.to_csv(PROCESSED_TRAIN, index=False, compression="gzip")
    te.to_csv(PROCESSED_TEST, index=False, compression="gzip")

    demo = _proportional_sample(combined, n=40000)
    demo.to_csv(PROCESSED_DEMO, index=False)
    print("\nSaved:")
    print("  cleaned :", PROCESSED_CLEAN)
    print("  train   :", PROCESSED_TRAIN, "(%d rows)" % len(tr))
    print("  test    :", PROCESSED_TEST, "(%d rows)" % len(te))
    print("  demo    :", PROCESSED_DEMO, "(%d rows, stratified by LabelGroup)" % len(demo))
    print("Demo class mix:")
    for c in CLASS_ORDER:
        if c in demo["LabelGroup"].value_counts().index:
            print("  %-12s %6d" % (c, int(demo["LabelGroup"].value_counts()[c])))

    info = {
        "dataset": "CIC-IDS2017",
        "raw_files": len(files),
        "raw_rows": int(sum(r["loaded_rows"] for r in per_file_log)),
        "cleaned_rows": int(len(combined)),
        "features": len(FEATURE_ORDER),
        "removed_rows": int(sum(r["loaded_rows"] for r in per_file_log) - len(combined)),
        "class_counts": {k: int(v) for k, v in cd.items()},
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "demo_rows": int(len(demo)),
        "per_file_log": per_file_log,
        "note": "Values derived from each live CSV file; 'Infinity' literals were "
                "winsorized to the column maximum; remaining NaN (Flow Bytes/s of "
                "0-duration flows) filled with 0; no silent deletions.",
    }
    with open(os.path.join(DATA_PROCESSED, "processed_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, default=str)
    print("\nWrote processed_info.json")


def _proportional_sample(df, n):
    total = len(df)
    parts = []
    for g, sub in df.groupby("LabelGroup"):
        take = max(1, int(round(n * len(sub) / total)))
        take = min(take, len(sub))
        parts.append(sub.sample(take, random_state=RANDOM_STATE))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)


if __name__ == "__main__":
    run_preprocessing()