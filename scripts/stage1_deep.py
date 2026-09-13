"""Stage 1 (deep): full per-file and aggregate inspection of CIC-IDS2017 raw CSVs."""
import os
import sys
import glob
import time
import json
import warnings
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
warnings.filterwarnings("ignore")
START = time.time()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_dataset_root():
    for root, dirs, files in os.walk(BASE):
        for d in dirs:
            if "IDS2017" in d:
                return os.path.join(root, d)
    return None

ROOT = find_dataset_root()
FILES = sorted(glob.glob(os.path.join(ROOT, "*.csv")))

def detect_encoding(path):
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as fh:
                fh.read(200000)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "latin-1"

def downcast(df):
    for c in df.columns:
        if df[c].dtype == np.int64:
            mx = df[c].abs().max(skipna=True)
            if not pd.isna(mx) and np.isfinite(mx) and mx <= np.iinfo(np.int32).max:
                df[c] = df[c].astype(np.int32)
        elif df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df

files_report = {}
class_counts = {}
missing_tot = {}
inf_tot = {}
features_stats = {}
all_cols = None

for f in FILES:
    name = os.path.basename(f).split(".")[0]
    enc = detect_encoding(f)
    df = pd.read_csv(f, encoding=enc, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    n0 = len(df)
    files_report[name] = {"encoding": enc, "rows": n0, "cols": len(df.columns)}

    dup = int(df.duplicated().sum())
    files_report[name]["duplicates"] = dup

    missing = df.isna().sum()
    inf = np.isinf(df.select_dtypes(include=np.number)).sum()
    files_report[name]["missing_total"] = int(missing.sum())
    files_report[name]["inf_total"] = int(inf.sum())
    mis = missing[missing > 0]
    infc = inf[inf > 0]
    if len(mis):
        files_report[name]["missing_by_col"] = {c: int(v) for c, v in mis.items()}
    if len(infc):
        files_report[name]["inf_by_col"] = {c: int(v) for c, v in infc.items()}

    for c, v in missing.items():
        missing_tot[c] = missing_tot.get(c, 0) + int(v)
    for c, v in inf.items():
        inf_tot[c] = inf_tot.get(c, 0) + int(v)

    cnt = df["Label"].value_counts()
    for k, v in cnt.items():
        class_counts[str(k).strip()] = class_counts.get(str(k).strip(), 0) + int(v)

    if all_cols is None:
        all_cols = list(df.columns)

    pdf = downcast(df.copy())
    num = pdf.select_dtypes(include=np.number)
    for c in num.columns:
        s = num[c].dropna()
        fin = s[np.isfinite(s)]
        n = int(fin.count())
        st = features_stats.setdefault(c, {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": np.inf, "max": -np.inf})
        st["count"] += n
        st["sum"] += float(fin.sum())
        st["sumsq"] += float((fin.astype(np.float64) ** 2).sum())
        st["min"] = min(st["min"], float(fin.min()) if n else np.inf)
        st["max"] = max(st["max"], float(fin.max()) if n else -np.inf)

    print(f"[{name}] rows={n0:,} dup={dup:,} NaN={int(missing.sum()):,} "
          f"Inf={int(inf.sum()):,} enc={enc}", flush=True)
    del df, pdf

TOTAL = sum(r["rows"] for r in files_report.values())

gstats = []
for c, st in features_stats.items():
    if st["count"] == 0:
        continue
    mean = st["sum"] / st["count"]
    var = max(st["sumsq"] / st["count"] - mean ** 2, 0.0)
    gstats.append((c, st["count"], mean, np.sqrt(var), st["min"], st["max"]))
gdf = pd.DataFrame(gstats, columns=["feature", "count", "mean", "std", "min", "max"])
gdf = gdf.sort_values("feature")
os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
gdf.to_csv(os.path.join(BASE, "results", "stage1_numeric_stats.csv"), index=False)

print("\n=== AGGREGATE SUMMARY ===")
print(f"Total raw samples      : {TOTAL:,}")
print(f"Total feature columns  : {len(all_cols)}")

tot_lbl = sum(class_counts.values())
print(f"\nClass distribution (all {len(FILES)} files combined, {tot_lbl:,} labelled rows):")
for k in sorted(class_counts, key=lambda x: -class_counts[x]):
    print(f"  {k:<24} {class_counts[k]:>10,}  ({100.0*class_counts[k]/tot_lbl:6.2f}%)")

missing_any = {c: v for c, v in missing_tot.items() if v > 0}
inf_any = {c: v for c, v in inf_tot.items() if v > 0}
print(f"\nColumns with missing values: {len(missing_any)} (total cells: {sum(missing_any.values()):,})")
for c, v in sorted(missing_any.items(), key=lambda x: -x[1])[:12]:
    print(f"  {c:<45} {v:>10,}  ({100.0*v/(TOTAL*len(all_cols)):.3f}% of all cells)")
print(f"\nColumns with Inf values: {len(inf_any)} (total cells: {sum(inf_any.values()):,})")
for c, v in sorted(inf_any.items(), key=lambda x: -x[1])[:12]:
    print(f"  {c:<45} {v:>10,}")

print(f"\nTotal duplicate rows: {sum(r['duplicates'] for r in files_report.values()):,} "
      f"({100.0*sum(r['duplicates'] for r in files_report.values())/TOTAL:.2f}% of rows)")

const = [c for c, st in features_stats.items() if st["count"] > 0 and st["min"] == st["max"]]
print(f"\nConstant numeric columns (zero variance): {len(const)}")
for c in const:
    print("  ", c)

print("\nFull column list (normalised names):")
for i, c in enumerate(all_cols):
    print(f"  {i:3d}. {c}")

with open(os.path.join(BASE, "results", "stage1_inspect_summary.json"), "w", encoding="utf-8") as fh:
    json.dump({"files": files_report, "classes": class_counts, "total": TOTAL,
               "constant_cols": const,
               "missing_cols": missing_any, "inf_cols": inf_any}, fh, indent=2, default=str)

print("\nInspection finished in %.1f s" % (time.time() - START))