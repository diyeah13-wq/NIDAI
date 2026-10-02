#!/usr/bin/env python3
"""Stage D step 2 - verify the collected lab dataset BEFORE any retraining.

This script is READ-ONLY with respect to data/lab. It loads the raw lab flow
records and their manifests, validates them, applies the proposed sampling
plan, measures how far the lab data drifts from CIC-IDS2017, and writes a
report. It never writes into data/lab, never loads a model, and never fits
anything. If it finds a problem it says so rather than fixing it silently.

WHAT IT CHECKS
---------------
  A. structure    69 features, exact config.FEATURE_ORDER, labels in
                  CLASS_ORDER, label_code consistent, no NaN/Inf
  B. provenance   every record maps to a manifest, SHA-256 of the raw file
                  matches, record timestamps lie inside the run's window
  C. read-only    SHA-256 of every raw file re-checked at the end and proven
                  unchanged by this run
  D. sampling     scoreable filter -> exact dedup -> per-signature cap,
                  reporting the exact row cost of each stage
  E. collapse     the signature grouping must NOT merge genuinely different
                  PortScan flow types; distinct-type count is compared before
                  and after capping
  F. leakage      no feature vector is shared across two runs, and the proposed
                  run-disjoint holdout is verified to be run-disjoint
  G. drift        per-feature lab-vs-CIC distribution differences, ranked

USAGE
-----
    python scripts/verify_lab_dataset.py
    python scripts/verify_lab_dataset.py --cap 150 --cic test
    python scripts/verify_lab_dataset.py --out results/metrics/stage13_lab_verification.json
"""
import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.config import (CLASS_ORDER, FEATURE_ORDER, LABEL_GROUP_MAP,
                        PROCESSED_CLEAN, PROCESSED_TEST, PROCESSED_TRAIN)

LAB_DIR = os.path.join(BASE_DIR, "data", "lab")
RAW_GLOB = os.path.join(LAB_DIR, "raw", "*", "*.jsonl")
MANIFEST_DIR = os.path.join(LAB_DIR, "manifests")


def set_lab_dir(path):
    """Repoint the verifier at a different lab root (--lab-dir)."""
    global LAB_DIR, RAW_GLOB, MANIFEST_DIR
    LAB_DIR = os.path.abspath(path) if os.path.isabs(path) \
        else os.path.join(BASE_DIR, path)
    RAW_GLOB = os.path.join(LAB_DIR, "raw", "*", "*.jsonl")
    MANIFEST_DIR = os.path.join(LAB_DIR, "manifests")

# The live gate, as implemented in src/live/flow_features.py (Stage A).
GATE_MIN_FWD = 1
GATE_MIN_BWD = 1

# Interpretable coarse bins, in the units the features actually use.
DURATION_EDGES_US = (1e3, 1e4, 1e5, 1e6)   # 1ms, 10ms, 100ms, 1s
BYTE_EDGES = (64, 256, 1024, 4096)
PKT_CAP = 8                                 # packet counts saturate at 8

LAB_LABELS = ("BENIGN", "PortScan")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _bucket(value, edges):
    """0 for zero, else 1 + index of the first edge the value exceeds."""
    if not value or value <= 0:
        return 0
    for i, e in enumerate(edges):
        if value <= e:
            return i + 1
    return len(edges) + 1


def flow_signature(label, proto, f):
    """Coarse grouping key: what KIND of flow this is, not which scan it came from.

    Built only from the 69 features plus protocol and label, so it can be
    recomputed identically at training time. Deliberately coarse (flag presence,
    saturated packet counts, coarse duration/byte bins) so that near-duplicate
    probes from a single scan collapse together, while genuinely different flow
    shapes stay in different buckets.
    """
    return (
        label,
        int(proto),
        int(f["SYN Flag Count"] > 0), int(f["ACK Flag Count"] > 0),
        int(f["RST Flag Count"] > 0), int(f["FIN Flag Count"] > 0),
        int(f["PSH Flag Count"] > 0), int(f["URG Flag Count"] > 0),
        min(int(f["Total Fwd Packets"]), PKT_CAP),
        min(int(f["Total Backward Packets"]), PKT_CAP),
        _bucket(f["Flow Duration"], DURATION_EDGES_US),
        _bucket(f["Total Length of Fwd Packets"], BYTE_EDGES),
        _bucket(f["Total Length of Bwd Packets"], BYTE_EDGES),
    )


def flow_type(label, proto, f, verdict):
    """A FINER key than the signature, used to prove the signature is not
    collapsing genuinely different flow types."""
    return (
        label,
        verdict,
        int(f["SYN Flag Count"] > 0), int(f["RST Flag Count"] > 0),
        int(f["PSH Flag Count"] > 0), int(f["FIN Flag Count"] > 0),
        min(int(f["Total Fwd Packets"]), 4),
        min(int(f["Total Backward Packets"]), 4),
        _bucket(f["Flow Duration"], DURATION_EDGES_US),
    )


def sig_str(sig):
    return ("L=%s|p=%d|flags=S%dA%dR%dF%dP%dU%d|fwd=%d|bwd=%d|dur=%d|fb=%d|bb=%d" % sig)


def type_str(t):
    return ("L=%s|v=%s|S%dR%dP%dF%d|f=%d|b=%d|dur=%d" % t)


# --------------------------------------------------------------------------- #
# A. load + structure
# --------------------------------------------------------------------------- #
def load_lab():
    problems = []
    rows = []
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        raise SystemExit("No lab raw files found under %s. Run scripts/collect_lab.py "
                         "first." % LAB_DIR)

    for path in files:
        rel = os.path.relpath(path, BASE_DIR).replace("\\", "/")
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    problems.append("%s:%d bad JSON: %s" % (rel, lineno, exc))
                    continue
                rec["_file"] = rel
                rec["_line"] = lineno
                rows.append(rec)

    n_feat_mismatch = n_label_bad = n_code_bad = n_source_bad = 0
    n_nan = 0
    feature_matrix = np.empty((len(rows), len(FEATURE_ORDER)), dtype=np.float32)
    feature_keys = []

    for i, r in enumerate(rows):
        f = r.get("features")
        if not isinstance(f, dict) or list(f.keys()) != list(FEATURE_ORDER):
            n_feat_mismatch += 1
            vals = [float(f.get(n, float("nan"))) if isinstance(f, dict)
                    else float("nan") for n in FEATURE_ORDER]
        else:
            vals = [float(f[n]) for n in FEATURE_ORDER]
        arr = np.asarray(vals, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            n_nan += 1
        feature_matrix[i] = arr
        feature_keys.append(tuple(round(float(v), 6) for v in arr))

        lab = r.get("label")
        if lab not in LAB_LABELS:
            n_label_bad += 1
        elif r.get("label_code") != CLASS_ORDER.index(lab):
            n_code_bad += 1
        if r.get("label_source") != "executed_scenario":
            n_source_bad += 1

    df = pd.DataFrame({
        "run_id": [r.get("run_id") for r in rows],
        "scenario": [r.get("scenario") for r in rows],
        "label": [r.get("label") for r in rows],
        "verdict": [r.get("wire_verdict") for r in rows],
        "proto": [r.get("proto") for r in rows],
        "n_fwd": [r.get("n_fwd") for r in rows],
        "n_bwd": [r.get("n_bwd") for r in rows],
        "first_ts": [r.get("first_ts") for r in rows],
        "last_ts": [r.get("last_ts") for r in rows],
        "file": [r["_file"] for r in rows],
    })

    if n_feat_mismatch:
        problems.append("%d record(s) do not have features in exact "
                        "FEATURE_ORDER sequence" % n_feat_mismatch)
    if n_label_bad:
        problems.append("%d record(s) with a label outside %s" % (n_label_bad, LAB_LABELS))
    if n_code_bad:
        problems.append("%d record(s) where label_code disagrees with CLASS_ORDER"
                        % n_code_bad)
    if n_source_bad:
        problems.append("%d record(s) whose label_source is not executed_scenario"
                        % n_source_bad)
    if n_nan:
        problems.append("%d record(s) contain NaN or Inf" % n_nan)

    return df, feature_matrix, feature_keys, files, problems


def load_manifests():
    mans = {}
    bad = []
    for p in sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.json"))):
        with open(p, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        mans[m["run_id"]] = m
        raw = os.path.join(BASE_DIR, m["raw_file"])
        if not os.path.exists(raw):
            bad.append("manifest %s references missing raw file %s"
                       % (m["run_id"], m["raw_file"]))
        elif sha256_of(raw) != m["raw_file_sha256"]:
            bad.append("manifest %s SHA-256 does not match %s"
                       % (m["run_id"], m["raw_file"]))
    return mans, bad


# --------------------------------------------------------------------------- #
# D. sampling plan
# --------------------------------------------------------------------------- #
def apply_scoreable_gate(df):
    mask = ((df["n_fwd"] >= GATE_MIN_FWD) & (df["n_bwd"] >= GATE_MIN_BWD)).to_numpy()
    return mask


def apply_exact_dedup(feature_keys, labels, gate_mask):
    """Drop repeated 69-vectors. Report label conflicts instead of hiding them."""
    seen = {}
    keep = np.zeros(len(feature_keys), dtype=bool)
    conflicts = []
    for i, key in enumerate(feature_keys):
        if not gate_mask[i]:
            continue
        if key in seen:
            if labels[seen[key]] != labels[i]:
                conflicts.append({"features_hash": hash(key),
                                  "labels": sorted({labels[seen[key]], labels[i]})})
            continue
        seen[key] = i
        keep[i] = True
    return keep, len(seen), conflicts


def round_robin_cap(df, cap):
    """Cap each signature, taking rows round-robin across runs.

    Round-robin matters: a plain head-of-list cap would let whichever scan ran
    first fill the entire bucket, which would silently delete whole runs and
    break the run-disjoint holdout.
    """
    df = df.copy()
    df["sig"] = list(df["_signature"])
    kept_idx = []
    dropped_by_sig = {}
    for sig, grp in df.groupby("sig", sort=True):
        if len(grp) <= cap:
            kept_idx.extend(grp.index.tolist())
            continue
        by_run = [g.index.tolist() for _r, g in grp.groupby("run_id", sort=True)]
        picked = []
        round_no = 0
        while len(picked) < cap and any(round_no < len(b) for b in by_run):
            for b in by_run:
                if round_no < len(b) and len(picked) < cap:
                    picked.append(b[round_no])
            round_no += 1
        kept_idx.extend(picked)
        dropped_by_sig[sig] = len(grp) - len(picked)
    out = df.loc[sorted(kept_idx)].copy()
    return out, dropped_by_sig


# --------------------------------------------------------------------------- #
# G. drift
# --------------------------------------------------------------------------- #
def two_sample(a, b):
    """Cohen's d, median ratio, KS statistic and out-of-support fraction."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return None
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                       / max(na + nb - 2, 1)) if (na + 1 > 1 and nb + 1 > 1) else 0.0
    d = (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0
    ma, mb = float(np.median(a)), float(np.median(b))
    # KS via empirical CDF on a shared grid (no scipy dependency)
    grid = np.union1d(np.quantile(a, np.linspace(0, 1, 400)),
                      np.quantile(b, np.linspace(0, 1, 400)))
    fa = np.searchsorted(np.sort(a), grid, side="right") / na
    fb = np.searchsorted(np.sort(b), grid, side="right") / nb
    ks = float(np.max(np.abs(fa - fb)))
    lo, hi = float(a.min()), float(a.max())
    outside = float(np.mean((b < lo) | (b > hi))) if nb else 0.0
    return {
        "cic_n": na, "lab_n": nb,
        "cic_mean": float(a.mean()), "lab_mean": float(b.mean()),
        "cic_median": ma, "lab_median": mb,
        "median_ratio": (mb / ma if ma not in (0.0,) and ma != 0 else None),
        "cohens_d": float(d), "ks": ks,
        "cic_min": lo, "cic_max": hi,
        "lab_outside_cic_range": outside,
    }


def load_cic(which):
    path = {"test": PROCESSED_TEST, "train": PROCESSED_TRAIN,
            "clean": PROCESSED_CLEAN}[which]
    if not os.path.exists(path):
        raise SystemExit("CIC file not found: %s" % path)
    df = pd.read_csv(path, compression="gzip",
                     usecols=FEATURE_ORDER + ["LabelGroup"],
                     low_memory=False)
    for c in FEATURE_ORDER:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[FEATURE_ORDER] = df[FEATURE_ORDER].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify the collected lab dataset.")
    ap.add_argument("--cap", type=int, default=200,
                    help="max rows per flow signature (default: %(default)s)")
    ap.add_argument("--cic", choices=("test", "train", "clean"), default="test",
                    help="which CIC split to compare against (default: %(default)s)")
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="fraction of runs reserved as holdout (default: %(default)s)")
    ap.add_argument("--lab-dir", default="data/lab",
                    help="lab root to verify (default: %(default)s)")
    ap.add_argument("--out", default=os.path.join("results", "metrics",
                                                  "stage13_lab_verification.json"))
    args = ap.parse_args(argv)

    set_lab_dir(args.lab_dir)

    report = {"generated_at": datetime.datetime.now().astimezone().isoformat(),
              "cap": args.cap, "cic_reference": args.cic,
              "lab_dir": os.path.relpath(LAB_DIR, BASE_DIR).replace("\\", "/"),
              "gate": {"min_fwd": GATE_MIN_FWD, "min_bwd": GATE_MIN_BWD}}
    problems = []

    # ---- C. snapshot before -------------------------------------------------
    before = {os.path.relpath(p, BASE_DIR).replace("\\", "/"): sha256_of(p)
              for p in sorted(glob.glob(RAW_GLOB))
              + sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.json")))}

    print("=" * 78)
    print("  NIDAI Stage D step 2 - LAB DATASET VERIFICATION (read-only)")
    print("=" * 78)

    # ---- A. load ------------------------------------------------------------
    df, X, fkeys, files, probs = load_lab()
    problems += probs
    mans, man_problems = load_manifests()
    problems += man_problems
    raw_rows = len(df)
    print("\n[A] STRUCTURE")
    print("  raw files              : %d" % len(files))
    print("  raw rows               : %d" % raw_rows)
    print("  features per row       : %d (expected %d)" % (len(FEATURE_ORDER),
                                                          len(FEATURE_ORDER)))
    print("  feature order exact    : %s" % (not any("FEATURE_ORDER" in p for p in problems)))
    print("  labels present         : %s" % sorted(df["label"].unique()))
    print("  labels valid           : %s" % (not any("outside" in p for p in problems)))
    print("  label_source verified  : %s"
          % (not any("label_source" in p for p in problems)))
    print("  NaN / Inf rows         : 0" if not any("NaN" in p for p in problems)
          else "  NaN / Inf rows         : PRESENT")
    report["structure"] = {"raw_files": len(files), "raw_rows": raw_rows,
                           "features": len(FEATURE_ORDER),
                           "labels": sorted(df["label"].unique())}

    # ---- B. provenance ------------------------------------------------------
    print("\n[B] PROVENANCE")
    no_manifest = sorted(set(df["run_id"]) - set(mans))
    out_of_window = 0
    for run_id, grp in df.groupby("run_id"):
        m = mans.get(run_id)
        if m is None:
            continue
        lo, hi = m["t_start_epoch"], m["window_end_epoch"]
        out_of_window += int(((grp["first_ts"] < lo) | (grp["last_ts"] > hi)).sum())
    runs_with_data = df["run_id"].nunique()
    print("  manifests              : %d" % len(mans))
    print("  runs with rows         : %d" % runs_with_data)
    print("  rows with no manifest  : %d" % (raw_rows - int(df["run_id"].isin(mans).sum())))
    print("  manifest SHA-256 fails : %d" % len([p for p in man_problems if "SHA-256" in p]))
    print("  rows outside window    : %d" % out_of_window)
    if no_manifest:
        problems.append("runs present in data with no manifest: %s" % no_manifest)
    if out_of_window:
        problems.append("%d rows lie outside their run window" % out_of_window)
    report["provenance"] = {"manifests": len(mans), "runs_with_rows": runs_with_data,
                            "rows_without_manifest":
                                int(raw_rows - df["run_id"].isin(mans).sum()),
                            "sha256_failures": len([p for p in man_problems
                                                    if "SHA-256" in p]),
                            "rows_outside_window": out_of_window}

    # ---- D. sampling --------------------------------------------------------
    print("\n[D] SAMPLING PLAN")
    labels = df["label"].tolist()
    gate = apply_scoreable_gate(df)
    n_scoreable = int(gate.sum())

    keep, n_unique, conflicts = apply_exact_dedup(fkeys, labels, gate)
    df["_signature"] = [flow_signature(df["label"][i], df["proto"][i],
                                       dict(zip(FEATURE_ORDER, X[i])))
                        for i in range(len(df))]
    df["_type"] = [flow_type(df["label"][i], df["proto"][i],
                             dict(zip(FEATURE_ORDER, X[i])), df["verdict"][i])
                   for i in range(len(df))]

    dedup_df = df[keep].copy()
    n_dedup_removed = n_scoreable - n_unique
    capped_df, dropped_by_sig = round_robin_cap(dedup_df, args.cap)
    n_cap_removed = len(dedup_df) - len(capped_df)

    before_counts = df["label"].value_counts().to_dict()
    scoreable_counts = df[gate]["label"].value_counts().to_dict()
    dedup_counts = dedup_df["label"].value_counts().to_dict()
    final_counts = capped_df["label"].value_counts().to_dict()

    print("  raw rows                          : %6d" % raw_rows)
    print("  scoreable rows (fwd>=%d, bwd>=%d)     : %6d   (%d dropped)"
          % (GATE_MIN_FWD, GATE_MIN_BWD, n_scoreable, raw_rows - n_scoreable))
    print("  exact duplicates removed          : %6d   (%d unique remain)"
          % (n_dedup_removed, n_unique))
    print("  rows removed by signature cap     : %6d" % n_cap_removed)
    print("  EFFECTIVE SAMPLE COUNT            : %6d" % len(capped_df))
    print("  exact-duplicate label conflicts   : %6d" % len(conflicts))
    print("\n  %-14s %8s %8s %8s %8s" % ("label", "raw", "scoreable", "dedup", "final"))
    for lab in LAB_LABELS:
        print("  %-14s %8d %8d %8d %8d"
              % (lab, before_counts.get(lab, 0), scoreable_counts.get(lab, 0),
                 dedup_counts.get(lab, 0), final_counts.get(lab, 0)))
    print("  %-14s %8d %8d %8d %8d"
          % ("TOTAL", raw_rows, n_scoreable, len(dedup_df), len(capped_df)))
    if conflicts:
        problems.append("%d feature vectors appear with conflicting labels" % len(conflicts))
    report["sampling"] = {
        "raw_rows": raw_rows, "scoreable_rows": n_scoreable,
        "exact_duplicates_removed": n_dedup_removed,
        "unique_after_dedup": n_unique,
        "removed_by_signature_cap": n_cap_removed,
        "effective_sample_count": len(capped_df),
        "label_conflicts": len(conflicts),
        "counts": {"raw": before_counts, "scoreable": scoreable_counts,
                   "dedup": dedup_counts, "final": final_counts},
        "signatures_before_cap": int(dedup_df["_signature"].nunique()),
        "signatures_after_cap": int(capped_df["_signature"].nunique()),
        "top_signatures_removed": [{"signature": sig_str(k), "removed": v}
                                   for k, v in sorted(dropped_by_sig.items(),
                                                       key=lambda kv: -kv[1])[:10]],
    }

    # ---- E. collapse check --------------------------------------------------
    print("\n[E] SIGNATURE GROUPING - DOES IT COLLAPSE DIFFERENT FLOW TYPES?")
    types_before = set(dedup_df["_type"])
    types_after = set(capped_df["_type"])
    lost = types_before - types_after
    print("  distinct flow types before cap      : %d" % len(types_before))
    print("  distinct flow types after  cap      : %d" % len(types_after))
    print("  flow types LOST by capping          : %d" % len(lost))
    for t in sorted(lost)[:10]:
        print("      lost: %s" % type_str(t))
    sig_het = dedup_df.groupby("_signature")["_type"].nunique()
    print("  signatures                         : %d" % dedup_df["_signature"].nunique())
    print("  distinct types per signature (max) : %d" % (sig_het.max() if len(sig_het) else 0))
    print("  distinct types per signature (mean): %.2f"
          % (sig_het.mean() if len(sig_het) else 0.0))
    mixed = sig_het[sig_het > 1]
    print("  signatures holding >1 flow type    : %d (these are the risky ones)"
          % len(mixed))
    # PortScan wire-verdict survival
    psb = dedup_df[dedup_df["label"] == "PortScan"]["verdict"].value_counts().to_dict()
    psa = capped_df[capped_df["label"] == "PortScan"]["verdict"].value_counts().to_dict()
    print("  PortScan wire verdicts before      : %s" % psb)
    print("  PortScan wire verdicts after       : %s" % psa)
    ps_verdicts_lost = set(psb) - set(psa)
    if lost:
        problems.append("signature capping removed %d distinct PortScan flow type(s)"
                        % len(lost))
    if ps_verdicts_lost:
        problems.append("signature capping removed PortScan wire verdict(s): %s"
                        % sorted(ps_verdicts_lost))
    report["collapse_check"] = {
        "types_before": len(types_before), "types_after": len(types_after),
        "types_lost": [type_str(t) for t in sorted(lost)],
        "signatures": int(dedup_df["_signature"].nunique()),
        "max_types_per_signature": int(sig_het.max()) if len(sig_het) else 0,
        "mean_types_per_signature": float(sig_het.mean()) if len(sig_het) else 0.0,
        "signatures_with_multiple_types": int(len(mixed)),
        "worst_mixed_signatures": [
            {"signature": sig_str(s), "n_types": int(n),
             "types": [type_str(t) for t in
                       sorted(dedup_df[dedup_df["_signature"] == s]["_type"].unique())[:6]]}
            for s, n in mixed.sort_values(ascending=False).head(5).items()],
        "portscan_verdicts_before": psb, "portscan_verdicts_after": psa,
        "signature_composition": [
            {"signature": sig_str(s), "rows": int(n)}
            for s, n in dedup_df["_signature"].value_counts().head(10).items()],
    }

    # ---- F. leakage ---------------------------------------------------------
    print("\n[F] RUN LEAKAGE / SPLIT PLAN")
    lab_X = X[capped_df.index.to_numpy()]
    keys = [tuple(round(float(x), 6) for x in row) for row in lab_X]
    capped_df = capped_df.assign(_feature_key=keys)
    shared = capped_df.groupby("_feature_key")["run_id"].nunique()
    n_shared = int((shared > 1).sum())
    runs = sorted(capped_df["run_id"].unique())
    # Stratify the holdout by label. Sorting run_ids alphabetically would put
    # every benign_* run first and leave a PortScan-only holdout, which could
    # never measure the lab BENIGN false-positive rate.
    runs_by_label = {}
    for run_id, grp in capped_df.groupby("run_id"):
        runs_by_label.setdefault(grp["label"].mode().iloc[0], []).append(run_id)
    holdout = set()
    for lab, rs in sorted(runs_by_label.items()):
        rs = sorted(rs)
        n = max(1, int(round(len(rs) * args.holdout_frac)))
        holdout.update(rs[-n:])          # deterministic, tail of each label group
    train = [r for r in runs if r not in holdout]
    tr_df = capped_df[capped_df["run_id"].isin(train)]
    ho_df = capped_df[capped_df["run_id"].isin(holdout)]
    print("  runs available                     : %d" % len(runs))
    print("  train runs                         : %d" % len(train))
    print("  holdout runs                       : %d" % len(holdout))
    print("  vectors shared by >1 run           : %d  %s"
          % (n_shared, "(LEAKAGE RISK)" if n_shared else "(clean)"))
    print("  run overlap train/holdout          : %d (must be 0)"
          % len(set(train) & set(holdout)))
    print("  train rows   %d   PortScan %d / BENIGN %d"
          % (len(tr_df), int((tr_df["label"] == "PortScan").sum()),
             int((tr_df["label"] == "BENIGN").sum())))
    print("  holdout rows %d   PortScan %d / BENIGN %d"
          % (len(ho_df), int((ho_df["label"] == "PortScan").sum()),
             int((ho_df["label"] == "BENIGN").sum())))
    for lab in LAB_LABELS:
        if int((ho_df["label"] == lab).sum()) == 0:
            problems.append("holdout split contains no %s runs" % lab)
    if n_shared:
        problems.append("%d feature vectors are shared across runs" % n_shared)
    report["leakage"] = {"runs": len(runs), "train_runs": len(train),
                         "holdout_runs": len(holdout),
                         "vectors_shared_across_runs": n_shared,
                         "train_rows": len(tr_df), "holdout_rows": len(ho_df),
                         "train_label_counts": tr_df["label"].value_counts().to_dict(),
                         "holdout_label_counts": ho_df["label"].value_counts().to_dict(),
                         "holdout_runs": sorted(holdout)}

    # ---- cap sensitivity ----------------------------------------------------
    sens = []
    for c in (50, 100, 200, 400, 10 ** 9):
        capped_c, _ = round_robin_cap(dedup_df, c)
        vc = capped_c["label"].value_counts().to_dict()
        t_b = set(capped_c["_type"])
        sens.append({"cap": (None if c >= 10 ** 9 else c),
                     "effective": len(capped_c),
                     "PortScan": vc.get("PortScan", 0),
                     "BENIGN": vc.get("BENIGN", 0),
                     "types_kept": len(t_b),
                     "types_lost": len(types_before - t_b)})
    print("\n  cap sensitivity (effective rows / distinct flow types kept):")
    print("    %-12s %10s %10s %10s %8s %8s"
          % ("cap", "effective", "PortScan", "BENIGN", "types", "lost"))
    for s in sens:
        print("    %-12s %10d %10d %10d %8d %8d"
              % (s["cap"] if s["cap"] is not None else "no cap",
                 s["effective"], s["PortScan"], s["BENIGN"],
                 s["types_kept"], s["types_lost"]))
    report["cap_sensitivity"] = sens

    # ---- G. drift -----------------------------------------------------------
    print("\n[G] DRIFT: lab vs CIC-IDS2017 (%s)" % args.cic)
    cic = load_cic(args.cic)
    drift = []
    lab_X = X[capped_df.index.to_numpy()]
    lab_lab = capped_df["label"].to_numpy()
    for lab in LAB_LABELS:
        cic_c = cic[cic["LabelGroup"] == lab]
        mask = lab_lab == lab
        if len(cic_c) == 0 or mask.sum() == 0:
            continue
        for j, feat in enumerate(FEATURE_ORDER):
            st = two_sample(cic_c[feat].to_numpy(), lab_X[mask, j])
            if st:
                st.update({"label": lab, "feature": feat})
                drift.append(st)
    dd = pd.DataFrame(drift)
    dd["abs_d"] = dd["cohens_d"].abs()
    top = dd.sort_values(["label", "abs_d"], ascending=[True, False])

    for lab in LAB_LABELS:
        sub = top[top["label"] == lab]
        if not len(sub):
            continue
        print("\n  --- %s (lab n=%d vs CIC n=%d) ---"
              % (lab, int(sub["lab_n"].iloc[0]), int(sub["cic_n"].iloc[0])))
        print("  %-30s %10s %10s %8s %7s %9s"
              % ("feature", "cic_med", "lab_med", "|d|", "KS", "outside"))
        for _, r in sub.head(12).iterrows():
            print("  %-30s %10.4g %10.4g %8.2f %7.3f %8.1f%%"
                  % (r["feature"], r["cic_median"], r["lab_median"],
                     abs(r["cohens_d"]), r["ks"], 100 * r["lab_outside_cic_range"]))
    report["drift"] = {
        "cic_reference": args.cic,
        "cic_rows": int(len(cic)),
        "cic_counts": cic["LabelGroup"].value_counts().to_dict(),
        "top_features_by_abs_d": [
            {"label": r["label"], "feature": r["feature"],
             "abs_cohens_d": round(abs(r["cohens_d"]), 4),
             "cohens_d": round(r["cohens_d"], 4),
             "ks": round(r["ks"], 4),
             "cic_median": r["cic_median"], "lab_median": r["lab_median"],
             "lab_outside_cic_range_pct": round(100 * r["lab_outside_cic_range"], 2)}
            for _, r in top.iterrows()],
    }

    # ---- C. snapshot after --------------------------------------------------
    after = {os.path.relpath(p, BASE_DIR).replace("\\", "/"): sha256_of(p)
             for p in sorted(glob.glob(RAW_GLOB))
             + sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.json")))}
    changed = [k for k in before if before[k] != after.get(k)]
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    print("\n[C] READ-ONLY PROOF")
    print("  lab files hashed before/after     : %d" % len(before))
    print("  files modified by this verifier   : %d" % len(changed))
    print("  files added by this verifier      : %d" % len(added))
    print("  files deleted by this verifier    : %d" % len(removed))
    if changed or added or removed:
        problems.append("verifier altered data/lab: changed=%s added=%s removed=%s"
                        % (changed, added, removed))
    report["read_only_proof"] = {"files": len(before), "changed": changed,
                                 "added": added, "removed": removed}

    # ---- verdict ------------------------------------------------------------
    print("\n" + "=" * 78)
    if problems:
        print("  PROBLEMS FOUND (%d)" % len(problems))
        for p in problems:
            print("    ! %s" % p)
    else:
        print("  NO PROBLEMS FOUND - dataset is internally consistent")
    print("=" * 78)
    report["problems"] = problems
    report["verdict"] = "PASS" if not problems else "PASS_WITH_PROBLEMS"

    out = os.path.join(BASE_DIR, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("\nreport written: %s" % os.path.relpath(out, BASE_DIR))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())