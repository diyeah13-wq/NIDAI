"""Stage 9 - cross-dataset transfer: do detectors built on one benchmark fire on another?

CIC-IDS2017 and UNSW-NB15 share almost nothing at the schema level (69 vs 49
features, different names/definitions/classes). To make transfer measurable we
cut both down to the same 10 "packet-level anatomy" features, defined so both
sides use the same construction:

   1. Flow Duration            dur
   2. Total Fwd Packets        spkts
   3. Total Backward Packets   dpkts
   4. Total Length of Fwd Pkts sbytes
   5. Total Length of Bwd Pkts dbytes
   6. Flow Bytes/s             (sbytes + dbytes) / dur
   7. Flow Packets/s           (spkts + dpkts) / dur
   8. Fwd Packet Length Mean   smean
   9. Bwd Packet Length Mean   dmean

Rows 6-7 are re-derived on both sides so the definition cannot drift. Classes
are reduced to the only comparable signal: benign (0) vs attack (1). UNSW's
nine attack categories still matter, so we record recall per category after a
cross-domain run.

(UNSW processed splits carry no raw port column, so the bridge is 9 features;
the Destination Port feature used in Studies 4-7 has no UNSW counterpart.)

The experiment is symmetric: fit a balanced HGB (same hyperparams as Stage 7)
on each dataset's training sample, then evaluate in-domain and cross-domain,
and compare AUC / attack recall / false-alarm rate.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (MODEL_SAMPLE_SIZE, RANDOM_STATE, CLASS_ORDER,
                    FEATURE_ORDER, METRICS_DIR, PLOTS_DIR)
from common import load_train, load_test, rare_aware_sample

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNSW_DIR = os.path.join(BASE_DIR, "UNSW-NB15 — secondary dataset")

COMMON_NAMES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Bytes/s", "Flow Packets/s",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean",
]

UNSW_PRIMS = {
    "Flow Duration": "dur",
    "Total Fwd Packets": "spkts",
    "Total Backward Packets": "dpkts",
    "Total Length of Fwd Packets": "sbytes",
    "Total Length of Bwd Packets": "dbytes",
    "Fwd Packet Length Mean": "smean",
    "Bwd Packet Length Mean": "dmean",
}

STRAT_KEEP = ["Backdoors", "Shellcode", "Worms", "Analysis", "Reconnaissance"]


def hgb():
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
        min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE)


# ----------------------------- UNSW loading -----------------------------


def load_unsw(is_train):
    """Return (X float32, y binary, attack_cat array) for the UNSW split."""
    path = os.path.join(UNSW_DIR, "UNSW_NB15_training-set.csv" if is_train
                        else "UNSW_NB15_testing-set.csv")
    df = pd.read_csv(path, usecols=["dur", "spkts", "dpkts", "sbytes",
                                    "dbytes", "smean", "dmean",
                                    "attack_cat", "label"],
                     dtype={"spkts": "float32", "dpkts": "float32",
                            "sbytes": "float32", "dbytes": "float32",
                            "smean": "float32", "dmean": "float32",
                            "dur": "float32"})
    cat = df["attack_cat"].astype(str).str.strip().to_numpy()
    y = (cat != "Normal").astype(int)

    prims = df[["dur", "spkts", "dpkts", "sbytes", "dbytes",
                "smean", "dmean"]].to_numpy(np.float64)
    dur = np.maximum(prims[:, 0], 1e-6)
    feats = np.column_stack([
        prims[:, 0],                 # Flow Duration
        prims[:, 1],                 # Total Fwd Packets
        prims[:, 2],                 # Total Backward Packets
        prims[:, 3],                 # Total Length of Fwd Packets
        prims[:, 4],                 # Total Length of Bwd Packets
        (prims[:, 3] + prims[:, 4]) / dur,   # Flow Bytes/s
        (prims[:, 1] + prims[:, 2]) / dur,   # Flow Packets/s
        prims[:, 5],                 # Fwd Packet Length Mean
        prims[:, 6],                 # Bwd Packet Length Mean
    ])
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats.astype(np.float32), y, cat


def strat_sample(cat, n=MODEL_SAMPLE_SIZE):
    """n-row sample keeping EVERY row of the rarest UNSW attack categories."""
    idx = np.arange(len(cat))
    keep = np.isin(cat, STRAT_KEEP)
    rest = idx[~keep]
    take = n - int(keep.sum())
    if take < len(rest):
        rng = np.random.RandomState(RANDOM_STATE)
        rest = rng.choice(rest, size=take, replace=False)
    return np.sort(np.concatenate([idx[keep], rest]))


def cic_common(X):
    """Select the 10 common columns from a full 69-feature CIC array."""
    cols = [FEATURE_ORDER.index(n) for n in COMMON_NAMES]
    return X[:, cols]


# ----------------------------- evaluation -----------------------------


def binary_metrics(y_true, y_pred):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "accuracy": round((tp + tn) / len(y_true), 4),
        "attack_recall": round(rec, 4),
        "attack_precision": round(prec, 4),
        "attack_f1": round(f1, 4),
        "false_alarm_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "predicted_attack": int(tp + fp),
        "true_attack": int(tp + fn),
        "true_benign": int(tn + fp),
    }


def auc(y_true, proba):
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(y_true, proba)), 4)


def save_binary(name, metrics, extra):
    payload = {"experiment": name}
    payload.update(metrics)
    payload["extra"] = extra
    path = os.path.join(METRICS_DIR, name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    return path


def per_category(prob, y_true, out_cat):
    out = {}
    for c in np.unique(out_cat):
        mask = out_cat == c
        pred = prob[mask] >= 0.5
        out[str(c)] = {
            "n": int(mask.sum()),
            "predicted_attack": int(pred.sum()),
            "recall": round(float(pred.mean()), 4) if mask.any() else 0.0,
            "true_attack": int(y_true[mask].sum()),
        }
    return out


def fit_and_run(name, xtr, ytr, in_X, in_y, out_X, out_y, out_cat=None):
    t0 = time.time()
    clf = hgb()
    clf.fit(xtr, ytr)
    fit_t = time.time() - t0

    p_in = clf.predict_proba(in_X)[:, 1]
    p_out = clf.predict_proba(out_X)[:, 1]
    m_in = binary_metrics(in_y, (p_in >= 0.5).astype(int))
    m_out = binary_metrics(out_y, (p_out >= 0.5).astype(int))
    m_in["auc"] = auc(in_y, p_in)
    m_out["auc"] = auc(out_y, p_out)

    extra = {"fit_seconds": round(fit_t, 1),
             "common_features": COMMON_NAMES}
    save_binary(name + "_in", m_in, dict(extra, domain="in-domain"))
    save_binary(name + "_cross", m_out, dict(extra, domain="cross-domain"))
    if out_cat is not None:
        save_binary(name + "_per_category", per_category(p_out, out_y, out_cat),
                    dict(extra, domain="cross-domain"))
    return m_in, m_out


def main():
    t0 = time.time()

    print("Loading CIC train/test -> 9 common features...", flush=True)
    Xtr_c, ytr_c, _ = load_train()
    idx = rare_aware_sample(
        pd.DataFrame(ytr_c).assign(LabelGroup=ytr_c),
        MODEL_SAMPLE_SIZE).index.to_numpy()
    Xc = cic_common(Xtr_c)[idx]
    del Xtr_c
    yc = (ytr_c != CLASS_ORDER.index("BENIGN")).astype(int)[idx]

    Xte_c, yte_c, _ = load_test()
    Xc_test = cic_common(Xte_c)
    yc_test = (yte_c != CLASS_ORDER.index("BENIGN")).astype(int)
    del Xte_c
    print("CIC sample=%d  CIC test=%d" % (len(idx), len(yc_test)), flush=True)

    print("\nLoading UNSW train/test...", flush=True)
    Xu, yu, cu = load_unsw(is_train=True)
    uidx = strat_sample(cu)
    Xus, yus = Xu[uidx], yu[uidx]
    Xut, yut, cut = load_unsw(is_train=False)
    print("UNSW sample=%d (rare cats kept)  UNSW test=%d" % (
        len(uidx), len(yut)), flush=True)

    summary = {}

    print("\n=== Detector trained on CIC-IDS2017 ===", flush=True)
    m_in, m_out = fit_and_run("cross_cic", Xc, yc,                # train
                              Xc_test, yc_test,                   # in-domain
                              Xut, yut, out_cat=cut)              # cross UNSW
    summary["cic_in"] = m_in
    summary["cic_to_unsw"] = m_out
    print("  in-domain AUC %.3f rec %.3f  |  cross AUC %.3f rec %.3f fa %.3f"
          % (m_in["auc"], m_in["attack_recall"], m_out["auc"],
             m_out["attack_recall"], m_out["false_alarm_rate"]))

    print("\n=== Detector trained on UNSW-NB15 ===", flush=True)
    m_in, m_out = fit_and_run("cross_unsw", Xus, yus,             # train
                              Xut, yut,                           # in-domain
                              Xc_test, yc_test)                   # cross CIC
    summary["unsw_in"] = m_in
    summary["unsw_to_cic"] = m_out
    print("  in-domain AUC %.3f rec %.3f  |  cross AUC %.3f rec %.3f fa %.3f"
          % (m_in["auc"], m_in["attack_recall"], m_out["auc"],
             m_out["attack_recall"], m_out["false_alarm_rate"]))

    with open(os.path.join(METRICS_DIR, "cross_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\nSummary -> results/metrics/cross_summary.json")

    _plot_directions(summary)
    _plot_categories()
    print("Total stage time %.1f s" % (time.time() - t0))


def _plot_directions(summary):
    labels = ["CIC->CIC\n(in)", "CIC->UNSW\n(cross)",
              "UNSW->UNSW\n(in)", "UNSW->CIC\n(cross)"]
    aucs = [summary["cic_in"]["auc"], summary["cic_to_unsw"]["auc"],
            summary["unsw_in"]["auc"], summary["unsw_to_cic"]["auc"]]
    recs = [summary["cic_in"]["attack_recall"],
            summary["cic_to_unsw"]["attack_recall"],
            summary["unsw_in"]["attack_recall"],
            summary["unsw_to_cic"]["attack_recall"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, values, title in ((ax[0], aucs, "AUC of attack-vs-benign detector"),
                             (ax[1], recs, "attack recall (default threshold)")):
        a.bar(labels, values, color=["#27ae60", "#c0392b", "#27ae60", "#c0392b"])
        a.set_ylim(0, 1)
        a.set_title(title)
        a.grid(axis="y", alpha=0.3)
        for i, v in enumerate(values):
            a.text(i, v + 0.01, "%.3f" % v, ha="center", fontsize=9)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "stage9_transfer_directions.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("Plot ->", out)


def _plot_categories():
    path = os.path.join(METRICS_DIR, "cross_cic_per_category.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        per = json.load(fh)
    items = [(c, per[c]) for c in per
             if c != "experiment" and per[c].get("n", 0) > 0]
    items.sort(key=lambda kv: kv[1]["recall"])
    cats, recs = zip(*[(c, p["recall"]) for c, p in items])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(cats, recs, color=["#c0392b" if r >= 0.5 else "#7f8c8d" for r in recs])
    ax.set_xlim(0, 1)
    ax.set_xlabel("recall (fraction detected as attack)")
    ax.set_title("CIC-trained detector on UNSW-NB15 - recall per attack type")
    for i, (c, p) in enumerate(items):
        ax.text(p["recall"] + 0.01, i, "%.2f" % p["recall"],
                va="center", fontsize=8)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "stage9_unsw_category_recall.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("Plot ->", out)


if __name__ == "__main__":
    main()
