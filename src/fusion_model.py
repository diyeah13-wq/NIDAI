"""Stage 5 - multi-technique fusion: one Random Forest per feature group.

The 5 feature groups (flow volume, packet statistics, timing/IAT, TCP flags,
endpoint+window) each look at traffic from a different angle - a "technique"
that is strong for some attacks and blind to others:

  - Volume      -> strong for DoS/DDoS floods, scans
  - Packet stat -> separates bulk flood traffic from single-packet probes
  - Timing/IAT  -> separates machine-paced attack tools from human users
  - TCP flags   -> declares SYN floods / half-open scans / asymmetry
  - Endpoint    -> ports and TCP windows betray services being scanned

Each group trains its own Random Forest on the SAME rare-aware 300k sample as
the Stage 4 baseline (identical budget => fair comparison) and predicts on the
FULL test set. The fused prediction is the simple average of the five per-group
class probabilities (soft voting) - a group that is unsure is outvoted by the
groups that are certain, and a rare attack missed by one technique is usually
caught by another.

We keep the 5 Group RFs (not one big forest) so each can be reasoned about and
audited individually; the per-group JSON files document each technique alone.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (FEATURE_GROUPS, FEATURE_ORDER, CLASS_ORDER,
                    RANDOM_STATE, MODEL_SAMPLE_SIZE,
                    MODELS_DIR, PREDICTIONS_DIR, METRICS_DIR)
from common import load_train, load_test, rare_aware_sample
from evaluation import (compute_metrics, save_experiment_metrics,
                        plot_confusion_matrix, plot_f1_comparison,
                        print_table)

GROUP_KEYS = ["A_Flow_Volume", "B_Packet_Statistics", "C_Timing_IAT",
              "D_TCP_Flags_Connection", "E_Endpoint_Window"]

SHORT_NAMES = {
    "A_Flow_Volume": "Volume",
    "B_Packet_Statistics": "Packet",
    "C_Timing_IAT": "Timing",
    "D_TCP_Flags_Connection": "TCP flags",
    "E_Endpoint_Window": "Endpoint",
}


def make_rf():
    return RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE)


def mode_votes(rows):
    """Hard-vote (majority) prediction along axis 0 of an (G, N) vote matrix."""
    n, m = rows.shape
    out = np.empty(m, dtype=np.int64)
    minlen = len(CLASS_ORDER)
    for j in range(m):
        counts = np.bincount(rows[:, j], minlength=minlen)
        out[j] = int(counts.argmax())
    return out


def main():
    t0 = time.time()
    print("Loading train/test (float32)...", flush=True)
    Xtr, ytr, ntr = load_train()
    Xte, yte, nte = load_test()
    print("train=%d rows  test=%d rows  features=%d" % (
        ntr, nte, len(FEATURE_ORDER)), flush=True)
    print("test class mix:")
    for c in CLASS_ORDER:
        print("  %-12s %8d" % (c, int((yte == CLASS_ORDER.index(c)).sum())))

    idx = rare_aware_sample(
        pd.DataFrame(Xtr, columns=FEATURE_ORDER).assign(LabelGroup=ytr),
        MODEL_SAMPLE_SIZE).index.to_numpy()
    Xs, ys = Xtr[idx], ytr[idx]
    print("model sample=%d (all rare classes kept):" % len(idx))
    for c in CLASS_ORDER:
        print("  %-12s %d" % (c, int((ys == CLASS_ORDER.index(c)).sum())))
    del Xtr, ytr

    col_index = {f: i for i, f in enumerate(FEATURE_ORDER)}
    group_cols = {g: [col_index[f] for f in FEATURE_GROUPS[g]["features"]]
                  for g in GROUP_KEYS}

    os.makedirs(os.path.join(MODELS_DIR, "fusion"), exist_ok=True)
    group_models, group_proba, group_metrics = {}, {}, {}
    for g in GROUP_KEYS:
        cols = group_cols[g]
        print("\n=== technique %-26s (%2d features) ===" % (g, len(cols)),
              flush=True)
        fit_t0 = time.time()
        clf = make_rf()
        clf.fit(Xs[:, cols], ys)
        print("fit %.1f s" % (time.time() - fit_t0), flush=True)

        proba = clf.predict_proba(Xte[:, cols])
        ypred = clf.classes_[proba.argmax(axis=1)]
        metrics = compute_metrics(yte, ypred)
        group_metrics[g] = metrics
        print_table(metrics)

        dump(clf, os.path.join(MODELS_DIR, "fusion", g + ".joblib"))
        save_experiment_metrics("group_" + g, metrics, extra={
            "technique": SHORT_NAMES[g],
            "model": "RandomForest(n_estimators=100, max_features=sqrt, "
                     "min_samples_leaf=2, class_weight=balanced)",
            "features": len(cols),
            "feature_names": FEATURE_GROUPS[g]["features"],
            "trained_on_rows": int(len(idx)),
            "test_rows": int(nte),
            "scaling": "none (tree thresholds are per-feature)",
            "model_file": os.path.relpath(
                os.path.join(MODELS_DIR, "fusion", g + ".joblib")),
        })
        cm = plot_confusion_matrix(SHORT_NAMES[g],
                                   np.asarray(metrics["confusion_matrix"]),
                                   CLASS_ORDER)
        print("confusion ->", cm, flush=True)
        group_models[g] = clf
        group_proba[g] = proba

    dump(group_models, os.path.join(MODELS_DIR, "fusion", "group_models.joblib"))

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    stack = np.stack([group_proba[g] for g in GROUP_KEYS], axis=0)
    fused_proba = stack.mean(axis=0)
    fused_pred = fused_proba.argmax(axis=1)
    np.save(os.path.join(PREDICTIONS_DIR, "fusion_test_proba.npy"), fused_proba)
    np.save(os.path.join(PREDICTIONS_DIR, "fusion_test_pred.npy"), fused_pred)

    votes = np.stack([group_proba[g].argmax(axis=1) for g in GROUP_KEYS], axis=0)
    hard_pred = mode_votes(votes)

    def _extra():
        return {
            "fusion": "soft voting (mean of 5 per-group predict_proba)",
            "techniques": [SHORT_NAMES[g] for g in GROUP_KEYS],
            "trained_on_rows": int(len(idx)),
            "test_rows": int(nte),
            "models": {SHORT_NAMES[g]: os.path.relpath(
                os.path.join(MODELS_DIR, "fusion", g + ".joblib"))
                for g in GROUP_KEYS},
            "scaling": "none (tree thresholds are per-feature)",
        }

    print("\n=== FUSION (soft voting) - TEST SET ===")
    soft_metrics = compute_metrics(yte, fused_pred)
    print_table(soft_metrics)
    soft_path = save_experiment_metrics("fusion_soft", soft_metrics, _extra())
    plot_confusion_matrix("fusion_soft",
                          np.asarray(soft_metrics["confusion_matrix"]),
                          CLASS_ORDER)

    print("\n=== FUSION (hard / majority voting) - TEST SET ===")
    hard_metrics = compute_metrics(yte, hard_pred)
    print_table(hard_metrics)
    hard_path = save_experiment_metrics("fusion_hard", hard_metrics, {
        **_extra(), "fusion": "hard voting (majority of 5 per-group argmax)"})
    print("\nMetrics ->", soft_path, "|", hard_path)

    _comparison_plot(group_metrics, soft_metrics, hard_metrics)
    print("Total stage time %.1f s" % (time.time() - t0))


def _comparison_plot(group_metrics, soft_metrics, hard_metrics):
    """Bar chart: macro-F1 of each technique, both fusions, and Stage 4 RF."""
    baseline_path = os.path.join(METRICS_DIR, "baseline_rf.json")
    baseline_f1 = 0.0
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as fh:
            baseline_f1 = json.load(fh)["macro"]["f1"]

    labels = [SHORT_NAMES[g] for g in GROUP_KEYS] + \
             ["Fused (soft)", "Fused (majority)", "RF all 69 (Stage 4)"]
    f1s = [group_metrics[g]["macro"]["f1"] for g in GROUP_KEYS] + \
          [soft_metrics["macro"]["f1"], hard_metrics["macro"]["f1"],
           baseline_f1]
    colors = (["#7f8c8d"] * len(GROUP_KEYS) +
              ["#27ae60", "#16a085", "#2980b9"])
    out = plot_f1_comparison(
        "stage5_fusion_f1", labels, f1s, colors=colors,
        title="Stage 5 - per-technique vs fused macro-F1", ylim=(0.6, 1.0))
    print("Comparison plot ->", out)


if __name__ == "__main__":
    main()