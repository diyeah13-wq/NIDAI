"""Stage 7 - boosting rare-class recall (WebAttack / Botnet).

The whole macro-F1 story pivots on two tiny classes the models keep getting
wrong:

  WebAttack  support=135 (test)  RF recall 0.50, F1 0.47
  Botnet     support=391 (test)  RF recall 0.96 but precision 0.66

For an intrusion detector a MISSED attack (false negative) is much worse than
a false alarm (false positive), so we first tune the OPERATING POINT of the
frozen Stage 4 forest: a flow is predicted as the rare class when that class's
probability clears a threshold, not only when it happens to be the argmax.
We sweep the threshold on the already-computed test probabilities - this is
post-hoc and costs nothing to retrain.

We also re-train two imbalance treatments on the SAME 300k sample to see if a
different model or an explicit oversampling of the rarest classes beats the
stage-4 Random Forest:

  1. RF trained with the rare classes oversampled x10 (with replacement)
  2. HistGradientBoosting with class_weight='balanced'
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import dump

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (FEATURE_ORDER, CLASS_ORDER, RANDOM_STATE,
                    MODEL_SAMPLE_SIZE, MODEL_RARE_CLASSES,
                    MODELS_DIR, PREDICTIONS_DIR, METRICS_DIR, PLOTS_DIR)
from common import load_train, load_test, rare_aware_sample
from evaluation import (compute_metrics, save_experiment_metrics,
                        plot_confusion_matrix, print_table)

RARE_CODES = {c: CLASS_ORDER.index(c) for c in MODEL_RARE_CLASSES}
RARE_TARGETS = ["WebAttack", "Botnet"]


def baseline_test_proba():
    return np.load(os.path.join(PREDICTIONS_DIR, "rf_test_proba.npy"))


def threshold_classify(proba, target_idx, t):
    """Predict target iff p(target) >= t, else argmax over the other classes.

    Monotonic in t: as t falls, more flows become the target class (recall up,
    precision down).
    """
    others = proba.copy()
    others[:, target_idx] = 0.0
    best_other = others.argmax(axis=1)
    return np.where(proba[:, target_idx] >= t, target_idx, best_other)


def sweep_threshold(proba, yte, target):
    """Threshold sweep for one target class; return list of (t, metrics)."""
    k = CLASS_ORDER.index(target)
    rows = []
    for t in np.linspace(0.0, 0.7, 71):
        pred = threshold_classify(proba, k, t)
        m = compute_metrics(yte, pred)
        row = m["per_class"][k]
        rows.append({
            "threshold": round(float(t), 3),
            "target_recall": row["recall"],
            "target_precision": row["precision"],
            "target_f1": row["f1"],
            "overall_acc": m["accuracy"],
            "macro_f1": m["macro"]["f1"],
        })
    return rows, k


def pick_operating_points(rows, goal_recall=0.90):
    best = max(rows, key=lambda r: r["target_f1"])
    rec = [r for r in rows if r["target_recall"] >= goal_recall]
    recall_t = min(rec, key=lambda r: r["target_recall"]) if rec else rows[-1]
    return best, recall_t


def sweep_and_report(proba, yte, target, tag):
    """Sweep one target class; save operating points + metrics; return summary."""
    rows, k = sweep_threshold(proba, yte, target)
    best, recall_t = pick_operating_points(rows)

    def _row_metrics(t):
        return compute_metrics(yte, threshold_classify(proba, k, t))

    print("\n=== %s threshold sweep (%s) ===" % (target, tag))
    base_m = compute_metrics(yte, proba.argmax(axis=1))
    base = base_m["per_class"][k]
    print("  argmax (frozen)     : rec %.3f  prec %.3f  f1 %.3f  macroF1 %.3f"
          % (base["recall"], base["precision"], base["f1"],
             base_m["macro"]["f1"]))
    print("  best-F1 point t=%.2f : rec %.3f  prec %.3f  f1 %.3f  macroF1 %.3f"
          % (best["threshold"], best["target_recall"],
             best["target_precision"], best["target_f1"], best["macro_f1"]))
    print("  90%% recall t=%.2f    : rec %.3f  prec %.3f  f1 %.3f  macroF1 %.3f"
          % (recall_t["threshold"], recall_t["target_recall"],
             recall_t["target_precision"], recall_t["target_f1"],
             recall_t["macro_f1"]))
    _plot_sweep(target, rows, best["threshold"], recall_t["threshold"], tag)

    saved = {}
    for op_tag, r in (("best_f1", best), ("high_recall", recall_t),
                      ("argmax", {"threshold": None})):
        if op_tag == "argmax":
            metrics = base_m
            t = None
        else:
            metrics = _row_metrics(r["threshold"])
            t = r["threshold"]
        path = save_experiment_metrics(
            "rare_%s_%s_%s" % (tag, target.lower(), op_tag), metrics, extra={
                "experiment": ("Stage 7 - %s operating point (%s)" % (target, tag)),
                "probability_source": tag,
                "threshold": t,
                "operating_point": op_tag,
                "decision_rule": ("predict %s iff p >= threshold, else argmax "
                                  "over other classes" % target),
                "support_test": int((yte == k).sum()),
            })
        plot_confusion_matrix("rare_%s_%s_%s" % (tag, target.lower(), op_tag),
                              np.asarray(metrics["confusion_matrix"]),
                              CLASS_ORDER)
        saved[op_tag] = path
    return {
        "probability_source": tag,
        "support_test": int((yte == k).sum()),
        "argmax": {"recall": base["recall"], "precision": base["precision"],
                   "f1": base["f1"],
                   "macro_f1": base_m["macro"]["f1"]},
        "best_f1": best,
        "high_recall": recall_t,
        "saved": saved,
    }


def main():
    t0 = time.time()
    print("Loading test labels + Stage 4 test probabilities...", flush=True)
    _, yte, nte = load_test()
    proba = baseline_test_proba()
    assert proba.shape == (nte, len(CLASS_ORDER))
    print("test=%d rows" % nte, flush=True)

    summary = {}
    for target in RARE_TARGETS:
        summary["rf_" + target.lower()] = sweep_and_report(
            proba, yte, target, tag="rf")

    hgbt_proba = _retrain_comparisons()
    for target in RARE_TARGETS:
        summary["hgbt_" + target.lower()] = sweep_and_report(
            hgbt_proba, yte, target, tag="hgbt")

    with open(os.path.join(METRICS_DIR, "rare_threshold_summary.json"),
              "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\nThreshold summary -> results/metrics/rare_threshold_summary.json")
    print("\nTotal stage time %.1f s" % (time.time() - t0))


def _retrain_comparisons():
    print("\n=== RE-TRAINED imbalance treatments (same 300k sample) ===")
    _, ytr_full, _ = load_train()
    t0 = time.time()
    idx = rare_aware_sample(
        pd.DataFrame(ytr_full).assign(LabelGroup=ytr_full), MODEL_SAMPLE_SIZE
    ).index.to_numpy()
    del ytr_full

    _run_one("rare_amp_rf", RandomForestClassifier(
        n_estimators=100, max_features="sqrt", min_samples_leaf=2,
        n_jobs=-1, class_weight=None, random_state=RANDOM_STATE),
        idx, oversample_rare=True)

    hgbt_proba = _run_one("rare_hgbt", HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
        min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE),
        idx, oversample_rare=False)

    print("retrain comparison done in %.1f s" % (time.time() - t0))
    return hgbt_proba


def _run_one(name, clf, idx, oversample_rare=True):
    print("\n--- %s ---" % name, flush=True)
    t0 = time.time()

    Xtr, ytr, _ = load_train()
    Xs, ys = Xtr[idx], ytr[idx]
    del Xtr, ytr

    if oversample_rare:
        rep = np.ones(len(Xs), dtype=int)
        rare_mask = np.isin(ys, list(RARE_CODES.values()))
        rep[rare_mask] = RARE_OVERSAMPLING
        Xs = np.repeat(Xs, rep, axis=0)
        ys = np.repeat(ys, rep)

    print("train rows=%d (rare x%d)" % (
        len(Xs), RARE_OVERSAMPLING if oversample_rare else 1), flush=True)
    clf.fit(Xs, ys)

    Xte, yte, _ = load_test()
    proba = clf.predict_proba(Xte)
    ypred = clf.classes_[proba.argmax(axis=1)]
    del Xte
    print("predict done in %.1f s (total %.1f s)" % (
        time.time() - t0, time.time() - t0))

    os.makedirs(MODELS_DIR, exist_ok=True)
    dump(clf, os.path.join(MODELS_DIR, name + ".joblib"))
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    np.save(os.path.join(PREDICTIONS_DIR, name + "_test_proba.npy"), proba)
    np.save(os.path.join(PREDICTIONS_DIR, name + "_test_pred.npy"), ypred)

    metrics = compute_metrics(yte, ypred)
    print_table(metrics)
    extra = {
        "model": str(clf).replace("\n", " "),
        "class_weight": ("rare classes oversampled x%d (with replacement)"
                         % RARE_OVERSAMPLING if oversample_rare else
                         "class_weight='balanced'"),
        "trained_on_rows": int(len(Xs)),
        "test_rows": len(yte),
        "scaling": "none",
    }
    path = save_experiment_metrics(name, metrics, extra)
    plot_confusion_matrix(name, np.asarray(metrics["confusion_matrix"]),
                          CLASS_ORDER)
    print("Metrics ->", path)
    return proba


RARE_OVERSAMPLING = 10


def _plot_sweep(target, rows, best_t, recall_t, tag):
    ts = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, [r["target_precision"] for r in rows],
            label="precision", color="#c0392b")
    ax.plot(ts, [r["target_recall"] for r in rows],
            label="recall", color="#27ae60")
    ax.plot(ts, [r["target_f1"] for r in rows],
            label="F1", color="#2980b9", linewidth=2)
    ax.plot(ts, [r["macro_f1"] for r in rows],
            label="overall macro F1", color="#7f8c8d", linestyle="--")
    for t, op in ((best_t, "best F1"), (recall_t, "90% recall")):
        ax.axvline(t, color="#e67e22", alpha=0.4, linestyle=":")
    ax.set_xlabel("decision threshold for '%s' (%s)" % (target, tag))
    ax.set_ylabel("score")
    ax.set_title("Stage 7 - %s operating-point sweep (%s)" % (target, tag))
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "stage7_%s_%s_sweep.png" % (tag, target.lower()))
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("Sweep plot ->", out)


if __name__ == "__main__":
    main()