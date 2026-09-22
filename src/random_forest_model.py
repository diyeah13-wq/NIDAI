"""Stage 4 - Random Forest baseline on all 69 features.

Why a training sample? A full Random Forest on 2.0M rows x 69 features is
infeasible on a laptop. We train on a ~300k row sample (see MODEL_SAMPLE_SIZE
in config.py) that keeps EVERY rare attack row (WebAttack, Botnet, BruteForce)
so the model has seen them, and evaluate on the FULL held-out test set.
The same training budget is used for every model so comparisons are fair.

Scaling: Random Forest is a tree method - each split is a threshold on ONE
feature, so absolute feature scales do not matter and we must NOT scale.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import dump
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (FEATURE_ORDER, CLASS_ORDER,
                    RANDOM_STATE, MODEL_SAMPLE_SIZE,
                    MODELS_DIR, PREDICTIONS_DIR, PLOTS_DIR)
from common import load_train, load_test, rare_aware_sample
from evaluation import (compute_metrics, save_experiment_metrics,
                        plot_confusion_matrix, print_table)


def main():
    t0 = time.time()
    print("Loading train/test (float32)...", flush=True)
    Xtr, ytr, ntr = load_train()
    Xte, yte, nte = load_test()
    print("train=%d rows  test=%d rows  features=%d" % (ntr, nte, len(FEATURE_ORDER)),
          flush=True)

    idx = rare_aware_sample(
        pd.DataFrame(Xtr, columns=FEATURE_ORDER).assign(LabelGroup=ytr),
        MODEL_SAMPLE_SIZE).index.to_numpy()
    Xs, ys = Xtr[idx], ytr[idx]
    print("model training sample=%d (all rare classes kept) class mix:" % len(idx))
    for c in CLASS_ORDER:
        print("  %-12s %d" % (c, int((ys == CLASS_ORDER.index(c)).sum())))
    del Xtr, ytr

    clf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE)

    print("\nTraining RandomForest (100 trees)...", flush=True)
    fit_t0 = time.time()
    clf.fit(Xs, ys)
    print("fit done in %.1f s" % (time.time() - fit_t0), flush=True)

    print("Predicting on full test set...", flush=True)
    proba = clf.predict_proba(Xte)
    ypred = clf.classes_[proba.argmax(axis=1)]

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    dump(clf, model_path)

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    np.save(os.path.join(PREDICTIONS_DIR, "rf_test_proba.npy"), proba)
    np.save(os.path.join(PREDICTIONS_DIR, "rf_test_pred.npy"), ypred)

    print("\n=== Random Forest baseline - TEST SET PERFORMANCE ===")
    print(classification_report(yte, ypred, target_names=CLASS_ORDER, zero_division=0))
    metrics = compute_metrics(yte, ypred)
    print_table(metrics)

    extra = {
        "model": "RandomForest(n_estimators=100, max_features=sqrt, "
                 "min_samples_leaf=2, class_weight=balanced)",
        "trained_on_rows": int(len(idx)),
        "trained_on_classes": {CLASS_ORDER[c]: int((ys == c).sum())
                               for c in range(len(CLASS_ORDER))},
        "test_rows": int(nte),
        "scaling": "none (tree thresholds are per-feature)",
        "fit_seconds": round(time.time() - fit_t0, 1),
        "model_file": os.path.relpath(model_path),
    }
    path = save_experiment_metrics("baseline_rf", metrics, extra)
    print("\nMetrics ->", path)

    cm_path = plot_confusion_matrix("rf", np.asarray(metrics["confusion_matrix"]),
                                    CLASS_ORDER)
    print("Confusion matrix ->", cm_path)

    imp = pd.Series(clf.feature_importances_, index=FEATURE_ORDER).sort_values()
    top = imp.tail(25)
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(top.index, top.values, color="#2980b9")
    ax.set_xlabel("importance (mean decrease in Gini impurity)")
    ax.set_title("Top 25 features by Random Forest importance")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "rf_feature_importance.png"), dpi=140)
    plt.close(fig)
    print("Feature importance plot -> results/plots/rf_feature_importance.png")
    print("Total stage time %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()