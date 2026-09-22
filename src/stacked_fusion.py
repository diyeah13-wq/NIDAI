"""Stage 6 - stacked fusion: a meta-learner over per-technique opinions.

Stage 5 showed the naive fusion (average of per-group probabilities, macro F1
0.73) LOSES to the single all-69-feature forest (0.89) on the rare classes.
Why: every technique is trained on a confined feature subset, so each one is
locally incapable of separating the very rarest attacks, and simple averaging
dilutes the handful of groups that DO see them.

Instead of trusting the average, we train a STACKED meta-learner on the five
per-group probability vectors - 35 numbers per flow that say "what do the
volume/packet/timing/flags/endpoint techniques think?". The meta-learner learns
WHEN to trust WHICH technique. Same 300k-row training budget and FULL test set
as every other experiment, so the comparison stays fair.

Training the meta-learner on the in-sample predictions of the group models
would let it memorise their training-set over-confidence, so we use
out-of-fold predictions: the 300k sample is split into folds; each group model
is refitted once per fold on the other folds and predicts the held-out fold,
so a meta-row is always a prediction on data the corresponding group model
never trained on.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (FEATURE_GROUPS, FEATURE_ORDER, CLASS_ORDER,
                    RANDOM_STATE, MODEL_SAMPLE_SIZE, N_FUSION_FOLDS,
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

COL_INDEXES = None


def make_rf():
    return RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE)


def init_col_indexes():
    global COL_INDEXES
    col_index = {f: i for i, f in enumerate(FEATURE_ORDER)}
    COL_INDEXES = {g: [col_index[f] for f in FEATURE_GROUPS[g]["features"]]
                   for g in GROUP_KEYS}


def fit_groups(Xs, ys, rows_idx):
    """Fit the 5 group forests on rows_idx only; return {group: clf}."""
    cls = {}
    for g in GROUP_KEYS:
        clf = make_rf()
        clf.fit(Xs[rows_idx][:, COL_INDEXES[g]], ys[rows_idx])
        cls[g] = clf
    return cls


def group_probabilities(cls, X):
    """Return [G x (N x K)]: a class-probability matrix per technique."""
    return [cls[g].predict_proba(X[:, COL_INDEXES[g]]) for g in GROUP_KEYS]


def meta_features(blocks):
    """Concatenate per-technique proba blocks along the feature axis."""
    return np.hstack(blocks)


def main():
    t0 = time.time()
    init_col_indexes()
    print("Loading train/test (float32)...", flush=True)
    Xtr, ytr, ntr = load_train()
    Xte, yte, nte = load_test()
    print("train=%d rows  test=%d rows" % (ntr, nte), flush=True)

    idx = rare_aware_sample(
        pd.DataFrame(Xtr, columns=FEATURE_ORDER).assign(LabelGroup=ytr),
        MODEL_SAMPLE_SIZE).index.to_numpy()
    Xs, ys = Xtr[idx], ytr[idx]
    del Xtr, ytr
    print("model sample=%d rows" % len(idx), flush=True)

    rng = np.random.RandomState(RANDOM_STATE)
    fold_ids = rng.randint(0, N_FUSION_FOLDS, size=len(Xs))
    print("building out-of-fold meta features across %d folds ..."
          % N_FUSION_FOLDS, flush=True)

    fold_blocks = []
    fold_labels = []
    for f in range(N_FUSION_FOLDS):
        cls = fit_groups(Xs, ys, fold_ids != f)
        in_fold = fold_ids == f
        fold_blocks.append(group_probabilities(cls, Xs[in_fold]))
        fold_labels.append(ys[in_fold])
        del cls
        print("  fold %d: %d rows" % (f + 1, int(in_fold.sum())), flush=True)

    meta_train = meta_features([
        np.concatenate([fb[g] for fb in fold_blocks], axis=0)
        for g in range(len(GROUP_KEYS))])
    meta_train_y = np.concatenate(fold_labels, axis=0)

    cls_out = fit_groups(Xs, ys, np.arange(len(Xs)))
    meta_test = meta_features(group_probabilities(cls_out, Xte))

    print("meta-feature matrix: train=%s  test=%s" % (meta_train.shape,
                                                      meta_test.shape),
          flush=True)

    meta = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                  min_samples_leaf=1, n_jobs=-1,
                                  class_weight="balanced",
                                  random_state=RANDOM_STATE)
    print("Fitting stacked meta-learner (RF, 200 trees)...", flush=True)
    fit_t0 = time.time()
    meta.fit(meta_train, meta_train_y)
    print("meta fit %.1f s" % (time.time() - fit_t0), flush=True)

    proba = meta.predict_proba(meta_test)
    ypred = meta.classes_[proba.argmax(axis=1)]

    os.makedirs(os.path.join(MODELS_DIR, "fusion"), exist_ok=True)
    dump({"group_models": cls_out, "meta_learner": meta},
         os.path.join(MODELS_DIR, "fusion", "stacked_fusion.joblib"))

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    np.save(os.path.join(PREDICTIONS_DIR, "stacked_test_proba.npy"), proba)
    np.save(os.path.join(PREDICTIONS_DIR, "stacked_test_pred.npy"), ypred)

    print("\n=== STACKED FUSION - TEST SET ===")
    metrics = compute_metrics(yte, ypred)
    print_table(metrics)

    extra = {
        "fusion": "stacked meta-learner (RF 200 trees, class_weight=balanced) "
                  "on the 5 per-technique probability vectors",
        "meta_features": "%d techniques x %d classes = %d" % (
            len(GROUP_KEYS), len(CLASS_ORDER),
            len(GROUP_KEYS) * len(CLASS_ORDER)),
        "out_of_fold_folds": N_FUSION_FOLDS,
        "techniques": [SHORT_NAMES[g] for g in GROUP_KEYS],
        "base_models": "per-group RandomForest (100 trees, same hyperparams "
                       "and seed as Stage 5)",
        "scaling": "none (tree thresholds are per-feature)",
        "trained_on_rows": int(len(idx)),
        "test_rows": int(nte),
    }
    path = save_experiment_metrics("fusion_stacked", metrics, extra)
    print("\nMetrics ->", path)

    plot_confusion_matrix("fusion_stacked",
                          np.asarray(metrics["confusion_matrix"]),
                          CLASS_ORDER)

    _comparison_plot(metrics)
    print("Total stage time %.1f s" % (time.time() - t0))


def _comparison_plot(stacked_metrics):
    with open(os.path.join(METRICS_DIR, "baseline_rf.json"),
              encoding="utf-8") as fh:
        baseline_f1 = json.load(fh)["macro"]["f1"]

    group_f1s = []
    for g in GROUP_KEYS:
        with open(os.path.join(METRICS_DIR, "group_" + g + ".json"),
                  encoding="utf-8") as fh:
            group_f1s.append(json.load(fh)["macro"]["f1"])

    labels = [SHORT_NAMES[g] for g in GROUP_KEYS]
    f1s = list(group_f1s)
    for fname, tag in (("fusion_soft.json", "Fused (soft)"),
                       ("fusion_hard.json", "Fused (majority)")):
        p = os.path.join(METRICS_DIR, fname)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                f1s.append(json.load(fh)["macro"]["f1"])
            labels.append(tag)
    labels += ["Stacked", "RF all 69"]
    f1s += [stacked_metrics["macro"]["f1"], baseline_f1]

    colors = (["#7f8c8d"] * len(GROUP_KEYS) +
              ["#27ae60", "#16a085", "#e67e22", "#2980b9"])[:len(labels)]
    out = plot_f1_comparison(
        "stage6_stacked_f1", labels, f1s, colors=colors,
        title="Stage 6 - stacked fusion vs naive fusion vs single forest",
        ylim=(0.6, 1.0))
    print("Comparison plot ->", out)


if __name__ == "__main__":
    main()