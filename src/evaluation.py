"""Shared evaluation helpers for the imbalanced multiclass NIDS experiments.

Every experiment (Random Forest, KNN, fusion) produces the same JSON schema so
results are directly comparable. NOTE: for an intrusion-detection system the
per-class recall (did we catch the attack?) matters more than accuracy.
"""
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (CLASS_ORDER, LABEL_DISPLAY, METRICS_DIR)

ZERO_DIV = 0


def compute_metrics(y_true, y_pred, class_names=None):
    """Return accuracy + macro/weighted + per-class precision/recall/F1/FPR.

    Classes are encoded as integer indices 0..K-1 (matching CLASS_ORDER); only
    class NAMES are used for display. FPR per class = FP / (FP + TN).
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    class_names = CLASS_ORDER if class_names is None else class_names
    labels = list(range(len(class_names)))
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=ZERO_DIV)
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    f1 = np.asarray(f1, dtype=float)
    support = np.asarray(support, dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tn = cm.sum() - (fp + fn + np.diag(cm))
    fpr = np.where((fp + tn) > 0, fp / (fp + tn), 0.0)

    per_class = []
    for i, c in enumerate(class_names):
        per_class.append({
            "class": c,
            "display": LABEL_DISPLAY.get(c, c),
            "support": int(support[i]),
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f1[i]), 4),
            "fpr": round(float(fpr[i]), 4),
        })
    wsum = float(support.sum())
    weighted = {
        "precision": round(float((p * support).sum() / wsum), 4),
        "recall": round(float((r * support).sum() / wsum), 4),
        "f1": round(float((f1 * support).sum() / wsum), 4),
    }
    return {
        "accuracy": round(acc, 4),
        "macro": {
            "precision": round(float(p.mean()), 4),
            "recall": round(float(r.mean()), 4),
            "f1": round(float(f1.mean()), 4),
        },
        "weighted": weighted,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": class_names,
    }


def save_experiment_metrics(name, metrics, extra=None):
    """Persist one experiment to results/metrics/<name>.json (tracked in git)."""
    os.makedirs(METRICS_DIR, exist_ok=True)
    payload = {"experiment": name}
    payload.update(metrics)
    if extra:
        payload["extra"] = extra
    path = os.path.join(METRICS_DIR, name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    return path


def plot_confusion_matrix(name, cm, labels):
    """Confusion matrix heatmap with log colour scale (classes are imbalanced)."""
    from matplotlib.colors import LogNorm
    os.makedirs(os.path.join(os.path.dirname(METRICS_DIR), "plots"), exist_ok=True)
    short = [LABEL_DISPLAY.get(c, c) for c in labels]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(cm, cmap="Blues", norm=LogNorm(vmin=1, vmax=cm.max()))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(short, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i][j]
            if v > 0:
                ax.text(j, i, "%d" % v, ha="center", va="center",
                        fontsize=8, color="white" if v > cm.max() * 0.35 else "#333333")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, shrink=0.8, label="count (log)")
    ax.set_title("Confusion matrix - %s" % LABEL_DISPLAY.get(
        name, name.replace("_", " ").title()))
    fig.tight_layout()
    out = os.path.join(os.path.dirname(METRICS_DIR), "plots", name + "_confusion_matrix.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_f1_comparison(name, labels, f1s, colors=None,
                       title=None, ylim=(0.6, 1.0)):
    """Bar chart of macro-F1 per system; saved to results/plots/<name>.png."""
    if colors is None:
        colors = ["#7f8c8d"] * len(labels)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(labels)), 5))
    bars = ax.bar(labels, f1s, color=colors)
    ax.set_ylabel("macro F1 (test set)")
    ax.set_ylim(*ylim)
    ax.set_title(title or "Macro-F1 comparison")
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, "%.3f" % v,
                ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(METRICS_DIR), "plots", name + ".png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def print_table(metrics):
    head = "%-12s %8s %8s %8s %8s %8s" % ("class", "prec", "rec", "f1", "fpr", "sup")
    print(head)
    print("-" * len(head))
    for row in metrics["per_class"]:
        print("%-12s %8.4f %8.4f %8.4f %8.4f %8d" % (
            row["class"], row["precision"], row["recall"],
            row["f1"], row["fpr"], row["support"]))
    m = metrics["macro"]
    w = metrics["weighted"]
    print("-" * len(head))
    print("acc=%.4f  macro P/R/F1=%.4f/%.4f/%.4f  weighted=%.4f/%.4f/%.4f" % (
        metrics["accuracy"], m["precision"], m["recall"], m["f1"],
        w["precision"], w["recall"], w["f1"]))