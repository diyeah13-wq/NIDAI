"""Memory-efficient EDA: streams the cleaned CSV in chunks, exact aggregates."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PROCESSED_CLEAN, PLOTS_DIR, RESULTS_DIR, FEATURE_ORDER, CLASS_ORDER, LABEL_DISPLAY

CHUNK = 400_000
PLOT_SAMPLE = 24_000


def _pass1():
    acc = {"classes": {}, "raw_classes": {}, "stats": {}, "zeros": {}, "n": 0}
    nrows = 0
    for chunk in pd.read_csv(PROCESSED_CLEAN, chunksize=CHUNK,
                             usecols=FEATURE_ORDER + ["Label", "LabelGroup"]):
        for k, v in chunk["LabelGroup"].value_counts().items():
            acc["classes"][k] = acc["classes"].get(k, 0) + int(v)
        for k, v in chunk["Label"].value_counts().items():
            acc["raw_classes"][k] = acc["raw_classes"].get(k, 0) + int(v)
        x = chunk[FEATURE_ORDER]
        mx = x.max(axis=0).replace(0, 1.0)
        xs = x / mx
        s = xs.sum()
        ss = (xs ** 2).where(xs.notna(), 0).sum()
        n = xs.notna().sum()
        for c in FEATURE_ORDER:
            st = acc["stats"].setdefault(c, {"count": 0, "sum": 0.0, "sumsq": 0.0,
                                             "min": np.inf, "max": -np.inf})
            st["count"] += int(n[c])
            st["sum"] += float(s[c])
            st["sumsq"] += float(ss[c])
            st["min"] = min(st["min"], float(xs[c].min()))
            st["max"] = max(st["max"], float(xs[c].max()))
        z = chunk[FEATURE_ORDER].eq(0).sum()
        for c in FEATURE_ORDER:
            acc["zeros"][c] = acc["zeros"].get(c, 0) + int(z[c])
        acc["n"] += len(chunk)
        nrows += len(chunk)
        print("pass1 streamed %d rows" % nrows, flush=True)
    return acc


def _summary_from(acc):
    rows = []
    for c, st in acc["stats"].items():
        mean = st["sum"] / st["count"]
        var = max(st["sumsq"] / st["count"] - mean ** 2, 0.0)
        rows.append({"feature": c, "count": st["count"], "mean": mean,
                     "std": np.sqrt(var), "min": st["min"], "max": st["max"],
                     "zeros": acc["zeros"].get(c, 0)})
    return pd.DataFrame(rows).sort_values("feature")


def _correlation(acc, summary):
    feats = FEATURE_ORDER
    n = len(feats)
    mean = summary.set_index("feature").loc[feats, "mean"].values
    std = summary.set_index("feature").loc[feats, "std"].values
    N = float(summary["count"].iloc[0])
    outer = np.zeros((n, n), dtype=np.float64)
    print("pass2 computing correlation...")
    it = 0
    for chunk in pd.read_csv(PROCESSED_CLEAN, chunksize=CHUNK, usecols=FEATURE_ORDER):
        x = chunk[feats].values.astype(np.float64)
        mx = x.max(axis=0)
        mx[mx == 0] = 1.0
        xs = x / mx
        outer += xs.T @ xs
        it += 1
        print("  chunk %d done" % it, flush=True)
        del chunk, x, xs
    cov = outer / N - np.outer(mean, mean)
    d = np.outer(std, std)
    corr = cov / np.maximum(d, 1e-15)
    corr[np.isnan(corr)] = 0.0
    corr[np.abs(corr) > 1] = np.sign(corr[np.abs(corr) > 1]) * 1.0
    np.fill_diagonal(corr, 1.0)
    np.save(os.path.join(RESULTS_DIR, "correlation_matrix.npy"), corr)
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(feats, rotation=90, fontsize=6)
    ax.set_yticklabels(feats, fontsize=6)
    fig.colorbar(im, shrink=0.6)
    ax.set_title("Correlation matrix of the %d selected features (exact, streamed)" % n)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=130)
    plt.close(fig)
    off = corr[np.triu_indices(n, 1)]
    print("Correlation heatmap saved. |r| mean=%.3f  max=%.3f"
          % (np.mean(np.abs(off)), np.max(np.abs(off))))
    return corr


def _attack_relevance():
    feats = FEATURE_ORDER
    n = len(feats)
    sb = np.zeros(n); sa = np.zeros(n); cb = np.zeros(n); ca = np.zeros(n)
    ssb = np.zeros(n); ssa = np.zeros(n)
    for chunk in pd.read_csv(PROCESSED_CLEAN, chunksize=CHUNK,
                             usecols=feats + ["LabelGroup"]):
        y = (chunk["LabelGroup"] != "BENIGN").values
        x = chunk[feats].values.astype(np.float64)
        a = x[y]
        b = x[~y]
        sa += a.sum(axis=0); ssb += (b ** 2).sum(axis=0)
        sb += b.sum(axis=0); ssa += (a ** 2).sum(axis=0)
        cb += (~y).sum(); ca += y.sum()
        del chunk, x, a, b
    isa = np.zeros(n)
    for i in range(n):
        if cb[i] > 0 and ca[i] > 0:
            m0 = sb[i] / cb[i]; m1 = sa[i] / ca[i]
            v0 = ssb[i] / cb[i] - m0 ** 2; v1 = ssa[i] / ca[i] - m1 ** 2
            isa[i] = (m1 - m0) / (np.sqrt(max(v0, 0)) + np.sqrt(max(v1, 0)) + 1e-9)
    order = np.argsort(-np.abs(isa))
    fig, ax = plt.subplots(figsize=(9, 7))
    top = order[:20]
    cols = ["#e67e22" if isa[i] > 0 else "#2980b9" for i in top][::-1]
    ax.barh([feats[i] for i in top][::-1], isa[top][::-1], color=cols)
    ax.set_xlabel("standardized mean shift (attack minus benign)")
    ax.set_title("Attack-relevance of features: positive = higher during attacks")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "attack_relevance.png"), dpi=140)
    plt.close(fig)
    print("Attack-relevance plot saved. Top features:")
    for i in top[:10]:
        print("   %s  (%.3f)" % (feats[i], isa[i]))
    return [feats[i] for i in top[:6]], isa


def _distributions(top):
    parts = []
    got = 0
    for chunk in pd.read_csv(PROCESSED_CLEAN, chunksize=PLOT_SAMPLE,
                             usecols=top + ["LabelGroup"]):
        parts.append(chunk)
        got += len(chunk)
        if got >= 2 * PLOT_SAMPLE:
            break
    samp = pd.concat(parts).sample(PLOT_SAMPLE, random_state=42)
    colors = {"BENIGN": "#27ae60", "ATTACK": "#c0392b"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = np.atleast_1d(axes).ravel()
    for a, f in zip(axes, top):
        sf = samp[["LabelGroup", f]].copy()
        sf["grp"] = np.where(sf["LabelGroup"] == "BENIGN", "BENIGN", "ATTACK")
        log = bool((sf[f] > 0).all())
        for g, col in colors.items():
            sel = sf[sf["grp"] == g][f] + 1e-9
            a.hist(sel, bins=80, alpha=0.45, color=col, density=True, label=g)
        if log:
            a.set_xscale("log")
        a.set_title(f, fontsize=10)
        a.set_ylabel("density")
        a.legend(fontsize=8)
    fig.suptitle("Top discriminative features - BENIGN vs attack (sample of %d real rows)" % PLOT_SAMPLE)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "feature_distributions.png"), dpi=140)
    plt.close(fig)


def _class_plots(acc):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    order = CLASS_ORDER
    for ax, labels, title, log in [
            (axes[0], [LABEL_DISPLAY[c] for c in order],
             "Coarse classes (7)", False),
            (axes[1], None, "Fine-grained raw labels", True)]:
        if labels:
            src = [acc["classes"].get(c, 0) for c in order]
            used = order
        else:
            rk = sorted(acc["raw_classes"], key=lambda k: -acc["raw_classes"][k])
            used = rk
            src = [acc["raw_classes"][c] for c in rk]
            labels = [c.replace("\ufffd", "-") for c in rk]
        y = np.arange(len(used))
        cols = ["#27ae60" if c == "BENIGN" else "#c0392b" for c in used]
        ax.barh(y, src, color=cols)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel("count" + (" (log)" if log else ""))
        for yy, v in zip(y, src):
            ax.text(v * 1.02, yy, "%s" % f"{v:,}", va="center", fontsize=7, color="#444444")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=140)
    plt.close(fig)


def run_eda():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    acc = _pass1()
    summary = _summary_from(acc)
    summary.to_csv(os.path.join(RESULTS_DIR, "eda_numeric_summary.csv"), index=False)

    total = acc["n"]
    print("\n=== EDA SUMMARY ===")
    print("Total cleaned samples: %d, selected features: %d" % (total, len(FEATURE_ORDER)))
    print("Class distribution (group):")
    for c in CLASS_ORDER:
        print("  %-12s %10d  (%5.2f%%)" % (LABEL_DISPLAY[c], acc["classes"].get(c, 0),
                                            100.0 * acc["classes"].get(c, 0) / total))

    pd.DataFrame({"class": CLASS_ORDER,
                  "class_display": [LABEL_DISPLAY[c] for c in CLASS_ORDER],
                  "count": [acc["classes"].get(c, 0) for c in CLASS_ORDER],
                  "percent": [round(100.0 * acc["classes"].get(c, 0) / total, 3)
                              for c in CLASS_ORDER]}).to_csv(
        os.path.join(RESULTS_DIR, "class_distribution.csv"), index=False)

    _class_plots(acc)
    corr = _correlation(acc, summary)
    del corr
    top, isa = _attack_relevance()
    _distributions(top)

    lines = []
    lines.append("=== CIC-IDS2017 EDA (cleaned) ===")
    lines.append("Total samples: %s" % f"{total:,}")
    lines.append("Selected features: %d" % len(FEATURE_ORDER))
    lines.append("")
    lines.append("Class distribution (7-class model target):")
    for c in CLASS_ORDER:
        lines.append("  %s: %d (%.2f%%)" % (c, acc["classes"].get(c, 0),
                                            100.0 * acc["classes"].get(c, 0) / total))
    lines.append("")
    lines.append("Plots: results/plots/")
    lines.append("  class_distribution.png")
    lines.append("  correlation_heatmap.png")
    lines.append("  attack_relevance.png")
    lines.append("  feature_distributions.png")
    lines.append("")
    lines.append("Numeric stats : results/eda_numeric_summary.csv")
    lines.append("Class counts  : results/class_distribution.csv")
    with open(os.path.join(RESULTS_DIR, "eda_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\nEDA finished. Plots saved to %s" % PLOTS_DIR)


if __name__ == "__main__":
    run_eda()