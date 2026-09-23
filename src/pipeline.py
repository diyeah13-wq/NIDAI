"""Stage 8 - operational pipeline: detect with one model, explain with five.

Detection and explanation are deliberately separate (per docs/experiments.md):

  - DETECTION core: a single strong classifier over all 69 features. Stage 7
    showed the balanced HistGradientBoosting (rare_hgbt.joblib) is the best
    rare-class learner; the Stage 4 Random Forest is the fallback.
  - EXPLANATION layer: the 5 per-group Random Forests from Stage 5. Each looks
    at the flow from one angle (volume, packet stats, timing, TCP flags,
    endpoint). For a flagged flow we answer "which technique fired and on
    which features?" by perturbing one feature at a time inside its group and
    measuring how much that group's confidence in the alert class drops.

``FlowScorer`` wraps both layers and is what the Streamlit dashboard calls.
"""
import os
import sys

import numpy as np

from joblib import load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (MODELS_DIR, FEATURE_GROUPS, FEATURE_ORDER, CLASS_ORDER)
from fusion_model import GROUP_KEYS, SHORT_NAMES

DETECTORS = {
    "HGB (balanced) - recommended": os.path.join(MODELS_DIR, "rare_hgbt.joblib"),
    "Random Forest (all 69)": os.path.join(MODELS_DIR, "random_forest.joblib"),
    "Random Forest (oversampled)": os.path.join(MODELS_DIR, "rare_amp_rf.joblib"),
}

GROUP_MODELS_PATH = os.path.join(MODELS_DIR, "fusion", "group_models.joblib")

GROUP_COLS = {g: [FEATURE_ORDER.index(f) for f in FEATURE_GROUPS[g]["features"]]
              for g in GROUP_KEYS}


def load_detector(key):
    """Load a detector model by its label in DETECTORS. Cached."""
    clf = load(DETECTORS[key])
    assert hasattr(clf, "predict_proba")
    return clf


def load_techniques():
    """Load the 5 per-group forests. Cached."""
    models = load(GROUP_MODELS_PATH)
    assert set(GROUP_KEYS).issubset(models)
    return models


class FlowScorer:
    """Score + explain flows: detector for the label, techniques for the why."""

    def __init__(self, detector, techniques, col_mean, col_std):
        self.detector = detector
        self.techniques = techniques
        self.col_mean = col_mean
        self.col_std = col_std

    def predict_proba(self, X):
        return self.detector.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.detector.classes_[proba.argmax(axis=1)]

    def explain(self, flow):
        """Explain ONE flow (1D float array). Return structured explanation."""
        x = np.asarray(flow, dtype=np.float64)
        p = self.predict_proba(x[None, :])[0]
        alert = int(self.detector.classes_[int(p.argmax())])
        alert_class = CLASS_ORDER[alert]

        techniques = []
        for g in GROUP_KEYS:
            cols = GROUP_COLS[g]
            clf = self.techniques[g]
            p_group = clf.predict_proba(x[None, cols])[0]
            class_codes = clf.classes_
            gi = int(np.where(class_codes == alert)[0][0])
            base = p_group[gi]
            votes = np.argsort(-p_group)
            top_codes = [int(class_codes[v]) for v in votes[:3]]

            contrib = []
            for fi in cols:
                xp = x.copy()
                xp[fi] = self.col_mean[fi]
                p_pert = clf.predict_proba(xp[None, cols])[0][gi]
                contrib.append((fi, float(base - p_pert)))
            contrib.sort(key=lambda t: -t[1])
            top = [{
                "feature": FEATURE_ORDER[fi],
                "drop": round(d, 4),
                "value": round(float(x[fi]), 3),
                "zscore": round(float((x[fi] - self.col_mean[fi]) /
                                      max(self.col_std[fi], 1e-6)), 2),
            } for fi, d in contrib[:3]]

            techniques.append({
                "key": g,
                "name": SHORT_NAMES[g],
                "confidence_alert": round(float(base), 4),
                "fired": int(top_codes[0]) == alert,
                "top_votes": [CLASS_ORDER[c] for c in top_codes],
                "top_features": top,
            })

        return {
            "predicted_code": alert,
            "predicted_class": alert_class,
            "severity": _severity(alert_class),
            "probabilities": {CLASS_ORDER[c]: round(float(v), 4)
                              for c, v in enumerate(p)},
            "techniques": techniques,
        }


_SEVERITY = {"BENIGN": "SAFE", "PortScan": "MEDIUM", "BruteForce": "HIGH",
             "WebAttack": "HIGH", "DoS": "HIGH", "Botnet": "CRITICAL",
             "DDoS": "CRITICAL"}


def _severity(cls):
    return _SEVERITY.get(cls, "UNKNOWN")


def main():
    """CLI smoke test: score a real test flow and print the explanation."""
    from common import load_test
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    Xte, yte, _ = load_test()
    mean, std = Xte.mean(axis=0), Xte.std(axis=0)

    det = load_detector("HGB (balanced) - recommended")
    techs = load_techniques()
    scorer = FlowScorer(det, techs, mean, std)

    rng = np.random.RandomState(0)
    while True:
        i = int(rng.randint(len(Xte)))
        if yte[i] in (CLASS_ORDER.index("WebAttack"),
                      CLASS_ORDER.index("Botnet")):
            break

    expl = scorer.explain(Xte[i])
    print("flow %d  true=%s" % (i, CLASS_ORDER[yte[i]]))
    print("detector:", expl["predicted_class"], expl["severity"])
    for t in expl["techniques"]:
        mark = "FIRED" if t["fired"] else "--"
        print("  %-12s %-6s conf=%.3f top=%s" % (
            t["name"], mark, t["confidence_alert"],
            ", ".join(t["top_votes"])))
        for f in t["top_features"]:
            print("      %-32s drop %.3f  value %10s  z %.2f" % (
                f["feature"], f["drop"], f["value"], f["zscore"]))


if __name__ == "__main__":
    main()