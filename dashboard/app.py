"""Stage 8 - “alert + reason” dashboard over frozen models.

Two layers are fused in one view:
  1. DETECTION - a single strong classifier (balanced HGB by default) flags a
     flow; we show the alert queue, severities, and the detector's own class
     probabilities.
  2. EXPLANATION - picking a flow replays the 5 per-group forests (volume,
     packet, timing, TCP flags, endpoint) and reports which techniques fired
     and on which features the alert class confidence drops when the feature
     is removed.

Run:  streamlit run dashboard/app.py
"""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (FEATURE_ORDER, CLASS_ORDER, SEVERITY,
                        PROCESSED_TEST)
from src.pipeline import DETECTORS, load_detector, load_techniques, FlowScorer

st.set_page_config(page_title="NIDS 7 - alert + reason", layout="wide")

SEVERITY_RANK = {"SAFE": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

st.title("NIDS - alert + reason dashboard")
st.caption(
    "Frozen models from Studies 4-7. Detection core = balanced HGB on all 69 "
    "features; explanation = 5 per-group Random Forests (Stage 5).")

st.sidebar.header("Controls")
detector_key = st.sidebar.selectbox("Detection core", list(DETECTORS.keys()),
                                    index=0)
n_rows = st.sidebar.slider("Test rows to scan", 5_000, 100_000, 20_000, 5_000)
min_severity = st.sidebar.selectbox("Minimum severity to show",
                                    ["SAFE", "MEDIUM", "HIGH", "CRITICAL"],
                                    index=0)
random_flow = st.sidebar.button("Draw a random flagged flow",
                                width='stretch')


@st.cache_data(show_spinner=False)
def load_chunk(n):
    """First n labelled rows of the held-out test set (float32)."""
    df = pd.read_csv(PROCESSED_TEST,
                     usecols=FEATURE_ORDER + ["LabelGroup"],
                     nrows=n,
                     dtype={c: "float32" for c in FEATURE_ORDER})
    df = df[df["LabelGroup"] != "UNKNOWN"].reset_index(drop=True)
    return df


@st.cache_resource(show_spinner="Loading models...")
def models_for(detector_key):
    det = load_detector(detector_key)
    techs = load_techniques()
    return det, techs


@st.cache_data(show_spinner="Scoring flows...")
def detect(df, detector_key):
    det, _ = models_for(detector_key)
    X = df[FEATURE_ORDER].to_numpy(np.float32)
    proba = det.predict_proba(X)
    codes = det.classes_[proba.argmax(axis=1)]
    cols = {"pred": codes}
    for i, c in enumerate(CLASS_ORDER):
        cols["p_" + c] = proba[:, i]
    return pd.DataFrame(cols), X


@st.cache_data(show_spinner=False)
def explain_flow(X, detector_key, flow_i):
    det, techs = models_for(detector_key)
    mean, std = X.mean(axis=0), X.std(axis=0)
    scorer = FlowScorer(det, techs, mean, std)
    return scorer.explain(X[flow_i])


df = load_chunk(n_rows)
table, X = detect(df, detector_key)

table["true"] = df["LabelGroup"].astype(str).values
table["pred_label"] = table["pred"].map(dict(enumerate(CLASS_ORDER)))
table["severity"] = table["pred_label"].map(SEVERITY)
table["confidence"] = table[["p_" + c for c in CLASS_ORDER]].max(axis=1)
table["flow"] = np.arange(len(table))

flagged = table[table["severity"].map(SEVERITY_RANK)
                .ge(SEVERITY_RANK[min_severity])].copy()
flagged = flagged.sort_values("severity", ascending=False)

st.subheader("Alert queue")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Flows scanned", f"{len(table):,}")
c2.metric("Alerts shown", f"{len(flagged):,}")
c3.metric("Critical alerts", f"{(flagged['severity'] == 'CRITICAL').sum():,}")
c4.metric("Attacks in data (true)",
          f"{table['true'].ne('BENIGN').sum():,}")

st.bar_chart(table["pred_label"].value_counts())

tru = table["true"].ne("BENIGN")
prd = table["pred_label"].ne("BENIGN")
st.caption(
    f"Chunk-level: {(tru == prd).mean():.4%} accuracy | attacks caught "
    f"{(tru & prd).sum():,}/{tru.sum():,} | false alarms {(~tru & prd).sum():,}")

st.dataframe(
    flagged[["flow", "true", "pred_label", "severity", "confidence"]]
    .rename(columns={"pred_label": "detected", "confidence": "conf"}),
    width='stretch', hide_index=True)

st.subheader("Why was a flow flagged?")
if flagged.empty:
    st.info("No flows at the selected severity - lower the threshold.")
else:
    if random_flow:
        choice = int(flagged.sample(1, random_state=0).iloc[0]["flow"])
    else:
        labels = ["#%d  det=%s  true=%s" % (
            r["flow"], r["pred_label"], r["true"])
            for _, r in flagged.head(200).iterrows()]
        sel = st.selectbox("Flagged flow", labels, index=0,
                           format_func=lambda s: s)
        choice = int(sel[1:sel.find("  ")])
    st.caption("Explaining flow #%d" % choice)

    expl = explain_flow(X, detector_key, choice)
    st.markdown("**Detected: %s (%s)**  |  detector probabilities:  %s" % (
        expl["predicted_class"], expl["severity"],
        "  ".join("%s %.2f" % (k, v) for k, v in sorted(
            expl["probabilities"].items(), key=lambda kv: -kv[1])[:4])))

    tech_df = pd.DataFrame({
        "technique": [t["name"] for t in expl["techniques"]],
        "fired": ["YES" if t["fired"] else "" for t in expl["techniques"]],
        "conf(alert)": [t["confidence_alert"] for t in expl["techniques"]],
        "group top votes": [" ".join(t["top_votes"])
                            for t in expl["techniques"]],
    })
    st.dataframe(tech_df, width='stretch', hide_index=True)
    st.bar_chart({t["name"]: t["confidence_alert"]
                  for t in expl["techniques"]})

    for t in expl["techniques"]:
        title = "%s %s" % (t["name"],
                           "FIRED" if t["fired"] else "(no signal)")
        with st.expander(title):
            st.dataframe(pd.DataFrame(t["top_features"]),
                         width='stretch', hide_index=True)
