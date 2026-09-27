"""Live NIDAI dashboard - a watch-it-work view normal people can read.

Three sources, one friendly alert queue:

  1. LAB REPLAY (default) - real held-out CIC-IDS2017 rows scored live by the
     real detector, paced so alerts arrive over time. True labels are shown
     for honesty. Explanations arrive a few seconds after each rare alert.
  2. PACKET REPLAY - the actual Stage 10 capture path over a generated demo
     pcap (real packets, honest, mostly quiet - see docs/experiments.md).
  3. LIVE SNIFF - scapy on an interface; needs Npcap + admin (Windows).

Nothing here shows raw 69-feature numbers: src/live/plain.py turns features
and the 5-technique explanation into sentences.

Run:  streamlit run dashboard/live.py
"""
import os
import sys
import time
from collections import Counter

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CLASS_ORDER
from src.live.detector import _load_scorer, DETECTOR_KEY
from src.live.livefeed import RowFeed
from src.live.plain import (alert_headline, explain_lines, severity_rank)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_PCAP = os.path.join(BASE, "results", "demo_live.pcap")

SOURCES = [
    "Lab replay - real CIC test rows (recommended)",
    "Packet replay - Stage 10 capture on a demo pcap",
    "Live sniff - scapy on an interface (needs Npcap)",
]

REPAINT_S = 0.8


# ---------------------------------------------------------------- feeds ---- #
def _start_feed(source, speed, detector_key):
    """Build/start the feed for the chosen source and return it."""
    scorer = _load_scorer(detector_key, with_explain=True)
    if source.startswith("Lab replay"):
        from src.live.demo_traffic import sample_test_rows
        with st.spinner("Loading labelled test rows..."):
            X, labels = sample_test_rows(220)
        feed = RowFeed(scorer, X, labels, delay=max(speed, 0.02))
        feed.lab = True
        feed.start()
        return feed
    if source.startswith("Packet replay"):
        return _packet_feed(scorer, delay=max(speed, 0.02))
    return _sniff_feed(scorer)


def _packet_feed(scorer, delay):
    """Replay the generated demo pcap through LiveDetector, paced per alert."""
    import threading

    from scapy.utils import PcapReader

    from src.live.detector import LiveDetector

    if not os.path.exists(DEMO_PCAP):
        from src.live.demo_traffic import write_demo_pcap
        with st.spinner("Generating demo traffic..."):
            write_demo_pcap(DEMO_PCAP, 220)

    feed = RowFeed(scorer, [], [], delay=0.0)
    feed.lab = False

    def _replay():
        det = LiveDetector(scorer, flow_timeout_s=1.5)
        with PcapReader(DEMO_PCAP) as rd:
            for pkt in rd:
                for a in det.ingest(bytes(pkt), ts=getattr(pkt, "time", None)):
                    a["true"] = None
                    feed.alerts.append(a)
                    if delay > 0:
                        time.sleep(delay)
        feed.finished.set()

    threading.Thread(target=_replay, daemon=True).start()
    return feed


def _sniff_feed(scorer):
    """Live sniff driver - guard against the missing-Npcap case up front."""
    import threading

    from src.live.detector import LiveDetector

    st.sidebar.warning(
        "Interface sniff needs scapy + Npcap with admin rights. If nothing "
        "appears, install Npcap and run the terminal with admin.")
    feed = RowFeed(scorer, [], [], delay=0.0)
    feed.lab = False

    def _sniff():
        det = LiveDetector(scorer, flow_timeout_s=15.0)

        def _emit(pkt):
            for a in det.ingest(bytes(pkt), ts=getattr(pkt, "time", None)):
                a["true"] = None
                feed.alerts.append(a)

        from scapy.sendrecv import sniff
        sniff(prn=_emit, store=0)

    threading.Thread(target=_sniff, daemon=True).start()
    return feed


# ----------------------------------------------------------------- view ---- #
def _paint_distribution(rows):
    counts = Counter(a["predicted_class"] for a in rows)
    st.bar_chart({c: counts.get(c, 0) for c in CLASS_ORDER if counts.get(c)})


def _technique_table(expl):
    df = pd.DataFrame({
        "angle": [t["name"] for t in expl["techniques"]],
        "fired": ["YES" if t.get("fired") else "" for t in expl["techniques"]],
        "confidence": [t["confidence_alert"] for t in expl["techniques"]],
    })
    st.caption("Detection angles (each looks at one side of the traffic):")
    st.dataframe(df, width="stretch", hide_index=True)


def _paint(feed, min_sev, placeholders):
    rows = list(reversed(feed.alerts))
    if not rows:
        placeholders[0].metric("Flows seen", "0")
        placeholders[1].metric("Alerts", "0")
        placeholders[2].metric("Critical", "0")
        placeholders[3].metric("Ground truth known", "0")
        st.info("Waiting for the first flow...")
        return

    nb = [a for a in rows if a["predicted_class"] != "BENIGN"]
    crit = sum(1 for a in rows if a["severity"] == "CRITICAL")
    labelled = sum(1 for a in rows if a.get("true"))
    placeholders[0].metric("Flows seen", f"{len(rows):,}")
    placeholders[1].metric("Alerts", f"{len(nb):,}")
    placeholders[2].metric("Critical", f"{crit:,}")
    placeholders[3].metric("Ground truth known", f"{labelled:,}")

    st.subheader("Alert queue (newest first)")
    view = pd.DataFrame({
        "flow": [a["num"] for a in rows],
        "severity": [a["severity"] for a in rows],
        "detected as": [a["predicted_class"] for a in rows],
        "confidence": [a["confidence"] for a in rows],
        "port": [a["dport"] or "?" for a in rows],
        "ground truth": [a.get("true") or "on the wire" for a in rows],
    })
    view = view[view["severity"].map(severity_rank)
                .ge(severity_rank(min_sev))]
    st.dataframe(view, width="stretch", hide_index=True)

    st.caption("Ground truth 'on the wire' means the flow was captured as "
               "packets; 'BENIGN/DoS/...' means a labelled lab row. A detector "
               "that differs from the label is not wrong - labels and "
               "predictions are independent opinions.")
    _paint_distribution(rows)

    st.subheader("What happened, in plain words")
    alarms = [a for a in rows if a["predicted_class"] != "BENIGN"
              and severity_rank(a["severity"]) >= severity_rank(min_sev)]
    if not alarms:
        st.info("No non-normal flows at this severity yet.")
        return
    labels = ["#%d  %s  (%.0f%% sure)" % (a["num"], a["predicted_class"],
                                          100 * a["confidence"])
              for a in alarms]
    choice = st.selectbox("Pick an alarm", labels, index=0)
    alert = alarms[labels.index(choice)]
    st.markdown("**%s**" % alert_headline(alert))
    if alert["src"]:
        st.caption("from %s  ->  %s : port %s (proto %d)" % (
            alert["src"], alert["dst"], alert["dport"], alert["proto"]))
    if alert.get("explanation"):
        for line in explain_lines(alert["explanation"]):
            st.markdown(line)
        _technique_table(alert["explanation"])
    elif feed.pending_explanations():
        st.info("Explanation is being analysed (seconds per alert) - "
                "check back shortly.")
    else:
        st.info("This flow was scored by the combined detector; no single "
                "technique was confident enough to explain on its own.")


# ------------------------------------------------------------------ app ---- #
st.set_page_config(page_title="NIDAI live watch", layout="wide")

if "cfg" not in st.session_state:
    st.session_state.cfg = None
if "feed" not in st.session_state:
    st.session_state.feed = None
if "detector_key" not in st.session_state:
    st.session_state.detector_key = DETECTOR_KEY

st.title("NIDAI - what your network is doing, right now")
st.caption("Detection: balanced HGB on all 69 CIC features. Explanation: 5 "
           "per-group forests. Sources and speeds are live controls.")

st.sidebar.header("LIVE feed controls")
source = st.sidebar.selectbox("Traffic source", SOURCES, index=0)
speed = st.sidebar.slider("Alert arrival speed", 0.0, 2.0, 0.15, 0.05,
                          help="seconds the feed waits between alerts (0 = instant)")
min_sev = st.sidebar.selectbox("Minimum severity to show",
                               ["SAFE", "MEDIUM", "HIGH", "CRITICAL"], index=0)
restart = st.sidebar.button("Restart feed")

cfg = (source, speed, restart)
if restart or st.session_state.cfg != cfg or st.session_state.feed is None:
    st.session_state.cfg = cfg
    st.session_state.feed = _start_feed(source, speed,
                                        st.session_state.detector_key)

feed = st.session_state.feed
placeholders = st.columns(4)
now = time.time()
if now - st.session_state.get("last_paint", 0.0) > REPAINT_S:
    st.session_state.last_paint = now
    _paint(feed, min_sev, placeholders)
    if not feed.finished.is_set() and speed > 0.05:
        time.sleep(0.05)
        st.rerun()
else:
    _paint(feed, min_sev, placeholders)
