"""Live NIDAI Dashboard — Real-Time Network Security Monitoring & Intrusion Detection.

Visualises live network flows, classifications, and 5-technique AI explanations
in a cybersecurity SOC interface.

Supports 3 operational sources:
  1. LAB REPLAY — Streams real held-out CIC-IDS2017 test flows paced over time.
  2. PACKET REPLAY — Feeds a generated PCAP stream through the live capture engine.
  3. LIVE SNIFF — Captures and classifies live packets directly from the network adapter.

Run:
  streamlit run dashboard/live.py
"""
import os
import sys
import time
from collections import Counter

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CLASS_ORDER, SEVERITY
from src.live.detector import _load_scorer, DETECTOR_KEY, resolve_interface
from src.live.livefeed import RowFeed
from src.live.plain import (alert_headline, explain_lines, severity_rank, severity_color)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_PCAP = os.path.join(BASE, "results", "demo_live.pcap")

SOURCES = [
    "Lab replay — real CIC test rows (recommended)",
    "Packet replay — Stage 10 capture on a demo pcap",
    "Live sniff — scapy on an interface (needs Npcap)",
]

REPAINT_S = 0.8

# --------------------------------------------------------------- CSS Styling ---- #
CUSTOM_CSS = """
<style>
  /* Main background & typography */
  .stApp {
      background-color: #0b0f19;
      color: #e2e8f0;
  }
  
  /* Cyber header banner */
  .nidai-banner {
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 24px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
  }
  .nidai-title {
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.5px;
      color: #38bdf8;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 12px;
  }
  .nidai-subtitle {
      font-size: 14px;
      color: #94a3b8;
      margin-top: 6px;
      margin-bottom: 0;
  }

  /* Metric cards */
  div[data-testid="stMetric"] {
      background-color: #1e293b;
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 14px 18px;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
  }
  div[data-testid="stMetricLabel"] {
      color: #94a3b8 !important;
      font-size: 13px !important;
      font-weight: 600 !important;
      text-transform: uppercase;
      letter-spacing: 0.5px;
  }
  div[data-testid="stMetricValue"] {
      color: #f8fafc !important;
      font-size: 28px !important;
      font-weight: 700 !important;
  }

  /* Status badge pulse */
  .live-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background-color: rgba(34, 197, 94, 0.15);
      border: 1px solid rgba(34, 197, 94, 0.4);
      color: #4ade80;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
  }
  .live-dot {
      width: 8px;
      height: 8px;
      background-color: #22c55e;
      border-radius: 50%;
      box-shadow: 0 0 10px #22c55e;
  }

  /* Threat detail panel */
  .threat-card {
      background-color: #1e293b;
      border: 1px solid #475569;
      border-left: 6px solid #ef4444;
      border-radius: 8px;
      padding: 16px 20px;
      margin-top: 12px;
      margin-bottom: 16px;
  }
</style>
"""


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
        feed.source_label = "LAB REPLAY (Held-out CIC-IDS2017)"
        feed.start()
        return feed
    if source.startswith("Packet replay"):
        feed = _packet_feed(scorer, delay=max(speed, 0.02))
        feed.source_label = "PCAP REPLAY (Simulated Traffic)"
        return feed
    feed = _sniff_feed(scorer)
    feed.source_label = "LIVE SNIFF (Network Interface)"
    return feed


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
                    a["true"] = "PCAP Packet"
                    feed.alerts.append(a)
                    if delay > 0:
                        time.sleep(delay)
        for a in det.flush_all():
            a["true"] = "PCAP Packet"
            feed.alerts.append(a)
        feed.finished.set()

    threading.Thread(target=_replay, daemon=True).start()
    return feed


def _sniff_feed(scorer):
    """Live sniff driver using Scapy + Npcap on the active interface."""
    import threading
    from src.live.detector import LiveDetector

    chosen_iface = resolve_interface()
    iface_name = getattr(chosen_iface, "name", str(chosen_iface))
    st.sidebar.info(f"Sniffing on interface: {iface_name}")

    feed = RowFeed(scorer, [], [], delay=0.0)
    feed.lab = False

    def _sniff():
        det = LiveDetector(scorer, flow_timeout_s=3.0)

        def _emit(alert):
            alert["true"] = "Live Capture"
            feed.alerts.append(alert)

        det.live_sniff(iface=chosen_iface, emit=_emit)

    threading.Thread(target=_sniff, daemon=True).start()
    return feed


# ----------------------------------------------------------------- view ---- #
def _paint_distribution(rows):
    counts = Counter(a["predicted_class"] for a in rows)
    data = {c: counts.get(c, 0) for c in CLASS_ORDER}
    chart_df = pd.DataFrame(list(data.items()), columns=["Traffic Class", "Count"]).set_index("Traffic Class")
    st.bar_chart(chart_df, color="#38bdf8")


def _technique_table(expl):
    df = pd.DataFrame({
        "Detection Technique": [t["name"] for t in expl["techniques"]],
        "Triggered": ["🔴 FIRED" if t.get("fired") else "⚪ Inactive" for t in expl["techniques"]],
        "Confidence": [f"{t['confidence_alert'] * 100:.1f}%" for t in expl["techniques"]],
        "Top Supporting Votes": [", ".join(t.get("top_votes", [])) for t in expl["techniques"]],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)


def _paint(feed, min_sev, placeholders):
    rows = list(reversed(feed.alerts))
    if not rows:
        placeholders[0].metric("Flows Evaluated", "0")
        placeholders[1].metric("Threat Alerts", "0")
        placeholders[2].metric("Critical Incidents", "0")
        placeholders[3].metric("Ground Truth Known", "0")
        st.info("⚡ Waiting for first network flows to finalize...")
        return

    nb = [a for a in rows if a["predicted_class"] != "BENIGN"]
    crit = sum(1 for a in rows if a["severity"] == "CRITICAL")
    labelled = sum(1 for a in rows if a.get("true") and a.get("true") not in ("PCAP Packet", "Live Capture"))

    placeholders[0].metric("Flows Evaluated", f"{len(rows):,}")
    placeholders[1].metric("Threat Alerts", f"{len(nb):,}")
    placeholders[2].metric("Critical Incidents", f"{crit:,}")
    placeholders[3].metric("Verified Truth Known", f"{labelled:,}")

    # Alert queue
    st.markdown("### 📋 Live Flow & Threat Stream")
    view_data = []
    for idx, a in enumerate(rows):
        flow_num = a.get("num", len(rows) - idx)
        view_data.append({
            "Flow ID": f"#{flow_num}",
            "Severity": a["severity"],
            "Classification": a["predicted_class"],
            "Confidence": f"{a['confidence'] * 100:.1f}%",
            "Src -> Dst": f"{a['src'] or 'Lab'} -> {a['dst'] or 'Target'}:{a['dport'] or '?'}",
            "Protocol": "TCP" if a.get("proto") == 6 else ("UDP" if a.get("proto") == 17 else "IP"),
            "Ground Truth": a.get("true") or "On-the-wire",
        })

    view_df = pd.DataFrame(view_data)
    # Severity filter
    filtered_df = view_df[view_df["Severity"].map(severity_rank).ge(severity_rank(min_sev))]
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    # Class distribution
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("### 📊 Attack Class Distribution")
        _paint_distribution(rows)
    with col2:
        st.markdown("### 🛡️ Threat Summary & Insights")
        total_non_benign = len(nb)
        if total_non_benign == 0:
            st.success("🟢 No intrusions detected in current traffic window.")
        else:
            top_threat = Counter(a["predicted_class"] for a in nb).most_common(1)[0]
            st.warning(f"⚠️ Primary threat: **{top_threat[0]}** ({top_threat[1]} flows detected)")
            st.write(f"Total anomalous flows: **{total_non_benign}** / {len(rows)} evaluated")

    # Threat Drill-Down
    st.markdown("---")
    st.markdown("### 🔍 Threat Forensics & AI Explanation")
    alarms = [a for a in rows if a["predicted_class"] != "BENIGN"
              and severity_rank(a["severity"]) >= severity_rank(min_sev)]
    if not alarms:
        st.info("No non-benign flows currently meet the selected severity threshold.")
        return

    labels = [
        f"Flow #{a.get('num', idx + 1)}: {a['predicted_class']} ({a['confidence'] * 100:.1f}% conf) — Port {a['dport']}"
        for idx, a in enumerate(alarms)
    ]
    choice = st.selectbox("Select Flagged Flow to Investigate", labels, index=0)
    selected_idx = labels.index(choice)
    alert = alarms[selected_idx]

    # Threat Card
    sev = alert["severity"]
    border_color = severity_color(sev)
    st.markdown(f"""
    <div class="threat-card" style="border-left: 6px solid {border_color};">
        <h4 style="margin: 0; color: #f8fafc;">{alert_headline(alert)}</h4>
        <p style="margin: 6px 0 0 0; color: #94a3b8; font-family: monospace;">
            Source: {alert['src'] or 'External'} &nbsp;➔&nbsp; Destination: {alert['dst'] or 'Internal'}:{alert['dport']} &nbsp;|&nbsp; Protocol: {alert.get('proto', 6)}
        </p>
    </div>
    """, unsafe_allow_html=True)

    if alert.get("explanation"):
        st.markdown("#### 🔬 Why Did NIDAI Flag This Flow?")
        for line in explain_lines(alert["explanation"]):
            st.markdown(f"> {line}")
        st.markdown("#### 📐 Per-Technique Group Breakdown")
        _technique_table(alert["explanation"])
    elif feed.pending_explanations():
        st.info("⏳ Computing 5-technique feature perturbation explanation... (arrives shortly)")
    else:
        st.info("Scored by the primary detection core; individual feature groups remained sub-threshold.")


# ------------------------------------------------------------------ App ---- #
st.set_page_config(
    page_title="NIDAI — Live Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "cfg" not in st.session_state:
    st.session_state.cfg = None
if "feed" not in st.session_state:
    st.session_state.feed = None
if "detector_key" not in st.session_state:
    st.session_state.detector_key = DETECTOR_KEY

# Header banner
st.markdown("""
<div class="nidai-banner">
    <div class="nidai-title">
        <span>🛡️ NIDAI — Live Network Intrusion Detection System</span>
        <span class="live-badge"><span class="live-dot"></span> LIVE ENGINE ACTIVE</span>
    </div>
    <p class="nidai-subtitle">
        Real-time packet inspection, 69-feature flow aggregation, and multi-technique attack attribution powered by CIC-IDS2017 pretrained models.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("⚙️ Control Panel")
source = st.sidebar.selectbox("Traffic Source", SOURCES, index=0)
speed = st.sidebar.slider(
    "Alert Emission Delay (seconds)", 0.0, 2.0, 0.15, 0.05,
    help="Delay between displayed alerts in lab replay mode (0 = maximum throughput)",
)
min_sev = st.sidebar.selectbox(
    "Minimum Severity Filter",
    ["SAFE", "MEDIUM", "HIGH", "CRITICAL"], index=0,
)
restart = st.sidebar.button("🔄 Restart Stream", use_container_width=True)

cfg = (source, speed, restart)
if restart or st.session_state.cfg != cfg or st.session_state.feed is None:
    st.session_state.cfg = cfg
    st.session_state.feed = _start_feed(source, speed, st.session_state.detector_key)

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
