"""Plain-language layer for alerts - the "normal person" dashboard vocabulary.

The dashboard never shows raw feature numbers. This module maps the engine's
69-feature names and the 5-technique explanation onto human sentences: a
friendly label + unit for each feature, a severity color, and a conscience of
what "fired" means in words. Everything here is pure (unit-testable).
"""
from src.config import SEVERITY

# feature -> (human label, unit). Units follow the CIC-IDS2017 convention
# (microseconds for timings, bytes, counts, per-second rates).
FRIENDLY = {
    "Flow Duration": ("how long the connection lasted", "s"),
    "Total Fwd Packets": ("packets sent", "pkts"),
    "Total Backward Packets": ("packets received", "pkts"),
    "Total Length of Fwd Packets": ("data sent", "B"),
    "Total Length of Bwd Packets": ("data received", "B"),
    "Flow Bytes/s": ("data rate", "B/s"),
    "Flow Packets/s": ("packet rate", "pkts/s"),
    "Fwd Packets/s": ("outgoing rate", "pkts/s"),
    "Bwd Packets/s": ("incoming rate", "pkts/s"),
    "Fwd Packet Length Mean": ("average packet size sent", "B"),
    "Fwd Packet Length Max": ("largest packet sent", "B"),
    "Fwd Packet Length Min": ("smallest packet sent", "B"),
    "Fwd Packet Length Std": ("how uneven the sent packet sizes were", "B"),
    "Bwd Packet Length Mean": ("average packet size received", "B"),
    "Bwd Packet Length Max": ("largest packet received", "B"),
    "Bwd Packet Length Min": ("smallest packet received", "B"),
    "Bwd Packet Length Std": ("how uneven the received packet sizes were", "B"),
    "Packet Length Mean": ("average packet size", "B"),
    "Packet Length Std": ("how uneven the packet sizes were", "B"),
    "Packet Length Variance": ("spread of the packet sizes", "B\u00b2"),
    "Average Packet Size": ("average packet size", "B"),
    "Avg Fwd Segment Size": ("average data chunk sent", "B"),
    "Avg Bwd Segment Size": ("average data chunk received", "B"),
    "Min Packet Length": ("smallest packet", "B"),
    "Max Packet Length": ("largest packet", "B"),
    "Flow IAT Mean": ("average time between packets", "\u00b5s"),
    "Flow IAT Max": ("longest pause between packets", "\u00b5s"),
    "Flow IAT Min": ("shortest pause between packets", "\u00b5s"),
    "Flow IAT Std": ("how uneven those pauses were", "\u00b5s"),
    "Fwd IAT Mean": ("avg time between outgoing packets", "\u00b5s"),
    "Fwd IAT Max": ("longest pause between outgoing packets", "\u00b5s"),
    "Fwd IAT Min": ("shortest pause between outgoing packets", "\u00b5s"),
    "Fwd IAT Std": ("how uneven those outgoing pauses were", "\u00b5s"),
    "Fwd IAT Total": ("total time spent sending", "\u00b5s"),
    "Bwd IAT Mean": ("avg time between incoming packets", "\u00b5s"),
    "Bwd IAT Max": ("longest pause between incoming packets", "\u00b5s"),
    "Bwd IAT Min": ("shortest pause between incoming packets", "\u00b5s"),
    "Bwd IAT Std": ("how uneven those incoming pauses were", "\u00b5s"),
    "Bwd IAT Total": ("total time spent receiving", "\u00b5s"),
    "Active Mean": ("average activity-burst length", "\u00b5s"),
    "Active Max": ("longest activity burst", "\u00b5s"),
    "Active Min": ("shortest activity burst", "\u00b5s"),
    "Active Std": ("how uneven the activity bursts were", "\u00b5s"),
    "Idle Mean": ("average idle stretch", "\u00b5s"),
    "Idle Max": ("longest idle stretch", "\u00b5s"),
    "Idle Min": ("shortest idle stretch", "\u00b5s"),
    "Idle Std": ("how uneven the idle stretches were", "\u00b5s"),
    "SYN Flag Count": ("connections started (SYN)", "flags"),
    "ACK Flag Count": ("acknowledgements sent (ACK)", "flags"),
    "FIN Flag Count": ("connections closed (FIN)", "flags"),
    "PSH Flag Count": ("push payloads (PSH)", "flags"),
    "RST Flag Count": ("connection resets (RST)", "flags"),
    "URG Flag Count": ("urgent data flags", "flags"),
    "CWE Flag Count": ("congestion-window flags", "flags"),
    "ECE Flag Count": ("early-congestion flags", "flags"),
    "Fwd PSH Flags": ("push payloads sent", "flags"),
    "Fwd URG Flags": ("urgent data sent", "flags"),
    "Down/Up Ratio": ("received-to-sent ratio", ""),
    "Destination Port": ("port contacted", ""),
    "Fwd Header Length": ("header bytes sent", "B"),
    "Bwd Header Length": ("header bytes received", "B"),
    "Subflow Fwd Packets": ("packets sent in this chunk", "pkts"),
    "Subflow Bwd Packets": ("packets received in this chunk", "pkts"),
    "Subflow Fwd Bytes": ("data sent in this chunk", "B"),
    "Subflow Bwd Bytes": ("data received in this chunk", "B"),
    "Init_Win_bytes_forward": ("sender advertised window", "B"),
    "Init_Win_bytes_backward": ("receiver advertised window", "B"),
    "act_data_pkt_fwd": ("packets carrying data sent", "pkts"),
    "min_seg_size_forward": ("smallest segment sent", "B"),
}

DEFAULT_LABEL = ("feature value", "")

SEVERITY_COLOR = {
    "SAFE": "#27ae60", "MEDIUM": "#f39c12", "HIGH": "#e67e22",
    "CRITICAL": "#c0392b",
}

TECHNIQUE_WORDS = {
    "A_Flow_Volume": "how much data moved and how fast",
    "B_Packet_Statistics": "packet sizes and shapes",
    "C_Timing_IAT": "the rhythm and pauses of the exchange",
    "D_TCP_Flags_Connection": "connection control signals (SYN/ACK/RST...)",
    "E_Endpoint_Window": "endpoints and connection windows",
}


def feature_label(name):
    """(human label, unit) for a feature name, with a safe fallback."""
    if name in FRIENDLY:
        return FRIENDLY[name]
    return DEFAULT_LABEL


def severity_color(severity):
    return SEVERITY_COLOR.get(severity, "#7f8c8d")


def describe_feature(feat):
    """One feature's deviation in words: '... about 3.4x the usual amount'."""
    label, unit = feature_label(feat["feature"])
    z = abs(feat["zscore"])
    if z < 1.0:
        guess = "about the usual amount"
    elif z < 3.0:
        guess = "somewhat %s than usual" % ("higher" if feat["zscore"] > 0
                                            else "lower")
    else:
        guess = "far %s than usual (%s\u00d7 the normal spread)" % (
            "higher" if feat["zscore"] > 0 else "lower", round(z, 1))
    val = _fmt(feat["value"])
    suffix = (" " + unit) if unit else ""
    return "%s = %s%s \u2014 %s" % (label, val, suffix, guess)


def _fmt(v):
    a = abs(v)
    if a and (a >= 1000 or a < 0.01):
        return ("%.1e" % v if a >= 1e6 else "%.3f" % v).rstrip("0").rstrip(".")
    if a >= 100:
        return "%.0f" % v
    return "%.2f" % v


def alert_headline(alert):
    """One-line verdict for the alert queue table."""
    sv = alert["severity"]
    cls = alert["predicted_class"]
    if cls == "BENIGN":
        return "%s \u2014 looked normal" % sv
    return "%s \u2014 flagged as %s (confidence %.0f%%)" % (
        sv, cls, 100.0 * alert["confidence"])


def explain_lines(expl):
    """Turn the 5-technique explanation into a short human report."""
    if not expl:
        return ["No explanation available."]
    lines = []
    fired = [t for t in expl["techniques"] if t.get("fired")]
    if not fired:
        lines.append(
            "No single technique was strongly confident on its own \u2014 the "
            "%s verdict came from the combined detector." % expl["predicted_class"])
        return lines
    lines.append("What stood out about this %s:"
                 % expl["predicted_class"])
    for t in fired:
        reason = TECHNIQUE_WORDS.get(t["key"], t["name"].lower())
        top = t["top_features"][:2]
        if not top:
            continue
        f = top[0]
        lines.append("\u2022 %s: %s" % (reason, describe_feature(f)))
        if len(top) > 1:
            lines.append("     also %s" % describe_feature(top[1]))
    return lines


def severity_rank(severity):
    return {"SAFE": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(severity, 0)