"""Phase 4 - live detector: raw frames in, alert + reason out.

Ties three layers into one stream pipeline:

  * CAPTURE   - ``capture.frame_to_event`` (scapy) turns raw Ethernet frames
                into PacketEvents.
  * FLOW MATH - ``flow_features.FlowTable`` accumulates bidirectional flows and
                finalizes CIC-IDS2017-compatible 69-feature vectors when a flow
                goes idle for ``flow_timeout_s``.
  * SCORING   - a ``pipeline.FlowScorer`` (balanced HGB detector + 5 per-group
                explanation forests, optional) scores the finished vector.

``LiveDetector.ingest(raw, ts)`` is the whole loop in one call; it returns
alerts for every flow that timed out, so a scrubber or the dashboard never has
to touch capture internals. Two drivers are provided:

  * ``replay(pcap_path)``  - offline, pcap replay. No admin, no Npcap; the
    same parse path as live sniffing, so results are comparable.
  * ``live_sniff()``       - scapy.sniff on an interface (needs Npcap/admin).

CLI:  python src/live/detector.py --pcap traces/demo.pcap [--explain]
      python src/live/detector.py --live --iface eth0 --count 500
"""
import argparse
import os
import sys

import numpy as np

from scapy.utils import PcapReader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.config import (FEATURE_ORDER, CLASS_ORDER, SEVERITY, PROCESSED_TEST)
from src.live.capture import packet_to_event
from src.live.flow_features import FlowTable, Flow

DETECTOR_KEY = "HGB (balanced) - recommended"
FLOW_TIMEOUT_S = 15.0


def _load_scorer(detector_key, stats_n=50_000, with_explain=True):
    """Build a FlowScorer for the given detector, stats from test head."""
    from src.pipeline import load_detector, load_techniques, FlowScorer
    det = load_detector(detector_key)
    if not with_explain:
        return FlowScorer(det, {}, None, None)
    import pandas as pd
    df = pd.read_csv(PROCESSED_TEST, usecols=FEATURE_ORDER,
                     nrows=stats_n, dtype={c: "float32" for c in FEATURE_ORDER})
    X = df.to_numpy(np.float32)
    techs = load_techniques()
    return FlowScorer(det, techs, X.mean(axis=0), X.std(axis=0))


class LiveDetector:
    """Capture -> FlowTable -> scorer. Feed frames, get alerts."""

    def __init__(self, scorer, flow_timeout_s=FLOW_TIMEOUT_S,
                 capture=packet_to_event):
        self.scorer = scorer
        self.table = FlowTable(flow_timeout_s)
        self.capture = capture
        self.n_flushed = 0
        self.n_packets = 0

    def score_vector(self, vec):
        """Predict one feature vector -> (class name, severity, conf, proba)."""
        proba = self.scorer.predict_proba(np.asarray(vec, np.float32)[None, :])[0]
        code = int(self.scorer.detector.classes_[int(proba.argmax())])
        cls = CLASS_ORDER[code]
        return cls, SEVERITY.get(cls, "UNKNOWN"), float(proba.max()), \
            {CLASS_ORDER[c]: float(v) for c, v in enumerate(proba)}

    def ingest(self, raw, ts=None, explain=False):
        """Feed one raw frame; return alerts for flows that just expired."""
        ev = self.capture(raw, ts)
        if ev is None:
            return []
        self.n_packets += 1
        self.table.add(ev)
        alerts = []
        for key, feats in self.table.flush_expired_keyed(ev.ts):
            self.n_flushed += 1
            vec = _vector(feats)
            cls, severity, conf, proba = self.score_vector(vec)
            alert = {
                "key": key,
                "src": key[0], "dst": key[1], "dport": key[3], "proto": key[4],
                "features": {n: float(v) for n, v in feats.items()},
                "predicted_class": cls,
                "severity": severity,
                "confidence": conf,
                "probabilities": proba,
            }
            if explain and getattr(self.scorer, "techniques", None):
                alert["explanation"] = self.scorer.explain(vec)
            else:
                alert["explanation"] = None
            alerts.append(alert)
        return alerts

    def replay(self, pcap_path, explain=False, emit=None):
        """Stream a pcap offline (same parse path as live sniffing)."""
        with PcapReader(pcap_path) as rd:
            for pkt in rd:
                alerts = self.ingest(bytes(pkt), ts=getattr(pkt, "time", None),
                                     explain=explain)
                for a in alerts:
                    if emit is not None:
                        emit(a)

    def live_sniff(self, iface, count, explain=False, emit=None):
        """scapy sniff on an interface; requires Npcap + admin on Windows."""
        from scapy.sendrecv import sniff

        def _cb(pkt):
            alerts = self.ingest(bytes(pkt), ts=getattr(pkt, "time", None),
                                 explain=explain)
            for a in alerts:
                if emit is not None:
                    emit(a)

        sniff(iface=iface, store=0, count=count, prn=_cb)


def _vector(feats):
    return Flow.vector_from_dict(feats)


def _fmt(alert):
    cls = alert["predicted_class"]
    mark = "  " if cls == "BENIGN" else "!!"
    return "%s %-9s %-8s -> %-8s :%-5s %-8s %.3f %s" % (
        mark, alert["severity"], alert["src"], alert["dst"], alert["dport"],
        cls, alert["confidence"],
        "+ explain" if alert.get("explanation") else "")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Live NIDAI detector: pcap replay or interface sniff.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcap", help="offline replay of a pcap file")
    src.add_argument("--live", action="store_true",
                     help="sniff a live interface (Npcap/admin required)")
    ap.add_argument("--iface", default=None, help="interface for --live")
    ap.add_argument("--count", type=int, default=500,
                    help="max packets for --live")
    ap.add_argument("--timeout", type=float, default=FLOW_TIMEOUT_S,
                    help="idle seconds before a flow is finalized")
    ap.add_argument("--detector", default=DETECTOR_KEY,
                    help="detector key in src/pipeline.DETECTORS")
    ap.add_argument("--explain", action="store_true",
                    help="attach the 5-technique explanation to each alert")
    ap.add_argument("--show-safe", action="store_true",
                    help="also print BENIGN flows")
    opts = ap.parse_args(argv)

    scorer = _load_scorer(opts.detector, with_explain=opts.explain)
    det = LiveDetector(scorer, flow_timeout_s=opts.timeout)

    def emit(a):
        if a["predicted_class"] == "BENIGN" and not opts.show_safe:
            return
        print(_fmt(a), flush=True)

    if opts.live:
        if opts.iface is None:
            ap.error("--live requires --iface")
        print("Sniffing on %s (max %d packets)... Ctrl-C to stop"
              % (opts.iface, opts.count), flush=True)
        det.live_sniff(opts.iface, opts.count, explain=opts.explain, emit=emit)
    else:
        print("Replaying %s..." % opts.pcap, flush=True)
        det.replay(opts.pcap, explain=opts.explain, emit=emit)

    print("packets=%d flows_finalized=%d" % (det.n_packets, det.n_flushed))


if __name__ == "__main__":
    main()