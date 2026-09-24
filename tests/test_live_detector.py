"""Unit tests for the live detector engine (Phase 4) - no models, no Npcap.

The scorer is a fake so the tests exercise the capture->FlowTable->scoring
plumbing deterministically: flow merging across directions, timeout flushing,
alert payload shape, non-IP skipping, and pcap replay via a written temp file.

Run:  python tests/test_live_detector.py
  or:  python -m unittest tests.test_live_detector -v
"""
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import Ether
from scapy.utils import wrpcap

from src.config import CLASS_ORDER, FEATURE_ORDER
from src.live.detector import LiveDetector
from src.live.flow_features import Flow

A, B = "10.0.0.1", "10.0.0.2"
C_S, D_S = "10.0.6.1", "10.0.6.2"


def frame(ts, src, dst, sport, dport, flags="A", window=100, payload=b""):
    pkt = Ether() / IP(src=src, dst=dst) / \
        TCP(sport=sport, dport=dport, flags=flags, window=window) / payload
    pkt.time = ts
    return bytes(pkt)


class FakeScorer:
    """Always predicts `cls`; exposes the detector.classes_ contract."""

    def __init__(self, cls="BENIGN"):
        self.cls = cls
        self.detector = SimpleNamespace(classes_=list(range(len(CLASS_ORDER))))
        self.techniques = {"fake": None}

    def predict_proba(self, X):
        p = np.zeros((len(X), len(CLASS_ORDER)))
        code = CLASS_ORDER.index(self.cls)
        p[:, code] = 1.0
        return p

    def explain(self, vec):
        return {"predicted_class": self.cls,
                "techniques": [{"name": "fake"}]}


class TestBidirectionalMerging(unittest.TestCase):
    def test_reverse_packets_merge_and_flush(self):
        det = LiveDetector(FakeScorer("Botnet"), flow_timeout_s=1.0)
        det.ingest(frame(0.0, A, B, 12345, 443, flags="S"), ts=0.0)
        det.ingest(frame(0.5, B, A, 443, 12345, flags="SA"), ts=0.5)
        self.assertEqual(det.table.active_flows(), 1)
        alerts = det.ingest(frame(3.0, A, C_S, 1111, 2222, flags="A"),
                            ts=3.0)  # first flow now idle > 1s
        self.assertEqual(len(alerts), 1)
        a = alerts[0]
        self.assertEqual(a["predicted_class"], "Botnet")
        self.assertEqual(a["severity"], "CRITICAL")
        self.assertAlmostEqual(a["confidence"], 1.0)
        self.assertEqual(a["src"], A)
        self.assertEqual(a["dport"], 443)
        self.assertEqual(a["features"]["Total Fwd Packets"], 1)
        self.assertEqual(a["features"]["Total Backward Packets"], 1)
        self.assertEqual(det.n_flushed, 1)
        self.assertEqual(alerts[0]["explanation"], None)  # explain off

    def test_explain_attached(self):
        det = LiveDetector(FakeScorer(), flow_timeout_s=1.0)
        det.ingest(frame(0.0, A, B, 1, 80), ts=0.0)
        alerts = det.ingest(frame(5.0, C_S, D_S, 7, 9), ts=5.0, explain=True)
        a = alerts[0]
        self.assertEqual(a["predicted_class"], "BENIGN")
        self.assertEqual(a["explanation"]["techniques"][0]["name"], "fake")


class TestSkippedAndCounters(unittest.TestCase):
    def test_non_ip_frame_ignored(self):
        det = LiveDetector(FakeScorer(), flow_timeout_s=1.0)
        self.assertEqual(det.ingest(b"\xff\xff\xff\xff\xff\xff" + b"\x00" * 20,
                                    ts=0.0), [])
        self.assertEqual(det.n_packets, 0)

    def test_n_packets_counts(self):
        det = LiveDetector(FakeScorer(), flow_timeout_s=60.0)
        det.ingest(frame(0.0, A, B, 1, 80), ts=0.0)
        det.ingest(frame(0.1, A, B, 2, 90), ts=0.1)
        self.assertEqual(det.n_packets, 2)
        self.assertEqual(det.n_flushed, 0)  # nothing idle yet


class TestPcapReplay(unittest.TestCase):
    def test_replay_produces_alerts(self):
        # Three bidirectional flows, each packet 5s apart (> flow_timeout),
        # plus one trailing packet so the last flow also gets flushed.
        flows = [
            [(A, B, 1000, 80), (B, A, 80, 1000)],          # HTTP
            [("10.0.0.9", "10.0.0.8", 9, 53),
             ("10.0.0.8", "10.0.0.9", 53, 9)],             # DNS
            [(C_S, D_S, 1443, 443), (D_S, C_S, 443, 1443)],  # TLS
        ]
        pkts = []
        t = 0.0
        for src, dst, sport, dport in flows[0] + flows[1] + flows[2]:
            p = Ether() / IP(src=src, dst=dst) / \
                TCP(sport=sport, dport=dport, flags="A") / b"x"
            p.time = t
            pkts.append(p)
            t += 5.0
        tail = Ether() / IP(src="10.0.9.1", dst="10.0.9.2") / \
            TCP(sport=11, dport=1337, flags="S")
        tail.time = t
        pkts.append(tail)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "demo.pcap")
            wrpcap(path, pkts)
            det = LiveDetector(FakeScorer("DDoS"), flow_timeout_s=1.0)
            got = []
            det.replay(path, emit=got.append)
            self.assertEqual(det.n_packets, 7)
            self.assertEqual(det.n_flushed, 3)
            self.assertEqual(len(got), 3)
            self.assertEqual(sorted(a["dport"] for a in got), [53, 80, 443])
            vec = Flow.vector_from_dict(got[0]["features"])
            self.assertEqual(len(vec), 69)
            self.assertEqual(vec.dtype, "float32")


if __name__ == "__main__":
    unittest.main(verbosity=2)