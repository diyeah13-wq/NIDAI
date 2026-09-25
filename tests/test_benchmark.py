"""Unit tests for the Stage 11 benchmark generator/runner - no models, no Npcap.

Run:  python tests/test_benchmark.py
  or:  python -m unittest tests.test_benchmark -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live.benchmark import synthetic_stream
from src.live.detector import LiveDetector


class _Scorer:
    def __init__(self):
        from types import SimpleNamespace
        self.detector = SimpleNamespace(classes_=[0])
        self.techniques = None

    def predict_proba(self, X):
        import numpy as np
        p = np.zeros((len(X), 1))
        p[:, 0] = 1.0
        return p


class TestSyntheticStream(unittest.TestCase):
    def test_deterministic(self):
        p1, f1 = synthetic_stream(n_flows=50, flows_per_sec=100, seed=7)
        p2, f2 = synthetic_stream(n_flows=50, flows_per_sec=100, seed=7)
        self.assertEqual([b[0].time for b in p1], [b[0].time for b in p2])
        self.assertEqual(list(f1), list(f2))

    def test_tail_flushes_every_flow(self):
        packets, first = synthetic_stream(n_flows=40, flows_per_sec=50,
                                          seed=3, tail_timeout_buf=2.0)
        det = LiveDetector(_Scorer(), flow_timeout_s=1.0)
        done = 0
        for pkt in packets:
            done += len(det.ingest(pkt, ts=pkt.time))
        # the trailing packet itself opens one extra flow that never flushes
        self.assertEqual(done, 40)

    def test_seed_changes_stream(self):
        p1, _ = synthetic_stream(n_flows=20, seed=1)
        p2, _ = synthetic_stream(n_flows=20, seed=2)
        self.assertNotEqual([b[0].time for b in p1[0:5]],
                            [b[0].time for b in p2[0:5]])


if __name__ == "__main__":
    unittest.main(verbosity=2)