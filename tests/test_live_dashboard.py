"""Unit tests for the plain-language layer and the live feed (Stage 12).

No models, no Npcap: the feed is driven by a fake scorer over tiny synthetic
rows; plain.py is pure formatting.

Run:  python tests/test_live_dashboard.py
  or:  python -m unittest tests.test_live_dashboard -v
"""
import os
import sys
import time
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import CLASS_ORDER, FEATURE_ORDER
from src.live.livefeed import RowFeed
from src.live.plain import (FRIENDLY, feature_label, describe_feature,
                            alert_headline, explain_lines, severity_color,
                            severity_rank)


def feature_label_names():
    return list(FRIENDLY)


def _row(cls_code, port=80):
    row = np.zeros(len(FEATURE_ORDER), dtype=np.float32)
    row[FEATURE_ORDER.index("Destination Port")] = port
    row[FEATURE_ORDER.index("Total Fwd Packets")] = 100
    row[FEATURE_ORDER.index("Flow Bytes/s")] = 5000.0
    row[FEATURE_ORDER.index("ACK Flag Count")] = 50
    row[FEATURE_ORDER.index("SYN Flag Count")] = 40
    return row


class FakeScorer:
    """Mimics the sklearn contract strictly: predict_proba wants 2D."""

    def __init__(self, cls="Botnet"):
        self.cls = cls
        self.detector = SimpleNamespace(classes_=list(range(len(CLASS_ORDER))))
        self.explain_calls = []

    def predict_proba(self, X):
        X = np.asarray(X)
        assert X.ndim == 2 and X.shape[1] == len(FEATURE_ORDER), X.shape
        p = np.zeros((len(X), len(CLASS_ORDER)))
        code = CLASS_ORDER.index(self.cls)
        p[:, code] = 1.0
        return p

    def explain(self, vec):
        vec = np.asarray(vec, dtype=np.float64)
        assert vec.ndim == 1 and vec.shape[0] == len(FEATURE_ORDER), vec.shape
        self.explain_calls.append(vec)
        return {"predicted_class": self.cls,
                "techniques": [
                    {"key": "A_Flow_Volume", "name": "Volume", "fired": True,
                     "confidence_alert": 0.9,
                     "top_features": [
                         {"feature": "Flow Bytes/s", "value": 5000.0,
                          "zscore": 3.2, "drop": 0.5},
                         {"feature": "Destination Port", "value": 80.0,
                          "zscore": 0.1, "drop": 0.2}]}]}


class TestPlainLayer(unittest.TestCase):
    def test_every_feature_has_a_plain_label(self):
        generic = [f for f in FEATURE_ORDER
                   if feature_label(f) == ("feature value", "")]
        self.assertEqual(generic, [], "features shown as 'feature value'")
        stale = [f for f in feature_label_names() if f not in FEATURE_ORDER]
        self.assertEqual(stale, [], "labels for features that no longer exist")

    def test_feature_label_fallback(self):
        self.assertEqual(feature_label("Flow Duration"), ("how long the "
                          "connection lasted", "s"))
        self.assertEqual(feature_label("Made Up Feature"), ("feature value", ""))

    def test_describe_feature(self):
        high = describe_feature({"feature": "Flow Bytes/s", "value": 5000.0,
                                 "zscore": 3.2})
        self.assertIn("5000", high)
        self.assertIn("B/s", high)
        self.assertIn("far higher", high)
        low = describe_feature({"feature": "Flow Bytes/s", "value": 0.05,
                                "zscore": -1.8})
        self.assertIn("somewhat lower", low)

    def test_headline_and_severity(self):
        alert = {"predicted_class": "Botnet", "severity": "CRITICAL",
                 "confidence": 0.86}
        self.assertIn("Botnet", alert_headline(alert))
        self.assertIn("86%", alert_headline(alert))
        self.assertEqual(severity_color("CRITICAL"), "#c0392b")
        self.assertEqual(severity_rank("CRITICAL"), 3)

    def test_explain_lines(self):
        expl = FakeScorer().explain(_row(CLASS_ORDER.index("Botnet")))
        lines = explain_lines(expl)
        self.assertTrue(any("data moved" in l or "data rate" in l
                            for l in lines))
        self.assertTrue(any("data rate" in l for l in lines))


class TestRowFeed(unittest.TestCase):
    def test_paced_alerts_with_true_labels(self):
        X = np.vstack([_row(CLASS_ORDER.index("BENIGN")),
                       _row(CLASS_ORDER.index("Botnet")),
                       _row(CLASS_ORDER.index("DoS"))])
        labels = np.array(["BENIGN", "Botnet", "DoS"])
        feed = RowFeed(FakeScorer("Botnet"), X, labels, delay=0.02,
                       explain_cap=1)
        feed.start()
        feed.join(timeout=10)
        self.assertTrue(feed.finished.is_set())
        self.assertEqual(len(feed.alerts), 3)
        first = feed.alerts[0]
        self.assertEqual(first["true"], "BENIGN")
        self.assertEqual(first["dport"], 80)
        self.assertEqual(first["predicted_class"], "Botnet")
        self.assertTrue(any(a["explanation"] is not None for a in feed.alerts))


if __name__ == "__main__":
    unittest.main(verbosity=2)