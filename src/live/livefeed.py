"""Live feed of alerts for the dashboard - real test rows, paced like a stream.

Serve two masters without lying:

  * The dashboard wants a *feel* of live traffic - alerts arriving over time -
    with real, ranked, explainable detections.
  * Replayed packets (Stage 10/11) are honest about capture but, for CIC's own
    rarest rows, the signal does not survive aggregate-level packet rebuild
    (see docs Stage 12), so a packet demo is quiet.

``RowFeed`` therefore streams REAL held-out CIC-IDS2017 rows through the real
scorer - one alert per row, paced by ``delay`` - exactly the rows Stage 8's
dashboard scanned statically, now presented as a time-ordered feed with true
labels attached for honesty. Explanations for the rarest alerts are computed
in a second thread (they cost seconds per alert; Stage 11), capped so the UI
never blocks. Packet replay stays available in ``dashboard/live.py`` as a
separate source for the true capture path.
"""
import threading
import time
from collections import deque

import numpy as np

from src.config import CLASS_ORDER, FEATURE_ORDER, SEVERITY


class RowFeed(threading.Thread):
    """Paced alert feed from real test rows. One alert per row."""

    def __init__(self, scorer, X, labels, delay=0.15, explain_cap=25,
                 maxlen=2000):
        super().__init__(daemon=True)
        self.scorer = scorer
        self.X = X
        self.labels = labels
        self.delay = delay
        self.explain_cap = explain_cap
        self.alerts = deque(maxlen=maxlen)
        self._explain_count = 0
        self.finished = threading.Event()
        self._pending = deque()

    # ------------------------------------------------------------------ #
    def score_row(self, row, true_label):
        cls, severity, conf, proba = _score(self.scorer, row)
        return {
            "key": None,
            "src": None, "dst": None,
            "dport": int(row[FEATURE_ORDER.index("Destination Port")]),
            "proto": 6,
            "features": {name: float(v) for name, v in zip(FEATURE_ORDER, row)},
            "predicted_class": cls,
            "severity": SEVERITY.get(cls, "UNKNOWN"),
            "confidence": conf,
            "probabilities": proba,
            "explanation": None,
            "true": true_label,
            "num": len(self.alerts),
        }

    def run(self):
        for row, label in zip(self.X, self.labels):
            alert = self.score_row(row, label)
            self.alerts.append(alert)
            if alert["predicted_class"] != "BENIGN":
                self._pending.append(alert)
            self._maybe_explain()
            time.sleep(self.delay)
        self.finished.set()
        while self._pending:
            self._maybe_explain()
            if self._pending:
                time.sleep(0.05)

    def _maybe_explain(self):
        """One explanation per outgoing-alarm, off the hot path (if budget)."""
        if not self._pending:
            return
        if self._explain_count >= self.explain_cap:
            self._pending.popleft()  # drop it; budget spent (never block)
            return
        alert = self._pending[0]
        try:
            alert["explanation"] = self.scorer.explain(
                _vector(alert["features"]))
            self._explain_count += 1
        except Exception:
            pass
        self._pending.popleft()

    def pending_explanations(self):
        return len(self._pending)


def _score(scorer, row):
    proba = scorer.predict_proba(_vector(row)[None, :])[0]
    code = int(scorer.detector.classes_[int(proba.argmax())])
    cls = CLASS_ORDER[code]
    return (cls, SEVERITY.get(cls, "UNKNOWN"), float(proba.max()),
            {CLASS_ORDER[c]: float(v) for c, v in enumerate(proba)})


def _vector(feats):
    """Model input for one flow: a 1D float32 array in FEATURE_ORDER."""
    if hasattr(feats, "shape"):  # already a row array
        return np.asarray(feats, dtype=np.float32).reshape(-1)
    return np.asarray([feats[n] for n in FEATURE_ORDER], dtype=np.float32)