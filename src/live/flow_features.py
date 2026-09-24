"""Live flow-feature extraction - CICFlowMeter-compatible, Phase 1.

This module converts raw packets (as a lightweight, capture-agnostic
``PacketEvent``) into the EXACT 69-feature representation the NIDAI models
were trained on (``config.FEATURE_ORDER``, float32, no scaling).

Design rules:
  * No scapy/OS dependencies here -- the capture layer (Phase 3) translates
    raw frames into ``PacketEvent`` objects, so this module is testable
    without requiring Npcap or root/admin privileges.
  * The 69 numeric features stay internal: consumers should map from
    ``FEATURE_META`` (tier + human meaning + unit) rather than showing the raw
    numbers on the dashboard.
  * Units follow CIC-IDS2017: Duration / IAT / Active / Idle are microseconds,
    all rates are per-second, sizes in bytes, counts are plain counts.

Feature origin tiers (also in FEATURE_META):
  exact  - computable exactly from a bidirectional packet stream
  approx - reproducible with CICFlowMeter-compatible rules over tunable
           parameters (idle/active timeouts, subflow gap); may differ from
           the original lab capture at the margins
  fill0  - derived fields that are structurally absent here -> 0.0, which the
           training pipeline already sanctioned for missing columns
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FEATURE_ORDER  # single source of truth for the 69 names/order

# TCP flag bitmask per the 6-bit flag byte (matches scapy's TCP().flags),
# kept independent of scapy so synthetic tests need no packet crafting.
FIN = 0x01
SYN = 0x02
RST = 0x04
PSH = 0x08
ACK = 0x10
URG = 0x20
ECE = 0x40
CWR = 0x80

# Tunable parameters matching CICFlowMeter's burst/subflow behaviour.
# Active/Idle bursts and subflow splits are defined around these gaps.
ACTIVE_TIMEOUT_S = 2.0
IDLE_TIMEOUT_S = 2.0
SUBFLOW_GAP_S = 2.0

GROUP_OF = {
    "A_Flow_Volume": [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Flow Bytes/s", "Flow Packets/s", "Fwd Header Length",
        "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
        "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
        "Subflow Bwd Bytes",
    ],
    "B_Packet_Statistics": [
        "Fwd Packet Length Max", "Fwd Packet Length Min",
        "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min",
        "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Min Packet Length", "Max Packet Length", "Packet Length Mean",
        "Packet Length Std", "Packet Length Variance", "Average Packet Size",
        "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    ],
    "C_Timing_IAT": [
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
        "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
        "Bwd IAT Max", "Bwd IAT Min", "Active Mean", "Active Std",
        "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max",
        "Idle Min",
    ],
    "D_TCP_Flags_Connection": [
        "Fwd PSH Flags", "Fwd URG Flags", "FIN Flag Count", "SYN Flag Count",
        "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
        "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    ],
    "E_Endpoint_Window": [
        "Destination Port", "Init_Win_bytes_forward",
        "Init_Win_bytes_backward", "act_data_pkt_fwd",
        "min_seg_size_forward",
    ],
}

_EXACT = set()
_APPROX = set()
_FILL0 = set()

_EXACT.update(GROUP_OF["A_Flow_Volume"][:11])
_EXACT.update(["Total Length of Fwd Packets", "Total Length of Bwd Packets"])
_EXACT.update(GROUP_OF["B_Packet_Statistics"])
_EXACT.update([
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Fwd URG Flags",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Destination Port",
])
_APPROX.update(["Subflow Fwd Packets", "Subflow Fwd Bytes",
                "Subflow Bwd Packets", "Subflow Bwd Bytes",
                "Active Mean", "Active Std", "Active Max", "Active Min",
                "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
                "Init_Win_bytes_forward", "Init_Win_bytes_backward",
                "act_data_pkt_fwd", "min_seg_size_forward"])
_FILL0.update(["Fwd Header Length", "Bwd Header Length"])

T1_EXACT = "exact"
T2_APPROX = "approx"
T3_FILL = "fill0"

FEATURE_META = {f: (T1_EXACT if f in _EXACT else
                    T2_APPROX if f in _APPROX else T3_FILL)
                for f in FEATURE_ORDER}

assert len(FEATURE_META) == 69
assert len(set(FEATURE_META)) == 69


class PacketEvent:
    """One captured packet normalised into the form the flow math needs.

    Fields are the minimum required to compute every CIC feature. The capture
    layer builds these from scapy frames; tests build them directly.
    """

    __slots__ = ("ts", "src", "dst", "sport", "dport", "proto",
                 "payload_len", "header_len", "flags", "win_size")

    def __init__(self, ts, src, dst, sport, dport, proto,
                 payload_len=0, header_len=0, flags=0, win_size=0):
        self.ts = float(ts)
        self.src = src
        self.dst = dst
        self.sport = int(sport)
        self.dport = int(dport)
        self.proto = int(proto)
        self.payload_len = int(payload_len)
        self.header_len = int(header_len)
        self.flags = int(flags)
        self.win_size = int(win_size)

    def length(self):
        return self.payload_len + self.header_len


def _stats(values):
    """Population max/min/mean/std over non-empty lists (CIC uses pop std)."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return (float(arr.max()), float(arr.min()), float(arr.mean()),
            float(arr.std(ddof=0)))


def _iats(ts_list):
    """Inter-arrival gaps (microseconds) between consecutive sort-by-time pkts."""
    if len(ts_list) < 2:
        return []
    ts = np.sort(np.asarray(ts_list, dtype=np.float64))
    return list(np.diff(ts) * 1e6)


def _bursts(ts_list):
    """Group packets into active bursts: consecutive packets within the gap."""
    bursts = []
    cur = []
    for i in range(len(ts_list)):
        if not cur:
            cur = [ts_list[i]]
        elif ts_list[i] - ts_list[i - 1] <= ACTIVE_TIMEOUT_S:
            cur.append(ts_list[i])
        else:
            bursts.append(cur)
            cur = [ts_list[i]]
    if cur:
        bursts.append(cur)
    return bursts


def _idle_gaps(ts_list):
    """Inter-burst gaps longer than the idle timeout (time *between* bursts)."""
    return [ts_list[i] - ts_list[i - 1] for i in range(1, len(ts_list))
            if ts_list[i] - ts_list[i - 1] > IDLE_TIMEOUT_S]


class Flow:
    """A bidirectional flow accumulating packets; finalize() yields the 69 features."""

    def __init__(self, key, src, dst, sport, dport, proto, first_ts):
        self.key = key
        self.src = src
        self.dst = dst
        self.sport = sport
        self.dport = dport
        self.proto = proto
        self.first_ts = first_ts
        self.last_ts = first_ts
        self._fwd = []
        self._bwd = []

    def feed(self, pkt):
        """Append a packet, assigning direction relative to the flow's first packet."""
        if pkt.ts < self.first_ts:
            self.first_ts = pkt.ts
        if pkt.ts > self.last_ts:
            self.last_ts = pkt.ts
        if (pkt.src == self.src and pkt.dst == self.dst
                and pkt.sport == self.sport and pkt.dport == self.dport):
            self._fwd.append(pkt)
        else:
            self._bwd.append(pkt)

    def is_expired(self, now, timeout_s):
        return now - self.last_ts >= timeout_s

    # ------------------------------------------------------------------ #
    def _flag_counts(self):
        counts = {}
        for name, bit in (("FIN", FIN), ("SYN", SYN), ("RST", RST),
                          ("PSH", PSH), ("ACK", ACK), ("URG", URG),
                          ("CWE", CWR), ("ECE", ECE)):
            counts[name] = sum(1 for p in self._fwd + self._bwd if p.flags & bit)
        return counts

    def _subflow(self, packets):
        """Packets before the first forward gap longer than SUBFLOW_GAP_S."""
        out = []
        for i, p in enumerate(packets):
            if i > 0 and p.ts - packets[i - 1].ts > SUBFLOW_GAP_S:
                break
            out.append(p)
        return out

    def to_features(self):
        f_fwd = sorted(self._fwd, key=lambda p: p.ts)
        f_bwd = sorted(self._bwd, key=lambda p: p.ts)
        all_pkts = sorted(self._fwd + self._bwd, key=lambda p: p.ts)
        all_ts = [p.ts for p in all_pkts]

        fwd_len = [p.length() for p in f_fwd]
        bwd_len = [p.length() for p in f_bwd]
        all_len = [p.length() for p in all_pkts]
        nf, nb = len(f_fwd), len(f_bwd)

        fmax, fmin, fmean, fstd = _stats(fwd_len)
        bmax, bmin, bmean, bstd = _stats(bwd_len)
        amax, amin, amean, astd = _stats(all_len)

        dur_sec = max(self.last_ts - self.first_ts, 0.0)
        dur_us = dur_sec * 1e6
        tot_f = float(sum(fwd_len))
        tot_b = float(sum(bwd_len))
        tot = tot_f + tot_b
        n_all = nf + nb

        bit_counts = self._flag_counts()
        fwd_flags = [p.flags for p in f_fwd]
        fwd_psh = sum(1 for f in fwd_flags if f & PSH)
        fwd_urg = sum(1 for f in fwd_flags if f & URG)

        fwd_ts = [p.ts for p in f_fwd]
        bwd_ts = [p.ts for p in f_bwd]
        flow_iat = _iats(all_ts)
        fwd_iat = _iats(fwd_ts)
        bwd_iat = _iats(bwd_ts)
        i_max, i_min, i_mean, i_std = _stats(flow_iat)
        f_max, f_min, f_mean, f_std = _stats(fwd_iat)
        b_max, b_min, b_mean, b_std = _stats(bwd_iat)
        f_tot = float(np.sum(fwd_iat))
        b_tot = float(np.sum(bwd_iat))

        a_max, a_min, a_mean, a_std = _stats([
            (b[-1] - b[0]) * 1e6 for b in _bursts(all_ts)])
        id_max, id_min, id_mean, id_std = _stats([
            g * 1e6 for g in _idle_gaps(all_ts)])

        sb_fwd = self._subflow(f_fwd)
        sb_bwd = self._subflow(f_bwd)
        sb_fp = float(len(sb_fwd))
        sb_fb = float(sum(p.payload_len for p in sb_fwd))
        sb_bp = float(len(sb_bwd))
        sb_bb = float(sum(p.payload_len for p in sb_bwd))

        rate = n_all / dur_sec if dur_sec > 0 else 0.0
        bytes_rate = tot / dur_sec if dur_sec > 0 else 0.0
        fwd_rate = nf / dur_sec if dur_sec > 0 else 0.0
        bwd_rate = nb / dur_sec if dur_sec > 0 else 0.0

        fwd_header = float(sum(p.header_len for p in f_fwd))
        bwd_header = float(sum(p.header_len for p in f_bwd))
        avg_pkt = tot / n_all if n_all else 0.0
        avg_fwd_seg = tot_f / nf if nf else 0.0
        avg_bwd_seg = tot_b / nb if nb else 0.0
        down_up = tot_b / tot_f if tot_f else 0.0

        init_win_f = f_fwd[0].win_size if f_fwd else 0
        init_win_b = f_bwd[0].win_size if f_bwd else 0
        act_data_f = sum(1 for p in f_fwd if p.payload_len > 0)
        min_seg_f = float(min(fwd_len)) if fwd_len else 0.0

        features = {
            "Flow Duration": dur_us,
            "Total Fwd Packets": nf,
            "Total Backward Packets": nb,
            "Total Length of Fwd Packets": tot_f,
            "Total Length of Bwd Packets": tot_b,
            "Flow Bytes/s": bytes_rate,
            "Flow Packets/s": rate,
            "Fwd Header Length": fwd_header,
            "Bwd Header Length": bwd_header,
            "Fwd Packets/s": fwd_rate,
            "Bwd Packets/s": bwd_rate,
            "Subflow Fwd Packets": sb_fp,
            "Subflow Fwd Bytes": sb_fb,
            "Subflow Bwd Packets": sb_bp,
            "Subflow Bwd Bytes": sb_bb,
            "Fwd Packet Length Max": fmax,
            "Fwd Packet Length Min": fmin,
            "Fwd Packet Length Mean": fmean,
            "Fwd Packet Length Std": fstd,
            "Bwd Packet Length Max": bmax,
            "Bwd Packet Length Min": bmin,
            "Bwd Packet Length Mean": bmean,
            "Bwd Packet Length Std": bstd,
            "Min Packet Length": amin,
            "Max Packet Length": amax,
            "Packet Length Mean": amean,
            "Packet Length Std": astd,
            "Packet Length Variance": astd ** 2,
            "Average Packet Size": avg_pkt,
            "Avg Fwd Segment Size": avg_fwd_seg,
            "Avg Bwd Segment Size": avg_bwd_seg,
            "Flow IAT Mean": i_mean,
            "Flow IAT Std": i_std,
            "Flow IAT Max": i_max,
            "Flow IAT Min": i_min,
            "Fwd IAT Total": f_tot,
            "Fwd IAT Mean": f_mean,
            "Fwd IAT Std": f_std,
            "Fwd IAT Max": f_max,
            "Fwd IAT Min": f_min,
            "Bwd IAT Total": b_tot,
            "Bwd IAT Mean": b_mean,
            "Bwd IAT Std": b_std,
            "Bwd IAT Max": b_max,
            "Bwd IAT Min": b_min,
            "Active Mean": a_mean,
            "Active Std": a_std,
            "Active Max": a_max,
            "Active Min": a_min,
            "Idle Mean": id_mean,
            "Idle Std": id_std,
            "Idle Max": id_max,
            "Idle Min": id_min,
            "Fwd PSH Flags": fwd_psh,
            "Fwd URG Flags": fwd_urg,
            "FIN Flag Count": bit_counts["FIN"],
            "SYN Flag Count": bit_counts["SYN"],
            "RST Flag Count": bit_counts["RST"],
            "PSH Flag Count": bit_counts["PSH"],
            "ACK Flag Count": bit_counts["ACK"],
            "URG Flag Count": bit_counts["URG"],
            "CWE Flag Count": bit_counts["CWE"],
            "ECE Flag Count": bit_counts["ECE"],
            "Down/Up Ratio": down_up,
            "Destination Port": self.dport,
            "Init_Win_bytes_forward": init_win_f,
            "Init_Win_bytes_backward": init_win_b,
            "act_data_pkt_fwd": act_data_f,
            "min_seg_size_forward": min_seg_f,
        }
        assert set(features) == set(FEATURE_ORDER)
        return features

    def feature_vector(self):
        """np.float32 array in EXACTLY config.FEATURE_ORDER (the model input)."""
        return self.vector_from_dict(self.to_features())

    @staticmethod
    def vector_from_dict(feats):
        """Build the model input from a finished feature dict (any caller)."""
        return np.asarray([feats[name] for name in FEATURE_ORDER],
                          dtype=np.float32)


def _bursts(ts_list):
    bursts = []
    cur = []
    for i in range(len(ts_list)):
        if not cur:
            cur = [ts_list[i]]
        elif ts_list[i] - ts_list[i - 1] <= ACTIVE_TIMEOUT_S:
            cur.append(ts_list[i])
        else:
            bursts.append(cur)
            cur = [ts_list[i]]
    if cur:
        bursts.append(cur)
    return bursts


def _idle_gaps(ts_list):
    return [ts_list[i] - ts_list[i - 1] for i in range(1, len(ts_list))
            if ts_list[i] - ts_list[i - 1] > IDLE_TIMEOUT_S]


class FlowTable:
    """Keyed collection of live flows with timeout expiry (capture-agnostic)."""

    def __init__(self, flow_timeout_s=15.0):
        self.flow_timeout_s = flow_timeout_s
        self._flows = {}

    @staticmethod
    def key_of(pkt):
        return (pkt.src, pkt.dst, pkt.sport, pkt.dport, pkt.proto)

    def add(self, pkt):
        k = self.key_of(pkt)
        fl = self._flows.get(k)
        if fl is None:
            reverse = (pkt.dst, pkt.src, pkt.dport, pkt.sport, pkt.proto)
            fl = self._flows.get(reverse)  # backward side of an existing flow
        if fl is None:
            fl = Flow(k, pkt.src, pkt.dst, pkt.sport, pkt.dport,
                      pkt.proto, pkt.ts)
            self._flows[k] = fl
        fl.feed(pkt)
        return fl

    def flush_expired(self, now):
        """Return finalized FeatureDicts for flows idle past the timeout."""
        return [feats for _, feats in self.flush_expired_keyed(now)]

    def flush_expired_keyed(self, now):
        """Like flush_expired but keeps the flow key alongside its features."""
        expired = [k for k in self._flows
                   if self._flows[k].is_expired(now, self.flow_timeout_s)]
        return [(k, self._flows.pop(k).to_features()) for k in expired]

    def active_flows(self):
        return len(self._flows)