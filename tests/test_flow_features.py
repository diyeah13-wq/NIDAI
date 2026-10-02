"""Deterministic unit tests for src/live/flow_features.py (Phase 1).

Run:  python tests/test_flow_features.py
 or:  python -m unittest tests.test_flow_features -v

Synthetic packets are built directly as PacketEvent objects - no scapy, no
Npcap, no real traffic. Every expected value is hand-computed.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from src.live.flow_features import (PacketEvent, Flow, FlowTable,
                                    FEATURE_META, FEATURE_ORDER,
                                    ACK, SYN, FIN, PSH, URG, ECE)

A, B = "10.0.0.1", "10.0.0.2"


def pkt(ts, src, dst, sport, dport, proto=6, plen=0, hlen=20,
        flags=0, win=0):
    return PacketEvent(ts=ts, src=src, dst=dst, sport=sport, dport=dport,
                       proto=proto, payload_len=plen, header_len=hlen,
                       flags=flags, win_size=win)


def flow_with(packets):
    f = Flow(None, packets[0].src, packets[0].dst, packets[0].sport,
             packets[0].dport, packets[0].proto, packets[0].ts)
    for p in packets:
        f.feed(p)
    return f


class TestContract(unittest.TestCase):
    def test_69_features_f32_in_config_order(self):
        f = flow_with([pkt(0.0, A, B, 1, 2, plen=10, hlen=20)])
        vec = f.feature_vector()
        feats = f.to_features()
        self.assertEqual(len(vec), 69)
        self.assertEqual(vec.dtype, "float32")
        self.assertEqual(list(FEATURE_ORDER), [n for n in feats])

    def test_meta_covers_all_features_with_tiers(self):
        self.assertEqual(len(FEATURE_META), 69)
        self.assertEqual(set(FEATURE_META), set(FEATURE_ORDER))
        self.assertIn("exact", set(FEATURE_META.values()))
        self.assertIn("approx", set(FEATURE_META.values()))


class TestTwoPacketUdp(unittest.TestCase):
    def setUp(self):
        self.p1 = pkt(0.0, A, B, 12345, 80, proto=17, plen=100, hlen=20)
        self.p2 = pkt(0.005, B, A, 80, 12345, proto=17, plen=50, hlen=20)
        self.v = flow_with([self.p1, self.p2]).to_features()

    def test_duration_and_rates(self):
        self.assertAlmostEqual(self.v["Flow Duration"], 5000.0, places=2)
        self.assertAlmostEqual(self.v["Flow Bytes/s"], 38000.0, places=1)
        self.assertAlmostEqual(self.v["Flow Packets/s"], 400.0, places=3)
        self.assertAlmostEqual(self.v["Fwd Packets/s"], 200.0, places=3)
        self.assertAlmostEqual(self.v["Bwd Packets/s"], 200.0, places=3)

    def test_volume(self):
        self.assertEqual(self.v["Total Fwd Packets"], 1)
        self.assertEqual(self.v["Total Backward Packets"], 1)
        self.assertAlmostEqual(self.v["Total Length of Fwd Packets"], 120.0)
        self.assertAlmostEqual(self.v["Total Length of Bwd Packets"], 70.0)
        self.assertAlmostEqual(self.v["Fwd Header Length"], 20.0)
        self.assertAlmostEqual(self.v["Bwd Header Length"], 20.0)
        self.assertAlmostEqual(self.v["Subflow Fwd Packets"], 1.0)
        self.assertAlmostEqual(self.v["Subflow Fwd Bytes"], 100.0)
        self.assertAlmostEqual(self.v["Subflow Bwd Packets"], 1.0)
        self.assertAlmostEqual(self.v["Subflow Bwd Bytes"], 50.0)

    def test_packet_stats(self):
        self.assertAlmostEqual(self.v["Fwd Packet Length Mean"], 120.0)
        self.assertAlmostEqual(self.v["Bwd Packet Length Mean"], 70.0)
        self.assertAlmostEqual(self.v["Max Packet Length"], 120.0)
        self.assertAlmostEqual(self.v["Min Packet Length"], 70.0)
        self.assertAlmostEqual(self.v["Packet Length Mean"], 95.0)
        self.assertAlmostEqual(self.v["Packet Length Variance"], 625.0)
        self.assertAlmostEqual(self.v["Average Packet Size"], 95.0)
        self.assertAlmostEqual(self.v["Avg Fwd Segment Size"], 120.0)
        self.assertAlmostEqual(self.v["Avg Bwd Segment Size"], 70.0)

    def test_iat(self):
        self.assertAlmostEqual(self.v["Flow IAT Mean"], 5000.0)
        self.assertAlmostEqual(self.v["Flow IAT Max"], 5000.0)
        self.assertAlmostEqual(self.v["Flow IAT Std"], 0.0)
        self.assertAlmostEqual(self.v["Fwd IAT Mean"], 0.0)  # one fwd pkt
        self.assertAlmostEqual(self.v["Bwd IAT Mean"], 0.0)

    def test_active_idle(self):
        self.assertAlmostEqual(self.v["Active Mean"], 5000.0)
        self.assertAlmostEqual(self.v["Active Max"], 5000.0)
        self.assertAlmostEqual(self.v["Idle Mean"], 0.0)

    def test_endpoint_and_flags(self):
        self.assertEqual(self.v["Destination Port"], 80)
        self.assertAlmostEqual(self.v["Down/Up Ratio"], 70.0 / 120.0)
        self.assertEqual(self.v["SYN Flag Count"], 0)
        self.assertEqual(self.v["act_data_pkt_fwd"], 1)
        self.assertAlmostEqual(self.v["min_seg_size_forward"], 120.0)


class TestTcpHandshake(unittest.TestCase):
    def setUp(self):
        self.packets = [
            pkt(0.000, A, B, 12345, 443, plen=0, hlen=20, flags=SYN, win=64240),
            pkt(0.001, B, A, 443, 12345, plen=300, hlen=20,
                flags=SYN | ACK, win=65535),
            pkt(0.002, A, B, 12345, 443, plen=200, hlen=20, flags=ACK, win=100),
            pkt(0.003, B, A, 443, 12345, plen=0, hlen=20, flags=ACK | FIN,
                win=1000),
        ]
        self.v = flow_with(self.packets).to_features()

    def test_flag_counts(self):
        self.assertEqual(self.v["SYN Flag Count"], 2)
        self.assertEqual(self.v["ACK Flag Count"], 3)
        self.assertEqual(self.v["FIN Flag Count"], 1)
        self.assertEqual(self.v["PSH Flag Count"], 0)
        self.assertEqual(self.v["Fwd PSH Flags"], 0)
        self.assertEqual(self.v["Fwd URG Flags"], 0)
        self.assertEqual(self.v["CWE Flag Count"], 0)
        self.assertEqual(self.v["ECE Flag Count"], 0)

    def test_windows_and_segments(self):
        self.assertEqual(self.v["Init_Win_bytes_forward"], 64240)
        self.assertEqual(self.v["Init_Win_bytes_backward"], 65535)
        self.assertEqual(self.v["act_data_pkt_fwd"], 1)  # only pkt 3 has payload
        self.assertAlmostEqual(self.v["min_seg_size_forward"], 20.0)

    def test_more_iat_entries(self):
        self.assertAlmostEqual(self.v["Flow IAT Mean"], 1000.0)
        self.assertAlmostEqual(self.v["Fwd IAT Mean"], 2000.0)
        self.assertAlmostEqual(self.v["Bwd IAT Mean"], 2000.0)
        self.assertAlmostEqual(self.v["Flow Duration"], 3000.0)


class TestSinglePacketEdge(unittest.TestCase):
    def setUp(self):
        self.v = flow_with([pkt(0.0, A, B, 1, 53, proto=17, plen=40, hlen=28)]
                           ).to_features()

    def test_zero_duration_means_zero_rates(self):
        self.assertAlmostEqual(self.v["Flow Duration"], 0.0)
        self.assertAlmostEqual(self.v["Flow Bytes/s"], 0.0)
        self.assertAlmostEqual(self.v["Flow Packets/s"], 0.0)
        self.assertAlmostEqual(self.v["Fwd Packets/s"], 0.0)

    def test_empty_secondary_direction(self):
        self.assertEqual(self.v["Total Backward Packets"], 0)
        self.assertAlmostEqual(self.v["Bwd Packet Length Mean"], 0.0)
        self.assertAlmostEqual(self.v["Bwd IAT Mean"], 0.0)
        self.assertAlmostEqual(self.v["Down/Up Ratio"], 0.0)
        self.assertAlmostEqual(self.v["Active Mean"], 0.0)
        self.assertAlmostEqual(self.v["Flow IAT Mean"], 0.0)
        self.assertEqual(self.v["Destination Port"], 53)


class TestActiveIdleBursts(unittest.TestCase):
    def setUp(self):
        packets = [pkt(0.0, A, B, 1, 2, plen=10),
                   pkt(0.001, A, B, 1, 2, plen=10),
                   pkt(0.002, A, B, 1, 2, plen=10),
                   pkt(5.0, A, B, 1, 2, plen=10),
                   pkt(5.5, A, B, 1, 2, plen=10)]
        self.v = flow_with(packets).to_features()

    def test_two_bursts_one_idle(self):
        # bursts: [0,0.001,0.002] dur 2000us ; [5.0,5.5] dur 500000us
        self.assertAlmostEqual(self.v["Active Mean"], 251000.0, places=0)
        self.assertAlmostEqual(self.v["Active Max"], 500000.0, places=0)
        self.assertAlmostEqual(self.v["Active Min"], 2000.0, places=0)
        self.assertAlmostEqual(self.v["Active Std"], 249000.0, places=0)
        # idle gap = 5.0 - 0.002 = 4.998 s
        self.assertAlmostEqual(self.v["Idle Mean"], 4998000.0, places=0)
        self.assertAlmostEqual(self.v["Idle Std"], 0.0)

    def test_subflow_only_first_burst(self):
        self.assertAlmostEqual(self.v["Subflow Fwd Packets"], 3.0)
        self.assertAlmostEqual(self.v["Subflow Fwd Bytes"], 30.0)


class TestFlowTable(unittest.TestCase):
    def test_keyed_directions_and_expiry(self):
        tbl = FlowTable(flow_timeout_s=15.0)
        tbl.add(pkt(0.0, A, B, 1, 2, plen=10, hlen=20))
        tbl.add(pkt(0.5, B, A, 2, 1, plen=40, hlen=20))
        tbl.add(pkt(1.0, "10.0.0.9", "10.0.0.8", 55, 66, plen=5, hlen=20))
        tbl.add(pkt(1.5, "10.0.0.8", "10.0.0.9", 66, 55, plen=5, hlen=20))
        self.assertEqual(tbl.active_flows(), 2)
        expired = tbl.flush_expired(now=17.0)  # >15s idle for the 1.5s flow
        self.assertEqual(len(expired), 2)
        self.assertEqual(tbl.active_flows(), 0)
        feats = [f for f in expired if f["Destination Port"] == 2][0]
        self.assertEqual(feats["Total Fwd Packets"], 1)
        self.assertEqual(feats["Total Backward Packets"], 1)
        self.assertAlmostEqual(feats["Total Length of Fwd Packets"], 30.0)
        self.assertAlmostEqual(feats["Total Length of Bwd Packets"], 60.0)


class TestScoreableGate(unittest.TestCase):
    """The direction gate: only bidirectional flows reach the model.

    Justified by the class-conditional measurement in docs/experiments.md -
    one-way flows are 12% of BENIGN but only 0.04% of PortScan, so gating on
    direction discards multicast noise and keeps real attacks.
    """

    def test_one_way_flow_is_evicted_but_not_returned(self):
        tbl = FlowTable(flow_timeout_s=1.0)
        tbl.add(pkt(0.0, A, B, 1, 2, plen=10, hlen=20))
        self.assertEqual(tbl.flush_expired(now=5.0), [])
        self.assertEqual(tbl.active_flows(), 0)      # evicted, not leaked
        self.assertEqual(tbl.dropped_unscoreable, 1)  # and counted, not silent

    def test_bidirectional_flow_passes_the_gate(self):
        tbl = FlowTable(flow_timeout_s=1.0)
        tbl.add(pkt(0.0, A, B, 1, 2, plen=10, hlen=20))
        tbl.add(pkt(0.1, B, A, 2, 1, plen=40, hlen=20))
        self.assertEqual(len(tbl.flush_expired(now=5.0)), 1)
        self.assertEqual(tbl.dropped_unscoreable, 0)

    def test_gate_can_be_relaxed_for_inspection(self):
        tbl = FlowTable(flow_timeout_s=1.0, min_bwd_packets=0)
        tbl.add(pkt(0.0, A, B, 1, 2, plen=10, hlen=20))
        self.assertEqual(len(tbl.flush_expired(now=5.0)), 1)
        self.assertEqual(tbl.dropped_unscoreable, 0)

    def test_two_packet_portscan_shape_survives(self):
        # The real reason this is a direction gate and not a volume gate:
        # 98.5% of CIC PortScan rows are exactly 1 fwd + 1 bwd packet.
        tbl = FlowTable(flow_timeout_s=1.0)
        tbl.add(pkt(0.0, A, B, 40000, 22, plen=0, hlen=40, flags=PSH))
        tbl.add(pkt(0.00005, B, A, 22, 40000, plen=0, hlen=20, flags=ACK))
        got = tbl.flush_expired(now=5.0)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["Total Fwd Packets"], 1)
        self.assertEqual(got[0]["Total Backward Packets"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)