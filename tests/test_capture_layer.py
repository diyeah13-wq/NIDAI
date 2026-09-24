"""Deterministic unit tests for the scapy capture layer (Phase 3).

No Npcap, no real traffic, no root: frames are built with scapy, serialised
to bytes with ``bytes(pkt)`` (which finalises IP total length / options), then
re-parsed by ``frame_to_event`` - the exact same code path scapy's ``sniff``
feed and pcap replay use. Expected values are hand-computed from IP/TCP/UDP
header sizes.

Run:  python tests/test_capture_layer.py
  or:  python -m unittest tests.test_capture_layer -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether, ARP

from src.live.capture import frame_to_event, packet_to_event
from src.live.flow_features import (SYN, ACK, FIN, Flow, FEATURE_ORDER)

A = "10.0.0.1"
B = "10.0.0.2"


def build(layer, ts=0.0, src=A, dst=B):
    """Ether/IP/<layer> -> bytes -> frame_to_event (ts override)."""
    return frame_to_event(bytes(Ether() / IP(src=src, dst=dst) / layer),
                          ts=ts)


class TestTcpMapping(unittest.TestCase):
    def test_syn_packet(self):
        ev = build(TCP(sport=12345, dport=443, flags="S",
                       window=64240) / RawHello(b"hello"))
        self.assertEqual(ev.proto, 6)
        self.assertEqual(ev.src, "10.0.0.1")
        self.assertEqual(ev.dst, "10.0.0.2")
        self.assertEqual((ev.sport, ev.dport), (12345, 443))
        self.assertTrue(ev.flags & SYN)
        self.assertFalse(ev.flags & FIN)
        self.assertEqual(ev.win_size, 64240)
        self.assertEqual(ev.header_len, 20)      # no TCP options
        self.assertEqual(ev.payload_len, 5)      # 45 - 20(IP) - 20(TCP)
        self.assertEqual(ev.length(), 25)        # IP payload (no IP header)

    def test_options_inflate_header_len(self):
        ev = build(TCP(sport=1, dport=2, flags="SA", window=65535,
                       options=[("MSS", 1460)]), ts=1.5)
        self.assertEqual(ev.header_len, 24)      # 20 + 4-byte MSS option
        self.assertTrue(ev.flags & SYN and ev.flags & ACK)
        self.assertAlmostEqual(ev.ts, 1.5)

    def test_fin_ack_cleanup(self):
        ev = build(TCP(sport=1, dport=2, flags="FA", window=1000))
        self.assertTrue(ev.flags & FIN)
        self.assertTrue(ev.flags & ACK)


class TestUdpIcmp(unittest.TestCase):
    def test_udp(self):
        ev = build(UDP(sport=5000, dport=53) / RawHello(b"abcd"))
        self.assertEqual(ev.proto, 17)
        self.assertEqual((ev.sport, ev.dport), (5000, 53))
        self.assertEqual(ev.header_len, 8)
        self.assertEqual(ev.payload_len, 4)      # 32 - 20(IP) - 8(UDP)
        self.assertEqual(ev.flags, 0)
        self.assertEqual(ev.win_size, 0)

    def test_icmp_no_port(self):
        ev = build(ICMP())
        self.assertEqual(ev.proto, 1)
        self.assertEqual((ev.sport, ev.dport), (0, 0))
        self.assertEqual(ev.header_len, 8)
        self.assertEqual(ev.flags, 0)


class TestSkippedAndEdges(unittest.TestCase):
    def test_arp_skipped(self):
        self.assertIsNone(frame_to_event(bytes(Ether() / ARP()), ts=0.0))

    def test_garbage_frame_skipped(self):
        self.assertIsNone(frame_to_event(b"\x00" * 14))

    def test_ipv6(self):
        ev = frame_to_event(
            bytes(Ether() / IPv6(src="fe80::1", dst="fe80::2") /
                  TCP(sport=9, dport=443, flags="S", window=300)),
            ts=2.0)
        self.assertEqual(ev.src, "fe80::1")
        self.assertEqual(ev.proto, 6)
        self.assertEqual(ev.sport, 9)
        self.assertTrue(ev.flags & SYN)

    def test_packet_to_event_matches_frame_path(self):
        pkt = Ether() / IP(src=A, dst=B) / \
            TCP(sport=10, dport=20, flags="A", window=100) / RawHello(b"zz")
        from_frame = frame_to_event(bytes(pkt), ts=7.0)
        from_pkt = packet_to_event(pkt, ts=7.0)
        self.assertEqual(from_frame.src, from_pkt.src)
        self.assertEqual(from_frame.length(), from_pkt.length())
        self.assertEqual(from_frame.payload_len, from_pkt.payload_len)
        self.assertEqual(from_frame.flags, from_pkt.flags)


class TestRoundTripIntoTrainableFeatures(unittest.TestCase):
    def test_handshake_feeds_69_features(self):
        frames = [
            Ether() / IP(src=A, dst=B) / TCP(sport=12345, dport=443,
                                             flags="S", window=64240),
            Ether() / IP(src=B, dst=A) / TCP(sport=443, dport=12345,
                                             flags="SA", window=65535,
                                             options=[("MSS", 1460)]),
            Ether() / IP(src=A, dst=B) / TCP(sport=12345, dport=443,
                                             flags="A", window=100) /
            RawHello(b"GET / HTTP/1.1"),
        ]
        events = [frame_to_event(bytes(f), ts=0.0 + i * 0.001)
                  for i, f in enumerate(frames)]
        flow = Flow(None, A, B, 12345, 443, 6, events[0].ts)
        for ev in events:
            flow.feed(ev)
        vec = flow.feature_vector()
        feats = flow.to_features()
        self.assertEqual(len(vec), 69)
        self.assertEqual(vec.dtype, "float32")
        self.assertEqual(feats["SYN Flag Count"], 2)
        self.assertEqual(feats["ACK Flag Count"], 2)
        self.assertEqual(feats["Total Fwd Packets"], 2)
        self.assertEqual(feats["Total Backward Packets"], 1)
        self.assertEqual(feats["Init_Win_bytes_forward"], 64240)
        self.assertEqual(feats["Init_Win_bytes_backward"], 65535)
        self.assertAlmostEqual(feats["Flow Duration"], 2000.0, places=1)
        self.assertEqual([n for n in feats], FEATURE_ORDER)


class RawHello(bytes):
    """Marker for the constructors above (readable call sites)."""
    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)