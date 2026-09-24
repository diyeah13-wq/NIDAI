"""Phase 3 - capture layer: raw frames -> PacketEvent (scapy).

This is the only module in the live stack that touches scapy / Npcap. It is
kept tiny on purpose: everything after it (flow math, scoring, explanation)
works on PacketEvent, so the pipeline runs and tests without any packet
capture privilege.

Two entry points that share one implementation:

  * ``frame_to_event(frame_bytes, ts)``  - the live path. What ``sniff`` hands
    back (or what a pcap replay yields) is raw bytes; we re-parse them so all
    numeric fields (IP total length, TCP header length, option bytes) are
    self-consistent on-wire values.
  * ``packet_to_event(pkt, ts)``        - convenience for tests / callers that
    already hold a scapy Packet (``bytes(pkt)`` is applied first so lengths
    are final).

Field mapping (units follow the ``flow_features`` contract):

  * payload_len  = IP payload minus transport header (the data bytes)
  * header_len   = transport header length incl. TCP options (TCP/UDP/ICMP)
  * so ``length() = payload + header`` ~= IP payload length; matches the
    synthetic-packet convention pinned by tests/test_flow_features.py.
  * flags        = TCP 6-bit flag byte (FIN..ECE). Non-TCP -> 0.
  * win_size     = TCP window size. Non-TCP -> 0.
  * proto        = IP protocol number (TCP=6, UDP=17, ICMP=1, ...).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether

from src.live.flow_features import PacketEvent

# Transport header sizes (bytes) when scapy cannot tell us otherwise.
UDP_HDR = 8
ICMP_HDR = 8
NO_TS = 0.0


def _tcp_flag_byte(pkt):
    """TCP flags as the 6-bit byte used by flow_features (FIN..ECE)."""
    return int(pkt[TCP].flags) & 0xFF


def _ip_info(pkt):
    """(src, dst, proto, total_len, ip_hdr_len) from IP or IPv6."""
    if pkt.haslayer(IPv6):
        ip = pkt[IPv6]
        return (ip.src, ip.dst, int(ip.nh), 40 + int(ip.plen), 40)
    if pkt.haslayer(IP):
        ip = pkt[IP]
        return (ip.src, ip.dst, int(ip.proto),
                int(ip.len), int(ip.ihl) * 4)
    return None


def _transport(pkt, iplen, ip_hdr):
    """(sport, dport, header_len, payload_len, flags, win)."""
    if pkt.haslayer(TCP):
        tcp = pkt[TCP]
        hdr = int(tcp.dataofs) * 4   # on-wire TCP header length incl. options
        return (int(tcp.sport), int(tcp.dport), hdr,
                max(iplen - ip_hdr - hdr, 0), _tcp_flag_byte(pkt),
                int(tcp.window))
    if pkt.haslayer(UDP):
        udp = pkt[UDP]
        return (int(udp.sport), int(udp.dport), UDP_HDR,
                max(iplen - ip_hdr - UDP_HDR, 0), 0, 0)
    if pkt.haslayer(ICMP):
        return (0, 0, ICMP_HDR, max(iplen - ip_hdr - ICMP_HDR, 0), 0, 0)
    return (0, 0, 0, max(iplen - ip_hdr, 0), 0, 0)


def packet_to_event(pkt, ts=None):
    """scapy Packet -> PacketEvent (raw bytes are a no-op wrapper)."""
    if isinstance(pkt, (bytes, bytearray)):
        return frame_to_event(bytes(pkt), ts)
    try:
        frame = bytes(pkt)  # finalise computed fields (lengths, checksums)
    except Exception:
        return None
    return frame_to_event(frame, ts)


def frame_to_event(frame, ts=None):
    """Raw Ethernet frame bytes -> PacketEvent (or None -> skip)."""
    try:
        pkt = Ether(frame)
    except Exception:
        return None
    info = _ip_info(pkt)
    if info is None:
        return None
    src, dst, proto, iplen, ip_hdr = info
    sport, dport, hlen, plen, flags, win = _transport(pkt, iplen, ip_hdr)
    if ts is None:
        ts = getattr(pkt, "time", NO_TS)
    return PacketEvent(ts=float(ts), src=src, dst=dst, sport=sport,
                       dport=dport, proto=proto, payload_len=plen,
                       header_len=hlen, flags=flags, win_size=win)