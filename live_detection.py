#!/usr/bin/env python3
"""NIDAI — Live Network Intrusion Detection System.

Real-time packet capture, bidirectional flow construction, 69-feature extraction,
and machine-learning-based intrusion detection with the trained NIDAI models.

Usage:
    python live_detection.py                   # Sniff on default active interface
    python live_detection.py --list-ifaces     # List network interfaces
    python live_detection.py --iface Wi-Fi     # Sniff on specific interface
    python live_detection.py --explain         # Show 5-technique explanation for alerts
    python live_detection.py --pcap demo.pcap  # Test against a PCAP file
"""
import argparse
import datetime
import json
import os
import signal
import sys
import time
from collections import Counter

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CLASS_ORDER, SEVERITY
from src.live.detector import (
    LiveDetector, _load_scorer, resolve_interface,
    list_interfaces, DETECTOR_KEY,
)
from src.live.plain import explain_lines

# ANSI Color codes for clean terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

PROTO_MAP = {6: "TCP", 17: "UDP", 1: "ICMP"}


def color_for_severity(sev):
    if sev == "SAFE":
        return GREEN
    if sev == "MEDIUM":
        return YELLOW
    if sev == "HIGH":
        return RED
    if sev == "CRITICAL":
        return MAGENTA
    return RESET


def print_banner():
    print(f"""
{CYAN}{BOLD}======================================================================
  NIDAI — Live Network Intrusion Detection System
  CIC-IDS2017 Pretrained Model Engine
======================================================================{RESET}
""")


def print_alert(alert, show_explanation=False):
    """Format and print an intrusion alert card or benign status."""
    cls = alert["predicted_class"]
    sev = alert["severity"]
    conf = alert["confidence"]
    src_ip = alert["src"]
    dst_ip = alert["dst"]
    dport = alert["dport"]
    proto_num = alert["proto"]
    proto_name = PROTO_MAP.get(proto_num, str(proto_num))
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    color = color_for_severity(sev)

    if cls == "BENIGN":
        print(f"{DIM}[{now_str}]{RESET} {GREEN}[OK] BENIGN{RESET}  {src_ip} -> {dst_ip}:{dport} ({proto_name})  conf: {conf:.1%}")
        return

    # Attack alert card
    print(f"\n{color}{BOLD}{'!' * 60}{RESET}")
    print(f"{color}{BOLD}[!] NIDAI ALERT #{alert.get('num', 1)}: [{cls.upper()}] -- Severity: {sev}{RESET}")
    print(f"{color}{BOLD}{'!' * 60}{RESET}")
    print(f"  {BOLD}Source:{RESET}       {src_ip}")
    print(f"  {BOLD}Destination:{RESET}  {dst_ip}:{dport} ({proto_name})")
    print(f"  {BOLD}Prediction:{RESET}   {color}{cls}{RESET} ({conf * 100:.1f}% confidence)")
    print(f"  {BOLD}Timestamp:{RESET}    {now_str}")

    # Top class probabilities
    probs = alert.get("probabilities", {})
    sorted_probs = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
    prob_str = " | ".join(f"{c}: {p*100:.1f}%" for c, p in sorted_probs if p > 0.01)
    print(f"  {BOLD}Probabilities:{RESET} {prob_str}")

    if show_explanation and alert.get("explanation"):
        print(f"\n  {BOLD}Explanation:{RESET}")
        for line in explain_lines(alert["explanation"]):
            print(f"    {line}")
    print(f"{color}{'-' * 60}{RESET}\n", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NIDAI Live Network Intrusion Detection Engine"
    )
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument(
        "--live", action="store_true", default=True,
        help="Capture live packets from a network interface (default)",
    )
    src_group.add_argument(
        "--pcap", type=str, default=None,
        help="Replay packets from a .pcap file instead of live capture",
    )
    parser.add_argument(
        "--list-ifaces", action="store_true",
        help="List available network interfaces and exit",
    )
    parser.add_argument(
        "--iface", type=str, default=None,
        help="Network interface name, IP, or GUID (default: auto-detect active)",
    )
    parser.add_argument(
        "--count", type=int, default=0,
        help="Max packets to capture (0 = continuous until stopped)",
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0,
        help="Flow idle timeout in seconds (default: 5.0s)",
    )
    parser.add_argument(
        "--detector", type=str, default=DETECTOR_KEY,
        help=f"Detection model key (default: '{DETECTOR_KEY}')",
    )
    parser.add_argument(
        "--min-bwd-pkts", type=int, default=1,
        help="Minimum reverse-direction packets for a flow to be scored "
             "(default: 1). One-way multicast/SSDP noise is structurally unlike "
             "the bidirectional CIC training flows; set 0 to score everything.",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="Generate 5-technique explanation for each detected alert",
    )
    parser.add_argument(
        "--quiet-benign", action="store_true",
        help="Suppress one-line notifications for benign traffic (show alerts only)",
    )
    parser.add_argument(
        "--log-alerts", type=str, default=None,
        help="Optional path to write JSON alerts log file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_ifaces:
        print(list_interfaces())
        return

    print_banner()

    # Load scoring engine
    print(f"[*] Loading detection model: {args.detector}...")
    scorer = _load_scorer(args.detector, with_explain=args.explain)
    detector = LiveDetector(scorer, flow_timeout_s=args.timeout,
                            min_bwd_packets=args.min_bwd_pkts)
    print(f"[+] Model loaded successfully. Classes: {', '.join(CLASS_ORDER)}")

    stats = Counter()
    alert_log = []

    def on_alert(alert):
        cls = alert["predicted_class"]
        stats[cls] += 1
        stats["total_flows"] += 1

        if args.log_alerts:
            record = {
                "num": alert.get("num"),
                "time": datetime.datetime.now().isoformat(),
                "src": alert["src"],
                "dst": alert["dst"],
                "dport": alert["dport"],
                "proto": alert["proto"],
                "predicted_class": cls,
                "severity": alert["severity"],
                "confidence": round(alert["confidence"], 4),
            }
            alert_log.append(record)
            try:
                with open(args.log_alerts, "w", encoding="utf-8") as fh:
                    json.dump(alert_log, fh, indent=2)
            except Exception:
                pass

        if cls == "BENIGN" and args.quiet_benign:
            return

        print_alert(alert, show_explanation=args.explain)

    # Replay PCAP or run Live Capture
    if args.pcap:
        if not os.path.exists(args.pcap):
            print(f"{RED}[-] Error: PCAP file not found: {args.pcap}{RESET}")
            sys.exit(1)
        print(f"[*] Replaying PCAP: {args.pcap}")
        detector.replay(args.pcap, explain=args.explain, emit=on_alert)
        # Flush remaining flows at the end of the pcap
        for a in detector.flush_all(explain=args.explain):
            on_alert(a)
    else:
        chosen_iface = resolve_interface(args.iface)
        iface_name = getattr(chosen_iface, "name", str(chosen_iface))
        iface_desc = getattr(chosen_iface, "description", "")
        iface_ip = getattr(chosen_iface, "ip", "N/A")

        print(f"[*] Bound Interface: {BOLD}{iface_name}{RESET} ({iface_desc})")
        print(f"[*] Adapter IP:       {iface_ip}")
        print(f"[*] Flow Timeout:     {args.timeout}s (fast TCP teardown: 1.0s)")
        print(f"[*] Scoring Gate:     >= {args.min_bwd_pkts} reverse packet(s) "
              f"per flow (one-way traffic is skipped)")
        print(f"[*] Capture Limit:    {'Continuous (Ctrl+C to stop)' if args.count == 0 else f'{args.count} packets'}")
        print(f"\n{GREEN}[+] NIDAI Live Sniffer Active. Listening for traffic...{RESET}\n", flush=True)

        try:
            detector.live_sniff(
                iface=chosen_iface,
                count=args.count if args.count > 0 else None,
                explain=args.explain,
                emit=on_alert,
            )
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[*] Stopping packet capture on user request...{RESET}")

    # Summary
    print(f"\n{CYAN}{BOLD}======================================================================{RESET}")
    print(f"{BOLD}  SESSION SUMMARY{RESET}")
    print(f"{CYAN}{BOLD}======================================================================{RESET}")
    print(f"  Packets processed:   {detector.n_packets}")
    print(f"  Flows evaluated:     {detector.n_flushed}")
    dropped = detector.n_dropped_unscoreable
    if dropped:
        print(f"  Skipped (one-way):   {dropped}  "
              f"(no reverse traffic; needs --min-bwd-pkts 0 to force)")
    print(f"  Breakdown by class:")
    for c in CLASS_ORDER:
        cnt = stats.get(c, 0)
        col = GREEN if c == "BENIGN" else (RED if cnt > 0 else DIM)
        print(f"    {col}{c:<14}: {cnt}{RESET}")
    if args.log_alerts:
        print(f"\n  [+] Saved {len(alert_log)} alerts to: {args.log_alerts}")
    print(f"{CYAN}{BOLD}======================================================================{RESET}\n")


if __name__ == "__main__":
    main()
