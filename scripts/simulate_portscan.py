#!/usr/bin/env python3
"""Simulate a controlled, local TCP port scan probe for testing NIDAI live detection.

Sends non-destructive TCP SYN connection attempts across a range of ports on
a designated local/lab IP address to generate real port-scanning traffic.

Usage:
    python scripts/simulate_portscan.py [TARGET_IP]
    python scripts/simulate_portscan.py 155.155.3.91
"""
import socket
import sys
import time

TARGET_IP = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PROBE_PORTS = [
    21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445,
    993, 995, 1433, 1521, 3306, 3389, 5432, 5900, 8080, 8443,
]

print(f"[*] NIDAI Controlled Probe Generator")
print(f"[*] Target IP:     {TARGET_IP}")
print(f"[*] Target Ports:  {len(PROBE_PORTS)} ports: {PROBE_PORTS[:8]}...")
print(f"[*] Dispatching TCP probe connections...")

success_count = 0
closed_count = 0

for port in PROBE_PORTS:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.15)
        res = s.connect_ex((TARGET_IP, port))
        if res == 0:
            success_count += 1
        else:
            closed_count += 1
        s.close()
    except Exception as e:
        closed_count += 1
    time.sleep(0.02)  # slight spacing

print(f"[+] Probing complete: {success_count} open, {closed_count} closed/filtered.")
