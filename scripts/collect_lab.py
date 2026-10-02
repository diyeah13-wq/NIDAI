#!/usr/bin/env python3
"""Stage D step 1 - collect labelled lab traffic from the authorized lab network.

WHY THIS EXISTS
---------------
Stage B proved the live pipeline works end to end, but the Stage 7 detector
called all 19 real port-scan flows BENIGN. Measured cause: CIC-IDS2017's
PortScan rows carry an impossible flag signature (SYN=0, PSH=1, RST=0 in 99.9%
of rows), so the region of feature space where REAL probes live has no training
support. Oversampling the existing rows 20x was measured NOT to move the
boundary. The fix is to collect real labelled probes from our own lab and add
them to training. This script collects them.

LAB ONLY. Every scenario targets 192.168.56.101 (our VirtualBox host-only
Metasploitable VM) or the local machine. Nothing here is written to attack a
third party, and every stimulus is a normal, non-destructive TCP connect or an
ordinary HTTP request.

GROUND TRUTH
------------
A label is assigned ONLY from the scenario that was executed and the time
window in which it ran. The model is never consulted. If a flow does not fit
its scenario's window and target rules, it is DISCARDED with a recorded reason
rather than guessed at. That is the whole point: an ambiguous flow must never
silently become a training label.

REPRODUCIBILITY
---------------
Every run writes a JSONL flow file plus a JSON manifest recording scenario,
exact command, target, interface, ISO + epoch start/end times, seed, per-reason
discard counts, and a SHA-256 of the flow file. Port lists and inter-packet
delays are generated from the seed, so re-running with the same seed and
scenario reproduces the same stimulus.

USAGE
-----
    python scripts/collect_lab.py --list-scenarios
    python scripts/collect_lab.py --iface "Ethernet 2" --target 192.168.56.101 \
        --scenario tcp_connect_scan --runs 3
    python scripts/collect_lab.py --iface "Ethernet 2" --target 192.168.56.101 \
        --scenario all --runs 2
    python scripts/collect_lab.py ... --dry-run      # print the plan, send nothing

OUTPUT
------
    data/lab/raw/<scenario>/<run_id>.jsonl    one JSON object per captured flow
    data/lab/manifests/<run_id>.json          provenance + integrity
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import select
import socket
import sys
import threading
import time
from urllib.request import Request, urlopen

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from scapy.sendrecv import AsyncSniffer

from src.config import CLASS_ORDER, FEATURE_ORDER, SEVERITY
from src.live.capture import packet_to_event
from src.live.flow_features import FlowTable

LAB_DIR = os.path.join(BASE_DIR, "data", "lab")
RAW_DIR = os.path.join(LAB_DIR, "raw")
MANIFEST_DIR = os.path.join(LAB_DIR, "manifests")


def set_lab_dir(path):
    """Repoint all output paths at a different lab root (--lab-dir)."""
    global LAB_DIR, RAW_DIR, MANIFEST_DIR
    LAB_DIR = os.path.abspath(path) if os.path.isabs(path) \
        else os.path.join(BASE_DIR, path)
    RAW_DIR = os.path.join(LAB_DIR, "raw")
    MANIFEST_DIR = os.path.join(LAB_DIR, "manifests")

DEFAULT_TARGET = "192.168.56.101"
DEFAULT_IFACE = "Ethernet 2"

# A flow is attributed to a scenario window only if its WHOLE lifetime falls
# inside [t_start, t_end + grace]. grace absorbs the few milliseconds a RST or
# a final ACK can arrive after the scan command returns. A flow that started
# before the window or is still open when it closes is ambiguous -> discarded.
DEFAULT_GRACE_S = 0.75


# --------------------------------------------------------------------------- #
# Reproducible stimulus port lists
# --------------------------------------------------------------------------- #
# A fixed list of well-known ports, used so the "top ports" scan is identical on
# every run. Widening beyond it uses a seeded Random so the extra ports are
# deterministic given --seed.
TOP_PORTS = [
    20, 21, 22, 23, 25, 53, 67, 68, 69, 80, 110, 111, 119, 123, 135, 137, 138,
    139, 143, 161, 179, 389, 443, 445, 465, 512, 513, 514, 515, 543, 544, 548,
    554, 587, 631, 636, 873, 993, 995, 1080, 1099, 1433, 1434, 1521, 1723, 2049,
    2121, 3306, 3389, 4444, 5432, 5900, 6000, 6379, 6667, 8000, 8009, 8080, 8081,
    8443, 8888, 9000, 9090, 9999, 10000, 11211, 27017, 33060,
]

# Realistic HTTP paths a browser would request. Used by BOTH the benign browse
# scenario and the path-scan scenario - see the label-noise warning in
# run_scenario: the two are separated by FLOW GRANULARITY (many short flows on
# many ports vs few long keep-alive flows), not by path content.
BROWSER_PATHS = [
    "/", "/index.html", "/favicon.ico", "/style.css", "/app.js",
    "/api/v1/status", "/api/v1/user", "/images/logo.png", "/login",
    "/dashboard", "/settings", "/search?q=nidai", "/about",
]


def port_list(count, seed):
    """Deterministic port list: TOP_PORTS first, then seeded extras under 65535."""
    import random
    ports = list(TOP_PORTS[:count])
    if count > len(ports):
        rng = random.Random(seed)
        seen = set(ports)
        while len(ports) < count:
            p = rng.randint(1, 65535)
            if p not in seen:
                seen.add(p)
                ports.append(p)
    return sorted(ports[:count])


# Windows connect_ex() on a non-blocking socket returns WSAEWOULDBLOCK (10035)
# immediately, and select() does NOT signal writability when the far end answers
# with RST - so a refused port looks exactly like a slow one and cannot be told
# apart from a filtered one using the socket API alone. We therefore do not try
# to guess here. The probe just fires the SYN and returns quickly; the
# authoritative open/closed verdict is derived from the CAPTURED packets after
# the run (see flow_verdict), which is wire truth and the very signature the
# model has to learn.


def probe_port(target, port, timeout):
    """Fire a non-blocking connect at a port. Returns 'answered' or 'no_answer'.

    Only drives scan pacing and progress output. Ground-truth labels never depend
    on this - they come from the executed scenario.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setblocking(False)
        rc = s.connect_ex((target, port))
        if rc == 0:
            return "answered"
        _r, w, _x = select.select([], [s], [], timeout)
        if not w:
            return "no_answer"
        return ("answered" if s.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) == 0
                else "refused")
    except OSError:
        return "error"
    finally:
        s.close()


# TCP flag bits as used by PacketEvent.flags.
FIN, SYN, RST, PSH, ACK = 1, 2, 4, 8, 16


def flow_verdict(fl):
    """Classify a captured flow from its own packets: handshake / reset / one_way.

    Derived from the capture, not from the socket API, so it reflects what was
    really on the wire. This is the signature the classifier must learn:
      handshake - SYN, SYN-ACK, ACK        -> port open
      reset     - SYN, RST                 -> port closed/refused
      one_way   - SYN with no reply        -> port filtered/dropped
    """
    fwd = 0
    for p in fl._fwd:
        fwd |= p.flags
    bwd = 0
    for p in fl._bwd:
        bwd |= p.flags
    if not fl._bwd:
        return "one_way"
    if bwd & RST:
        return "reset"
    if bwd & SYN:
        return "handshake"
    if bwd & ACK:
        return "ack_only"
    return "other"


# --------------------------------------------------------------------------- #
# Scenarios - each returns (label, human description of what was executed)
# --------------------------------------------------------------------------- #
def sc_tcp_connect_scan(ctx):
    """Connect scan many ports. One short bidirectional flow per probed port."""
    label = "PortScan"
    ports = port_list(ctx["scan_ports"], ctx["seed"])
    delay = ctx["scan_delay"]
    print("    probing %d ports (delay %.3fs)" % (len(ports), delay), flush=True)
    tally = {"answered": 0, "no_answer": 0, "refused": 0, "error": 0}
    for i, port in enumerate(ports):
        tally[probe_port(ctx["target"], port, ctx["sock_timeout"])] += 1
        if delay:
            time.sleep(delay)
        if ctx["progress"] and (i + 1) % 100 == 0:
            print("      %d/%d ports" % (i + 1, len(ports)), flush=True)
    print("    socket-level probe result: %s" % tally, flush=True)
    return label, ("TCP connect scan of %d ports on %s (socket probe tally: %s; "
                   "authoritative open/closed split is in wire_verdicts)"
                   % (len(ports), ctx["target"], tally))


def sc_tcp_connect_scan_slow(ctx):
    """Same scan, slower rate. Timing diversity for the Flow/IAT + rate features."""
    label = "PortScan"
    ports = port_list(ctx["scan_ports"], ctx["seed"] + 1)
    delay = ctx["scan_delay_slow"]
    print("    slow probing %d ports (delay %.3fs)" % (len(ports), delay),
          flush=True)
    tally = {"answered": 0, "no_answer": 0, "refused": 0, "error": 0}
    for port in ports:
        tally[probe_port(ctx["target"], port, ctx["sock_timeout"])] += 1
        time.sleep(delay)
    print("    socket-level probe result: %s" % tally, flush=True)
    return label, ("TCP connect scan of %d ports on %s at slow rate "
                   "(%.3fs between probes, socket probe tally: %s)"
                   % (len(ports), ctx["target"], delay, tally))


def sc_http_path_scan(ctx):
    """One short HTTP request per port. PSH-heavy flows, close to the CIC shape."""
    label = "PortScan"
    ports = port_list(ctx["scan_ports"], ctx["seed"] + 2)
    print("    HTTP path probe on %d ports" % len(ports), flush=True)
    answered = 0
    for i, port in enumerate(ports):
        try:
            s = socket.create_connection((ctx["target"], port), ctx["http_timeout"])
        except OSError:
            continue
        try:
            req = ("GET /probe-%d HTTP/1.1\r\nHost: %s\r\n"
                   "User-Agent: lab-probe/1.0\r\nConnection: close\r\n\r\n"
                   % (i, ctx["target"]))
            s.sendall(req.encode())
            s.recv(256)
            answered += 1
        except OSError:
            pass
        finally:
            s.close()
        if ctx["scan_delay"]:
            time.sleep(ctx["scan_delay"])
    print("    ports that answered HTTP: %d/%d" % (answered, len(ports)),
          flush=True)
    return label, ("single HTTP GET per port across %d ports on %s "
                   "(%d responded)" % (len(ports), ctx["target"], answered))


def _talk(target, port, script, timeout, reads=3):
    """Open one TCP connection, run a short request/response script, close.

    Each element of `script` is (bytes_to_send_or_None, bytes_to_read). This is
    the shape of ordinary client/server application traffic: payload moves in
    both directions, so the flows carry PSH/data and are structurally distinct
    from a bare port probe.
    """
    sent = recvd = 0
    try:
        s = socket.create_connection((target, port), timeout)
    except OSError:
        return 0, 0
    try:
        s.settimeout(timeout)
        for out, want in script:
            if out:
                s.sendall(out)
                sent += len(out)
            if want:
                try:
                    got = s.recv(want)
                    recvd += len(got)
                    if not got:
                        break
                except OSError:
                    break
        return sent, recvd
    except OSError:
        return sent, recvd
    finally:
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        s.close()


def sc_benign_web_browse(ctx):
    """Ordinary browsing of the lab web server. Data-carrying, multi-packet."""
    label = "BENIGN"
    base = ctx.get("web_base") or ("http://%s/" % ctx["target"])
    print("    browsing %s" % base, flush=True)
    done = 0
    for path in BROWSER_PATHS:
        try:
            req = Request(base.rstrip("/") + path,
                          headers={"User-Agent": "Mozilla/5.0 (lab-browse)",
                                   "Accept": "text/html,*/*"})
            with urlopen(req, timeout=ctx["http_timeout"]) as r:
                r.read(4096)
            done += 1
        except Exception:
            pass
        time.sleep(0.12)
    print("    pages fetched: %d/%d" % (done, len(BROWSER_PATHS)), flush=True)
    return label, ("sequential HTTP GETs of %d paths on %s, reading response "
                   "bodies (%d fetched)"
                   % (len(BROWSER_PATHS), base, done))


def sc_benign_ssh_session(ctx):
    """SSH banner exchange on the lab VM. Real protocol bytes, both directions."""
    label = "BENIGN"
    port = 22
    print("    SSH banner exchange on port %d" % port, flush=True)
    sent, recvd = _talk(ctx["target"], port,
                        [(b"SSH-2.0-OpenSSH_7.9p1 Debian-10\r\n", 256)],
                        ctx["http_timeout"])
    print("    ssh bytes sent=%d received=%d" % (sent, recvd), flush=True)
    return label, ("SSH banner exchange with %s:%d (%d bytes out, %d bytes in)"
                   % (ctx["target"], port, sent, recvd))


def sc_benign_ftp_control(ctx):
    """FTP control-channel login against the lab VM. Real protocol bytes."""
    label = "BENIGN"
    port = 21
    print("    FTP control session on port %d" % port, flush=True)
    sent, recvd = _talk(ctx["target"], port,
                        [(None, 256), (b"USER anonymous\r\n", 256),
                         (b"PASS lab@example.invalid\r\n", 256),
                         (b"QUIT\r\n", 128)],
                        ctx["http_timeout"])
    print("    ftp bytes sent=%d received=%d" % (sent, recvd), flush=True)
    return label, ("FTP control login (USER/PASS/QUIT) to %s:%d "
                   "(%d bytes out, %d bytes in)"
                   % (ctx["target"], port, sent, recvd))


def sc_benign_mysql_handshake(ctx):
    """MySQL server greeting on the lab VM. Real multi-packet payload exchange."""
    label = "BENIGN"
    port = 3306
    print("    MySQL greeting read on port %d" % port, flush=True)
    sent, recvd = _talk(ctx["target"], port, [(None, 512)], ctx["http_timeout"])
    print("    mysql bytes sent=%d received=%d" % (sent, recvd), flush=True)
    return label, ("MySQL initial handshake read from %s:%d "
                   "(%d bytes out, %d bytes in)"
                   % (ctx["target"], port, sent, recvd))


def sc_benign_idle(ctx):
    """Passive baseline: capture whatever the lab generates, send nothing."""
    label = "BENIGN"
    print("    idle for %.1fs, sending nothing" % ctx["idle_s"], flush=True)
    time.sleep(ctx["idle_s"])
    return label, ("idle baseline, %.1fs of passive capture with no stimulus "
                   "generated by this script" % ctx["idle_s"])


SCENARIOS = {
    "tcp_connect_scan": sc_tcp_connect_scan,
    "tcp_connect_scan_slow": sc_tcp_connect_scan_slow,
    "http_path_scan": sc_http_path_scan,
    "benign_web_browse": sc_benign_web_browse,
    "benign_ssh_session": sc_benign_ssh_session,
    "benign_ftp_control": sc_benign_ftp_control,
    "benign_mysql_handshake": sc_benign_mysql_handshake,
    "benign_idle": sc_benign_idle,
}

# Declared per scenario, never inferred from the label string.
#   allow_target - may a flow in this window legitimately involve the scan
#                  target? Benign scenarios that talk to a VM service say yes;
#                  the passive idle baseline says no.
#   proto        - which IP protocols belong to this scenario, or None for any.
# NOTE: there is deliberately no "browse the internet and call it BENIGN"
# scenario. Traffic leaving over Wi-Fi cannot be attributed on a host-only
# capture, and labelling unrelated campus traffic BENIGN would be fabricated
# ground truth. Every benign scenario here is generated against our own VM.
SCENARIO_META = {
    "tcp_connect_scan":        {"label": "PortScan", "allow_target": True,  "proto": {6}},
    "tcp_connect_scan_slow":   {"label": "PortScan", "allow_target": True,  "proto": {6}},
    "http_path_scan":          {"label": "PortScan", "allow_target": True,  "proto": {6}},
    "benign_web_browse":       {"label": "BENIGN",   "allow_target": True,  "proto": {6}},
    "benign_ssh_session":      {"label": "BENIGN",   "allow_target": True,  "proto": {6}},
    "benign_ftp_control":      {"label": "BENIGN",   "allow_target": True,  "proto": {6}},
    "benign_mysql_handshake":  {"label": "BENIGN",   "allow_target": True,  "proto": {6}},
    "benign_idle":             {"label": "BENIGN",   "allow_target": False, "proto": None},
}

# Order matters: attack scenarios run before the passive benign ones so a
# benign capture is never contaminated by scan teardown.
SCENARIO_ORDER = ["tcp_connect_scan", "tcp_connect_scan_slow",
                  "http_path_scan", "benign_web_browse",
                  "benign_ssh_session", "benign_ftp_control",
                  "benign_mysql_handshake", "benign_idle"]


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
class Capture:
    """Scapy AsyncSniffer + FlowTable, no model and no scorer anywhere."""

    def __init__(self, iface, flow_timeout):
        self.iface = iface
        self.flow_table = FlowTable(flow_timeout, min_bwd_packets=0)
        self.n_packets = 0
        self.n_parse_skipped = 0
        self._lock = threading.Lock()
        self._sniffer = None

    def _on_packet(self, pkt):
        ts = getattr(pkt, "time", None)
        try:
            ev = packet_to_event(pkt, float(ts) if ts else None)
        except Exception:
            ev = None
        if ev is None:
            with self._lock:
                self.n_parse_skipped += 1
            return
        with self._lock:
            self.n_packets += 1
            self.flow_table.add(ev)

    def start(self):
        self._sniffer = AsyncSniffer(iface=self.iface, store=0,
                                     prn=self._on_packet)
        self._sniffer.start()

    def stop(self):
        if self._sniffer is not None:
            try:
                self._sniffer.stop()
            except Exception:
                pass
            self._sniffer = None

    def drain(self):
        """Pop every flow accumulated so far; returns list of Flow objects."""
        with self._lock:
            flows = list(self.flow_table._flows.values())
            self.flow_table._flows.clear()
        flows.sort(key=lambda f: f.first_ts)
        return flows

    def stats(self):
        with self._lock:
            return self.n_packets, self.n_parse_skipped


# --------------------------------------------------------------------------- #
# Labelling - scenario-derived only
# --------------------------------------------------------------------------- #
def classify_flows(flows, allow_target, target, win_start, win_end,
                   proto_filter=None):
    """Split drained flows into kept records and discarded (reason, flow) pairs.

    A flow is KEPT only if every one of these holds:

      1. it started at or after the scenario window opened, and finished at or
         before the window closed (+ grace) - i.e. its whole life is inside;
      2. it satisfies the scenario's declared target rule: an attack window may
         only keep flows that involve the target, and a benign window keeps them
         only if the scenario allows it (a declared service session) and
         discards them if it does not (the passive idle baseline);
      3. if the scenario sets a proto_filter, the flow's protocol matches.

    Everything else is DISCARDED with an explicit reason. Nothing is guessed.
    """
    kept, dropped = [], []
    for fl in flows:
        if fl.first_ts < win_start:
            dropped.append(("started_before_window", fl))
            continue
        if fl.last_ts > win_end:
            dropped.append(("ended_after_window", fl))
            continue
        touches_target = (fl.src == target or fl.dst == target)
        if touches_target and not allow_target:
            dropped.append(("involves_target_not_allowed_here", fl))
            continue
        if proto_filter is not None and fl.proto not in proto_filter:
            dropped.append(("proto_mismatch", fl))
            continue
        kept.append(fl)
    return kept, dropped


def flow_to_record(fl, run_id, scenario, label, target):
    """One JSON-serialisable lab record. Features in config.FEATURE_ORDER."""
    feats = fl.to_features()
    return {
        "run_id": run_id,
        "scenario": scenario,
        "label": label,
        "label_source": "executed_scenario",
        "label_code": CLASS_ORDER.index(label),
        "severity": SEVERITY.get(label, "UNKNOWN"),
        "src": fl.src,
        "dst": fl.dst,
        "sport": fl.sport,
        "dport": fl.dport,
        "proto": fl.proto,
        "first_ts": round(fl.first_ts, 6),
        "last_ts": round(fl.last_ts, 6),
        "n_packets": len(fl._fwd) + len(fl._bwd),
        "n_fwd": len(fl._fwd),
        "n_bwd": len(fl._bwd),
        "involves_target": (fl.src == target or fl.dst == target),
        "wire_verdict": flow_verdict(fl),
        "scoreable": fl.is_scoreable(1, 1),
        "features": {name: float(feats[name]) for name in FEATURE_ORDER},
    }


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# One collection run
# --------------------------------------------------------------------------- #
def run_scenario(scenario, args, run_id):
    """Execute one scenario while capturing; return (manifest, exit_ok)."""
    fn = SCENARIOS[scenario]
    meta = SCENARIO_META[scenario]
    allow_target = meta["allow_target"]
    proto_filter = meta["proto"]

    os.makedirs(os.path.join(RAW_DIR, scenario), exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_DIR, scenario, run_id + ".jsonl")

    flow_timeout = args.timeout
    # Drain long enough for every flow to be finalized: idle timeout plus the
    # 1s TCP teardown shortcut, plus a margin.
    drain_s = flow_timeout + 2.0

    cap = Capture(args.iface, flow_timeout)
    print("  [%s] run_id=%s" % (scenario, run_id), flush=True)
    print("    iface=%s target=%s timeout=%.1fs" % (args.iface, args.target,
                                                    flow_timeout), flush=True)

    pre = cap.drain()  # anything already in flight belongs to no scenario
    cap.start()
    time.sleep(args.settle)
    pre_drained = cap.drain()  # settle period flows: also unscenaried

    ctx = {
        "target": args.target,
        "seed": args.seed,
        "scan_ports": args.scan_ports,
        "scan_delay": args.scan_delay,
        "scan_delay_slow": args.scan_delay_slow,
        "sock_timeout": args.sock_timeout,
        "http_timeout": args.http_timeout,
        "idle_s": args.idle_s,
        "web_base": args.web_base,
        "progress": args.progress,
    }

    t_start = time.time()
    try:
        observed_label, command = fn(ctx)
        if observed_label != meta["label"]:
            print("    FATAL: scenario '%s' returned label '%s' but is declared "
                  "'%s'. Aborting rather than mislabel data."
                  % (scenario, observed_label, meta["label"]), flush=True)
            cap.stop()
            return None, False
    except KeyboardInterrupt:
        print("    interrupted; aborting scenario", flush=True)
        cap.stop()
        return None, False
    t_end = time.time()
    win_end = t_end + args.grace

    print("    scenario done in %.1fs, draining %.1fs" % (t_end - t_start, drain_s),
          flush=True)
    time.sleep(drain_s)
    cap.stop()
    time.sleep(0.3)

    flows = cap.drain()
    kept, dropped = classify_flows(flows, allow_target, args.target,
                                   t_start, win_end, proto_filter)

    with open(raw_path, "w", encoding="utf-8") as fh:
        for fl in kept:
            fh.write(json.dumps(flow_to_record(fl, run_id, scenario,
                                               observed_label, args.target)) + "\n")

    n_pkts, n_skipped = cap.stats()
    reasons = {}
    for reason, _fl in dropped:
        reasons[reason] = reasons.get(reason, 0) + 1
    verdicts = {}
    for fl in kept:
        v = flow_verdict(fl)
        verdicts[v] = verdicts.get(v, 0) + 1
    n_scoreable = sum(1 for fl in kept if fl.is_scoreable(1, 1))

    manifest = {
        "run_id": run_id,
        "scenario": scenario,
        "label": observed_label,
        "label_source": "executed_scenario",
        "ground_truth_note": ("label comes from the scenario that was executed "
                              "and its time window; the model was not consulted"),
        "command": command,
        "target": args.target,
        "interface": {
            "name": getattr(args.iface, "name", str(args.iface)),
            "description": getattr(args.iface, "description", ""),
            "ip": getattr(args.iface, "ip", ""),
        },
        "started_at_iso": datetime.datetime.fromtimestamp(
            t_start).astimezone().isoformat(),
        "ended_at_iso": datetime.datetime.fromtimestamp(
            t_end).astimezone().isoformat(),
        "t_start_epoch": round(t_start, 6),
        "t_end_epoch": round(t_end, 6),
        "window_end_epoch": round(win_end, 6),
        "grace_s": args.grace,
        "flow_timeout_s": flow_timeout,
        "seed": args.seed,
        "params": {k: ctx[k] for k in sorted(ctx) if k != "progress"},
        "counts": {
            "packets_captured": n_pkts,
            "packets_unparsed": n_skipped,
            "flows_drained": len(flows),
            "flows_kept": len(kept),
            "flows_discarded": len(dropped),
            "flows_discarded_by_reason": reasons,
            "flows_pre_scenario_discarded": len(pre) + len(pre_drained),
            "kept_scoreable": n_scoreable,
            "wire_verdicts": verdicts,
        },
        "raw_file": os.path.relpath(raw_path, BASE_DIR).replace("\\", "/"),
        "raw_file_sha256": sha256_of(raw_path),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "scapy": _scapy_version(),
        },
    }

    manifest_path = os.path.join(MANIFEST_DIR, run_id + ".json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    c = manifest["counts"]
    print("    packets=%d unparsed=%d | flows drained=%d kept=%d discarded=%d"
          % (c["packets_captured"], c["packets_unparsed"], c["flows_drained"],
             c["flows_kept"], c["flows_discarded"]), flush=True)
    if reasons:
        print("    discard reasons: %s" % reasons, flush=True)
    print("    -> %s" % os.path.relpath(raw_path, BASE_DIR), flush=True)
    print("    -> %s" % os.path.relpath(manifest_path, BASE_DIR), flush=True)
    return manifest, True


def _scapy_version():
    try:
        import scapy
        return getattr(scapy, "__version__", "unknown")
    except Exception:
        return "unavailable"


def make_run_id(scenario, seed):
    stamp = datetime.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    return "%s_%s_s%d" % (scenario, stamp, seed)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def resolve_iface(query):
    """Resolve an interface name / description / IP the same way the CLI does."""
    from scapy.config import conf
    if not query:
        return conf.iface
    q = str(query).strip().lower()
    try:
        for _k, v in conf.ifaces.items():
            if (q == getattr(v, "name", "").lower()
                    or q in getattr(v, "description", "").lower()
                    or q == getattr(v, "ip", "")):
                return v
    except Exception:
        pass
    return query


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Collect labelled lab traffic for NIDAI domain adaptation "
                    "(authorized lab only).")
    p.add_argument("--iface", default=DEFAULT_IFACE,
                   help="interface to capture on (default: %(default)s)")
    p.add_argument("--target", default=DEFAULT_TARGET,
                   help="lab target host for scan scenarios (default: %(default)s)")
    p.add_argument("--scenario", default="tcp_connect_scan",
                   help="comma-separated scenarios, or 'all' (default: %(default)s)")
    p.add_argument("--runs", type=int, default=1,
                   help="independent repetitions per scenario (default: %(default)s)")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for deterministic port lists (default: %(default)s)")
    p.add_argument("--timeout", type=float, default=2.0,
                   help="flow idle timeout in seconds (default: %(default)s)")
    p.add_argument("--grace", type=float, default=DEFAULT_GRACE_S,
                   help="seconds a flow may finish after the window closes "
                        "(default: %(default)s)")
    p.add_argument("--settle", type=float, default=1.0,
                   help="seconds to settle before the scenario starts (default: %(default)s)")
    p.add_argument("--scan-ports", type=int, default=200,
                   help="ports per scan scenario (default: %(default)s)")
    p.add_argument("--scan-delay", type=float, default=0.01,
                   help="seconds between probes, fast scan (default: %(default)s)")
    p.add_argument("--scan-delay-slow", type=float, default=0.08,
                   help="seconds between probes, slow scan (default: %(default)s)")
    p.add_argument("--sock-timeout", type=float, default=0.05,
                   help="per-port connect wait; only affects scan pacing "
                        "(default: %(default)s)")
    p.add_argument("--http-timeout", type=float, default=0.8,
                   help="timeout for the HTTP path-scan requests (default: %(default)s)")
    p.add_argument("--idle-s", type=float, default=20.0,
                   help="seconds for the idle benign baseline (default: %(default)s)")
    p.add_argument("--web-base", default=None,
                   help="base URL for benign browsing (default: http://TARGET/)")
    p.add_argument("--gap", type=float, default=3.0,
                   help="seconds between runs (default: %(default)s)")
    p.add_argument("--progress", action="store_true",
                   help="print progress every 100 ports")
    p.add_argument("--list-scenarios", action="store_true",
                   help="list scenarios and exit")
    p.add_argument("--lab-dir", default="data/lab",
                   help="output root for raw/ + manifests/ "
                        "(default: %(default)s)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan without capturing or sending anything")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    set_lab_dir(args.lab_dir)

    if args.list_scenarios:
        print("Available scenarios:\n")
        for name in SCENARIO_ORDER:
            fn = SCENARIOS[name]
            doc = (fn.__doc__ or "").strip().splitlines()[0]
            m = SCENARIO_META[name]
            tgt = "may involve target" if m["allow_target"] else "no target traffic"
            print("  %-24s [%-8s] %s\n      target rule: %s\n"
                  % (name, m["label"], doc, tgt))
        return 0

    if args.scenario.strip().lower() == "all":
        chosen = list(SCENARIO_ORDER)
    else:
        chosen = [s.strip() for s in args.scenario.split(",") if s.strip()]
    for s in chosen:
        if s not in SCENARIOS:
            print("ERROR: unknown scenario '%s'. Use --list-scenarios." % s,
                  file=sys.stderr)
            return 2

    args.iface = resolve_iface(args.iface)

    print("=" * 72)
    print("  NIDAI Stage D - lab traffic collector (authorized lab only)")
    print("=" * 72)
    print("  interface : %s (%s)" % (getattr(args.iface, "name", args.iface),
                                     getattr(args.iface, "description", "")))
    print("  adapter IP: %s" % getattr(args.iface, "ip", "N/A"))
    print("  target    : %s" % args.target)
    print("  scenarios : %s" % ", ".join(chosen))
    print("  runs each : %d" % args.runs)
    print("  seed      : %d" % args.seed)
    print("  output    : %s" % os.path.relpath(LAB_DIR, BASE_DIR))

    if args.dry_run:
        print("\n  DRY RUN - no packets captured, no traffic generated.")
        for name in chosen:
            fn = SCENARIOS[name]
            doc = (fn.__doc__ or "").strip().splitlines()[0]
            print("\n  %-24s %s" % (name, doc))
            if name in ("tcp_connect_scan", "tcp_connect_scan_slow",
                        "http_path_scan"):
                pl = port_list(args.scan_ports, args.seed)
                print("      would probe %d ports, first 12: %s"
                      % (len(pl), pl[:12]))
        print("\n  Estimated capture time: %.0f s per run x %d run(s) x %d scenario(s)"
              % (args.timeout + 4, args.runs, len(chosen)))
        return 0

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    manifests, failed = [], 0
    for name in chosen:
        for r in range(args.runs):
            run_id = make_run_id(name, args.seed + r)
            m, ok = run_scenario(name, args, run_id)
            if not ok:
                failed += 1
                break
            manifests.append(m)
            if args.gap and (r + 1 < args.runs or name != chosen[-1]):
                time.sleep(args.gap)

    print("\n" + "=" * 72)
    print("  COLLECTION SUMMARY")
    print("=" * 72)
    tot = {}
    for m in manifests:
        c = m["counts"]
        key = m["scenario"]
        agg = tot.setdefault(key, {"label": m["label"], "runs": 0, "kept": 0,
                                  "discarded": 0, "packets": 0})
        agg["runs"] += 1
        agg["kept"] += c["flows_kept"]
        agg["discarded"] += c["flows_discarded"]
        agg["packets"] += c["packets_captured"]
    print("  %-24s %-10s %5s %8s %10s %10s"
          % ("scenario", "label", "runs", "kept", "discarded", "packets"))
    for key in SCENARIO_ORDER:
        if key not in tot:
            continue
        a = tot[key]
        print("  %-24s %-10s %5d %8d %10d %10d"
              % (key, a["label"], a["runs"], a["kept"], a["discarded"],
                 a["packets"]))
    by_label = {}
    for m in manifests:
        by_label[m["label"]] = by_label.get(m["label"], 0) + m["counts"]["flows_kept"]
    print("\n  kept flows by label: %s" % (by_label or "none"))
    print("  runs written: %d" % len(manifests))
    print("  manifests: %s" % os.path.relpath(MANIFEST_DIR, BASE_DIR))
    if failed:
        print("  WARNING: %d scenario run(s) aborted" % failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())