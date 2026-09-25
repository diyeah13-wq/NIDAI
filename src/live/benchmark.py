"""Stage 11 - live pipeline performance: throughput, latency, explain cost.

Four things an operator actually needs numbers for:

  1. THROUGHPUT: how many flows can ``LiveDetector`` finalize + score per second
     on this machine (packet ingestion, flow math, one HGB predict per flow).
  2. DETECTION LATENCY vs FLOW TIMEOUT: timeout is a knob (idle seconds before a
     flow is declared finished); bigger timeout = richer features but slower
     alerts. Measured in SIMULATED seconds (wall clock is pipelined away).
  3. EXPLAIN COST: the 5-technique explanation runs off the hot path (per-alert,
     on demand). We time single calls on real models and report ms/alert.
  4. DETECTOR COMPARISON: score-only flows/sec for the HGB (recommended) vs the
     Random Forest, so the detection-core choice has an operational number too.

Traffic is a deterministic synthetic stream (scapy-crafted frames, no Npcap):
``flows_per_sec`` sessions per simulated second, each a 2-8 packet
SYN/ACK/data exchange against mixed ports and one offered load. Everything is
flushed by one trailing packet, so flow accounting is exact.

Run:  python src/live/benchmark.py [--flows 10000] [--fps 1000] [--seed 42]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.config import METRICS_DIR, PLOTS_DIR
from src.live.detector import LiveDetector, _load_scorer, DETECTOR_KEY

PORTS = [80, 443, 53, 22, 8080, 3389]


def synthetic_stream(n_flows=10_000, flows_per_sec=1_000, seed=42,
                     pkt_range=(2, 8), tail_timeout_buf=2.0):
    """Deterministic synthetic traffic -> (packet list, flow_first map).

    Each flow starts a few ms apart (``flows_per_sec`` controls the offered
    load); packets inside a flow are spaced by 5-45 ms. The stream ends with a
    single trailing packet ``tail_timeout_buf`` seconds after the last flow so
    a detector with timeout <= tail_timeout_buf flushes every flow.
    """
    rng = np.random.RandomState(seed)
    packets = []
    flow_first = {}
    t = 0.0
    arrival_gap = 1.0 / flows_per_sec if flows_per_sec > 0 else 0.0

    for i in range(n_flows):
        start = t
        sport = 20000 + i
        dport = int(PORTS[rng.randint(0, len(PORTS))])
        src = "10.%d.%d.%d" % (rng.randint(1, 10), rng.randint(0, 254),
                               rng.randint(1, 254))
        dst = "192.168.%d.%d" % (rng.randint(1, 10), rng.randint(1, 254))
        web = rng.randint(0, 4)  # 20% UDP (DNS-ish), 80% TCP
        is_tcp = web != 0
        n_pkts = int(rng.randint(*pkt_range))
        span = 0.0
        key = (src, dst, sport, dport, 6 if is_tcp else 17)
        flow_first[key] = start

        for k in range(n_pkts):
            ts = start + span
            direction = (k % 2 == 0)
            p_src, p_dst = (src, dst) if direction else (dst, src)
            p_sport, p_dport = (sport, dport) if direction else (dport, sport)
            if is_tcp:
                flags = {0: "S", 1: "SA"}.get(k, "A")
                p = Ether() / IP(src=p_src, dst=p_dst) / \
                    TCP(sport=p_sport, dport=p_dport, flags=flags,
                        window=int(rng.randint(29200, 65535))) / \
                    bytes(rng.randint(0, 1460))
            else:
                p = Ether() / IP(src=p_src, dst=p_dst) / \
                    UDP(sport=p_sport, dport=p_dport) / bytes(rng.randint(0, 512))
            p.time = ts
            packets.append(p)
            span += 0.005 + rng.uniform(0.0, 0.04)
        t = start + span + max(arrival_gap, 0.0)

    tail = Ether() / IP(src="10.99.99.99", dst="10.99.99.98") / \
        TCP(sport=9999, dport=80, flags="S")
    tail.time = t + tail_timeout_buf
    packets.append(tail)
    return packets, flow_first


def run_stream(packets, scorer, timeout, explain=False, flow_first=None):
    """Feed a stream through LiveDetector; return (wall_s, alerts, latencies)."""
    det = LiveDetector(scorer, flow_timeout_s=timeout)
    t0 = time.perf_counter()
    alerts = []
    lat = []
    for pkt in packets:
        now = pkt.time
        for a in det.ingest(pkt, ts=now, explain=explain):
            alerts.append(a)
            if flow_first is not None:
                lat.append(now - flow_first.get(a["key"], now))
    wall = time.perf_counter() - t0
    return wall, alerts, lat


def med(x):
    return float(np.median(x)) if len(x) else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flows", type=int, default=2_000)
    ap.add_argument("--fps", type=float, default=1_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeouts", type=float, nargs="+", default=[2.0, 15.0, 60.0])
    ap.add_argument("--explain-samples", type=int, default=30)
    args = ap.parse_args(argv)

    print("Generating %d flows @ %.0f/s ..." % (args.flows, args.fps), flush=True)
    packets, first = synthetic_stream(args.flows, args.fps, args.seed,
                                      tail_timeout_buf=max(args.timeouts) + 0.5)

    summary = {
        "generator": {"n_flows": args.flows, "fps": args.fps, "seed": args.seed},
    }

    # --- score-only throughput + detection latency across timeouts ---------
    scorer_plain = _load_scorer(DETECTOR_KEY, with_explain=False)
    latency_rows = []
    for timeout in args.timeouts:
        wall, alerts, lat = run_stream(packets, scorer_plain, timeout,
                                       flow_first=first)
        tput = len(alerts) / wall if wall else 0.0
        summary.setdefault("throughput", {})[
            "timeout_%.0f_s" % timeout] = round(tput, 1)
        latency_rows.append({"timeout_s": timeout, "median_latency_s":
                             round(med(lat), 3)})
        print("timeout=%4.0fs  flows/s=%8.1f  median latency=%.3fs  (wall %.2fs)"
              % (timeout, tput, med(lat), wall), flush=True)
    summary["latency_by_timeout_s"] = latency_rows

    # --- per-alert explanation cost (real models, off the hot path) ---------
    scorer_full = _load_scorer(DETECTOR_KEY, with_explain=True)
    det = LiveDetector(scorer_full, flow_timeout_s=args.timeouts[0])
    vectors = []
    for pkt in packets:
        for a in det.ingest(pkt, ts=pkt.time):
            if len(vectors) >= args.explain_samples:
                break
            vectors.append(_vec(a))
        if len(vectors) >= args.explain_samples:
            break
    costs = []
    for vec in vectors:
        t0 = time.perf_counter()
        scorer_full.explain(vec)
        costs.append((time.perf_counter() - t0) * 1000)
    summary["explain_cost_ms_per_alert"] = {
        "n": len(costs),
        "mean": round(float(np.mean(costs)), 2),
        "median": round(float(np.median(costs)), 2),
        "max": round(float(np.max(costs)), 2),
    }
    print("explain: mean %.1f ms / median %.1f ms per alert (n=%d)"
          % (np.mean(costs), np.median(costs), len(costs)), flush=True)

    # --- detector comparison (score-only, same stream) ----------------------
    # HGB throughput is already measured at timeouts[0] above; only rerun the
    # Random Forest to keep the comparison on one machine/one load.
    hgb_flows = summary["throughput"].get(
        "timeout_%.0f_s" % args.timeouts[0], None)
    rf_scorer = _load_scorer("Random Forest (all 69)", with_explain=False)
    wall_rf, alerts_rf, _ = run_stream(packets, rf_scorer, args.timeouts[0])
    pps = {"HGB": hgb_flows,
           "RandomForest": round(len(alerts_rf) / wall_rf, 1)}
    summary["detector_throughput_flows_per_s"] = pps
    print("detector flows/s: %s" % ", ".join("%s=%s" % kv for kv in pps.items()),
          flush=True)

    _plot(summary)
    path = os.path.join(METRICS_DIR, "stage11_benchmark.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("Saved ->", path)


def _vec(a):
    from src.live.flow_features import Flow
    return Flow.vector_from_dict(a["features"])


def _plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    lat = summary["latency_by_timeout_s"]
    ts = [r["timeout_s"] for r in lat]
    ls = [r["median_latency_s"] for r in lat]
    ax[0].plot(ts, ls, "o-", color="#c0392b")
    for x, y in zip(ts, ls):
        ax[0].text(x, y + 0.5, "%.1f" % y, ha="center", fontsize=8)
    ax[0].set_xlabel("flow timeout (s)")
    ax[0].set_ylabel("median detection latency (s, simulated)")
    ax[0].set_title("Detection latency vs timeout")
    ax[0].grid(alpha=0.3)

    tp = summary["throughput"]
    names = list(tp.keys())
    vals = list(tp.values())
    ax[1].bar(names, vals, color="#2980b9")
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.5, "%.0f" % v, ha="center", fontsize=8)
    ax[1].set_title("Finalization+scoring throughput")
    ax[1].tick_params(axis="x", rotation=20)
    ax[1].grid(axis="y", alpha=0.3)

    det = {k: v for k, v in summary["detector_throughput_flows_per_s"].items()
           if v is not None}
    ax[2].bar(list(det), list(det.values()), color="#27ae60")
    for i, (k, v) in enumerate(det.items()):
        ax[2].text(i, v + 0.5, "%.0f" % v, ha="center", fontsize=8)
    ax[2].set_title("Throughput by detector")

    fig.suptitle("Stage 11 - live pipeline performance")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "stage11_live_benchmark.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("Plot ->", out)


if __name__ == "__main__":
    main()