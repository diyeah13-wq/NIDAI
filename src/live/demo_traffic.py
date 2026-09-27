"""Demo traffic for the live dashboard - rebuilt from real CIC-IDS2017 rows.

The dashboard cannot wait for real attack traffic to happen on a laptop, so
``write_demo_pcap`` turns a stratified sample of REAL labelled CIC test rows
into approximate packet streams - one flow per row, sizes/ports/windows taken
from the row's own feature values. Replayed through the live detector this
shows, in one screen, what the system does with normal web traffic AND with
real attack flows (WebAttack, Botnet, DDoS, ...) it has never seen as packets.

Fidelity is deliberately coarse: a handful of handshake/data packets per flow
with the row's aggregate stats (packet sizes, byte totals, ports, windows,
rough timing). Exotic micro-features (exact IAT std, active/idle burst shapes)
are impossible to invert from aggregates; the visible signals - volume, sizes,
ports, windows, connection-control counts - stay close to the originals.
"""
import os
import sys

import numpy as np
import pandas as pd

from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import Ether
from scapy.utils import wrpcap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.config import (FEATURE_ORDER, CLASS_ORDER, PROCESSED_TEST,
                        METRICS_DIR)
from src.live.capture import packet_to_event
from src.live.flow_features import FEATURE_META, Flow, FlowTable

# caps: keep the stream small (constant replay time) and the sizes sane
MAX_PKTS_PER_DIR = 12
MAX_DURATION_S = 3.0
MIN_DURATION_S = 0.001
MAX_PAYLOAD = 1400  # keep IP total length well under the 16-bit field cap


def _flag_pattern(row, n_total):
    """Flag sequence reproducing the row's connection-control counts.

    The handshake/teardown skeleton in ``row_to_flow`` gives every flow the
    same SYN/ACK/FIN shape; real attacks distinguish themselves here (many
    RSTs, floods of bare SYNs, PSH-heavy transfers). So we emit a multiset of
    flags whose counts match the row's, then interleave them across the flow.
    """
    counts = {
        "S": int(_f("SYN Flag Count", row)),
        "R": int(_f("RST Flag Count", row)),
        "F": int(_f("FIN Flag Count", row)),
        "P": int(_f("PSH Flag Count", row)),
        "U": int(_f("URG Flag Count", row)),
        "A": max(int(_f("ACK Flag Count", row)), 1),
    }
    bag = (["R"] * counts["R"] + ["F"] * counts["F"] + ["U"] * counts["U"] +
           ["P"] * counts["P"] + ["S"] * counts["S"] + ["A"] * counts["A"])
    if not bag:
        return ["A"] * n_total
    if len(bag) < n_total:  # pad with plain ACKs -> keep the count columns right
        bag = bag + ["A"] * (n_total - len(bag))
    rng = np.random.RandomState(int(_f("SYN Flag Count", row) * 131 +
                                     _f("ACK Flag Count", row)))
    order = rng.permutation(len(bag))
    return [bag[i] for i in order[:n_total]]


def _f(name, row):
    return float(row[FEATURE_ORDER.index(name)])


def row_to_flow(row, t0=0.0, nonce=0):
    """Build (packet_list, dur_s) for one CIC row.

    Packets are TCP (all demo rows use a handshake + data exchange profile).
    ``t0`` places the flow on a global clock (so replaying the whole demo lets
    flows time out); ``nonce`` makes the client port unique per row so distinct
    rows never collapse onto one flow key.
    """
    port = int(_f("Destination Port", row))
    if port <= 0:
        return [], 0.0
    n_fwd = int(min(_f("Total Fwd Packets", row), MAX_PKTS_PER_DIR))
    n_bwd = int(min(_f("Total Backward Packets", row), MAX_PKTS_PER_DIR))
    if n_fwd + n_bwd < 2:
        return [], 0.0
    dur_s = float(np.clip(_f("Flow Duration", row) / 1e6,
                          MIN_DURATION_S, MAX_DURATION_S))

    def _sizes(n, mean_key, max_key, min_key):
        mean = _f(mean_key, row)
        mx = _f(max_key, row)
        mn = _f(min_key, row)
        if n <= 0 or mean <= 0:
            return [0] * n
        base = int(np.clip(mean, 20, MAX_PAYLOAD + 20)) - 20
        payloads = [base] * n
        if n >= 2:  # keep one evident min/max sized packet
            payloads[-1] = int(np.clip(min(mn, MAX_PAYLOAD + 20), 0,
                                       MAX_PAYLOAD))
        return payloads

    fwd_lens = _sizes(n_fwd, "Fwd Packet Length Mean", "Fwd Packet Length Max",
                      "Fwd Packet Length Min")
    bwd_lens = _sizes(n_bwd, "Bwd Packet Length Mean", "Bwd Packet Length Max",
                      "Bwd Packet Length Min")
    win_f = int(np.clip(_f("Init_Win_bytes_forward", row) or 20000,
                        0, 65535))
    win_b = int(np.clip(_f("Init_Win_bytes_backward", row) or 20000,
                        0, 65535))
    flags = _flag_pattern(row, n_fwd + n_bwd)
    src = "10.%d.0.%d" % (1 + int(port % 5), 20 + int(port % 150))
    dst = "192.168.1.%d" % (2 + int(port % 240))
    sport = 49152 + (nonce * 131 + int(port * 997)) % 10000

    t = t0 = float(t0)
    pkts = []
    gap = dur_s / max(max(n_fwd + n_bwd, 1), 1)

    def emit(a_len, direction, flags, win, t0):
        payload = bytes(a_len)
        if direction:
            p = Ether() / IP(src=src, dst=dst) / \
                TCP(sport=sport, dport=port, flags=flags, window=win) / payload
        else:
            p = Ether() / IP(src=dst, dst=src) / \
                TCP(sport=port, dport=sport, flags=flags, window=win) / payload
        p.time = t0
        pkts.append(p)

    n_total = n_fwd + n_bwd
    emit(fwd_lens[0] if fwd_lens else 0, True, flags[0], win_f, t)
    pos = 1
    for k in range(1, min(n_fwd, n_total)):
        emit(fwd_lens[k] if k < len(fwd_lens) else 0, True,
             flags[pos], win_f, t + gap * pos)
        pos += 1
    for k in range(n_bwd):
        fl = bwd_lens[k] if k < len(bwd_lens) else 0
        emit(fl, False, flags[pos], win_b, t + gap * pos)
        pos += 1
    if n_fwd + n_bwd >= 3:
        final = "FA" if "F" not in flags else "A"
        emit(0, True, final, win_f, t0 + dur_s)
    return pkts, dur_s


def sample_test_rows(n_total=220):
    """Stratified sample: ~half benign, ~half attacks across the classes."""
    df = pd.read_csv(PROCESSED_TEST, usecols=FEATURE_ORDER + ["LabelGroup"],
                     dtype={c: "float32" for c in FEATURE_ORDER})
    df = df[df["LabelGroup"] != "UNKNOWN"].reset_index(drop=True)
    rng = np.random.RandomState(42)
    take = {}
    per_attack = max(1, n_total // (len(CLASS_ORDER) * 2))
    for i, c in enumerate(CLASS_ORDER):
        take[i] = n_total // 2 if i == 0 else per_attack
    rows = []
    for code, n in take.items():
        sub = df[df["LabelGroup"] == CLASS_ORDER[code]]
        if not len(sub):
            continue
        idx = rng.choice(len(sub), size=min(n, len(sub)), replace=False)
        rows.append(sub.iloc[idx])
    out = pd.concat(rows, ignore_index=True)
    return out[FEATURE_ORDER].to_numpy(np.float32), \
        out["LabelGroup"].to_numpy()


def write_demo_pcap(path, n_total=220, finalize_gap_s=2.05):
    """Build the demo pcap from sample CIC test rows. -> (flows, attacks).

    ``finalize_gap_s`` is the simulated silence between consecutive flows
    (>= the detector's flow timeout) so replaying the stream keeps every flow
    finishing promptly - the dashboard shows one alert per flow as it arrives.
    """
    X, labels = sample_test_rows(n_total)
    pkts = []
    flow_info = []
    clock = 0.0
    for i, row in enumerate(X):
        ps, dur = row_to_flow(row, t0=clock, nonce=i)
        if not ps:
            clock += dur + finalize_gap_s
            continue
        pkts.extend(ps)
        flow_info.append((labels[i], len(ps)))
        clock += dur + finalize_gap_s
    tail = Ether() / IP(src="10.99.99.9", dst="10.99.99.8") / \
        TCP(sport=6666, dport=80, flags="S")
    tail.time = clock + finalize_gap_s  # flush the very last flow too
    pkts.append(tail)
    wrpcap(path, pkts)
    attacks = sum(1 for lab, _ in flow_info if lab != "BENIGN")
    return {"flows": len(flow_info), "packets": len(pkts), "attacks": attacks,
            "classes": {c: sum(1 for lab, _ in flow_info if lab == c)
                        for c in CLASS_ORDER}}


def rebuild_vector(row, nonce, flow_timeout_s=1.5):
    """One row -> generated packets -> the rebuilt 69-feature vector.

    The whole round trip the dashboard's packet source runs: packetizer ->
    capture parser -> FlowTable -> model input. Returns ``(vector, n_packets)``,
    or ``None`` when the row cannot be expressed as packets at all (no usable
    port, or fewer than two packets).
    """
    pkts, _ = row_to_flow(row, t0=0.0, nonce=nonce)
    if not pkts:
        return None
    table = FlowTable(flow_timeout_s)
    last = 0.0
    for p in pkts:
        ts = float(getattr(p, "time", 0.0) or 0.0)
        last = max(last, ts)
        ev = packet_to_event(bytes(p), ts)
        if ev is not None:
            table.add(ev)
    done = table.flush_expired(last + flow_timeout_s + 0.1)
    if not done:
        return None
    return Flow.vector_from_dict(done[0]), len(pkts)


def roundtrip_fidelity(n_total=120, n_ref=50_000, flow_timeout_s=1.5):
    """How much of a row's identity survives the packet round trip?

    Scores every sampled labelled row twice with the real Stage 7 detector -
    once as the original 69-feature vector, once as the vector rebuilt from
    generated packets - and reports verdict agreement, per-class survival, and
    per-feature error in SD units of the test distribution. This is the number
    that explains why the packet source is quiet: the packetizer is coarse, and
    what it drops is exactly the signal the detector leans on.
    """
    import json

    from src.live.detector import DETECTOR_KEY, _load_scorer

    X, labels = sample_test_rows(n_total)
    ref = pd.read_csv(PROCESSED_TEST, usecols=FEATURE_ORDER, nrows=n_ref,
                      dtype={c: "float32" for c in FEATURE_ORDER}).to_numpy(np.float32)
    sd = np.maximum(ref.std(axis=0), 1e-6)

    scorer = _load_scorer(DETECTOR_KEY, with_explain=False)
    classes = scorer.detector.classes_

    def _verdict(mat):
        return [CLASS_ORDER[int(classes[int(c)])] for c in mat.argmax(axis=1)]

    orig = _verdict(scorer.predict_proba(X))
    rebuilt_vecs, kept_idx, emitted = [], [], []
    for i, row in enumerate(X):
        got = rebuild_vector(row, i, flow_timeout_s)
        if got is not None:
            vec, n_pkts = got
            rebuilt_vecs.append(vec)
            kept_idx.append(i)
            emitted.append(n_pkts)
    if not rebuilt_vecs:
        raise RuntimeError(
            "no sampled row could be expressed as packets (%d rows sampled) - "
            "check Destination Port coverage in the test split" % len(X))
    rebuilt = _verdict(scorer.predict_proba(np.vstack(rebuilt_vecs)))
    labs = [str(labels[i]) for i in kept_idx]

    zerr = np.abs(np.vstack(rebuilt_vecs) - X[kept_idx]) / sd
    per_feature = {f: round(float(np.percentile(zerr[:, j], 90)), 2)
                   for j, f in enumerate(FEATURE_ORDER)}
    per_feature_med = {f: round(float(np.median(zerr[:, j])), 2)
                       for j, f in enumerate(FEATURE_ORDER)}
    tiers = {}
    for f, e in per_feature.items():
        tier = tiers.setdefault(FEATURE_META[f], {"median": [], "p90": []})
        tier["median"].append(per_feature_med[f])
        tier["p90"].append(e)
    tier_err = {t: {"median": round(float(np.median(v["median"])), 2),
                    "p90": round(float(np.median(v["p90"])), 2)}
                for t, v in tiers.items()}

    row_pkts = (X[kept_idx, FEATURE_ORDER.index("Total Fwd Packets")] +
                X[kept_idx, FEATURE_ORDER.index("Total Backward Packets")])

    per_class = {}
    for lab in CLASS_ORDER:
        idx = [k for k, l in enumerate(labs) if l == lab]
        if not idx:
            continue
        per_class[lab] = {
            "n": len(idx),
            "detector_correct_on_row": sum(1 for k in idx if orig[kept_idx[k]] == lab),
            "verdict_kept": sum(1 for k in idx
                                if rebuilt[k] == orig[kept_idx[k]]),
            "rebuilt_says_benign": sum(1 for k in idx if rebuilt[k] == "BENIGN"),
        }

    keep = sum(1 for k in range(len(labs)) if rebuilt[k] == orig[kept_idx[k]])
    out = {
        "sampled": int(len(X)),
        "rebuilt": len(labs),
        "agreement": round(keep / len(labs), 3),
        "rebuilt_benign_share": round(
            sum(1 for c in rebuilt if c == "BENIGN") / len(rebuilt), 3),
        "packets_per_flow": {
            "row_median": int(np.median(row_pkts)),
            "emitted_median": int(np.median(emitted)),
            "row_p90": int(np.percentile(row_pkts, 90)),
            "emitted_p90": int(np.percentile(emitted, 90)),
        },
        "tier_z_error": tier_err,
        "worst_features_p90": sorted(per_feature.items(),
                                      key=lambda t: -t[1])[:10],
        "per_class": per_class,
    }
    path = os.path.join(METRICS_DIR, "stage12_roundtrip_fidelity.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    out["path"] = path
    return out


def _print_fidelity(info):
    print("rows sampled        %d   packetizable %d" % (info["sampled"],
                                                        info["rebuilt"]))
    print("verdict agreement   %.1f%%   rebuilt-as-BENIGN %.1f%%" % (
        100 * info["agreement"], 100 * info["rebuilt_benign_share"]))
    pk = info["packets_per_flow"]
    print("packets per flow    median %d -> %d    p90 %d -> %d" % (
        pk["row_median"], pk["emitted_median"],
        pk["row_p90"], pk["emitted_p90"]))
    print("feature error in SD of the test distribution (median / p90):")
    for t, e in sorted(info["tier_z_error"].items(),
                       key=lambda x: -x[1]["p90"]):
        print("  %-6s median %5.2f   p90 %6.2f" % (t, e["median"], e["p90"]))
    print("worst features (p90): " + ", ".join("%s %.1f" % (f, e) for f, e
                                               in info["worst_features_p90"][:5]))
    print("per class:")
    print("  %-12s %4s %10s %8s %10s" % ("class", "n", "row-ok", "kept", "->BENIGN"))
    for c, d in info["per_class"].items():
        print("  %-12s %4d %10d %8d %10d" % (c, d["n"],
                                             d["detector_correct_on_row"],
                                             d["verdict_kept"],
                                             d["rebuilt_says_benign"]))
    print("saved %s" % info.get("path", ""))


def write_demo_pcap_cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Build the live-dashboard demo pcap.")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results", "demo_live.pcap"))
    ap.add_argument("--flows", type=int, default=220)
    ap.add_argument("--check", action="store_true",
                    help="measure the aggregate->packet->feature round trip "
                         "instead of writing a pcap")
    args = ap.parse_args(argv)
    if args.check:
        info = roundtrip_fidelity(args.flows)
        _print_fidelity(info)
        return info
    info = write_demo_pcap(args.out, args.flows)
    print("wrote", args.out)
    for k, v in info.items():
        print("  %-9s %s" % (k, v))
    return info


if __name__ == "__main__":
    write_demo_pcap_cli()