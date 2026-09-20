"""Feature grouping: maps every feature to a logical group with a reason."""
import os
import pandas as pd

from config import (FEATURE_GROUPS, FEATURE_ORDER, CONSTANT_COLUMNS,
                    DUPLICATE_COLUMNS, LEAKAGE_COLUMNS, RESULTS_DIR)

GROUP_REASONS = {
    "A_Flow_Volume": (
        "Saturation attacks (DoS/DDoS) and scans flood flows with packets and bytes, "
        "so volume/rate features are their primary signature."),
    "B_Packet_Statistics": (
        "Packet-size distributions differ sharply between bulk flood traffic, "
        "single-packet probes and normal browsing."),
    "C_Timing_IAT": (
        "Attack tools send packets at machine pace and drop human interaction "
        "latency; IAT rhythm separates automated behaviour from real users."),
    "D_TCP_Flags_Connection": (
        "Handshake/flag patterns detect SYN floods, port-scans and half-open "
        "connection attempts; Down/Up ratio reveals asymmetrical traffic."),
    "E_Endpoint_Window": (
        "Destination port identifies scanned services; TCP window scalings and "
        "segment sizes betray bot/C2 traffic."),
}

REASON_BY_FEATURE = {
    "Destination Port": ("connection", "PortScan opens many disjoint destination ports."),
    "Flow Duration": ("volume", "Duration of the flow; floods maintain long flows."),
    "Total Fwd Packets": ("volume", "Volume of packets sent."),
    "Total Backward Packets": ("volume", "Volume of packets received."),
    "Total Length of Fwd Packets": ("volume", "Volume of bytes sent."),
    "Total Length of Bwd Packets": ("volume", "Volume of bytes received."),
    "Flow Bytes/s": ("volume", "Throughput; DDoS generates extreme values."),
    "Flow Packets/s": ("volume", "Packet rate; flood tools saturate this."),
    "Fwd Header Length": ("volume", "Forward header overhead per flow."),
    "Bwd Header Length": ("volume", "Backward header overhead per flow."),
    "Fwd Packets/s": ("volume", "Forward packet rate."),
    "Bwd Packets/s": ("volume", "Backward packet rate."),
    "Subflow Fwd Packets": ("volume", "Forward subflow packet volume."),
    "Subflow Fwd Bytes": ("volume", "Forward subflow byte volume."),
    "Subflow Bwd Packets": ("volume", "Backward subflow packet volume."),
    "Subflow Bwd Bytes": ("volume", "Backward subflow byte volume."),
    "Fwd Packet Length Max": ("packet", "Forward packet-size extremes."),
    "Fwd Packet Length Min": ("packet", "Forward packet-size extremes."),
    "Fwd Packet Length Mean": ("packet", "Forward average packet size."),
    "Fwd Packet Length Std": ("packet", "Forward size variability."),
    "Bwd Packet Length Max": ("packet", "Backward packet-size extremes."),
    "Bwd Packet Length Min": ("packet", "Backward packet-size extremes."),
    "Bwd Packet Length Mean": ("packet", "Backward average packet size."),
    "Bwd Packet Length Std": ("packet", "Backward size variability."),
    "Min Packet Length": ("packet", "Smallest observed packet."),
    "Max Packet Length": ("packet", "Largest observed packet."),
    "Packet Length Mean": ("packet", "Overall average packet size."),
    "Packet Length Std": ("packet", "Overall size variability."),
    "Packet Length Variance": ("packet", "Overall size variability."),
    "Average Packet Size": ("packet", "Mean packet size of the flow."),
    "Avg Fwd Segment Size": ("packet", "Mean forward segment size."),
    "Avg Bwd Segment Size": ("packet", "Mean backward segment size."),
    "Flow IAT Mean": ("timing", "Mean inter-arrival time between packets."),
    "Flow IAT Std": ("timing", "Timing variability."),
    "Flow IAT Max": ("timing", "Largest packet gap."),
    "Flow IAT Min": ("timing", "Smallest packet gap."),
    "Fwd IAT Total": ("timing", "Total forward inter-arrival time."),
    "Fwd IAT Mean": ("timing", "Forward mean packet gap."),
    "Fwd IAT Std": ("timing", "Forward timing variability."),
    "Fwd IAT Max": ("timing", "Forward largest gap."),
    "Fwd IAT Min": ("timing", "Forward smallest gap."),
    "Bwd IAT Total": ("timing", "Total backward inter-arrival time."),
    "Bwd IAT Mean": ("timing", "Backward mean packet gap."),
    "Bwd IAT Std": ("timing", "Backward timing variability."),
    "Bwd IAT Max": ("timing", "Backward largest gap."),
    "Bwd IAT Min": ("timing", "Backward smallest gap."),
    "Active Mean": ("timing", "Mean duration of active bursts."),
    "Active Std": ("timing", "Active-burst variability."),
    "Active Max": ("timing", "Longest active burst."),
    "Active Min": ("timing", "Shortest active burst."),
    "Idle Mean": ("timing", "Mean idle period between bursts."),
    "Idle Std": ("timing", "Idle-period variability."),
    "Idle Max": ("timing", "Longest idle period."),
    "Idle Min": ("timing", "Shortest idle period."),
    "Fwd PSH Flags": ("flags", "Push-flag behaviour (PUSH priority)."),
    "Fwd URG Flags": ("flags", "Urgent-flag behaviour."),
    "FIN Flag Count": ("flags", "Session terminations; scans send FIN probes."),
    "SYN Flag Count": ("flags", "Connection initiations; SYN floods / scans."),
    "RST Flag Count": ("flags", "Resets; dropped connections in DoS."),
    "PSH Flag Count": ("flags", "Push flags in DoS floods."),
    "ACK Flag Count": ("flags", "Acknowledgement traffic; scan backscatter."),
    "URG Flag Count": ("flags", "Urgent flags are rare in normal traffic."),
    "CWE Flag Count": ("flags", "Congestion-window-reduced flags."),
    "ECE Flag Count": ("flags", "Explicit-congestion-notification flags."),
    "Down/Up Ratio": ("flags", "Asymmetry; botnet/C2 and floods are one-directional."),
    "Init_Win_bytes_forward": ("endpoint", "Advertised forward TCP window size."),
    "Init_Win_bytes_backward": ("endpoint", "Advertised backward TCP window size."),
    "act_data_pkt_fwd": ("endpoint", "Active data packets sent before push."),
    "min_seg_size_forward": ("endpoint", "Minimum forward segment size."),
}

DROP_REASONS = {
    "Flow ID": "Row identifier; presence of the row leaks the answer (and is unavailable in live traffic).",
    "Source IP": "Host identity, not behaviour; would leak the host that was attacked.",
    "Destination IP": "Host identity, not behaviour; would leak the target of the attack.",
    "Timestamp": "Capture timestamp; not available in live streaming detection.",
    "Source Port": "Source ephemeral port; behaviour learning is not improved.",
    "Protocol": "Protocol constant dominated by TCP in this capture; absorbed into group features.",
    "Bwd PSH Flags": "Constant column (zero variance) - no information.",
    "Bwd URG Flags": "Constant column (zero variance) - no information.",
    "Fwd Avg Bytes/Bulk": "Constant 0 column (bulk transfers not present).",
    "Fwd Avg Packets/Bulk": "Constant 0 column.",
    "Fwd Avg Bulk Rate": "Constant 0 column.",
    "Bwd Avg Bytes/Bulk": "Constant 0 column.",
    "Bwd Avg Packets/Bulk": "Constant 0 column.",
    "Bwd Avg Bulk Rate": "Constant 0 column.",
    "Fwd Header Length": "Exact duplicate of the first 'Fwd Header Length' column (100% identical values).",
}


def build_group_documentation():
    rows = []
    for group, meta in FEATURE_GROUPS.items():
        for f in meta["features"]:
            tag, detail = REASON_BY_FEATURE.get(
                f, ("-", "Included in the %s group." % group))
            rows.append({
                "Feature": f,
                "Feature Group": group,
                "Group description": meta["description"],
                "Why selected": detail,
                "Group reason": GROUP_REASONS[group],
            })
    for c in DROP_REASONS:
        rows.append({
            "Feature": c,
            "Feature Group": "DROPPED",
            "Group description": "",
            "Why selected": "",
            "Group reason": DROP_REASONS[c],
        })
    return pd.DataFrame(rows)


def get_features(group=None):
    if group is None:
        return list(FEATURE_ORDER)
    return list(FEATURE_GROUPS[group]["features"])


def get_group_keys():
    return list(FEATURE_GROUPS.keys())


def get_group_meta():
    return FEATURE_GROUPS


if __name__ == "__main__":
    doc = build_group_documentation()
    out = os.path.join(RESULTS_DIR, "feature_groups.csv")
    doc.to_csv(out, index=False)
    print("Features used in pipeline: %d" % len(FEATURE_ORDER))
    print("Features dropped: %d" % len(DROP_REASONS))
    total = len(FEATURE_ORDER) + len(DROP_REASONS)
    print("Detailed documentation written to %s" % out)
    for g, meta in FEATURE_GROUPS.items():
        print("  %-24s %d features  -  %s" % (g, len(meta["features"]), meta["description"]))