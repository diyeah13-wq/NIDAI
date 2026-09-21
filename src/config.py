"""Central configuration: paths, feature groups, and label mapping."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_SAMPLE_SIZE = 300_000
MODEL_RARE_CLASSES = ["WebAttack", "Botnet", "BruteForce"]

PROCESSED_CLEAN = os.path.join(DATA_PROCESSED, "cic_ids2017_cleaned.csv.gz")
PROCESSED_TRAIN = os.path.join(DATA_PROCESSED, "train.csv.gz")
PROCESSED_TEST = os.path.join(DATA_PROCESSED, "test.csv.gz")
PROCESSED_DEMO = os.path.join(DATA_PROCESSED, "demo_sample_40k.csv")

LEAKAGE_COLUMNS = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]

CONSTANT_COLUMNS = [
    "Bwd PSH Flags", "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]

DUPLICATE_COLUMNS = ["Fwd Header Length"]

DROPPED_COLUMNS = LEAKAGE_COLUMNS + CONSTANT_COLUMNS + DUPLICATE_COLUMNS

FEATURE_GROUPS = {
    "A_Flow_Volume": {
        "features": [
            "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets",
            "Flow Bytes/s", "Flow Packets/s",
            "Fwd Header Length", "Bwd Header Length",
            "Fwd Packets/s", "Bwd Packets/s",
            "Subflow Fwd Packets", "Subflow Fwd Bytes",
            "Subflow Bwd Packets", "Subflow Bwd Bytes",
        ],
        "description": "Flow size and volume: how much traffic flows and how fast.",
    },
    "B_Packet_Statistics": {
        "features": [
            "Fwd Packet Length Max", "Fwd Packet Length Min",
            "Fwd Packet Length Mean", "Fwd Packet Length Std",
            "Bwd Packet Length Max", "Bwd Packet Length Min",
            "Bwd Packet Length Mean", "Bwd Packet Length Std",
            "Min Packet Length", "Max Packet Length",
            "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
            "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
        ],
        "description": "Packet-size statistics: shape and size of packets in both directions.",
    },
    "C_Timing_IAT": {
        "features": [
            "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
            "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
            "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
            "Active Mean", "Active Std", "Active Max", "Active Min",
            "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
        ],
        "description": "Timing / inter-arrival-time behaviour: rhythm and pauses of the flow.",
    },
    "D_TCP_Flags_Connection": {
        "features": [
            "Fwd PSH Flags", "Fwd URG Flags",
            "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
            "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
            "CWE Flag Count", "ECE Flag Count",
            "Down/Up Ratio",
        ],
        "description": "TCP flags and direction ratio: connection control behaviour.",
    },
    "E_Endpoint_Window": {
        "features": [
            "Destination Port",
            "Init_Win_bytes_forward", "Init_Win_bytes_backward",
            "act_data_pkt_fwd", "min_seg_size_forward",
        ],
        "description": "Endpoint + TCP window parameters: port dialled and window sizing.",
    },
}

FEATURE_ORDER = [f for g in [
    "A_Flow_Volume", "B_Packet_Statistics", "C_Timing_IAT",
    "D_TCP_Flags_Connection", "E_Endpoint_Window",
] for f in FEATURE_GROUPS[g]["features"]]

CLASS_ORDER = ["BENIGN", "DoS", "DDoS", "PortScan", "BruteForce", "Botnet", "WebAttack"]

LABEL_GROUP_MAP = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS slowloris": "DoS",
    "Heartbleed": "DoS",
    "PortScan": "PortScan",
    "Infiltration": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "BruteForce",
    "Bot": "Botnet",
    "Web Attack - SQL Injection": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Web Attack - XSS": "WebAttack",
}

LABEL_DISPLAY = {
    "BENIGN": "Normal (BENIGN)",
    "DoS": "DoS",
    "DDoS": "DDoS",
    "PortScan": "Port Scan",
    "BruteForce": "Brute Force",
    "Botnet": "Botnet",
    "WebAttack": "Web Attack",
}

SEVERITY = {
    "BENIGN": "SAFE",
    "DoS": "HIGH",
    "DDoS": "CRITICAL",
    "PortScan": "MEDIUM",
    "BruteForce": "HIGH",
    "Botnet": "CRITICAL",
    "WebAttack": "HIGH",
}

for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, METRICS_DIR,
          PLOTS_DIR, PREDICTIONS_DIR):
    os.makedirs(d, exist_ok=True)