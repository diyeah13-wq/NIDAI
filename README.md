# NIDAI — Real-Time Network Intrusion Detection System

NIDAI is a machine-learning-based Network Intrusion Detection System (NIDS) trained on the **CIC-IDS2017** benchmark dataset. It provides **real-time packet capture, bidirectional flow construction, 69-feature extraction, multi-class threat classification, and AI-driven attack attribution**.

---

## Architecture

```text
[Live Network Traffic (Ethernet/Wi-Fi)]
                    │
                    ▼
          [Packet Capture Layer]           (Scapy / Npcap)
                    │
                    ▼
          [Bidirectional Flow Engine]      (5-tuple tracking, TCP teardown)
                    │
                    ▼
          [69-Feature Extraction]          (CIC-IDS2017 feature schema)
                    │
                    ▼
          [Trained AI Detection Engine]    (Balanced HistGradientBoosting)
                    │
                    ▼
          ┌─────────────────────┐
          │  ✔ Normal (BENIGN)  │
          │  🚨 Intrusion Alert │
          └─────────────────────┘
                    │
                    ▼
     ┌─────────────────────────────┐
     ▼                             ▼
[Terminal Live Monitor]     [Interactive SOC Dashboard]
  python live_detection.py    streamlit run dashboard/live.py
```

---

## Features & Attack Classes

NIDAI detects and classifies network traffic into **7 distinct categories**:
- `BENIGN` — Normal legitimate network traffic (SAFE)
- `PortScan` — Host/port reconnaissance probes (MEDIUM)
- `DoS` — Denial-of-Service attacks (Hulk, GoldenEye, Slowloris) (HIGH)
- `BruteForce` — Authentication attacks (FTP/SSH Patator) (HIGH)
- `WebAttack` — Web vulnerabilities (SQL Injection, XSS) (HIGH)
- `Botnet` — Automated command-and-control communication (CRITICAL)
- `DDoS` — Distributed Denial-of-Service saturation floods (CRITICAL)

### The 69 Flow Features across 5 Logical Groups

1. **`A_Flow_Volume` (15 features):** Flow Duration, Packet/Byte counts (Fwd/Bwd), Throughput rates (Flow Bytes/s, Flow Packets/s), Header Lengths, Subflow rates.
2. **`B_Packet_Statistics` (16 features):** Packet length extremes, means, variances, standard deviations (Fwd, Bwd, and Total).
3. **`C_Timing_IAT` (22 features):** Inter-arrival times (Mean, Std, Max, Min across Flow, Fwd, and Bwd), Active/Idle burst timings.
4. **`D_TCP_Flags_Connection` (11 features):** TCP flags (SYN, ACK, FIN, RST, PSH, URG, CWE, ECE), Down/Up ratio.
5. **`E_Endpoint_Window` (5 features):** Destination Port, TCP Initial Window Sizes (Fwd/Bwd), Active Data Packets, Minimum Segment Size.

---

## Quick Start

### 1. Requirements & Prerequisites
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```
*Note for Windows:* Ensure **Npcap** or WinPcap is installed (with "WinPcap API-compatible mode" enabled) to allow raw packet sniffing.

---

### 2. Live Detection CLI

Run real-time traffic detection directly in your terminal:
```bash
python live_detection.py
```
*NIDAI will automatically bind to your active network adapter, sniff packets, build flows, and print predictions.*

#### CLI Options:
```bash
# List available network interfaces
python live_detection.py --list-ifaces

# Bind to a specific interface (e.g. Wi-Fi or Ethernet)
python live_detection.py --iface Wi-Fi

# Attach the 5-technique AI explanation to flagged attacks
python live_detection.py --explain

# Show attack alerts only (suppress benign traffic)
python live_detection.py --quiet-benign

# Log all alerts to a JSON file
python live_detection.py --log-alerts results/alerts.json

# Replay an offline PCAP file
python live_detection.py --pcap results/demo_live.pcap
```

---

### 3. Interactive Cybersecurity SOC Dashboard

Launch the Streamlit monitoring interface:
```bash
streamlit run dashboard/live.py
```

The dashboard provides:
- **Live Stream View:** Real-time KPI tiles (flows seen, threat count, critical alarms).
- **Attack Class Distribution:** Dynamic bar charts of categorized network events.
- **Threat Forensics & AI Explanation:** Deep-dive into any flagged flow to see which of the 5 feature groups triggered the alert and the exact feature deviations (z-scores).
- **Traffic Sources:**
  - `Lab replay`: Streams real held-out CIC-IDS2017 test flows (recommended for project demos).
  - `Live sniff`: Directly captures and classifies live network adapter traffic.
  - `Packet replay`: Streams generated PCAP traces through the live capture engine.

---

### 4. Controlled Testing & Validation in the Lab

#### Test Normal Traffic:
While NIDAI is running, perform typical web activities:
```bash
nslookup google.com
curl https://httpbin.org/get
```
*Result:* NIDAI evaluates the flows and outputs `[OK] BENIGN` (>99% confidence).

#### Test Port Scanning:
Run the provided non-destructive local probe generator:
```bash
python scripts/simulate_portscan.py 127.0.0.1
```

---

## Machine Learning Models

| Model | File | Macro F1 | WebAttack Recall | Botnet Recall | Inference Speed | Role |
|---|---|---|---|---|---|---|
| **Balanced HGB** *(Recommended)* | `models/rare_hgbt.joblib` | **0.891** | **91.1%** | **99.0%** | **~28 flows/s** | **Primary Detection Engine** |
| Random Forest (All 69) | `models/random_forest.joblib` | 0.887 | 50.4% | 96.4% | ~10 flows/s | Baseline / Fallback Detector |
| Group Models (5 Forests) | `models/fusion/group_models.joblib` | — | — | — | — | **5-Technique Explanation Layer** |
