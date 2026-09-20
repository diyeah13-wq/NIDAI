# Feature groups (CIC-IDS2017)

The 69 selected features are organised into 5 logical groups. Each group captures a different *type* of information about a network flow, which lets the multi-technique system study attack behaviour from complementary angles.

## A_Flow_Volume (15 features)

_Group description:_ Flow size and volume: how much traffic flows and how fast.

_Why this group helps:_ Saturation attacks (DoS/DDoS) and scans flood flows with packets and bytes, so volume/rate features are their primary signature.

| Feature | Why selected |
|---|---|
| `Flow Duration` | Duration of the flow; floods maintain long flows. |
| `Total Fwd Packets` | Volume of packets sent. |
| `Total Backward Packets` | Volume of packets received. |
| `Total Length of Fwd Packets` | Volume of bytes sent. |
| `Total Length of Bwd Packets` | Volume of bytes received. |
| `Flow Bytes/s` | Throughput; DDoS generates extreme values. |
| `Flow Packets/s` | Packet rate; flood tools saturate this. |
| `Fwd Header Length` | Forward header overhead per flow. |
| `Bwd Header Length` | Backward header overhead per flow. |
| `Fwd Packets/s` | Forward packet rate. |
| `Bwd Packets/s` | Backward packet rate. |
| `Subflow Fwd Packets` | Forward subflow packet volume. |
| `Subflow Fwd Bytes` | Forward subflow byte volume. |
| `Subflow Bwd Packets` | Backward subflow packet volume. |
| `Subflow Bwd Bytes` | Backward subflow byte volume. |

## B_Packet_Statistics (16 features)

_Group description:_ Packet-size statistics: shape and size of packets in both directions.

_Why this group helps:_ Packet-size distributions differ sharply between bulk flood traffic, single-packet probes and normal browsing.

| Feature | Why selected |
|---|---|
| `Fwd Packet Length Max` | Forward packet-size extremes. |
| `Fwd Packet Length Min` | Forward packet-size extremes. |
| `Fwd Packet Length Mean` | Forward average packet size. |
| `Fwd Packet Length Std` | Forward size variability. |
| `Bwd Packet Length Max` | Backward packet-size extremes. |
| `Bwd Packet Length Min` | Backward packet-size extremes. |
| `Bwd Packet Length Mean` | Backward average packet size. |
| `Bwd Packet Length Std` | Backward size variability. |
| `Min Packet Length` | Smallest observed packet. |
| `Max Packet Length` | Largest observed packet. |
| `Packet Length Mean` | Overall average packet size. |
| `Packet Length Std` | Overall size variability. |
| `Packet Length Variance` | Overall size variability. |
| `Average Packet Size` | Mean packet size of the flow. |
| `Avg Fwd Segment Size` | Mean forward segment size. |
| `Avg Bwd Segment Size` | Mean backward segment size. |

## C_Timing_IAT (22 features)

_Group description:_ Timing / inter-arrival-time behaviour: rhythm and pauses of the flow.

_Why this group helps:_ Attack tools send packets at machine pace and drop human interaction latency; IAT rhythm separates automated behaviour from real users.

| Feature | Why selected |
|---|---|
| `Flow IAT Mean` | Mean inter-arrival time between packets. |
| `Flow IAT Std` | Timing variability. |
| `Flow IAT Max` | Largest packet gap. |
| `Flow IAT Min` | Smallest packet gap. |
| `Fwd IAT Total` | Total forward inter-arrival time. |
| `Fwd IAT Mean` | Forward mean packet gap. |
| `Fwd IAT Std` | Forward timing variability. |
| `Fwd IAT Max` | Forward largest gap. |
| `Fwd IAT Min` | Forward smallest gap. |
| `Bwd IAT Total` | Total backward inter-arrival time. |
| `Bwd IAT Mean` | Backward mean packet gap. |
| `Bwd IAT Std` | Backward timing variability. |
| `Bwd IAT Max` | Backward largest gap. |
| `Bwd IAT Min` | Backward smallest gap. |
| `Active Mean` | Mean duration of active bursts. |
| `Active Std` | Active-burst variability. |
| `Active Max` | Longest active burst. |
| `Active Min` | Shortest active burst. |
| `Idle Mean` | Mean idle period between bursts. |
| `Idle Std` | Idle-period variability. |
| `Idle Max` | Longest idle period. |
| `Idle Min` | Shortest idle period. |

## D_TCP_Flags_Connection (11 features)

_Group description:_ TCP flags and direction ratio: connection control behaviour.

_Why this group helps:_ Handshake/flag patterns detect SYN floods, port-scans and half-open connection attempts; Down/Up ratio reveals asymmetrical traffic.

| Feature | Why selected |
|---|---|
| `Fwd PSH Flags` | Push-flag behaviour (PUSH priority). |
| `Fwd URG Flags` | Urgent-flag behaviour. |
| `FIN Flag Count` | Session terminations; scans send FIN probes. |
| `SYN Flag Count` | Connection initiations; SYN floods / scans. |
| `RST Flag Count` | Resets; dropped connections in DoS. |
| `PSH Flag Count` | Push flags in DoS floods. |
| `ACK Flag Count` | Acknowledgement traffic; scan backscatter. |
| `URG Flag Count` | Urgent flags are rare in normal traffic. |
| `CWE Flag Count` | Congestion-window-reduced flags. |
| `ECE Flag Count` | Explicit-congestion-notification flags. |
| `Down/Up Ratio` | Asymmetry; botnet/C2 and floods are one-directional. |

## E_Endpoint_Window (5 features)

_Group description:_ Endpoint + TCP window parameters: port dialled and window sizing.

_Why this group helps:_ Destination port identifies scanned services; TCP window scalings and segment sizes betray bot/C2 traffic.

| Feature | Why selected |
|---|---|
| `Destination Port` | PortScan opens many disjoint destination ports. |
| `Init_Win_bytes_forward` | Advertised forward TCP window size. |
| `Init_Win_bytes_backward` | Advertised backward TCP window size. |
| `act_data_pkt_fwd` | Active data packets sent before push. |
| `min_seg_size_forward` | Minimum forward segment size. |

## Dropped columns (15)

| Feature | Reason for dropping |
|---|---|
| `Flow ID` | Row identifier; presence of the row leaks the answer (and is unavailable in live traffic). |
| `Source IP` | Host identity, not behaviour; would leak the host that was attacked. |
| `Destination IP` | Host identity, not behaviour; would leak the target of the attack. |
| `Timestamp` | Capture timestamp; not available in live streaming detection. |
| `Source Port` | Source ephemeral port; behaviour learning is not improved. |
| `Protocol` | Protocol constant dominated by TCP in this capture; absorbed into group features. |
| `Bwd PSH Flags` | Constant column (zero variance) - no information. |
| `Bwd URG Flags` | Constant column (zero variance) - no information. |
| `Fwd Avg Bytes/Bulk` | Constant 0 column (bulk transfers not present). |
| `Fwd Avg Packets/Bulk` | Constant 0 column. |
| `Fwd Avg Bulk Rate` | Constant 0 column. |
| `Bwd Avg Bytes/Bulk` | Constant 0 column. |
| `Bwd Avg Packets/Bulk` | Constant 0 column. |
| `Bwd Avg Bulk Rate` | Constant 0 column. |
| `Fwd Header Length` | Exact duplicate of the first 'Fwd Header Length' column (100% identical values). |
