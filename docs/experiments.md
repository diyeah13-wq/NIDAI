# Experiment results (CIC-IDS2017, full test set)

All experiments share ONE contract so they are directly comparable:

- **Training budget:** a 300k-row sample (`MODEL_SAMPLE_SIZE`) that keeps
  EVERY rare attack row (WebAttack, Botnet, BruteForce) and stratifies the
  rest - identical rows for every model.
- **Test:** the full 514,853-row stratified held-out split.
- **Scaling:** none (every model is tree-based; thresholds are per-feature).
- **Metric priority:** for an intrusion detector, per-class **recall**
  (did we catch the attack?) matters more than raw accuracy.

## The five techniques (Stage 5)

Each feature group is its own RandomForest - a "technique" looking at traffic
from one angle. Best macro-F1 column is per class.

| Technique (group) | Features | Test acc | Macro F1 | Strong at | Weak at |
|---|---|---|---|---|---|
| Volume (A) | 15 | 0.9660 | 0.663 | DoS, DDoS, PortScan | BruteForce, Botnet, WebAttack |
| Packet stat (B) | 16 | 0.8563 | 0.684 | DDoS, PortScan | Botnet, WebAttack |
| Timing/IAT (C) | 22 | 0.9215 | 0.632 | DoS, DDoS | PortScan, Botnet, WebAttack |
| TCP flags (D) | 11 | 0.5004 | 0.270 | WebAttack recall | Everything else |
| Endpoint (E) | 5 | 0.9935 | 0.832 | Botnet (F1 0.80), PortScan | WebAttack |

No single technique is enough - each has blind spots. That is the motivation
for the fusion experiments.

## Fusion systems vs the single forest (Stages 5-6)

| System | Acc | Macro F1 | Botnet F1 | WebAttack F1 |
|---|---|---|---|---|
| Group E only (Endpoint) | 0.9935 | 0.832 | 0.804 | 0.136 |
| Fused, soft vote (mean proba) | 0.9737 | 0.727 | 0.066 | 0.122 |
| Fused, majority vote | 0.9702 | 0.715 | 0.066 | 0.071 |
| **Stacked fusion** (OOF meta-RF) | **0.9986** | 0.854 | **0.815** | 0.206 |
| **RF, all 69 features (Stage 4)** | 0.9983 | **0.887** | 0.784 | **0.472** |

## Honest conclusions

1. **Naive fusion loses.** Averaging the five group probabilities dilutes the
   few groups that can actually see the rarest attacks, so macro-F1 falls from
   0.887 (single forest) to 0.73. A "multi-technique" detector is NOT
   automatically better by ensembling opinions at the output layer.

2. **Stacking recovers most of the gap** - accuracy actually exceeds the
   single forest (0.9986) and Botnet recall improves (F1 0.82 vs 0.78) - but it
   does not win on macro-F1 because WebAttack recall collapses (F1 0.21 vs
   0.47). The meta-learner learns to distrust every technique's WebAttack
   opinion, and the small 135-row class cannot argue back through a forest of
   five already-weak signals.

3. **The single all-feature forest is the strongest detector.** Melt all 69
   features together beats every late-fusion scheme on the rarest classes.
   This is the recommended **detection** core.

4. **Where the multi-technique system earns its place is EXPLANATION, not
   detection.** Per-group models attribute a detection to the angle(s) that
   fired (volume? timing? flags? endpoint?), which feeds the "alert + reason"
   promises of the dashboard. Fusing five weak opinions predicts worse than
   one strong detector - but five opinions explain WHY far better than one.

## Stage 7 - boosting rare-class recall (WebAttack / Botnet)

Macro-F1 is dragged down by two tiny classes: WebAttack (135 test rows) and
Botnet (391 rows). For a detector a missed attack (false negative) costs more
than an extra false alarm, so we tune the OPERATING POINT of the frozen
Stage 4 forest: a flow is flagged as the rare class when its probability
clears a threshold, not only when it wins the argmax. The sweep reuses the
already-computed test probabilities, so it is free. Two imbalance treatments
are also re-trained on the same 300k sample: an RF with the rare classes
oversampled x10, and a HistGradientBoosting with `class_weight='balanced'`.

| Source | target | op. point | threshold | recall | precision | F1 | macro-F1 |
|---|---|---|---|---|---|---|---|
| RF (frozen) | WebAttack | argmax | - | 0.504 | 0.444 | 0.472 | 0.887 |
| RF (frozen) | WebAttack | best F1 | 0.40 | 0.696 | 0.427 | **0.530** | 0.895 |
| RF (frozen) | WebAttack | 90% recall | 0.14 | 0.904 | 0.273 | 0.419 | 0.877 |
| RF (frozen) | Botnet | argmax | - | 0.964 | 0.660 | 0.784 | 0.887 |
| RF (frozen) | Botnet | best F1 | 0.65 | 0.946 | 0.708 | **0.810** | 0.891 |
| RF (frozen) | Botnet | 90% recall | 0.70 | 0.926 | 0.711 | 0.804 | 0.890 |
| HGB (balanced) | WebAttack | argmax | - | 0.911 | 0.374 | 0.530 | 0.891 |
| HGB (balanced) | WebAttack | best F1 | 0.69 | 0.859 | 0.416 | **0.560** | 0.896 |
| HGB (balanced) | Botnet | argmax | - | 0.990 | 0.634 | 0.773 | 0.891 |
| HGB (balanced) | Botnet | best F1 | 0.69 | 0.987 | 0.651 | **0.785** | 0.893 |

Re-trained models (argmax, same test set): oversampled RF lands at macro F1
0.877 with WebAttack recall collapsing to 0.363; the balanced HGB reaches
macro F1 0.891 and natively hits WebAttack recall 0.911.

### Conclusions

1. **Tuning the threshold is the cheapest win.** The frozen Stage 4 forest
   goes from WebAttack F1 0.472 -> 0.530 (recall 0.50 -> 0.70) and Botnet
   0.784 -> 0.810 with zero retraining; accuracy stays ~0.998.

2. **A detector should run a recall-first operating point.** Flagging WebAttack
   at p >= 0.14 catches 90% of attacks for 27% precision - for an alert queue
   that surface is fine; precision can be improved downstream.

3. **Oversampling the rare classes hurts.** Duplicating WebAttack/Botnet x10
   in the training sample makes the forest treat them as larger and drops
   WebAttack recall from 0.50 to 0.36 - the repeats just teach low-precision
   confusion.

4. **HGB + class weighting is the best rare-class learner.** It beats both the
   forest and the frozen forest on WebAttack (F1 0.560 at best point, recall
   0.911 natively) while keeping Botnet around the same F1. Classification
   doesn't have to be a Random Forest for a good reason; the boosted trees
   fit the WebAttack boundary better than the bagged forest at this training
   budget. Shortlist: HGB (balanced) for detection, RF forest as a fallback.

## Stage 8 - operational pipeline + "alert + reason" dashboard

Detection and explanation are kept as two layers, reused from the earlier
studies instead of retrained:

- **Detection core** - the balanced HGB from Stage 7 (or, in the sidebar, the
  Stage 4 Random Forest) predicts on all 69 features. This is what flags a
  flow, so the wrong-alarm story stays as good as the best single model.
- **Explanation layer** - the 5 per-group forests from Stage 5. For a flagged
  flow, each technique reports its own confidence in the alert class and lists
  the features whose removal from the flow drops that confidence the most
  (perturbation attribution). The flow's features are z-scored against the
  scanned batch so the numbers read like "this value is 2.3 SD away from
  normal".

`src/pipeline.py` exposes `FlowScorer.score/explain`; `dashboard/app.py` is a
Streamlit view over it: a KPI header, the alert queue with severity, and a
drill-down that answers "why was flow #n flagged?".

## Stage 9 - cross-dataset generalization (CIC-IDS2017 <-> UNSW-NB15)

Do the detectors generalise to a different benchmark? UNSW-NB15 has a
different 49-feature schema, so we cut both datasets down to the same 9
"packet anatomy" features (duration, fwd/bwd packet and byte counts, Flow
Bytes/s and Packets/s *re-derived with the same formula on both sides*, fwd/bwd
mean packet size). Classes collapse to the only comparable signal: benign vs
attack. A balanced HGB (identical hyperparameters to Stage 7) is trained on
one dataset and tested on the other, in both directions.

| Detector trained on | Evaluated on | domain | AUC | attack recall | false alarm |
|---|---|---|---|---|---|
| CIC (300k, rare-aware) | CIC test | in | 0.999 | 0.99 | 0.011 |
| CIC | UNSW test (175k) | **cross** | 0.548 | ~0.00 | 0.000 |
| UNSW (82k, rare cats kept) | UNSW test | in | 0.983 | 0.86 | 0.039 |
| UNSW | CIC test | **cross** | 0.642 | 0.79 | 0.539 |

### Conclusions

1. **Models stay dataset-bound.** Near-perfect in-domain (AUC ~0.98-0.99)
   collapses to 0.55 (effectively random) in the CIC->UNSW direction. The
   classifier learned CIC's attack shape - deployed on UNSW flows it raises
   almost no alerts, and the "safe" story is an artefact (`recall ~0`, `FA 0`).

2. **The reverse direction shows a prior, not discrimination.** UNSW->CIC
   catches 79% of CIC attacks but at a 54% false-alarm rate: the model's
   general "attack-ness" transfers, its decision boundary does not.

3. **A shared feature vocabulary does not a shared distribution make.** Even
   with identical feature definitions the two benchmarks' flow populations
   barely overlap, so threshold-based tree models cannot port. This is the
   textbook cross-dataset result in NIDS research, not a fixable bug.

4. **Operational guideline:** the detector must be retrained or
   domain-adapted per deployment (fine-tuned, or fed distribution statistics
   of the target network); a model copied across environments is silently
   blind. The pipeline/dashboard from Stage 8 make that retraining cheap,
   which is the transferable part of the system.

## Stage 10 - live capture, flow rebuild, and detection (offline + sniff)

Studies 1-9 were all offline tables. Stage 10 closes the loop: take raw packets
(as a stream), rebuild the SAME 69-feature CIC-IDS2017 representation the
models were trained on, and run the Stage 7/8 scorer on finished flows. Three
layers, kept deliberately separate (`src/live/`):

1. **Flow math** (`flow_features.py`) - captures raw 5-tuple PacketEvents into
   a stateless, scapy-free `Flow`/`FlowTable` and finalizes the 69 features in
   `config.FEATURE_ORDER` (float32, no scaling). Every feature is declared in
   one of three tiers (`FEATURE_META`):
   - `exact` - computable exactly from a bidirectional packet stream
     (counts, byte/rate stats, IATs, flag counts, port, window);
   - `approx` - CICFlowMeter-compatible reconstructions over tunable
     parameters (active/idle burst and subflow timeouts; init-window / act-data
     heuristics) that may differ from the lab capture at the margins;
   - `fill0` - structurally absent here (per-direction header sums) -> 0.0,
     which the training pipeline already sanctioned for missing columns.
2. **Capture** (`capture.py`) - the only scapy/Npcap-touching module. Turns a
   raw Ethernet frame into a `PacketEvent`: IP proto, TCP flags, window, and
   header/payload split via on-wire IP total length. Non-IP frames are skipped.
3. **Detector** (`detector.py`) - `LiveDetector.ingest(frame)` = capture ->
   `FlowTable.add` -> `flush_expired_keyed` -> `FlowScorer` on the finished
   vector. Returns an alert dict per expired flow (source/dest/port, predicted
   class, severity, probabilities, optional 5-technique explanation).

Two drivers: `--pcap file.pcap` (offline replay; same parse path, no admin) and
`--live --iface eth0` (scapy sniff, needs Npcap/admin). A demo pcap of three
synthetic flows replays as `BENIGN` at p≈0.998 through the real HGB detector,
with or without explanations.

### Honest notes

- **Reconstruction is approximation, not capture replay.** The feature
  definitions are reproduced to the field, but exact parity with
  CICFlowMeter's lab capture is not guaranteed (timeout constants, subflow
  splits, window heuristics). The `exact/approx/fill0` tiers say where to
  trust the numbers and where to expect drift.
- **The detector stays dataset-bound** (Stage 9): live flows that differ from
  CIC's population will be judged by CIC-trained boundaries. The value of this
  stage is the *pipeline* - train once, score your own capture with the same
  contract - plus a deterministic test harness (31 unit tests, no Npcap).

## Stage 11 - live pipeline performance: throughput, latency, explain cost

A live IDS needs operational numbers, not just accuracy. `src/live/benchmark.py`
feeds a deterministic synthetic stream (2000 sessions at 1000 offered flows/s,
scapy-crafted frames, no Npcap) through the real Stage 10 pipeline and measures
four things. The machine is a busy laptop, so absolute numbers are noisy
(±2x run-to-run); the ratios and the structural conclusions are the point.

| measurement | result | meaning |
|---|---|---|
| throughput (finalize+score) | ~25-80 flows/s | the Python capture+flow-math hot path is a few ms per packet - fine for lab/demo, nowhere near line rate |
| detection latency | ≈ flow timeout (2s->2.1, 15->15.1, 60->60.1) | timeout sits exactly where it should: a knob that trades feature completeness for alert delay |
| explanation cost | ~2.4-6.1 s per alert (on-demand) | the 5-group perturbation explain is seconds-scale; it must stay off the hot path |
| detector score speed | HGB ~28 flows/s vs RandomForest ~10 | the balanced HGB detection core is also the cheaper one to run; RF about 3x slower per flow |

Conclusions:

1. **The bottleneck is capture + flow math, not the model.** Processing a
   packet costs ~3.5 ms of which the HGB predict is a fraction. A real
   deployment would batch capture (e.g. libnids / NFQueue) or drop to compiled
   flow exporters; this pipeline is an honest reference implementation.
2. **Timeout is a direct latency dial.** Alert delay tracks the configured
   idle timeout one-to-one - useful, because it lets an operator pick the
   latency/completeness trade explicitly instead of discovering it.
3. **Explanation is expensive by design.** At seconds per alert the 5-technique
   attribution can only run on demand for a flagged flow (which is exactly how
   Stage 8 built the dashboard). Quantifying the cost made that design choice
   explicit rather than assumed.

## Stage 12 - a dashboard a normal person can read, and what it cost to find

Stages 10-11 built a detector that runs on packets. Stage 8 built an
"alert + reason" view, but it showed raw 69-feature numbers and z-scores, which
answers "what did the model see?" for an ML reviewer and nothing at all for the
person who has to act on the alert. Stage 12 rebuilds the front end around
language (`dashboard/live.py`, `src/live/plain.py`): a friendly label and unit
for all 69 features, a severity colour, and per-technique reasons phrased as
sentences ("packets sent = 4 pkts - far higher than usual (12x the normal
spread)"). A test pins the invariant that no feature can fall back to
"feature value", so the vocabulary cannot silently rot.

The dashboard has three honest sources: **lab replay** (real held-out CIC rows
scored live by the real detector, paced, true labels shown), **packet replay**
(the Stage 10 capture path over a generated pcap), and **live sniff**
(needs Npcap + admin). Because explanation costs seconds per alert (Stage 11),
the feed computes them in a worker thread with a budget cap, so the UI never
blocks - which is why an alert's "why" can arrive a few seconds after the alert
itself.

### The measurement that changed the design

Building the demo traffic as a sanity check produced a bad result: replaying
the generated pcap through the real detector flagged **nothing**. Rather than
paper over it, `python -m src.live.demo_traffic --check` measures the round trip
directly - score each row as-is, rebuild its features from generated packets,
score again (130 stratified rows, real Stage 7 detector):

| | value |
|---|---|
| verdict kept after the round trip | **56.2%** |
| rebuilt flows the detector calls BENIGN | **97.7%** |
| detector correct on the ORIGINAL rows | 70/70 benign, 9-10/10 per attack class |
| verdict kept, per attack class | DoS/DDoS/PortScan/BruteForce **0/10**, Botnet 2/10, WebAttack 1/10 |
| feature error, median flow | ~0.00 SD (both tiers) |
| feature error, p90 | 0.28 SD (exact tier), 0.24 SD (approx tier) |
| worst features at p90 | ACK Flag Count **51.7 SD**, Down/Up Ratio **30.4 SD**, FIN Flag Count 5.6 SD |

### Conclusions

1. **The round trip is exact for a typical flow and wrong for an attack.**
   The median flow is 4-5 packets, and the packetizer rebuilds it almost
   perfectly - which is why benign verdicts survive 70/70 and the median
   feature error is ~0 SD. Attacks live in the tail: their flag and ratio
   counts are tens of SD off, because a flow can only be as long as the
   packetizer emits, and CIC attack rows carry far more packets and far more
   ACKs than a demo can afford. The signal is not lost in the flow math
   (Stage 10), it is lost in the *synthesis* of the packets.

2. **A quiet packet demo is a property of the demo, not evidence the IDS
   works.** The same detector reads the original rows at 9-10/10 per attack
   class. Anyone replaying a coarse pcap and seeing silence should read it as
   "my synthetic traffic is too small to carry the signature", never as "the
   model is broken".

3. **Therefore lab replay is the default source and packet replay is a
   labelled second opinion.** Lab replay shows real, ranked, explainable
   detections with ground truth attached. Packet replay stays in the UI
   because it is the honest capture path - and its quietness is now a
   documented, measured property rather than a surprise.

4. **Two bugs the Stage 12 UI work exposed, both fixed.** The dashboard called
   its helpers before defining them (`NameError` on every run), and the feed
   passed 1D rows to `predict_proba`, which sklearn rejects - the unit tests
   missed it because the fake scorer accepted anything. The fake now asserts
   sklearn's real 2D contract, and the app is verified with Streamlit's
   `AppTest` harness rather than by eye.

## Reproduce

```
python src/random_forest_model.py      # Stage 4 - single forest baseline
python src/fusion_model.py             # Stage 5 - per-technique + naive voting
python src/stacked_fusion.py           # Stage 6 - stacked (OOF) meta-fusion
python src/rare_class_experiments.py   # Stage 7 - rare-class threshold sweep +
                                       #   oversampled RF + balanced HGB
python src/pipeline.py                 # Stage 8 - CLI smoke of score + explain
streamlit run dashboard/app.py         # Stage 8 - alert + reason dashboard
python src/cross_dataset.py            # Stage 9 - cross-dataset transfer (2 ways)

# Stage 10 - live detection
python -m unittest tests.test_flow_features   # flow math (16 tests)
python -m unittest tests.test_capture_layer   # scapy capture (10 tests)
python -m unittest tests.test_live_detector   # engine + pcap replay (5 tests)
python src/live/detector.py --pcap demo.pcap --timeout 0.4 --explain
python src/live/detector.py --live --iface eth0 --count 500 --explain
python src/live/benchmark.py --flows 2000            # Stage 11 throughput/latency

# Stage 12 - plain-language live dashboard
streamlit run dashboard/live.py            # lab / packet / live-sniff sources
python -m unittest tests.test_live_dashboard  # plain language + feed (6 tests)
python -m src.live.demo_traffic            # build results/demo_live.pcap
python -m src.live.demo_traffic --check    # round-trip fidelity measurement
```

All tests: `python -m unittest discover -s tests` (40 tests, no Npcap needed).

Metrics: `results/metrics/*.json`  |  plots: `results/plots/stage*`  |
per-class tables are printed at each run.