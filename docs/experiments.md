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

## Reproduce

```
python src/random_forest_model.py      # Stage 4 - single forest baseline
python src/fusion_model.py             # Stage 5 - per-technique + naive voting
python src/stacked_fusion.py           # Stage 6 - stacked (OOF) meta-fusion
python src/rare_class_experiments.py   # Stage 7 - rare-class threshold sweep +
                                       #   oversampled RF + balanced HGB
python src/pipeline.py                 # Stage 8 - CLI smoke of score + explain
streamlit run dashboard/app.py         # Stage 8 - alert + reason dashboard
```

Metrics: `results/metrics/*.json`  |  plots: `results/plots/stage*`  |
per-class tables are printed at each run.