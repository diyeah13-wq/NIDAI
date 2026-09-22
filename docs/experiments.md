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

## Reproduce

```
python src/random_forest_model.py   # Stage 4 - single forest baseline
python src/fusion_model.py          # Stage 5 - per-technique + naive voting
python src/stacked_fusion.py        # Stage 6 - stacked (OOF) meta-fusion
```

Metrics: `results/metrics/*.json`  |  plots: `results/plots/stage*`  |
per-class tables are printed at each run.