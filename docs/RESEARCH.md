# Research Experiments

All experiments use the synthetic DGP in `data/generator.py`. Oracle CATEs exist only for evaluation.

## 1. Selection bias & confounding

**Setup.** 80% observational assignments with propensity rising in baseline risk and contactability; 20% RCT.

**Finding.** Naive ATE (control loss rate − treated loss rate) is biased because treated customers are systematically higher hazard. RCT / adjusted uplift estimates recover values closer to oracle mean CATE.

## 2. Treatment heterogeneity

**Setup.** Estimate CATE by risk segment and maintenance/hazard features for maintenance, inspection, and education interventions.

**Finding.** Effects are not constant. Maintenance recommendations help low-maintenance properties most; inspections help high-hazard profiles; education helps engaged customers more.

## 3. Policy optimization

**Policies compared**

- no action  
- random intervention  
- risk-based (inspect if high `P(loss)`)  
- uplift-ranked utility maximization  

**Metric.** Mean oracle CATE under the assigned intervention.

**Finding.** Uplift-aware ranking beats risk-only targeting by avoiding expensive interventions on unresponsive high-risk customers.

## 4. Exploration vs exploitation

**Setup.** Epsilon-greedy among top eligible interventions vs pure exploitation.

**Metrics.** Oracle policy value and intervention entropy.

**Finding.** Exploration increases diversity (coverage) with a small short-run value tradeoff — useful for continual learning.

## Evaluation metrics

| Metric | Purpose |
|---|---|
| ROC-AUC / Brier / ECE | Predictive accuracy & calibration |
| Qini / AUUC | Uplift ranking quality |
| Recommendation quality | `|τ̂ − τ_oracle|` complement |
| Intervention entropy | Diversity of recommended actions |
| Mean burden | Customer experience cost |
| Policy value | Expected true risk reduction |
| Naive ATE bias | How misleading observational comparisons are |
