# Demo Video Script — Loss Prevention Agent

Duration target: 3–4 minutes.

## Scene 1 — Opening
**Screen:** UI hero / Risk Overview  
**Action:** Show brand sidebar and pipeline stages.  
**Narration (EN):** “This is an autonomous insurance loss-prevention agent. It answers: which intervention is most likely to reduce future loss?”  
**Narration (ZH):** “这是一个自主保险损失预防智能体。它回答：哪种干预最有可能降低未来损失？”  
**Technical point:** Prediction and uplift are separated.

## Scene 2 — Architecture
**Screen:** Architecture tab  
**Action:** Open Architecture.  
**Explanation:** Customer → risk → uplift → constraints → ranking → grounded agent explanation.

## Scene 3 — Select customer & run
**Screen:** Controls  
**Action:** Select a high-hazard customer; choose X-learner; click Run recommendation.  
**What happens:** Status pill shows loading; pipeline stages animate; recommendation returns with agent_trace.

## Scene 4 — Risk Overview
**Screen:** Overview metrics + gauge + CATE bars  
**Explanation:** P(loss) is predictive. Recommended action uses estimated CATE and utility, not risk alone.

## Scene 5 — Risk Factors
**Screen:** Factors tab  
**Action:** Open Risk Factors.  
**Explanation:** Feature importances explain baseline risk, not treatment effect.

## Scene 6 — Interventions ranking
**Screen:** Interventions table  
**Action:** Scroll ranked list; highlight selected vs blocked rows.  
**Explanation:** Hard constraints cannot be overridden by the LLM.

## Scene 7 — Counterfactual simulator
**Screen:** Counterfactual tab  
**Action:** Choose Inspection; click Simulate.  
**What happens:** Baseline vs counterfactual P(loss), net utility, caveats.

## Scene 8 — Model compare
**Screen:** Model Compare  
**Action:** Compare S/T/X/causal forest for maintenance recommendation.  
**Explanation:** Same causal target, different inductive biases.

## Scene 9 — Research lab
**Screen:** Research snapshot  
**Action:** Click Research snapshot; wait for metrics.  
**Explanation:** Selection bias, policy value, Qini/AUUC, calibration.

## Scene 10 — Conclusion
**Screen:** Overview  
**Summary:** Predictive ML + uplift + ranking + grounded agent + research evaluation on synthetic data.
