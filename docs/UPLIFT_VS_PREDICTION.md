# Prediction vs Intervention Effect

## The mistake

Many “AI for insurance” demos train a model to predict claims, then recommend heavy interventions to whoever scores highest.

That optimizes **risk identification**, not **risk reduction**.

## Formal distinction

Let `Y` be a future loss indicator, `X` customer/property features, `T` an intervention.

**Predictive model**

```
μ(X) = P(Y = 1 | X)
```

Answers: *Who is likely to have a loss?*

**Potential outcomes / uplift**

```
μ0(X) = P(Y = 1 | X, do(T=0))
μ1(X) = P(Y = 1 | X, do(T=1))
τ(X)  = μ0(X) − μ1(X)   # absolute risk reduction (CATE)
```

Answers: *How much does intervention T change this customer's loss probability?*

## Why they diverge

1. **Non-responders** — high baseline risk, but the intervention does not address their mechanism (e.g. inspection when the real issue is coverage design).  
2. **Already safe** — low risk customers can still show small positive CATE for cheap education, but net utility may be negative after cost/burden.  
3. **Heterogeneity** — maintenance advice helps low-maintenance properties far more than well-maintained ones.  
4. **Confounding** — historically, ops treated the riskiest customers, so naive treated-vs-control comparisons mix selection with effect.

## Meta-learners implemented

| Method | Idea |
|---|---|
| S-learner | One model on `(X, T)`; CATE from contrasting T=0/1 predictions |
| T-learner | Separate models for treated and control outcomes |
| X-learner | Impute individual effects, regress them, propensity-weight |
| Causal forest | Honest trees that split on effect heterogeneity |

## Practical rule used by this system

```
recommend argmax_T  utility( τ̂(X,T), confidence, cost, burden, feasibility )
subject to eligibility constraints
```

Including `no_intervention` as a first-class action.
