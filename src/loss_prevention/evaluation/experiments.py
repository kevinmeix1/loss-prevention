"""Research experiments: confounding, heterogeneity, policy optimization, explore/exploit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from loss_prevention.data.generator import (
    FEATURE_COLUMNS,
    generate_customers,
    generate_experiment_dataset,
    oracle_policy_value,
    true_cate,
    customer_feature_dict,
)
from loss_prevention.models.bundle import train_all_models
from loss_prevention.ranking.interventions import RankingConfig, rank_interventions
from loss_prevention.schemas import ExperimentReport, InterventionType


def experiment_selection_bias(seed: int = 42) -> ExperimentReport:
    """Show that naive treated-vs-control comparisons are biased under confounding."""
    _, df = generate_experiment_dataset(n_customers=3000, n_historical=8000, seed=seed)
    rows = []
    for intervention, sub in df.groupby("intervention"):
        treated = sub[sub["treated"] == 1]["outcome_loss"]
        control = sub[sub["treated"] == 0]["outcome_loss"]
        if len(treated) < 30 or len(control) < 30:
            continue
        naive_ate = float(control.mean() - treated.mean())  # positive => appears helpful
        # RCT subset
        rct = sub[np.isclose(sub["propensity_score"], 0.5)]
        if len(rct) >= 40 and rct["treated"].nunique() == 2:
            rct_ate = float(
                rct[rct["treated"] == 0]["outcome_loss"].mean()
                - rct[rct["treated"] == 1]["outcome_loss"].mean()
            )
        else:
            rct_ate = np.nan
        # Oracle mean CATE
        oracle = []
        for _, row in sub.head(400).iterrows():
            feats = {c: float(row[c]) for c in FEATURE_COLUMNS}
            oracle.append(true_cate(InterventionType(intervention), feats))
        rows.append(
            {
                "intervention": intervention,
                "naive_ate": naive_ate,
                "rct_ate": rct_ate,
                "oracle_mean_cate": float(np.mean(oracle)),
                "treated_rate": float(sub["treated"].mean()),
                "mean_risk_treated": float(sub[sub["treated"] == 1]["hazard_exposure"].mean()),
                "mean_risk_control": float(sub[sub["treated"] == 0]["hazard_exposure"].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    bias = float((frame["naive_ate"] - frame["oracle_mean_cate"]).abs().mean())
    return ExperimentReport(
        name="selection_bias_confounding",
        summary=(
            "Ops preferentially treats higher-risk, more contactable customers. "
            "Naive ATE therefore mixes selection differences with true effects."
        ),
        metrics={
            "mean_abs_naive_bias": bias,
            "by_intervention": frame.to_dict(orient="records"),
        },
        takeaways=[
            "Treated customers have higher hazard exposure than controls — classic confounding.",
            "Naive ATE can understate (or flip) benefits when the untreated look healthier.",
            "RCT / properly adjusted uplift estimates recover effects closer to the oracle CATE.",
            "Predicting who is at risk is not the same as estimating who benefits.",
        ],
    )


def experiment_treatment_heterogeneity(seed: int = 42) -> ExperimentReport:
    """Demonstrate that CATE varies across customer segments."""
    customers = generate_customers(2500, seed=seed)
    bundle = train_all_models(n_customers=2500, n_historical=7000, seed=seed, fit_causal_forest=False)
    records = []
    for c in customers[:800]:
        feats = customer_feature_dict(c)
        for intervention in [
            InterventionType.MAINTENANCE_RECOMMENDATION,
            InterventionType.INSPECTION,
            InterventionType.EDUCATIONAL_GUIDANCE,
        ]:
            est = bundle.uplift.estimate(intervention, feats)
            records.append(
                {
                    "intervention": intervention.value,
                    "segment": c.true_segment.value if c.true_segment else "unknown",
                    "maintenance_score": c.maintenance_score,
                    "oracle_cate": true_cate(intervention, feats),
                    "est_cate": est.cate,
                }
            )
    frame = pd.DataFrame(records)
    het = (
        frame.groupby(["intervention", "segment"])[["oracle_cate", "est_cate"]]
        .mean()
        .reset_index()
    )
    return ExperimentReport(
        name="treatment_heterogeneity",
        summary="Average treatment effects hide large subgroup differences.",
        metrics={"heterogeneity_table": het.to_dict(orient="records")},
        takeaways=[
            "Maintenance recommendations help most when maintenance_score is already low.",
            "Inspections help high-hazard profiles more than low-risk digital-only customers.",
            "Policy should optimize individualized CATE, not one-size-fits-all ATE.",
        ],
    )


def experiment_policy_optimization(seed: int = 42) -> ExperimentReport:
    """Compare targeting policies by oracle policy value."""
    customers = generate_customers(2000, seed=seed)
    bundle = train_all_models(n_customers=2000, n_historical=6000, seed=seed, fit_causal_forest=False)
    cfg = RankingConfig()

    def policy_risk_only(c):
        # Treat highest predicted risk with expensive inspection
        risk = bundle.predictive.predict_customer(c)
        return (
            InterventionType.INSPECTION
            if risk.p_loss >= 0.25
            else InterventionType.EDUCATIONAL_GUIDANCE
        )

    def policy_uplift(c):
        ranked = rank_interventions(c, bundle, cfg, method="x_learner")
        return next(s.intervention for s in ranked if s.eligible)

    def policy_random(c):
        rng = np.random.default_rng(abs(hash(c.customer_id)) % (2**32))
        choices = [i for i in InterventionType]
        return choices[int(rng.integers(0, len(choices)))]

    def policy_no_action(c):
        return InterventionType.NO_INTERVENTION

    values = {
        "no_action": oracle_policy_value(customers, policy_no_action),
        "random": oracle_policy_value(customers, policy_random),
        "risk_based": oracle_policy_value(customers, policy_risk_only),
        "uplift_ranked": oracle_policy_value(customers, policy_uplift),
    }
    return ExperimentReport(
        name="policy_optimization",
        summary="Uplift-aware ranking captures more true risk reduction than risk-only targeting.",
        metrics={"oracle_policy_value_mean_cate": values},
        takeaways=[
            "Risk-based targeting over-spends on unresponsive high-risk customers.",
            "Uplift ranking improves expected risk reduction under the oracle DGP.",
            "No-action is a valid policy when expected benefit < cost/burden.",
        ],
    )


def experiment_explore_exploit(seed: int = 42) -> ExperimentReport:
    """Epsilon-greedy exploration vs pure exploitation on policy value + diversity."""
    customers = generate_customers(1500, seed=seed + 1)
    bundle = train_all_models(n_customers=1500, n_historical=5000, seed=seed, fit_causal_forest=False)
    from loss_prevention.agent.recommender import LossPreventionAgent
    from loss_prevention.evaluation.metrics import intervention_entropy

    agent = LossPreventionAgent(bundle, customers)
    labels_exploit, labels_explore = [], []
    value_exploit = value_explore = 0.0
    for c in customers[:600]:
        r0 = agent.recommend(c.customer_id, explore=False)
        r1 = agent.recommend(c.customer_id, explore=True)
        labels_exploit.append(r0.recommended_intervention.value)
        labels_explore.append(r1.recommended_intervention.value)
        feats = customer_feature_dict(c)
        value_exploit += true_cate(r0.recommended_intervention, feats)
        value_explore += true_cate(r1.recommended_intervention, feats)
    n = 600
    return ExperimentReport(
        name="explore_vs_exploit",
        summary="Exploration increases intervention diversity with a small short-run value tradeoff.",
        metrics={
            "exploit_policy_value": value_exploit / n,
            "explore_policy_value": value_explore / n,
            "exploit_entropy": intervention_entropy(labels_exploit),
            "explore_entropy": intervention_entropy(labels_explore),
            "epsilon": agent.cfg.explore_epsilon,
        },
        takeaways=[
            "Pure exploitation can concentrate on a few interventions and slow learning.",
            "Epsilon-greedy raises entropy (coverage) while modestly affecting short-run value.",
            "In production, couple exploration with logging for continual uplift re-estimation.",
        ],
    )


def run_research_suite(seed: int = 42) -> list[ExperimentReport]:
    return [
        experiment_selection_bias(seed),
        experiment_treatment_heterogeneity(seed),
        experiment_policy_optimization(seed),
        experiment_explore_exploit(seed),
    ]
