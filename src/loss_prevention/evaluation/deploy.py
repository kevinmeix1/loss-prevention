"""Simulate offline→online deployment of the recommendation policy."""

from __future__ import annotations

import numpy as np

from loss_prevention.data.generator import (
    FEATURE_COLUMNS,
    customer_feature_dict,
    generate_customers,
    generate_experiment_dataset,
    true_cate,
)
from loss_prevention.evaluation.metrics import (
    auuc,
    intervention_entropy,
    predictive_metrics,
    qini_coefficient,
)
from loss_prevention.models.bundle import train_all_models
from loss_prevention.ranking.interventions import RankingConfig, rank_interventions
from loss_prevention.schemas import DeploymentMetrics, InterventionType


def simulate_deployment(
    n_customers: int = 2000,
    n_historical: int = 6000,
    seed: int = 42,
) -> DeploymentMetrics:
    customers, df = generate_experiment_dataset(
        n_customers=n_customers, n_historical=n_historical, seed=seed
    )
    bundle = train_all_models(
        n_customers=n_customers, n_historical=n_historical, seed=seed, fit_causal_forest=False
    )

    # Predictive metrics on untreated holdout
    untreated = df[df["treated"] == 0]
    x = untreated[FEATURE_COLUMNS].astype(float).values
    y = untreated["outcome_loss"].astype(int).values
    p = bundle.predictive.predict_proba_matrix(x)
    pred = predictive_metrics(y, p)

    # Uplift metrics on a primary intervention
    primary = InterventionType.MAINTENANCE_RECOMMENDATION.value
    sub = df[df["intervention"] == primary]
    x_u = sub[FEATURE_COLUMNS].astype(float).values
    t = sub["treated"].astype(int).values
    y_u = sub["outcome_loss"].astype(int).values
    cate = bundle.uplift.models[primary].cate(x_u, method="x_learner")
    qini = qini_coefficient(y_u, t, cate)
    au = auuc(y_u, t, cate)

    # Deploy policy
    cfg = RankingConfig()
    labels = []
    burdens = []
    quality = []
    naive_biases = []
    for c in customers[:1000]:
        ranked = rank_interventions(c, bundle, cfg)
        top = next(s for s in ranked if s.eligible)
        labels.append(top.intervention.value)
        burdens.append(top.customer_burden)
        feats = customer_feature_dict(c)
        oracle = true_cate(top.intervention, feats)
        est = top.uplift.cate if top.uplift else 0.0
        quality.append(1.0 - abs(oracle - est))
        # Naive bias proxy from historical same intervention
        hist = df[df["intervention"] == top.intervention.value]
        if len(hist) > 40:
            naive = float(
                hist[hist["treated"] == 0]["outcome_loss"].mean()
                - hist[hist["treated"] == 1]["outcome_loss"].mean()
            )
            naive_biases.append(abs(naive - oracle))

    policy_value = float(
        np.mean(
            [
                true_cate(
                    InterventionType(labels[i]),
                    customer_feature_dict(customers[i]),
                )
                for i in range(len(labels))
            ]
        )
    )

    return DeploymentMetrics(
        n_customers=len(customers),
        predictive_auc=pred["auc"],
        predictive_brier=pred["brier"],
        uplift_qini=float(qini),
        uplift_auuc=float(au),
        calibration_ece=pred["ece"],
        mean_burden=float(np.mean(burdens)),
        intervention_entropy=intervention_entropy(labels),
        recommendation_quality=float(np.mean(quality)),
        policy_value=policy_value,
        naive_ate_bias=float(np.mean(naive_biases) if naive_biases else 0.0),
    )
