"""Constraint filtering and multi-objective intervention ranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loss_prevention.data.generator import customer_feature_dict
from loss_prevention.models.bundle import ModelBundle
from loss_prevention.schemas import (
    INTERVENTION_CATALOG,
    CustomerContext,
    InterventionScore,
    InterventionType,
)


@dataclass
class RankingConfig:
    w_benefit: float = 1.0
    w_confidence: float = 0.25
    w_cost: float = 0.35
    w_burden: float = 0.40
    w_feasibility: float = 0.30
    max_burden: float = 0.85
    min_confidence: float = 0.15
    suppress_recent_days: int = 90
    max_interventions_per_year: int = 3
    explore_epsilon: float = 0.08
    expected_severity: float = 5000.0


def _feasibility(customer: CustomerContext, intervention: InterventionType) -> float:
    meta = INTERVENTION_CATALOG[intervention]
    if intervention == InterventionType.NO_INTERVENTION:
        return 1.0
    score = 0.55
    # Digital channels need digital affinity
    if "email" in meta["channels"] or "app" in meta["channels"]:
        score += 0.25 * customer.digital_affinity
    if "field" in meta["channels"]:
        score += 0.20 * customer.contactability
        score -= 0.15 * (1.0 - customer.engagement_score)
    if "phone" in meta["channels"]:
        score += 0.20 * customer.contactability
    # Specialist / inspection require higher risk or complexity signals
    if intervention in {
        InterventionType.SPECIALIST_SUPPORT,
        InterventionType.INSPECTION,
    }:
        complexity = (
            0.4 * customer.hazard_exposure
            + 0.3 * (1 - customer.maintenance_score)
            + 0.3 * min(customer.prior_claims_3y / 3.0, 1.0)
        )
        score = 0.3 + 0.7 * complexity
    return float(np.clip(score, 0.05, 1.0))


def _eligibility(
    customer: CustomerContext,
    intervention: InterventionType,
    cfg: RankingConfig,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if intervention == InterventionType.NO_INTERVENTION:
        return True, reasons
    if customer.days_since_last_intervention < cfg.suppress_recent_days:
        reasons.append(
            f"Recent intervention within {cfg.suppress_recent_days} days "
            f"({customer.days_since_last_intervention}d ago)"
        )
    if customer.recent_interventions_12m >= cfg.max_interventions_per_year:
        reasons.append(
            f"Annual intervention cap reached ({customer.recent_interventions_12m})"
        )
    burden = INTERVENTION_CATALOG[intervention]["burden"]
    if burden > cfg.max_burden and customer.engagement_score < 0.35:
        reasons.append("High burden intervention unsuitable for low-engagement customer")
    if (
        intervention == InterventionType.FRAUD_AWARENESS_EDUCATION
        and customer.fraud_signal_score < 0.15
    ):
        reasons.append("Fraud signal too low for fraud-awareness education")
    if (
        intervention == InterventionType.SPECIALIST_SUPPORT
        and customer.prior_claims_3y == 0
        and customer.hazard_exposure < 0.35
    ):
        reasons.append("Specialist support reserved for complex / high-exposure risks")
    return len(reasons) == 0, reasons


def rank_interventions(
    customer: CustomerContext,
    bundle: ModelBundle,
    cfg: RankingConfig | None = None,
    method: str = "x_learner",
    include_ineligible: bool = True,
) -> list[InterventionScore]:
    cfg = cfg or RankingConfig()
    feats = customer_feature_dict(customer)
    scores: list[InterventionScore] = []

    for intervention in InterventionType:
        eligible, reasons = _eligibility(customer, intervention, cfg)
        meta = INTERVENTION_CATALOG[intervention]
        uplift = bundle.uplift.estimate(
            intervention,
            feats,
            method=method if method != "causal_forest" else "x_learner",
            expected_severity=cfg.expected_severity,
        )
        # Optional causal forest override for CATE
        if method == "causal_forest" and intervention.value in bundle.causal_forests:
            from loss_prevention.data.generator import FEATURE_COLUMNS

            x = np.array([[feats[c] for c in FEATURE_COLUMNS]])
            cate = float(bundle.causal_forests[intervention.value].predict(x)[0])
            uplift.cate = cate
            uplift.p_loss_treated = float(np.clip(uplift.p_loss_control - cate, 0, 1))
            uplift.expected_benefit = cate * cfg.expected_severity
            uplift.method = "causal_forest"

        feasibility = _feasibility(customer, intervention)
        cost_norm = meta["cost"] / 250.0
        burden = meta["burden"]

        utility = (
            cfg.w_benefit * (uplift.cate * 10.0)  # scale risk reduction
            + cfg.w_confidence * uplift.confidence
            + cfg.w_feasibility * feasibility
            - cfg.w_cost * cost_norm
            - cfg.w_burden * burden
        )
        if not eligible:
            utility = -1e6

        # Soft confidence gate for ranking (still returned for transparency)
        if uplift.confidence < cfg.min_confidence and intervention != InterventionType.NO_INTERVENTION:
            utility -= 0.5

        scores.append(
            InterventionScore(
                intervention=intervention,
                expected_benefit=uplift.expected_benefit,
                confidence=uplift.confidence,
                cost=meta["cost"],
                customer_burden=burden,
                feasibility=feasibility,
                utility=float(utility),
                rank=0,
                eligible=eligible,
                ineligibility_reasons=reasons,
                uplift=uplift,
            )
        )

    # Rank eligible first by utility
    eligible_scores = sorted(
        [s for s in scores if s.eligible], key=lambda s: s.utility, reverse=True
    )
    ineligible_scores = sorted(
        [s for s in scores if not s.eligible], key=lambda s: s.utility, reverse=True
    )
    ordered = eligible_scores + (ineligible_scores if include_ineligible else [])
    for i, s in enumerate(ordered, start=1):
        s.rank = i
    return ordered
