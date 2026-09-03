"""Synthetic insurance DGP with treatment effect heterogeneity and confounding.

Key design:
- True baseline loss risk depends on property/behavior features.
- Each intervention has a heterogeneous CATE (treatment effect).
- Historical assignment is *not* random: ops preferentially treat high-risk,
  high-contactability customers → selection bias / confounding.
- Naive treated-vs-control comparisons are therefore misleading.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from loss_prevention.schemas import (
    INTERVENTION_CATALOG,
    CustomerContext,
    HistoricalAssignment,
    InterventionType,
    LineOfBusiness,
    RiskSegment,
)

FEATURE_COLUMNS = [
    "tenure_months",
    "property_age_years",
    "prior_claims_3y",
    "claim_severity_avg",
    "maintenance_score",
    "safety_score",
    "hazard_exposure",
    "coverage_adequacy",
    "engagement_score",
    "digital_affinity",
    "recent_interventions_12m",
    "days_since_last_intervention",
    "region_risk_index",
    "deductible_ratio",
    "premium_band",
    "fraud_signal_score",
    "contactability",
]

ACTIVE_INTERVENTIONS = [i for i in InterventionType if i != InterventionType.NO_INTERVENTION]


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _true_baseline_risk(row: dict[str, float]) -> float:
    logit = (
        -2.1
        + 0.55 * row["prior_claims_3y"]
        + 0.9 * row["hazard_exposure"]
        + 0.7 * (1.0 - row["maintenance_score"])
        + 0.6 * (1.0 - row["safety_score"])
        + 0.35 * row["property_age_years"] / 40.0
        + 0.4 * row["region_risk_index"]
        + 0.25 * (1.0 - row["coverage_adequacy"])
        + 0.2 * row["fraud_signal_score"]
        - 0.15 * row["engagement_score"]
    )
    return float(np.clip(_sigmoid(logit), 0.02, 0.85))


def _segment(p: float) -> RiskSegment:
    if p >= 0.45:
        return RiskSegment.CRITICAL
    if p >= 0.28:
        return RiskSegment.HIGH
    if p >= 0.15:
        return RiskSegment.MODERATE
    return RiskSegment.LOW


def true_cate(intervention: InterventionType, row: dict[str, float]) -> float:
    """Ground-truth absolute risk reduction for treated vs control (positive helps)."""
    base = _true_baseline_risk(row)
    if intervention == InterventionType.NO_INTERVENTION:
        return 0.0

    # Heterogeneity: interventions help different profiles differently.
    effects = {
        InterventionType.EDUCATIONAL_GUIDANCE: 0.03
        + 0.05 * row["engagement_score"]
        + 0.02 * (1.0 - row["safety_score"]),
        InterventionType.SAFETY_ASSESSMENT: 0.04
        + 0.08 * (1.0 - row["safety_score"])
        + 0.04 * row["hazard_exposure"],
        InterventionType.MAINTENANCE_RECOMMENDATION: 0.035
        + 0.10 * (1.0 - row["maintenance_score"])
        + 0.03 * (row["property_age_years"] / 50.0),
        InterventionType.COVERAGE_REVIEW: 0.02
        + 0.07 * (1.0 - row["coverage_adequacy"])
        + 0.02 * row["prior_claims_3y"] / 3.0,
        InterventionType.FRAUD_AWARENESS_EDUCATION: 0.015
        + 0.08 * row["fraud_signal_score"],
        InterventionType.CLAIMS_PREVENTION_GUIDANCE: 0.03
        + 0.07 * min(row["prior_claims_3y"] / 3.0, 1.0)
        + 0.03 * row["claim_severity_avg"],
        InterventionType.INSPECTION: 0.05
        + 0.09 * row["hazard_exposure"]
        + 0.06 * (1.0 - row["maintenance_score"]),
        InterventionType.SPECIALIST_SUPPORT: 0.06
        + 0.10 * base
        + 0.05 * row["prior_claims_3y"] / 3.0,
    }
    raw = effects[intervention]
    # Diminishing returns for already-safe, low-risk customers
    raw *= 0.4 + 0.9 * base
    # Some interventions can mildly backfire for very low-engagement customers
    if intervention in {
        InterventionType.INSPECTION,
        InterventionType.SPECIALIST_SUPPORT,
    } and row["engagement_score"] < 0.25:
        raw -= 0.02
    return float(np.clip(raw, -0.03, 0.25))


def generate_customers(n: int = 5000, seed: int = 42) -> list[CustomerContext]:
    rng = np.random.default_rng(seed)
    lines = list(LineOfBusiness)
    customers: list[CustomerContext] = []
    for i in range(n):
        prior_claims = int(rng.choice([0, 1, 2, 3, 4], p=[0.55, 0.25, 0.12, 0.05, 0.03]))
        maintenance = float(np.clip(rng.beta(5, 3), 0, 1))
        safety = float(np.clip(rng.beta(5, 2.5), 0, 1))
        hazard = float(np.clip(rng.beta(2.2, 4.0), 0, 1))
        # Induce correlation: older properties → worse maintenance
        property_age = float(np.clip(rng.gamma(3.5, 6.0), 1, 80))
        maintenance = float(np.clip(maintenance - 0.15 * (property_age / 80.0), 0, 1))
        coverage = float(np.clip(rng.beta(4, 2.5), 0, 1))
        engagement = float(np.clip(rng.beta(3, 3), 0, 1))
        contactability = float(np.clip(0.55 * engagement + 0.45 * rng.random(), 0, 1))
        row = {
            "tenure_months": float(rng.integers(1, 240)),
            "property_age_years": property_age,
            "prior_claims_3y": float(prior_claims),
            "claim_severity_avg": float(np.clip(rng.beta(2, 5) + 0.1 * prior_claims, 0, 1)),
            "maintenance_score": maintenance,
            "safety_score": safety,
            "hazard_exposure": hazard,
            "coverage_adequacy": coverage,
            "engagement_score": engagement,
            "digital_affinity": float(np.clip(rng.beta(3, 2), 0, 1)),
            "recent_interventions_12m": float(rng.integers(0, 4)),
            "days_since_last_intervention": float(rng.integers(0, 400)),
            "region_risk_index": float(np.clip(rng.beta(2.5, 3.5), 0, 1)),
            "deductible_ratio": float(np.clip(rng.beta(2, 4), 0, 1)),
            "premium_band": float(np.clip(rng.beta(3, 3), 0, 1)),
            "fraud_signal_score": float(np.clip(rng.beta(1.5, 8), 0, 1)),
            "contactability": contactability,
        }
        baseline = _true_baseline_risk(row)
        customers.append(
            CustomerContext(
                customer_id=f"C{i:05d}",
                policy_id=f"P{i:05d}",
                line_of_business=lines[int(rng.integers(0, len(lines)))],
                tenure_months=int(row["tenure_months"]),
                property_age_years=row["property_age_years"],
                prior_claims_3y=prior_claims,
                claim_severity_avg=row["claim_severity_avg"],
                maintenance_score=row["maintenance_score"],
                safety_score=row["safety_score"],
                hazard_exposure=row["hazard_exposure"],
                coverage_adequacy=row["coverage_adequacy"],
                engagement_score=row["engagement_score"],
                digital_affinity=row["digital_affinity"],
                recent_interventions_12m=int(row["recent_interventions_12m"]),
                days_since_last_intervention=int(row["days_since_last_intervention"]),
                region_risk_index=row["region_risk_index"],
                deductible_ratio=row["deductible_ratio"],
                premium_band=row["premium_band"],
                fraud_signal_score=row["fraud_signal_score"],
                contactability=row["contactability"],
                true_baseline_risk=baseline,
                true_segment=_segment(baseline),
            )
        )
    return customers


def customer_feature_dict(c: CustomerContext) -> dict[str, float]:
    return {k: float(getattr(c, k)) for k in FEATURE_COLUMNS}


def _selection_propensity(intervention: InterventionType, row: dict[str, float]) -> float:
    """Historical ops propensity — correlates with risk & contactability (confounders)."""
    base_risk = _true_baseline_risk(row)
    cost = INTERVENTION_CATALOG[intervention]["cost"]
    # Higher risk + more contactable → more likely treated. Costly interventions rarer.
    logit = (
        -1.2
        + 2.4 * base_risk
        + 1.6 * row["contactability"]
        + 0.5 * row["prior_claims_3y"] / 3.0
        - 0.008 * cost
        + 0.3 * row["engagement_score"]
    )
    if intervention == InterventionType.FRAUD_AWARENESS_EDUCATION:
        logit += 1.5 * row["fraud_signal_score"]
    if intervention == InterventionType.INSPECTION:
        logit += 1.2 * row["hazard_exposure"]
    return float(np.clip(_sigmoid(logit), 0.02, 0.92))


def generate_historical_assignments(
    customers: list[CustomerContext] | None = None,
    n: int = 12000,
    seed: int = 42,
    randomized_fraction: float = 0.20,
) -> list[HistoricalAssignment]:
    """Generate observational + partially randomized historical interventions.

    ``randomized_fraction`` of assignments are true RCT arms (unconfounded).
    The rest follow ops selection — useful for demonstrating confounding bias.
    """
    rng = np.random.default_rng(seed)
    if customers is None:
        customers = generate_customers(max(n // 2, 1000), seed=seed)
    assignments: list[HistoricalAssignment] = []
    start = datetime(2023, 1, 1)

    for i in range(n):
        c = customers[int(rng.integers(0, len(customers)))]
        feats = customer_feature_dict(c)
        intervention = ACTIVE_INTERVENTIONS[int(rng.integers(0, len(ACTIVE_INTERVENTIONS)))]
        is_rct = bool(rng.random() < randomized_fraction)
        if is_rct:
            treated = bool(rng.random() < 0.5)
            propensity = 0.5
            selection = 0.5
        else:
            propensity = _selection_propensity(intervention, feats)
            treated = bool(rng.random() < propensity)
            selection = propensity

        cate = true_cate(intervention, feats)
        p_control = _true_baseline_risk(feats)
        p_outcome = p_control - cate if treated else p_control
        p_outcome = float(np.clip(p_outcome, 0.01, 0.95))
        loss = bool(rng.random() < p_outcome)
        severity = float(rng.lognormal(mean=7.5, sigma=0.8)) if loss else 0.0

        assignments.append(
            HistoricalAssignment(
                assignment_id=f"A{i:06d}",
                customer_id=c.customer_id,
                intervention=intervention,
                treated=treated,
                outcome_loss=loss,
                loss_severity=severity,
                propensity_score=propensity,
                assigned_at=start + timedelta(days=int(rng.integers(0, 700))),
                selection_score=selection,
                features=feats,
            )
        )
    return assignments


def assignments_to_frame(assignments: list[HistoricalAssignment]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for a in assignments:
        row = {
            "assignment_id": a.assignment_id,
            "customer_id": a.customer_id,
            "intervention": a.intervention.value,
            "treated": int(a.treated),
            "outcome_loss": int(a.outcome_loss),
            "loss_severity": a.loss_severity,
            "propensity_score": a.propensity_score,
            "selection_score": a.selection_score,
            "is_rct": float(abs(a.propensity_score - 0.5) < 1e-9),
        }
        row.update(a.features)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_experiment_dataset(
    n_customers: int = 5000,
    n_historical: int = 12000,
    seed: int = 42,
) -> tuple[list[CustomerContext], pd.DataFrame]:
    customers = generate_customers(n_customers, seed=seed)
    assignments = generate_historical_assignments(customers, n=n_historical, seed=seed + 7)
    return customers, assignments_to_frame(assignments)


def oracle_policy_value(
    customers: list[CustomerContext],
    policy_fn,
) -> float:
    """Expected risk reduction under a policy using ground-truth CATE (evaluation only)."""
    total = 0.0
    for c in customers:
        feats = customer_feature_dict(c)
        intervention = policy_fn(c)
        if isinstance(intervention, str):
            intervention = InterventionType(intervention)
        total += true_cate(intervention, feats)
    return total / max(len(customers), 1)
