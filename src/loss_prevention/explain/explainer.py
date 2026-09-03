"""Grounded explanations — only cite model outputs and observed features."""

from __future__ import annotations

from loss_prevention.schemas import (
    INTERVENTION_CATALOG,
    CustomerContext,
    InterventionScore,
    Recommendation,
    RiskPrediction,
)


def build_explanation(
    customer: CustomerContext,
    risk: RiskPrediction,
    ranked: list[InterventionScore],
) -> tuple[str, list[str], list[str], list[dict]]:
    top = next(s for s in ranked if s.eligible)
    label = INTERVENTION_CATALOG[top.intervention]["label"]
    bullets: list[str] = []
    tradeoffs: list[str] = []
    evidence: list[dict] = []

    bullets.append(
        f"Predicted baseline loss probability is {risk.p_loss:.1%} "
        f"({risk.risk_segment.value} segment) from {risk.model_version}."
    )
    for f in risk.top_risk_factors[:3]:
        bullets.append(
            f"Risk factor `{f['feature']}`={f['value']:.2f} {f['direction']} risk "
            f"(importance={f['importance']:.3f})."
        )
        evidence.append({"type": "risk_factor", **f})

    if top.uplift:
        u = top.uplift
        bullets.append(
            f"Uplift ({u.method}): P(loss|control)={u.p_loss_control:.1%}, "
            f"P(loss|treated)={u.p_loss_treated:.1%}, "
            f"CATE={u.cate:.1%} absolute risk reduction, "
            f"expected benefit=${u.expected_benefit:,.0f}, confidence={u.confidence:.2f}."
        )
        evidence.append(
            {
                "type": "uplift",
                "intervention": top.intervention.value,
                "method": u.method,
                "p_loss_control": u.p_loss_control,
                "p_loss_treated": u.p_loss_treated,
                "cate": u.cate,
                "confidence": u.confidence,
            }
        )

    bullets.append(
        f"Recommended `{label}` ranks #{top.rank} with utility={top.utility:.3f}, "
        f"cost=${top.cost:.0f}, burden={top.customer_burden:.2f}, "
        f"feasibility={top.feasibility:.2f}."
    )

    # Tradeoffs vs next best eligible alternative
    alts = [s for s in ranked if s.eligible and s.intervention != top.intervention]
    if alts:
        alt = alts[0]
        alt_label = INTERVENTION_CATALOG[alt.intervention]["label"]
        tradeoffs.append(
            f"vs {alt_label}: Δutility={top.utility - alt.utility:.3f}, "
            f"ΔCATE={(top.uplift.cate if top.uplift else 0) - (alt.uplift.cate if alt.uplift else 0):.1%}, "
            f"Δcost=${top.cost - alt.cost:.0f}, "
            f"Δburden={top.customer_burden - alt.customer_burden:.2f}."
        )
        if top.cost > alt.cost and (top.uplift and alt.uplift) and top.uplift.cate <= alt.uplift.cate + 0.005:
            tradeoffs.append(
                f"{label} is costlier without a clearly larger treatment effect than {alt_label}."
            )

    if risk.p_loss > 0.35 and top.uplift and top.uplift.cate < 0.02:
        tradeoffs.append(
            "High predicted risk but modest uplift — prediction ≠ treatment effect; "
            "prefer interventions with stronger heterogeneous benefit."
        )

    narrative = (
        f"For customer {customer.customer_id} (policy {customer.policy_id}), "
        f"the system recommends **{label}**. "
        f"This is based on calibrated risk {risk.p_loss:.1%} and estimated "
        f"treatment effect from the uplift model — not LLM speculation."
    )
    return narrative, bullets, tradeoffs, evidence


def attach_explanation(
    rec: Recommendation,
    customer: CustomerContext,
) -> Recommendation:
    narrative, bullets, tradeoffs, evidence = build_explanation(
        customer, rec.risk, rec.rank_list
    )
    rec.explanation = narrative
    rec.explanation_bullets = bullets
    rec.tradeoffs = tradeoffs
    rec.evidence = evidence
    return rec
