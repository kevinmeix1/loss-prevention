"""Deterministic tools the agent may call. Numerical effects come only from models."""

from __future__ import annotations

from typing import Any

from loss_prevention.data.generator import customer_feature_dict
from loss_prevention.models.bundle import ModelBundle
from loss_prevention.ranking.interventions import RankingConfig, rank_interventions
from loss_prevention.schemas import (
    INTERVENTION_CATALOG,
    CustomerContext,
    InterventionType,
)


class AgentTools:
    def __init__(self, bundle: ModelBundle, customers: dict[str, CustomerContext]):
        self.bundle = bundle
        self.customers = customers
        self.cfg = RankingConfig()

    def get_customer_profile(self, customer_id: str) -> dict[str, Any]:
        c = self.customers[customer_id]
        return c.model_dump(exclude={"true_baseline_risk", "true_segment"})

    def get_risk_factors(self, customer_id: str) -> dict[str, Any]:
        c = self.customers[customer_id]
        risk = self.bundle.predictive.predict_customer(c)
        return risk.model_dump()

    def get_candidate_interventions(self, customer_id: str) -> list[dict[str, Any]]:
        return [
            {
                "intervention": i.value,
                **{k: v for k, v in INTERVENTION_CATALOG[i].items()},
            }
            for i in InterventionType
        ]

    def get_uplift_estimates(
        self, customer_id: str, method: str = "x_learner"
    ) -> list[dict[str, Any]]:
        c = self.customers[customer_id]
        feats = customer_feature_dict(c)
        out = []
        for intervention in InterventionType:
            est = self.bundle.uplift.estimate(
                intervention, feats, method=method if method != "causal_forest" else "x_learner"
            )
            if method == "causal_forest" and intervention.value in self.bundle.causal_forests:
                from loss_prevention.data.generator import FEATURE_COLUMNS
                import numpy as np

                x = np.array([[feats[col] for col in FEATURE_COLUMNS]])
                cate = float(self.bundle.causal_forests[intervention.value].predict(x)[0])
                est.cate = cate
                est.p_loss_treated = float(max(0.0, min(1.0, est.p_loss_control - cate)))
                est.expected_benefit = cate * self.cfg.expected_severity
                est.method = "causal_forest"
            out.append(est.model_dump())
        return out

    def get_ranked_interventions(
        self, customer_id: str, method: str = "x_learner"
    ) -> list[dict[str, Any]]:
        c = self.customers[customer_id]
        ranked = rank_interventions(c, self.bundle, self.cfg, method=method)
        return [r.model_dump() for r in ranked]

    def simulate_intervention(
        self, customer_id: str, intervention: str, method: str = "x_learner"
    ) -> dict[str, Any]:
        c = self.customers[customer_id]
        iv = InterventionType(intervention)
        feats = customer_feature_dict(c)
        risk = self.bundle.predictive.predict_customer(c)
        uplift = self.bundle.uplift.estimate(iv, feats, method=method)
        if method == "causal_forest" and iv.value in self.bundle.causal_forests:
            from loss_prevention.data.generator import FEATURE_COLUMNS
            import numpy as np

            x = np.array([[feats[col] for col in FEATURE_COLUMNS]])
            cate = float(self.bundle.causal_forests[iv.value].predict(x)[0])
            uplift.cate = cate
            uplift.p_loss_treated = float(max(0.0, min(1.0, uplift.p_loss_control - cate)))
            uplift.expected_benefit = cate * self.cfg.expected_severity
            uplift.method = "causal_forest"
        meta = INTERVENTION_CATALOG[iv]
        # Keep counterfactual arithmetic internally consistent:
        # baseline = uplift P(Y|T=0), treated = uplift P(Y|T=1), delta = CATE.
        p0 = float(uplift.p_loss_control)
        p1 = float(uplift.p_loss_treated) if iv != InterventionType.NO_INTERVENTION else p0
        cate = float(p0 - p1)
        benefit = cate * self.cfg.expected_severity
        net = benefit - meta["cost"] - 200.0 * meta["burden"]
        return {
            "customer_id": customer_id,
            "intervention": iv.value,
            "p_loss_baseline": p0,
            "p_loss_with_intervention": p1,
            "expected_risk_reduction": cate,
            "expected_severity_reduction": benefit,
            "cost": meta["cost"],
            "burden": meta["burden"],
            "net_utility": net,
            "confidence": uplift.confidence,
            "method": uplift.method,
            "p_loss_predictive": risk.p_loss,
            "caveats": [
                "Counterfactual Δ uses uplift P(loss|control) − P(loss|treated), not the predictive score.",
                "Predictive P(loss|X) is shown separately because prediction ≠ treatment effect.",
                "Observational history includes confounding; prefer RCT-calibrated estimates when available.",
                "Individual outcomes remain stochastic even with positive average treatment effect.",
            ],
        }

    def get_evidence_summary(self, customer_id: str) -> dict[str, Any]:
        ranked = self.get_ranked_interventions(customer_id)
        risk = self.get_risk_factors(customer_id)
        return {
            "risk": risk,
            "top3": ranked[:3],
            "ineligible": [r for r in ranked if not r["eligible"]],
            "note": "All numeric fields originate from PredictiveRiskModel / UpliftModelSuite.",
        }
