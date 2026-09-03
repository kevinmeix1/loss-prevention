"""Loss-prevention recommendation agent.

Orchestrates tool calls; never invents treatment effects.
Optional LLM narrative is constrained to template grounding.
"""

from __future__ import annotations

import os
from typing import Any

from loss_prevention.agent.tools import AgentTools
from loss_prevention.explain.explainer import attach_explanation, build_explanation
from loss_prevention.models.bundle import ModelBundle
from loss_prevention.ranking.interventions import RankingConfig, rank_interventions
from loss_prevention.schemas import (
    INTERVENTION_CATALOG,
    CustomerContext,
    InterventionType,
    Recommendation,
)


class LossPreventionAgent:
    """Tool-using agent for intervention recommendation."""

    def __init__(
        self,
        bundle: ModelBundle,
        customers: list[CustomerContext] | dict[str, CustomerContext],
        ranking_cfg: RankingConfig | None = None,
        use_llm: bool = False,
    ):
        if isinstance(customers, list):
            self.customers = {c.customer_id: c for c in customers}
        else:
            self.customers = customers
        self.bundle = bundle
        self.cfg = ranking_cfg or RankingConfig()
        self.tools = AgentTools(bundle, self.customers)
        self.use_llm = use_llm and bool(os.getenv("OPENAI_API_KEY"))
        self.trace: list[dict[str, Any]] = []

    def _log(self, tool: str, args: dict[str, Any], result_preview: Any) -> None:
        self.trace.append({"tool": tool, "args": args, "preview": result_preview})

    def recommend(
        self,
        customer_id: str,
        method: str = "x_learner",
        explore: bool = False,
    ) -> Recommendation:
        self.trace = []
        customer = self.customers[customer_id]

        # 1. Inspect risk factors
        risk = self.bundle.predictive.predict_customer(customer)
        self._log("get_risk_factors", {"customer_id": customer_id}, risk.p_loss)

        # 2. Retrieve evidence / candidates
        evidence_pack = self.tools.get_evidence_summary(customer_id)
        self._log("get_evidence_summary", {"customer_id": customer_id}, "ok")

        # 3–5. Causal/risk tools + compare interventions
        ranked = rank_interventions(customer, self.bundle, self.cfg, method=method)
        self._log(
            "get_ranked_interventions",
            {"customer_id": customer_id, "method": method},
            [s.intervention.value for s in ranked[:3]],
        )

        eligible = [s for s in ranked if s.eligible]
        if explore and len(eligible) > 1 and self.cfg.explore_epsilon > 0:
            # Epsilon-greedy exploration among top-3 eligible
            import numpy as np

            rng = np.random.default_rng(abs(hash(customer_id)) % (2**32))
            if rng.random() < self.cfg.explore_epsilon:
                pick = eligible[int(rng.integers(0, min(3, len(eligible))))]
                # Move explored option to front while preserving scores
                eligible = [pick] + [s for s in eligible if s.intervention != pick.intervention]
                ranked = eligible + [s for s in ranked if not s.eligible]

        top = eligible[0] if eligible else ranked[0]
        narrative, bullets, tradeoffs, evidence = build_explanation(customer, risk, ranked)

        # Optional LLM polish — still cannot invent numbers
        if self.use_llm:
            narrative = self._llm_narrative(customer, risk, top, bullets) or narrative

        constraints = []
        for s in ranked:
            constraints.extend(s.ineligibility_reasons)
        constraints = sorted(set(constraints))

        rec = Recommendation(
            customer_id=customer.customer_id,
            policy_id=customer.policy_id,
            recommended_intervention=top.intervention,
            rank_list=ranked,
            risk=risk,
            explanation=narrative,
            explanation_bullets=bullets,
            evidence=evidence,
            tradeoffs=tradeoffs,
            model_versions={
                "predictive": self.bundle.predictive.model_version,
                "uplift": self.bundle.uplift.model_version,
                "method": method,
            },
            constraints_applied=constraints,
            alternatives=[
                s.intervention
                for s in eligible[1:4]
                if s.intervention != top.intervention
            ],
            counterfactual=self.tools.simulate_intervention(
                customer_id, top.intervention.value, method=method
            ),
        )
        # Ensure grounding
        return attach_explanation(rec, customer) if not bullets else rec

    def compare_interventions(
        self, customer_id: str, a: str, b: str, method: str = "x_learner"
    ) -> dict[str, Any]:
        sa = self.tools.simulate_intervention(customer_id, a, method=method)
        sb = self.tools.simulate_intervention(customer_id, b, method=method)
        self._log("compare_interventions", {"a": a, "b": b}, "ok")
        return {
            "a": sa,
            "b": sb,
            "winner": a
            if sa["net_utility"] >= sb["net_utility"]
            else b,
            "delta_risk_reduction": sa["expected_risk_reduction"] - sb["expected_risk_reduction"],
            "delta_net_utility": sa["net_utility"] - sb["net_utility"],
            "note": "Winner selected by model-derived net utility, not LLM judgment.",
        }

    def _llm_narrative(
        self,
        customer: CustomerContext,
        risk: Any,
        top: Any,
        bullets: list[str],
    ) -> str | None:
        """Optional narrative polish. Numbers must match tool outputs."""
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return None
        label = INTERVENTION_CATALOG[top.intervention]["label"]
        allowed = {
            "p_loss": risk.p_loss,
            "cate": top.uplift.cate if top.uplift else 0.0,
            "cost": top.cost,
            "burden": top.customer_burden,
            "intervention": label,
        }
        prompt = (
            "Rewrite the following grounded recommendation as a concise advisor note. "
            "You MUST use only these numbers and must not invent treatment effects:\n"
            f"{allowed}\nBullets:\n" + "\n".join(bullets)
        )
        try:
            client = OpenAI()
            resp = client.chat.completions.create(
                model=os.getenv("LOSS_PREVENTION_LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an insurance loss-prevention advisor. "
                            "Never invent probabilities, CATEs, costs, or confidence scores."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content
        except Exception:
            return None
