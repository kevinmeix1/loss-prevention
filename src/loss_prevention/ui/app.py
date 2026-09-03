"""Loss Prevention Control Center — Streamlit UI."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow `streamlit run src/loss_prevention/ui/app.py`
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from loss_prevention.pipeline import LossPreventionPipeline  # noqa: E402
from loss_prevention.schemas import INTERVENTION_CATALOG, InterventionType  # noqa: E402

st.set_page_config(
    page_title="Loss Prevention Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', serif !important; letter-spacing: -0.02em; }
.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, #d7ebe3 0%, transparent 55%),
    radial-gradient(900px 500px at 100% 0%, #f3e6d3 0%, transparent 50%),
    linear-gradient(180deg, #f7f4ef 0%, #eef2f0 100%);
}
div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.55);
  border: 1px solid rgba(30,60,50,0.08);
  padding: 0.75rem 1rem;
  border-radius: 12px;
}
.hero {
  padding: 1.2rem 1.4rem;
  border-radius: 16px;
  background: linear-gradient(135deg, #1f3d34 0%, #2f5d4c 55%, #3d6b52 100%);
  color: #f4f7f5;
  margin-bottom: 1rem;
}
.hero h1 { color: #f4f7f5 !important; margin: 0; font-size: 1.8rem; }
.hero p { color: #c9ddd4; margin: 0.35rem 0 0 0; }
.card {
  background: rgba(255,255,255,0.7);
  border: 1px solid rgba(31,61,52,0.08);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.75rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_pipeline() -> LossPreventionPipeline:
    pipe = LossPreventionPipeline()
    pipe.ensure_ready(n_customers=2000, n_historical=6000, seed=42)
    return pipe


def main() -> None:
    pipe = load_pipeline()
    st.markdown(
        """
        <div class="hero">
          <h1>Loss Prevention Agent</h1>
          <p>Predict risk. Estimate treatment effects. Recommend the intervention that actually reduces loss.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Customer")
        ids = [c.customer_id for c in pipe.customers]
        customer_id = st.selectbox("Customer ID", ids, index=0)
        method = st.selectbox(
            "Uplift method",
            ["x_learner", "t_learner", "s_learner", "causal_forest"],
            index=0,
        )
        st.caption(
            "Prediction ≠ uplift. High-risk customers are not always high-benefit."
        )

    customer = pipe.get_customer(customer_id)
    rec = pipe.recommend(customer_id, method=method)
    risk = rec.risk

    tab_overview, tab_factors, tab_recs, tab_impact, tab_evidence, tab_cf, tab_research = st.tabs(
        [
            "Risk Overview",
            "Risk Factors",
            "Recommended Interventions",
            "Expected Impact",
            "Evidence",
            "Counterfactual Simulator",
            "Research",
        ]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P(future loss)", f"{risk.p_loss:.1%}")
        c2.metric("Risk segment", risk.risk_segment.value)
        c3.metric(
            "Recommended",
            INTERVENTION_CATALOG[rec.recommended_intervention]["label"],
        )
        top = next(s for s in rec.rank_list if s.eligible)
        c4.metric("Est. CATE", f"{(top.uplift.cate if top.uplift else 0):.1%}")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(rec.explanation)
        st.markdown("</div>", unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.write("**Customer context**")
            st.json(
                {
                    "policy_id": customer.policy_id,
                    "line": customer.line_of_business.value,
                    "tenure_months": customer.tenure_months,
                    "prior_claims_3y": customer.prior_claims_3y,
                    "hazard_exposure": customer.hazard_exposure,
                    "maintenance_score": customer.maintenance_score,
                    "safety_score": customer.safety_score,
                    "coverage_adequacy": customer.coverage_adequacy,
                }
            )
        with right:
            st.write("**Why prediction ≠ intervention effect**")
            st.info(
                "The risk model estimates P(loss | features). "
                "The uplift model estimates how that probability *changes* under each intervention. "
                "A high-risk customer may not respond to a given treatment."
            )

    with tab_factors:
        st.subheader("Top risk factors")
        factor_df = pd.DataFrame(risk.top_risk_factors)
        if not factor_df.empty:
            st.dataframe(factor_df, use_container_width=True)
            st.bar_chart(factor_df.set_index("feature")["importance"])
        st.caption(f"Model: {risk.model_version} (calibrated={risk.calibrated})")

    with tab_recs:
        st.subheader("Ranked interventions")
        rows = []
        for s in rec.rank_list:
            rows.append(
                {
                    "rank": s.rank,
                    "intervention": INTERVENTION_CATALOG[s.intervention]["label"],
                    "eligible": s.eligible,
                    "CATE": s.uplift.cate if s.uplift else 0.0,
                    "expected_benefit_$": s.expected_benefit,
                    "confidence": s.confidence,
                    "cost": s.cost,
                    "burden": s.customer_burden,
                    "feasibility": s.feasibility,
                    "utility": s.utility if s.eligible else None,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.write("**Trade-offs**")
        for t in rec.tradeoffs:
            st.write(f"- {t}")
        if rec.constraints_applied:
            st.write("**Constraints applied**")
            for c in rec.constraints_applied:
                st.write(f"- {c}")

    with tab_impact:
        st.subheader("Expected impact of recommendation")
        if rec.counterfactual:
            cf = rec.counterfactual
            a, b, c, d = st.columns(4)
            a.metric("Baseline P(loss)", f"{cf['p_loss_baseline']:.1%}")
            b.metric("With intervention", f"{cf['p_loss_with_intervention']:.1%}")
            c.metric("Risk reduction", f"{cf['expected_risk_reduction']:.1%}")
            d.metric("Net utility", f"{cf['net_utility']:.1f}")
            st.write(
                f"Expected severity reduction: **${cf['expected_severity_reduction']:,.0f}** "
                f"· cost ${cf['cost']:.0f} · burden {cf['burden']:.2f} · "
                f"confidence {cf['confidence']:.2f} ({cf['method']})"
            )

    with tab_evidence:
        st.subheader("Grounded evidence")
        for bullet in rec.explanation_bullets:
            st.write(f"- {bullet}")
        st.write("**Structured evidence objects**")
        st.json(rec.evidence)
        st.caption("All numbers come from model tools — the agent does not invent treatment effects.")

    with tab_cf:
        st.subheader("What if we apply intervention X?")
        options = {INTERVENTION_CATALOG[i]["label"]: i for i in InterventionType}
        choice = st.selectbox("Intervention", list(options.keys()))
        if st.button("Simulate", type="primary"):
            result = pipe.counterfactual(customer_id, options[choice], method=method)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Baseline P(loss)", f"{result.p_loss_baseline:.1%}")
            m2.metric("Counterfactual P(loss)", f"{result.p_loss_with_intervention:.1%}")
            m3.metric("Δ risk", f"{result.expected_risk_reduction:.1%}")
            m4.metric("Net utility", f"{result.net_utility:.1f}")
            st.write(
                f"Benefit ${result.expected_severity_reduction:,.0f} · "
                f"cost ${result.cost:.0f} · burden {result.burden:.2f} · "
                f"confidence {result.confidence:.2f}"
            )
            for cave in result.caveats:
                st.caption(cave)

            # Compare all interventions chart
            sims = []
            for iv in InterventionType:
                r = pipe.counterfactual(customer_id, iv, method=method)
                sims.append(
                    {
                        "intervention": INTERVENTION_CATALOG[iv]["label"],
                        "risk_reduction": r.expected_risk_reduction,
                        "net_utility": r.net_utility,
                    }
                )
            sim_df = pd.DataFrame(sims).set_index("intervention")
            st.bar_chart(sim_df["risk_reduction"])

    with tab_research:
        st.subheader("Research experiments")
        st.write(
            "These demos investigate selection bias, confounding, treatment heterogeneity, "
            "and policy optimization on synthetic data."
        )
        if st.button("Run research suite (may take a minute)"):
            from loss_prevention.evaluation.deploy import simulate_deployment
            from loss_prevention.evaluation.experiments import (
                experiment_policy_optimization,
                experiment_selection_bias,
            )

            with st.spinner("Running experiments..."):
                bias = experiment_selection_bias(42)
                policy = experiment_policy_optimization(42)
                deploy = simulate_deployment(1200, 4000, 42)
            st.write(f"**{bias.name}**: {bias.summary}")
            for t in bias.takeaways:
                st.write(f"- {t}")
            st.dataframe(pd.DataFrame(bias.metrics["by_intervention"]), use_container_width=True)
            st.write(f"**{policy.name}**: {policy.summary}")
            st.json(policy.metrics)
            st.write("**Simulated deployment metrics**")
            st.json(deploy.model_dump())


if __name__ == "__main__":
    main()
