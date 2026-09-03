"""Agent edge cases: missing data, invalid args, grounding."""

import pytest

from loss_prevention.agent.recommender import LossPreventionAgent
from loss_prevention.schemas import InterventionType


def test_agent_trace_tools_called(pipeline):
    agent = LossPreventionAgent(pipeline.bundle, pipeline.customers)
    rec = agent.recommend(pipeline.customers[0].customer_id)
    tools = [t["tool"] for t in agent.trace]
    assert "get_risk_factors" in tools
    assert "get_evidence_summary" in tools
    assert "get_ranked_interventions" in tools
    assert rec.evidence


def test_agent_missing_customer(pipeline):
    agent = LossPreventionAgent(pipeline.bundle, pipeline.customers)
    with pytest.raises(KeyError):
        agent.recommend("NO_SUCH_CUSTOMER")


def test_agent_invalid_intervention_compare(pipeline):
    agent = LossPreventionAgent(pipeline.bundle, pipeline.customers)
    cid = pipeline.customers[0].customer_id
    with pytest.raises(ValueError):
        agent.compare_interventions(cid, "inspection", "not_real")


def test_recommendation_numbers_match_uplift(pipeline):
    agent = LossPreventionAgent(pipeline.bundle, pipeline.customers)
    cid = pipeline.customers[4].customer_id
    rec = agent.recommend(cid)
    top = next(s for s in rec.rank_list if s.intervention == rec.recommended_intervention)
    assert top.uplift is not None
    # Evidence uplift must match ranked recommendation numerics
    uplift_ev = [e for e in rec.evidence if e.get("type") == "uplift"]
    if uplift_ev:
        assert abs(uplift_ev[0]["cate"] - top.uplift.cate) < 1e-9


def test_no_intervention_always_eligible_utility_path(pipeline):
    agent = LossPreventionAgent(pipeline.bundle, pipeline.customers)
    c = pipeline.customers[0].model_copy(deep=True)
    c.days_since_last_intervention = 1
    c.recent_interventions_12m = 10
    agent.customers[c.customer_id] = c
    rec = agent.recommend(c.customer_id)
    assert rec.recommended_intervention == InterventionType.NO_INTERVENTION
