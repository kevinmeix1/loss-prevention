from loss_prevention.agent.recommender import LossPreventionAgent
from loss_prevention.ranking.interventions import RankingConfig, rank_interventions
from loss_prevention.schemas import InterventionType


def test_rank_includes_all_interventions(pipeline):
    ranked = rank_interventions(pipeline.customers[0], pipeline.bundle)
    assert len(ranked) == len(InterventionType)
    assert ranked[0].rank == 1


def test_constraints_can_block(pipeline):
    c = pipeline.customers[0].model_copy(deep=True)
    c.days_since_last_intervention = 5
    c.recent_interventions_12m = 5
    cfg = RankingConfig(suppress_recent_days=90, max_interventions_per_year=3)
    ranked = rank_interventions(c, pipeline.bundle, cfg)
    active = [s for s in ranked if s.intervention != InterventionType.NO_INTERVENTION]
    assert all(not s.eligible for s in active)


def test_agent_recommendation_grounded(pipeline):
    agent = LossPreventionAgent(pipeline.bundle, pipeline.customers)
    rec = agent.recommend(pipeline.customers[5].customer_id)
    assert rec.recommended_intervention in InterventionType
    assert rec.risk.p_loss > 0
    assert rec.explanation_bullets
    assert rec.evidence
    # Numbers in evidence must come from uplift objects
    uplift_ev = [e for e in rec.evidence if e.get("type") == "uplift"]
    assert uplift_ev
    assert "cate" in uplift_ev[0]


def test_agent_compare_uses_models(pipeline):
    agent = LossPreventionAgent(pipeline.bundle, pipeline.customers)
    cid = pipeline.customers[2].customer_id
    cmp = agent.compare_interventions(
        cid, "inspection", "educational_guidance"
    )
    assert "delta_risk_reduction" in cmp
    assert cmp["winner"] in {"inspection", "educational_guidance"}


def test_counterfactual(pipeline):
    result = pipeline.counterfactual(
        pipeline.customers[0].customer_id, InterventionType.INSPECTION
    )
    assert 0 <= result.p_loss_with_intervention <= 1
    assert result.method
    # Arithmetic must be internally consistent (uplift control/treated)
    assert abs(
        (result.p_loss_baseline - result.p_loss_with_intervention)
        - result.expected_risk_reduction
    ) < 1e-6
    assert result.p_loss_predictive is not None
