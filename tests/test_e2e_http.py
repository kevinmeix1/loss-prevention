"""HTTP end-to-end user journeys against the FastAPI app."""

from fastapi.testclient import TestClient

from loss_prevention.api import app as api_module
from loss_prevention.pipeline import LossPreventionPipeline


def test_full_recommendation_journey(small_bundle, small_customers):
    pipe = LossPreventionPipeline(bundle=small_bundle, customers=small_customers)
    if hasattr(api_module.get_pipeline, "cache_clear"):
        api_module.get_pipeline.cache_clear()
    api_module.get_pipeline = lambda: pipe  # type: ignore[assignment]
    client = TestClient(api_module.app)

    # Open app
    home = client.get("/")
    assert home.status_code == 200
    assert 'id="loadBtn"' in home.text
    assert "/assets/app.js" in home.text

    # Select entity list
    customers = client.get("/customers?limit=10").json()
    assert len(customers) >= 1
    cid = customers[3]["customer_id"] if len(customers) > 3 else customers[0]["customer_id"]

    # Run analysis
    rec = client.get(f"/customers/{cid}/recommend?method=x_learner").json()
    assert rec["customer_id"] == cid
    assert rec["recommended_intervention"]
    assert rec["rank_list"]
    assert rec["evidence"]

    # Inspect result / change configuration / run again
    rec2 = client.get(f"/customers/{cid}/recommend?method=t_learner").json()
    assert rec2["model_versions"]["method"] == "t_learner"

    # Counterfactual simulator
    iv = rec["recommended_intervention"]
    cf = client.post(
        "/counterfactual?method=x_learner",
        json={"customer_id": cid, "intervention": iv},
    ).json()
    assert "expected_risk_reduction" in cf
    assert "caveats" in cf

    # Method compare panel
    mc = client.get(
        f"/customers/{cid}/method-compare",
        params={"intervention": "inspection"},
    ).json()
    assert "methods" in mc

    # Research lab
    deploy = client.get("/research/deploy").json()
    assert "predictive_auc" in deploy
    assert "policy_value" in deploy


def test_failure_unknown_customer_recovers(small_bundle, small_customers):
    pipe = LossPreventionPipeline(bundle=small_bundle, customers=small_customers)
    api_module.get_pipeline = lambda: pipe  # type: ignore[assignment]
    client = TestClient(api_module.app)
    bad = client.get("/customers/NOPE/recommend")
    assert bad.status_code == 404
    ok = client.get(f"/customers/{pipe.customers[0].customer_id}/recommend")
    assert ok.status_code == 200
