"""Extended API contract tests for production UI endpoints."""

from fastapi.testclient import TestClient

from loss_prevention.api import app as api_module
from loss_prevention.pipeline import LossPreventionPipeline


def _client(small_bundle, small_customers):
    pipe = LossPreventionPipeline(bundle=small_bundle, customers=small_customers)
    if hasattr(api_module.get_pipeline, "cache_clear"):
        api_module.get_pipeline.cache_clear()
    api_module.get_pipeline = lambda: pipe  # type: ignore[assignment]
    return TestClient(api_module.app), pipe


def test_recommend_includes_agent_trace(small_bundle, small_customers):
    client, pipe = _client(small_bundle, small_customers)
    cid = pipe.customers[0].customer_id
    rec = client.get(f"/customers/{cid}/recommend").json()
    assert "agent_trace" in rec
    assert isinstance(rec["agent_trace"], list)
    assert rec["agent_trace"]
    assert "tool" in rec["agent_trace"][0]


def test_architecture_endpoint(small_bundle, small_customers):
    client, _ = _client(small_bundle, small_customers)
    arch = client.get("/architecture").json()
    assert "pipeline" in arch
    assert "agent_tools" in arch
    assert "get_risk_factors" in arch["agent_tools"]


def test_method_compare(small_bundle, small_customers):
    client, pipe = _client(small_bundle, small_customers)
    cid = pipe.customers[1].customer_id
    data = client.get(
        f"/customers/{cid}/method-compare",
        params={"intervention": "maintenance_recommendation"},
    ).json()
    assert "x_learner" in data["methods"]
    assert "t_learner" in data["methods"]
    assert "cate" in data["methods"]["x_learner"]


def test_compare_interventions(small_bundle, small_customers):
    client, pipe = _client(small_bundle, small_customers)
    cid = pipe.customers[2].customer_id
    data = client.get(
        f"/customers/{cid}/compare",
        params={"a": "inspection", "b": "educational_guidance"},
    ).json()
    assert data["winner"] in {"inspection", "educational_guidance"}


def test_unknown_customer_404(small_bundle, small_customers):
    client, _ = _client(small_bundle, small_customers)
    assert client.get("/customers/DOES_NOT_EXIST/risk").status_code == 404
    assert client.get("/customers/DOES_NOT_EXIST/recommend").status_code == 404


def test_invalid_intervention_400(small_bundle, small_customers):
    client, pipe = _client(small_bundle, small_customers)
    cid = pipe.customers[0].customer_id
    res = client.post(
        "/counterfactual",
        json={"customer_id": cid, "intervention": "not_a_real_intervention"},
    )
    assert res.status_code in {400, 422}


def test_static_assets_served(small_bundle, small_customers):
    client, _ = _client(small_bundle, small_customers)
    assert client.get("/").status_code == 200
    assert "Loss Prevention" in client.get("/").text
    assert client.get("/assets/app.css").status_code == 200
    assert client.get("/assets/app.js").status_code == 200
    css = client.get("/assets/app.css").text
    js = client.get("/assets/app.js").text
    assert "--accent" in css
    assert "loadRecommendation" in js
