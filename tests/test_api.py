from fastapi.testclient import TestClient

from loss_prevention.api.app import app
from loss_prevention.pipeline import LossPreventionPipeline


def test_api_endpoints(small_bundle, small_customers):
    pipe = LossPreventionPipeline(bundle=small_bundle, customers=small_customers)
    app.dependency_overrides.clear()

    # Bypass lru_cache by patching the symbol used inside route handlers
    import loss_prevention.api.app as api_module

    api_module.get_pipeline.cache_clear()
    api_module.get_pipeline = lambda: pipe  # type: ignore[assignment]

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"

    customers = client.get("/customers?limit=5").json()
    assert len(customers) == 5
    cid = customers[0]["customer_id"]

    risk = client.get(f"/customers/{cid}/risk").json()
    assert "p_loss" in risk

    rec = client.get(f"/customers/{cid}/recommend").json()
    assert rec["recommended_intervention"]
    assert rec["rank_list"]

    cf = client.post(
        "/counterfactual",
        json={"customer_id": cid, "intervention": "maintenance_recommendation"},
    ).json()
    assert "expected_risk_reduction" in cf

    assert client.get("/interventions").status_code == 200
