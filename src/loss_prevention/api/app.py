"""REST API for risk, recommendations, counterfactuals, and research metrics."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from loss_prevention.pipeline import LossPreventionPipeline
from loss_prevention.schemas import CounterfactualRequest

STATIC_DIR = Path(__file__).resolve().parents[1] / "ui" / "static"

app = FastAPI(
    title="Loss Prevention Recommendation API",
    version="1.0.0",
    description="Predictive risk + uplift-based intervention recommendations (synthetic data).",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/")
def ui_home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@lru_cache(maxsize=1)
def get_pipeline() -> LossPreventionPipeline:
    pipe = LossPreventionPipeline()
    pipe.ensure_ready(n_customers=2000, n_historical=6000, seed=42)
    return pipe


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/customers")
def list_customers(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    pipe = get_pipeline()
    out = []
    for c in pipe.customers[:limit]:
        out.append(
            {
                "customer_id": c.customer_id,
                "policy_id": c.policy_id,
                "line_of_business": c.line_of_business.value,
                "prior_claims_3y": c.prior_claims_3y,
                "hazard_exposure": c.hazard_exposure,
                "maintenance_score": c.maintenance_score,
                "true_segment": c.true_segment.value if c.true_segment else None,
            }
        )
    return out


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str) -> dict[str, Any]:
    pipe = get_pipeline()
    try:
        c = pipe.get_customer(customer_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown customer {customer_id}") from exc
    return c.model_dump(exclude={"true_baseline_risk"})


@app.get("/customers/{customer_id}/risk")
def get_risk(customer_id: str) -> dict[str, Any]:
    pipe = get_pipeline()
    try:
        c = pipe.get_customer(customer_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return pipe.bundle.predictive.predict_customer(c).model_dump()


@app.get("/customers/{customer_id}/recommend")
def recommend(
    customer_id: str,
    method: str = Query("x_learner", pattern="^(s_learner|t_learner|x_learner|causal_forest)$"),
    explore: bool = False,
) -> dict[str, Any]:
    pipe = get_pipeline()
    try:
        pipe.ensure_ready()
        rec = pipe.agent.recommend(customer_id, method=method, explore=explore)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown customer {customer_id}") from exc
    payload = rec.model_dump()
    payload["agent_trace"] = pipe.agent.trace
    return payload


@app.post("/counterfactual")
def counterfactual(
    req: CounterfactualRequest,
    method: str = Query("x_learner", pattern="^(s_learner|t_learner|x_learner|causal_forest)$"),
) -> dict[str, Any]:
    pipe = get_pipeline()
    try:
        result = pipe.counterfactual(req.customer_id, req.intervention, method=method)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown customer {req.customer_id}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return result.model_dump()


@app.get("/customers/{customer_id}/compare")
def compare_interventions(
    customer_id: str,
    a: str = Query(...),
    b: str = Query(...),
    method: str = Query("x_learner", pattern="^(s_learner|t_learner|x_learner|causal_forest)$"),
) -> dict[str, Any]:
    pipe = get_pipeline()
    try:
        pipe.ensure_ready()
        return pipe.agent.compare_interventions(customer_id, a, b, method=method)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown customer {customer_id}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/customers/{customer_id}/method-compare")
def method_compare(
    customer_id: str,
    intervention: str = Query("maintenance_recommendation"),
) -> dict[str, Any]:
    """Compare S/T/X learner and causal forest CATE for one intervention."""
    from loss_prevention.data.generator import FEATURE_COLUMNS, customer_feature_dict
    from loss_prevention.schemas import InterventionType
    import numpy as np

    pipe = get_pipeline()
    try:
        c = pipe.get_customer(customer_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown customer {customer_id}") from exc
    assert pipe.bundle is not None
    feats = customer_feature_dict(c)
    iv = InterventionType(intervention)
    out: dict[str, Any] = {"customer_id": customer_id, "intervention": intervention, "methods": {}}
    for method in ("s_learner", "t_learner", "x_learner"):
        est = pipe.bundle.uplift.estimate(iv, feats, method=method)
        out["methods"][method] = {
            "cate": est.cate,
            "p_loss_control": est.p_loss_control,
            "p_loss_treated": est.p_loss_treated,
            "confidence": est.confidence,
        }
    if iv.value in pipe.bundle.causal_forests:
        x = np.array([[feats[col] for col in FEATURE_COLUMNS]])
        cate = float(pipe.bundle.causal_forests[iv.value].predict(x)[0])
        p0 = out["methods"]["t_learner"]["p_loss_control"]
        out["methods"]["causal_forest"] = {
            "cate": cate,
            "p_loss_control": p0,
            "p_loss_treated": float(max(0.0, min(1.0, p0 - cate))),
            "confidence": None,
        }
    return out


@app.get("/interventions")
def interventions() -> list[dict[str, Any]]:
    from loss_prevention.schemas import INTERVENTION_CATALOG

    return [{"intervention": k.value, **v} for k, v in INTERVENTION_CATALOG.items()]


@app.get("/architecture")
def architecture() -> dict[str, Any]:
    return {
        "pipeline": [
            "customer",
            "risk_model",
            "risk_factors",
            "candidate_interventions",
            "uplift_causal_model",
            "expected_impact",
            "constraint_filtering",
            "recommendation_ranking",
            "ai_explanation",
            "customer_outcome",
        ],
        "models": ["predictive_gbm", "s_learner", "t_learner", "x_learner", "causal_forest"],
        "agent_tools": [
            "get_customer_profile",
            "get_risk_factors",
            "get_candidate_interventions",
            "get_uplift_estimates",
            "get_ranked_interventions",
            "simulate_intervention",
            "get_evidence_summary",
        ],
        "responsible_ai": [
            "no_protected_attributes",
            "hard_eligibility_constraints",
            "grounded_explanations_only",
            "prediction_vs_uplift_separation",
        ],
    }


@app.get("/research/deploy")
def research_deploy() -> dict[str, Any]:
    from loss_prevention.evaluation.deploy import simulate_deployment

    return simulate_deployment(n_customers=1200, n_historical=4000, seed=42).model_dump()


@app.get("/research/experiments")
def research_experiments() -> list[dict[str, Any]]:
    from loss_prevention.evaluation.experiments import (
        experiment_selection_bias,
        experiment_treatment_heterogeneity,
    )

    reports = [
        experiment_selection_bias(42),
        experiment_treatment_heterogeneity(42),
    ]
    return [r.model_dump() for r in reports]


@app.get("/models/metrics")
def model_metrics() -> dict[str, Any]:
    pipe = get_pipeline()
    assert pipe.bundle is not None
    return pipe.bundle.train_metrics
