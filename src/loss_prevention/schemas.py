"""Domain schemas for loss prevention, uplift, and recommendations.

Protected characteristics are intentionally excluded from features and decisions.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LineOfBusiness(str, Enum):
    HOME = "home"
    AUTO = "auto"
    COMMERCIAL = "commercial"
    RENTERS = "renters"


class RiskSegment(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class InterventionType(str, Enum):
    EDUCATIONAL_GUIDANCE = "educational_guidance"
    SAFETY_ASSESSMENT = "safety_assessment"
    MAINTENANCE_RECOMMENDATION = "maintenance_recommendation"
    COVERAGE_REVIEW = "coverage_review"
    FRAUD_AWARENESS_EDUCATION = "fraud_awareness_education"
    CLAIMS_PREVENTION_GUIDANCE = "claims_prevention_guidance"
    INSPECTION = "inspection"
    SPECIALIST_SUPPORT = "specialist_support"
    NO_INTERVENTION = "no_intervention"


INTERVENTION_CATALOG: dict[InterventionType, dict[str, Any]] = {
    InterventionType.EDUCATIONAL_GUIDANCE: {
        "label": "Educational guidance",
        "cost": 15.0,
        "burden": 0.15,
        "channels": ["email", "app"],
        "targets": ["behavior", "awareness"],
    },
    InterventionType.SAFETY_ASSESSMENT: {
        "label": "Safety assessment",
        "cost": 85.0,
        "burden": 0.45,
        "channels": ["phone", "virtual"],
        "targets": ["hazards", "safety"],
    },
    InterventionType.MAINTENANCE_RECOMMENDATION: {
        "label": "Maintenance recommendation",
        "cost": 25.0,
        "burden": 0.25,
        "channels": ["email", "app"],
        "targets": ["property_condition", "wear"],
    },
    InterventionType.COVERAGE_REVIEW: {
        "label": "Coverage review",
        "cost": 40.0,
        "burden": 0.30,
        "channels": ["phone", "agent"],
        "targets": ["coverage_gaps", "underinsurance"],
    },
    InterventionType.FRAUD_AWARENESS_EDUCATION: {
        "label": "Fraud-awareness education",
        "cost": 12.0,
        "burden": 0.10,
        "channels": ["email"],
        "targets": ["fraud_exposure"],
    },
    InterventionType.CLAIMS_PREVENTION_GUIDANCE: {
        "label": "Claims-prevention guidance",
        "cost": 20.0,
        "burden": 0.20,
        "channels": ["email", "app"],
        "targets": ["claims_history", "recurrence"],
    },
    InterventionType.INSPECTION: {
        "label": "Inspection",
        "cost": 180.0,
        "burden": 0.70,
        "channels": ["field"],
        "targets": ["property_condition", "hazards"],
    },
    InterventionType.SPECIALIST_SUPPORT: {
        "label": "Specialist support",
        "cost": 250.0,
        "burden": 0.80,
        "channels": ["phone", "field"],
        "targets": ["complex_risk", "high_severity"],
    },
    InterventionType.NO_INTERVENTION: {
        "label": "No intervention",
        "cost": 0.0,
        "burden": 0.0,
        "channels": [],
        "targets": [],
    },
}


class CustomerContext(BaseModel):
    customer_id: str
    policy_id: str
    line_of_business: LineOfBusiness
    tenure_months: int
    property_age_years: float
    prior_claims_3y: int
    claim_severity_avg: float
    maintenance_score: float = Field(ge=0, le=1)
    safety_score: float = Field(ge=0, le=1)
    hazard_exposure: float = Field(ge=0, le=1)
    coverage_adequacy: float = Field(ge=0, le=1)
    engagement_score: float = Field(ge=0, le=1)
    digital_affinity: float = Field(ge=0, le=1)
    recent_interventions_12m: int = 0
    days_since_last_intervention: int = 999
    region_risk_index: float = Field(ge=0, le=1)
    deductible_ratio: float = Field(ge=0, le=1)
    premium_band: float = Field(ge=0, le=1)
    fraud_signal_score: float = Field(ge=0, le=1)
    # Confounder observed by ops but not always adjusted for in naive analyses
    contactability: float = Field(ge=0, le=1)
    # Latent true propensity used only for synthetic DGP / evaluation oracle
    true_baseline_risk: float | None = None
    true_segment: RiskSegment | None = None


class HistoricalAssignment(BaseModel):
    assignment_id: str
    customer_id: str
    intervention: InterventionType
    treated: bool
    outcome_loss: bool
    loss_severity: float
    propensity_score: float
    assigned_at: datetime
    # Observational selection intensity used by ops historically
    selection_score: float
    features: dict[str, float]


class RiskPrediction(BaseModel):
    customer_id: str
    p_loss: float
    risk_segment: RiskSegment
    top_risk_factors: list[dict[str, Any]]
    model_version: str
    calibrated: bool = True


class UpliftEstimate(BaseModel):
    intervention: InterventionType
    p_loss_control: float
    p_loss_treated: float
    cate: float  # control - treated  (positive = risk reduction)
    expected_benefit: float
    confidence: float
    method: str
    heterogeneous: bool = False


class InterventionScore(BaseModel):
    intervention: InterventionType
    expected_benefit: float
    confidence: float
    cost: float
    customer_burden: float
    feasibility: float
    utility: float
    rank: int
    eligible: bool
    ineligibility_reasons: list[str] = Field(default_factory=list)
    uplift: UpliftEstimate | None = None


class Recommendation(BaseModel):
    customer_id: str
    policy_id: str
    recommended_intervention: InterventionType
    rank_list: list[InterventionScore]
    risk: RiskPrediction
    explanation: str
    explanation_bullets: list[str]
    evidence: list[dict[str, Any]]
    tradeoffs: list[str]
    model_versions: dict[str, str]
    constraints_applied: list[str]
    alternatives: list[InterventionType]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    counterfactual: dict[str, Any] | None = None


class CounterfactualRequest(BaseModel):
    customer_id: str
    intervention: InterventionType


class CounterfactualResult(BaseModel):
    customer_id: str
    intervention: InterventionType
    # Uplift-consistent control probability used for CATE arithmetic
    p_loss_baseline: float
    p_loss_with_intervention: float
    expected_risk_reduction: float
    expected_severity_reduction: float
    cost: float
    burden: float
    net_utility: float
    confidence: float
    method: str
    caveats: list[str]
    # Separate predictive baseline (may differ from uplift control — intentional)
    p_loss_predictive: float | None = None


class ExperimentReport(BaseModel):
    name: str
    summary: str
    metrics: dict[str, Any]
    figures: dict[str, Any] = Field(default_factory=dict)
    takeaways: list[str]


class DeploymentMetrics(BaseModel):
    n_customers: int
    predictive_auc: float
    predictive_brier: float
    uplift_qini: float
    uplift_auuc: float
    calibration_ece: float
    mean_burden: float
    intervention_entropy: float
    recommendation_quality: float
    policy_value: float
    naive_ate_bias: float
