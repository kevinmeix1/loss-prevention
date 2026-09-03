"""Predictive model for P(future loss).

This answers: "Who is at risk?"
It does NOT answer: "Who will benefit from intervention X?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from loss_prevention.data.generator import FEATURE_COLUMNS
from loss_prevention.schemas import CustomerContext, RiskPrediction, RiskSegment


def _segment(p: float) -> RiskSegment:
    if p >= 0.45:
        return RiskSegment.CRITICAL
    if p >= 0.28:
        return RiskSegment.HIGH
    if p >= 0.15:
        return RiskSegment.MODERATE
    return RiskSegment.LOW


@dataclass
class PredictiveRiskModel:
    """Calibrated gradient boosting classifier for loss probability."""

    model_version: str = "risk-gbm-v1"
    n_estimators: int = 120
    max_depth: int = 6
    random_state: int = 42
    _clf: Any = field(default=None, repr=False)
    metrics_: dict[str, float] = field(default_factory=dict)
    feature_importances_: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, outcome_col: str = "outcome_loss") -> PredictiveRiskModel:
        # Train on control / baseline outcomes where possible to avoid treatment contamination
        train_df = df.copy()
        if "treated" in train_df.columns:
            # Prefer untreated rows for baseline risk; fall back to all if sparse
            untreated = train_df[train_df["treated"] == 0]
            if len(untreated) >= 200:
                train_df = untreated

        x = train_df[FEATURE_COLUMNS].astype(float).values
        y = train_df[outcome_col].astype(int).values
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=0.25, random_state=self.random_state, stratify=y
        )
        base = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.06,
            random_state=self.random_state,
        )
        self._clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self._clf.fit(x_tr, y_tr)
        proba = self._clf.predict_proba(x_te)[:, 1]
        self.metrics_ = {
            "auc": float(roc_auc_score(y_te, proba)),
            "brier": float(brier_score_loss(y_te, proba)),
            "log_loss": float(log_loss(y_te, proba)),
            "n_train": float(len(y_tr)),
            "n_test": float(len(y_te)),
            "base_rate": float(y.mean()),
        }
        # Importances from an uncalibrated fit for interpretability
        base.fit(x_tr, y_tr)
        self.feature_importances_ = {
            f: float(v) for f, v in zip(FEATURE_COLUMNS, base.feature_importances_, strict=True)
        }
        return self

    def predict_proba_matrix(self, x: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Model not fitted")
        return self._clf.predict_proba(x)[:, 1]

    def predict_customer(self, customer: CustomerContext) -> RiskPrediction:
        x = np.array([[float(getattr(customer, c)) for c in FEATURE_COLUMNS]])
        p = float(self.predict_proba_matrix(x)[0])
        imps = sorted(self.feature_importances_.items(), key=lambda kv: -kv[1])
        factors = []
        for name, importance in imps[:6]:
            value = float(getattr(customer, name))
            direction = "elevates" if _risk_direction(name, value) else "mitigates"
            factors.append(
                {
                    "feature": name,
                    "value": value,
                    "importance": importance,
                    "direction": direction,
                }
            )
        return RiskPrediction(
            customer_id=customer.customer_id,
            p_loss=p,
            risk_segment=_segment(p),
            top_risk_factors=factors,
            model_version=self.model_version,
            calibrated=True,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> PredictiveRiskModel:
        return joblib.load(path)


def _risk_direction(feature: str, value: float) -> bool:
    high_bad = {
        "prior_claims_3y",
        "claim_severity_avg",
        "hazard_exposure",
        "region_risk_index",
        "fraud_signal_score",
        "property_age_years",
    }
    low_bad = {
        "maintenance_score",
        "safety_score",
        "coverage_adequacy",
        "engagement_score",
    }
    if feature in high_bad:
        return value >= 0.4 if feature != "prior_claims_3y" else value >= 1
    if feature in low_bad:
        return value <= 0.45
    return False
