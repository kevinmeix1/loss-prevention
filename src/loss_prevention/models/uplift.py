"""Uplift / CATE estimation via meta-learners.

Core distinction taught here:

    Predictive model:  P(Y=1 | X)           — who is at risk?
    Uplift model:      P(Y=1 | X, T=0) - P(Y=1 | X, T=1)  — who benefits?

High risk ≠ high treatment effect. A customer can be high-risk yet unresponsive
to a given intervention (or helped more by a cheaper alternative).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from loss_prevention.data.generator import FEATURE_COLUMNS
from loss_prevention.schemas import InterventionType, UpliftEstimate

MetaLearner = Literal["s_learner", "t_learner", "x_learner"]


def _gb_clf(seed: int = 42) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.08,
        random_state=seed,
    )


def _gb_reg(seed: int = 42) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.08,
        random_state=seed,
    )


@dataclass
class InterventionUpliftModel:
    """Per-intervention uplift model suite (S / T / X learners + optional CF)."""

    intervention: InterventionType
    preferred_method: MetaLearner = "x_learner"
    random_state: int = 42
    _s_model: Any = field(default=None, repr=False)
    _t0: Any = field(default=None, repr=False)
    _t1: Any = field(default=None, repr=False)
    _x_tau: Any = field(default=None, repr=False)
    _propensity: Any = field(default=None, repr=False)
    metrics_: dict[str, float] = field(default_factory=dict)
    n_treated_: int = 0
    n_control_: int = 0

    def fit(self, df: pd.DataFrame) -> InterventionUpliftModel:
        sub = df[df["intervention"] == self.intervention.value].copy()
        if len(sub) < 80:
            # Fall back to all rows labeled with this intervention or empty-safe stub
            sub = df.copy()
            sub["intervention"] = self.intervention.value

        x = sub[FEATURE_COLUMNS].astype(float).values
        t = sub["treated"].astype(int).values
        y = sub["outcome_loss"].astype(int).values
        self.n_treated_ = int(t.sum())
        self.n_control_ = int((1 - t).sum())

        # --- S-learner: single model with treatment indicator ---
        x_s = np.column_stack([x, t])
        self._s_model = _gb_clf(self.random_state)
        self._s_model.fit(x_s, y)

        # --- T-learner: separate outcome models ---
        self._t0 = _gb_clf(self.random_state)
        self._t1 = _gb_clf(self.random_state + 1)
        if self.n_control_ >= 20:
            self._t0.fit(x[t == 0], y[t == 0])
        else:
            self._t0.fit(x, y)
        if self.n_treated_ >= 20:
            self._t1.fit(x[t == 1], y[t == 1])
        else:
            self._t1.fit(x, y)

        # --- Propensity ---
        self._propensity = _gb_clf(self.random_state + 2)
        self._propensity.fit(x, t)

        # --- X-learner ---
        mu0 = self._safe_proba(self._t0, x)
        mu1 = self._safe_proba(self._t1, x)
        d1 = mu0[t == 1] - y[t == 1]  # imputed treatment effect on treated
        d0 = y[t == 0] - mu1[t == 0]  # imputed on control
        tau1 = _gb_reg(self.random_state + 3)
        tau0 = _gb_reg(self.random_state + 4)
        if len(d1) >= 20:
            tau1.fit(x[t == 1], d1)
        else:
            tau1.fit(x, mu0 - mu1)
        if len(d0) >= 20:
            tau0.fit(x[t == 0], d0)
        else:
            tau0.fit(x, mu0 - mu1)
        # Store both; combine at predict with propensity
        self._x_tau = (tau0, tau1)

        # Simple diagnostics on held-in sample
        cate = self.cate(x, method="x_learner")
        self.metrics_ = {
            "n": float(len(sub)),
            "n_treated": float(self.n_treated_),
            "n_control": float(self.n_control_),
            "mean_cate": float(np.mean(cate)),
            "std_cate": float(np.std(cate)),
            "frac_positive_cate": float(np.mean(cate > 0)),
        }
        return self

    @staticmethod
    def _safe_proba(model: Any, x: np.ndarray) -> np.ndarray:
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            # Only one class seen during fit
            cls = model.classes_[0]
            return np.full(len(x), float(cls))
        return proba[:, 1]

    def p_loss(self, x: np.ndarray, treated: bool, method: MetaLearner | None = None) -> np.ndarray:
        method = method or self.preferred_method
        if method == "s_learner":
            t = np.full((len(x), 1), 1.0 if treated else 0.0)
            return self._safe_proba(self._s_model, np.column_stack([x, t]))
        if method == "t_learner":
            model = self._t1 if treated else self._t0
            return self._safe_proba(model, x)
        # x_learner: reconstruct from control + cate
        p0 = self._safe_proba(self._t0, x)
        cate = self.cate(x, method="x_learner")
        if treated:
            return np.clip(p0 - cate, 0.0, 1.0)
        return p0

    def cate(self, x: np.ndarray, method: MetaLearner | None = None) -> np.ndarray:
        """Estimated absolute risk reduction: P(Y|T=0) - P(Y|T=1)."""
        method = method or self.preferred_method
        if method == "s_learner":
            p0 = self.p_loss(x, treated=False, method="s_learner")
            p1 = self.p_loss(x, treated=True, method="s_learner")
            return p0 - p1
        if method == "t_learner":
            p0 = self._safe_proba(self._t0, x)
            p1 = self._safe_proba(self._t1, x)
            return p0 - p1
        # X-learner
        tau0, tau1 = self._x_tau
        e = np.clip(self._safe_proba(self._propensity, x), 0.05, 0.95)
        return e * tau0.predict(x) + (1 - e) * tau1.predict(x)

    def estimate_row(
        self,
        features: dict[str, float],
        method: MetaLearner | None = None,
        expected_severity: float = 5000.0,
    ) -> UpliftEstimate:
        method = method or self.preferred_method
        x = np.array([[float(features[c]) for c in FEATURE_COLUMNS]])
        p0 = float(self.p_loss(x, treated=False, method=method)[0])
        p1 = float(self.p_loss(x, treated=True, method=method)[0])
        cate = float(p0 - p1)
        # Confidence: higher with more data and when propensity not extreme
        e = float(np.clip(self._safe_proba(self._propensity, x)[0], 0.05, 0.95))
        n_eff = min(self.n_treated_, self.n_control_)
        conf = float(
            np.clip(
                0.25
                + 0.45 * (n_eff / (n_eff + 80))
                + 0.20 * (1.0 - abs(e - 0.5) * 2)
                + 0.10 * (1.0 if abs(cate) > 0.01 else 0.3),
                0.05,
                0.95,
            )
        )
        return UpliftEstimate(
            intervention=self.intervention,
            p_loss_control=p0,
            p_loss_treated=p1,
            cate=cate,
            expected_benefit=cate * expected_severity,
            confidence=conf,
            method=method,
            heterogeneous=abs(cate - float(self.metrics_.get("mean_cate", cate))) > 0.02,
        )


@dataclass
class UpliftModelSuite:
    """Collection of per-intervention uplift models."""

    preferred_method: MetaLearner = "x_learner"
    random_state: int = 42
    models: dict[str, InterventionUpliftModel] = field(default_factory=dict)
    model_version: str = "uplift-meta-v1"

    def fit(self, df: pd.DataFrame) -> UpliftModelSuite:
        for intervention in InterventionType:
            if intervention == InterventionType.NO_INTERVENTION:
                continue
            m = InterventionUpliftModel(
                intervention=intervention,
                preferred_method=self.preferred_method,
                random_state=self.random_state,
            )
            m.fit(df)
            self.models[intervention.value] = m
        return self

    def estimate(
        self,
        intervention: InterventionType,
        features: dict[str, float],
        method: MetaLearner | None = None,
        expected_severity: float = 5000.0,
    ) -> UpliftEstimate:
        if intervention == InterventionType.NO_INTERVENTION:
            x_p = 0.0
            # baseline risk from any fitted t0 if available
            if self.models:
                any_m = next(iter(self.models.values()))
                x = np.array([[float(features[c]) for c in FEATURE_COLUMNS]])
                x_p = float(any_m.p_loss(x, treated=False, method="t_learner")[0])
            return UpliftEstimate(
                intervention=intervention,
                p_loss_control=x_p,
                p_loss_treated=x_p,
                cate=0.0,
                expected_benefit=0.0,
                confidence=0.99,
                method="identity",
            )
        return self.models[intervention.value].estimate_row(
            features, method=method, expected_severity=expected_severity
        )
