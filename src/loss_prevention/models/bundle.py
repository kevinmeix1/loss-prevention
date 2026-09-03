"""Train and persist the full model bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from loss_prevention.data.generator import FEATURE_COLUMNS, generate_experiment_dataset
from loss_prevention.models.causal_forest import CausalForest
from loss_prevention.models.predictive import PredictiveRiskModel
from loss_prevention.models.uplift import UpliftModelSuite
from loss_prevention.schemas import InterventionType


@dataclass
class ModelBundle:
    predictive: PredictiveRiskModel
    uplift: UpliftModelSuite
    causal_forests: dict[str, CausalForest] = field(default_factory=dict)
    train_metrics: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "model_bundle.joblib")

    @staticmethod
    def load(directory: Path) -> ModelBundle:
        return joblib.load(directory / "model_bundle.joblib")


def train_all_models(
    n_customers: int = 4000,
    n_historical: int = 10000,
    seed: int = 42,
    fit_causal_forest: bool = True,
) -> ModelBundle:
    customers, df = generate_experiment_dataset(
        n_customers=n_customers, n_historical=n_historical, seed=seed
    )
    predictive = PredictiveRiskModel(random_state=seed).fit(df)
    uplift = UpliftModelSuite(preferred_method="x_learner", random_state=seed).fit(df)

    forests: dict[str, CausalForest] = {}
    cf_metrics: dict[str, Any] = {}
    if fit_causal_forest:
        for intervention in InterventionType:
            if intervention == InterventionType.NO_INTERVENTION:
                continue
            sub = df[df["intervention"] == intervention.value]
            if len(sub) < 200:
                continue
            x = sub[FEATURE_COLUMNS].astype(float).values
            t = sub["treated"].astype(int).values
            y = sub["outcome_loss"].astype(int).values
            cf = CausalForest(
                n_trees=60, max_depth=4, min_leaf=35, random_state=seed
            ).fit(x, t, y)
            forests[intervention.value] = cf
            cate = cf.predict(x)
            cf_metrics[intervention.value] = {
                "mean_cate": float(np.mean(cate)),
                "std_cate": float(np.std(cate)),
            }

    # Compare methods on a small diagnostic slice
    method_compare = _compare_methods(df, uplift, forests)

    bundle = ModelBundle(
        predictive=predictive,
        uplift=uplift,
        causal_forests=forests,
        train_metrics={
            "predictive": predictive.metrics_,
            "uplift": {k: v.metrics_ for k, v in uplift.models.items()},
            "causal_forest": cf_metrics,
            "method_compare": method_compare,
            "n_customers": len(customers),
            "n_historical": len(df),
        },
    )
    return bundle


def _compare_methods(
    df: pd.DataFrame,
    uplift: UpliftModelSuite,
    forests: dict[str, CausalForest],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for intervention, model in uplift.models.items():
        sub = df[df["intervention"] == intervention].head(300)
        if len(sub) < 50:
            continue
        x = sub[FEATURE_COLUMNS].astype(float).values
        row = {
            "s_learner_mean": float(np.mean(model.cate(x, method="s_learner"))),
            "t_learner_mean": float(np.mean(model.cate(x, method="t_learner"))),
            "x_learner_mean": float(np.mean(model.cate(x, method="x_learner"))),
        }
        if intervention in forests:
            row["causal_forest_mean"] = float(np.mean(forests[intervention].predict(x)))
        out[intervention] = row
    return out
