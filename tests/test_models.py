import numpy as np

from loss_prevention.data.generator import FEATURE_COLUMNS
from loss_prevention.models.causal_forest import CausalForest
from loss_prevention.schemas import InterventionType


def test_predictive_metrics(small_bundle):
    m = small_bundle.predictive.metrics_
    assert m["auc"] > 0.55
    assert 0 <= m["brier"] <= 1


def test_predict_customer(small_bundle, small_customers):
    risk = small_bundle.predictive.predict_customer(small_customers[0])
    assert 0 < risk.p_loss < 1
    assert risk.top_risk_factors


def test_uplift_methods_run(small_bundle, small_customers):
    c = small_customers[10]
    feats = {k: float(getattr(c, k)) for k in FEATURE_COLUMNS}
    for method in ("s_learner", "t_learner", "x_learner"):
        est = small_bundle.uplift.estimate(
            InterventionType.MAINTENANCE_RECOMMENDATION, feats, method=method
        )
        assert est.method == method
        assert -0.5 <= est.cate <= 0.5


def test_no_intervention_zero_cate(small_bundle, small_customers):
    feats = {k: float(getattr(small_customers[0], k)) for k in FEATURE_COLUMNS}
    est = small_bundle.uplift.estimate(InterventionType.NO_INTERVENTION, feats)
    assert est.cate == 0.0


def test_causal_forest_predict_shape(hist_df):
    sub = hist_df[hist_df["intervention"] == "maintenance_recommendation"]
    x = sub[FEATURE_COLUMNS].astype(float).values
    t = sub["treated"].astype(int).values
    y = sub["outcome_loss"].astype(int).values
    cf = CausalForest(n_trees=20, max_depth=3, min_leaf=25, random_state=0).fit(x, t, y)
    pred = cf.predict(x[:10])
    assert pred.shape == (10,)
    assert np.isfinite(pred).all()
