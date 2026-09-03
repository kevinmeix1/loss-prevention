from loss_prevention.evaluation.experiments import (
    experiment_selection_bias,
    experiment_policy_optimization,
)
from loss_prevention.evaluation.metrics import (
    auuc,
    calibration_ece,
    intervention_entropy,
    qini_coefficient,
)
import numpy as np


def test_calibration_ece_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert calibration_ece(y, p) == 0.0


def test_qini_auuc_smoke():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    t = rng.integers(0, 2, 200)
    score = rng.normal(size=200)
    assert np.isfinite(qini_coefficient(y, t, score))
    assert np.isfinite(auuc(y, t, score))


def test_entropy():
    assert intervention_entropy(["a", "a", "a"]) < intervention_entropy(["a", "b", "c"])


def test_selection_bias_experiment_runs():
    report = experiment_selection_bias(seed=1)
    assert report.name == "selection_bias_confounding"
    assert "mean_abs_naive_bias" in report.metrics
    assert report.takeaways


def test_policy_optimization_uplift_beats_random():
    report = experiment_policy_optimization(seed=2)
    values = report.metrics["oracle_policy_value_mean_cate"]
    assert values["uplift_ranked"] >= values["no_action"]
