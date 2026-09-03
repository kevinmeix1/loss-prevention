"""Evaluation metrics for prediction, uplift, and recommendations."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def calibration_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def qini_coefficient(y_true: np.ndarray, t: np.ndarray, uplift_score: np.ndarray) -> float:
    """Qini coefficient for uplift ranking quality."""
    order = np.argsort(-uplift_score)
    y = y_true[order]
    treat = t[order]
    n = len(y)
    if n == 0 or treat.sum() == 0 or (1 - treat).sum() == 0:
        return 0.0
    cum_t = np.cumsum(treat)
    cum_c = np.cumsum(1 - treat)
    cum_yt = np.cumsum(y * treat)
    cum_yc = np.cumsum(y * (1 - treat))
    with np.errstate(divide="ignore", invalid="ignore"):
        qini_curve = (cum_yc / np.maximum(cum_c, 1)) - (cum_yt / np.maximum(cum_t, 1))
        qini_curve = np.nan_to_num(qini_curve) * (np.arange(1, n + 1) / n)
    # Random baseline roughly linear to overall ATE
    ate = y[treat == 0].mean() - y[treat == 1].mean()
    random = ate * (np.arange(1, n + 1) / n)
    return float(np.trapezoid(qini_curve - random, dx=1 / n))


def auuc(y_true: np.ndarray, t: np.ndarray, uplift_score: np.ndarray) -> float:
    """Area Under Uplift Curve (higher is better)."""
    order = np.argsort(-uplift_score)
    y = y_true[order]
    treat = t[order]
    n = len(y)
    gains = []
    for k in range(1, n + 1):
        yt = y[:k][treat[:k] == 1]
        yc = y[:k][treat[:k] == 0]
        if len(yt) == 0 or len(yc) == 0:
            gains.append(0.0)
        else:
            gains.append(float(yc.mean() - yt.mean()))
    return float(np.trapezoid(gains, dx=1 / n))


def intervention_entropy(labels: list[str]) -> float:
    if not labels:
        return 0.0
    values, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def predictive_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": calibration_ece(y_true, y_prob),
    }
