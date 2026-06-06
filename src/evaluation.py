"""Classification, calibration, interval, and business metrics for Deliverable A."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.feature_engineering import approved_matured_mask


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) with equal-width bins on [0, 1]."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def evaluate_pd_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Primary classification and calibration metrics (PRD evaluation section)."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    out: dict[str, float] = {
        "log_loss": float(log_loss(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }
    if len(np.unique(y)) > 1:
        out["auc_roc"] = float(roc_auc_score(y, p))
    else:
        out["auc_roc"] = float("nan")
    return out


def interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of outcomes captured by [lower, upper] (target ~90%)."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))


def portfolio_expected_npv(
    expected_npv_values: np.ndarray,
    decisions: np.ndarray,
) -> float:
    """Sum of E[NPV] on approved rows."""
    d = np.asarray(decisions, dtype=float)
    enpv = np.asarray(expected_npv_values, dtype=float)
    return float((d * enpv).sum())


def evaluate_validation(
    validation: pd.DataFrame,
    *,
    pd_hat: np.ndarray,
    pd_lower: np.ndarray,
    pd_upper: np.ndarray,
    expected_npv_values: np.ndarray,
    decisions: np.ndarray,
    realized_profit: float,
) -> dict[str, float]:
    """Aggregate validation metrics for the evaluation report."""
    mask = approved_matured_mask(validation)
    labeled = validation.loc[mask]
    y = labeled["default_flag"].astype(int).to_numpy()
    idx = labeled.index
    pos = validation.index.get_indexer(idx)

    pd_metrics = evaluate_pd_predictions(y, pd_hat[pos])
    coverage = interval_coverage(y, pd_lower[pos], pd_upper[pos])

    d = np.asarray(decisions, dtype=int)
    return {
        **pd_metrics,
        "interval_coverage": coverage,
        "approval_rate": float(np.mean(d)),
        "portfolio_expected_npv": portfolio_expected_npv(expected_npv_values, d),
        "portfolio_realized_profit": float(realized_profit),
        "mean_predicted_pd": float(np.mean(pd_hat)),
    }
