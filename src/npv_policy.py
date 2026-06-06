"""Expected NPV and approve/decline policy for Deliverable A."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import APR, DEFAULT_WINDOW_DAYS, ORIGINATION_FEE_RATE, TERM_DAYS

# Threshold tuning: require a real profit lift before moving off break-even (tau=0).
THRESHOLD_IMPROVEMENT_FRAC = 0.005
THRESHOLD_IMPROVEMENT_MIN = 1_000.0
THRESHOLD_PROFIT_TIE_EPS = 1.0


def loan_cashflows(requested_amount: pd.Series | np.ndarray) -> dict[str, np.ndarray]:
    """Compute fixed product cash-flow components for each loan."""
    r = np.asarray(requested_amount, dtype=float)
    origination_fee = ORIGINATION_FEE_RATE * r
    total_repayment = r * (1.0 + APR * TERM_DAYS / 365.0)
    interest_income = total_repayment - r
    draw = total_repayment / TERM_DAYS
    return {
        "origination_fee": origination_fee,
        "total_repayment": total_repayment,
        "interest_income": interest_income,
        "daily_draw": draw,
    }


def npv_if_repaid(requested_amount: pd.Series | np.ndarray) -> np.ndarray:
    """NPV when the borrower repays in full by day T."""
    cf = loan_cashflows(requested_amount)
    return cf["origination_fee"] + cf["interest_income"]


def npv_if_default(
    requested_amount: pd.Series | np.ndarray,
    default_day: pd.Series | np.ndarray,
    recovery_amount: pd.Series | np.ndarray,
) -> np.ndarray:
    """NPV when the borrower defaults on default_day with recovery_amount collected."""
    cf = loan_cashflows(requested_amount)
    r = np.asarray(requested_amount, dtype=float)
    t = np.clip(np.asarray(default_day, dtype=float), 1.0, float(DEFAULT_WINDOW_DAYS))
    rec = np.maximum(np.asarray(recovery_amount, dtype=float), 0.0)
    payments_collected = t * cf["daily_draw"]
    return cf["origination_fee"] + payments_collected + rec - r


def expected_npv(
    requested_amount: pd.Series | np.ndarray,
    pd_hat: pd.Series | np.ndarray,
    default_day: pd.Series | np.ndarray,
    recovery_amount: pd.Series | np.ndarray,
) -> np.ndarray:
    """E[NPV] = (1-p)*NPV_repay + p*NPV_default."""
    p = np.clip(np.asarray(pd_hat, dtype=float), 0.0, 1.0)
    npv_repay = npv_if_repaid(requested_amount)
    npv_def = npv_if_default(requested_amount, default_day, recovery_amount)
    return (1.0 - p) * npv_repay + p * npv_def


def approval_decision(
    expected_npv_values: pd.Series | np.ndarray,
    *,
    npv_threshold: float = 0.0,
) -> np.ndarray:
    """Approve iff E[NPV] > npv_threshold (threshold tuned on validation profit)."""
    return (np.asarray(expected_npv_values, dtype=float) > npv_threshold).astype(int)


def realized_npv(df: pd.DataFrame) -> np.ndarray:
    """Observed NPV for matured rows using actual outcomes (validation tuning only)."""
    amount = df["requested_amount"].to_numpy(dtype=float)
    defaulted = df["default_flag"].astype(bool).to_numpy()
    npv = np.zeros(len(df), dtype=float)
    if (~defaulted).any():
        npv[~defaulted] = npv_if_repaid(amount[~defaulted])
    if defaulted.any():
        default_day = df.loc[defaulted, "days_to_default"].fillna(DEFAULT_WINDOW_DAYS)
        recovery = df.loc[defaulted, "final_recovered_amount"].fillna(0.0)
        npv[defaulted] = npv_if_default(
            amount[defaulted],
            default_day.to_numpy(),
            recovery.to_numpy(),
        )
    return npv


def portfolio_realized_profit(df: pd.DataFrame, decisions: np.ndarray) -> float:
    """Sum of realized NPV on approved rows (requires outcome columns)."""
    d = np.asarray(decisions, dtype=int)
    realized = realized_npv(df)
    return float((d * realized).sum())


def tune_npv_threshold(
    df: pd.DataFrame,
    expected_npv_values: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    anchor_tau: float = 0.0,
    improvement_frac: float = THRESHOLD_IMPROVEMENT_FRAC,
    improvement_min: float = THRESHOLD_IMPROVEMENT_MIN,
    profit_tie_eps: float = THRESHOLD_PROFIT_TIE_EPS,
) -> tuple[float, float]:
    """Pick threshold maximizing validation profit with anti-overfit guardrails.

    Starts from the break-even anchor (tau=0 by default). A candidate threshold
    replaces the incumbent only if it beats the best profit by a meaningful margin
    (relative + absolute floor). On near-ties, prefer the threshold closest to zero
    so tuning does not chase fold noise with extreme cutoffs.
    """
    if thresholds is None:
        enpv = np.asarray(expected_npv_values, dtype=float)
        lo = float(np.quantile(enpv, 0.05))
        hi = float(np.quantile(enpv, 0.95))
        thresholds = np.linspace(lo, hi, 41)

    enpv = np.asarray(expected_npv_values, dtype=float)
    anchor_tau = float(anchor_tau)
    best_tau = anchor_tau
    best_profit = portfolio_realized_profit(
        df, approval_decision(enpv, npv_threshold=anchor_tau)
    )
    margin = max(improvement_min, improvement_frac * abs(best_profit))

    for tau in thresholds:
        tau = float(tau)
        profit = portfolio_realized_profit(
            df, approval_decision(enpv, npv_threshold=tau)
        )
        if profit > best_profit + margin:
            best_profit = profit
            best_tau = tau
        elif abs(profit - best_profit) <= profit_tie_eps and abs(tau) < abs(best_tau):
            best_profit = profit
            best_tau = tau

    return best_tau, best_profit
