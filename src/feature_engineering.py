"""Application-time features for PD and NPV models (Deliverable A)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import APR, TERM_DAYS

# Raw columns used directly in the PD / recovery models (excluding prior-lender score).
RAW_NUMERIC_FEATURES = [
    "requested_amount",
    "stated_annual_revenue",
    "stated_time_in_business",
    "aggregate_credit_utilization",
    "invoice_payment_delinquency_rate",
    "existing_debt_obligations",
    "recent_inquiries_count_6mo",
    "multi_lender_inquiry_count_30d",
    "prior_loans_count",
    "prior_loans_default_count",
    "repeat_application_count",
    "requested_amount_to_observed_revenue",
    "observed_monthly_revenue_avg_3mo",
    "observed_cash_balance_p10",
    "payroll_regularity_score",
    "observed_overdraft_count_3mo",
    "observed_revenue_trend_3mo",
    "observed_revenue_volatility",
]

# Optional: encodes the *previous* lender policy — ablated by default (see train_models).
PRIOR_LENDER_FEATURES = ["prior_underwriter_score"]

RAW_CATEGORICAL_FEATURES = [
    "sector",
    "geography_region",
    "employee_count_bucket",
    "application_channel",
    "intended_use_of_funds",
    "owner_personal_credit_band",
]

RAW_BOOLEAN_FEATURES = ["has_linked_bank_feed"]

ENGINEERED_FEATURES = [
    "daily_draw_burden",
    "requested_to_stated_revenue",
    "debt_to_observed_revenue",
    "debt_to_stated_revenue",
    "stated_vs_observed_revenue_ratio",
    "stated_vs_vintage_time_gap",
    "prior_default_rate",
    "has_prior_default",
    "inquiry_intensity",
    "external_decline_recency",
]

DEFAULT_NO_DECLINE_SENTINEL = 9999.0


def feature_columns(*, include_prior_underwriter_score: bool = False) -> list[str]:
    """Application-time feature list for modeling."""
    numeric = list(RAW_NUMERIC_FEATURES)
    if include_prior_underwriter_score:
        numeric = numeric + list(PRIOR_LENDER_FEATURES)
    return numeric + RAW_CATEGORICAL_FEATURES + RAW_BOOLEAN_FEATURES + ENGINEERED_FEATURES


# Default feature set (no prior-lender score).
FEATURE_COLUMNS = feature_columns(include_prior_underwriter_score=False)


def daily_draw(requested_amount: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Daily ACH draw: D_i = R_i * (1 + r*T/365) / T."""
    total_repayment = requested_amount * (1.0 + APR * TERM_DAYS / 365.0)
    return total_repayment / TERM_DAYS


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    fill_value: float = np.nan,
) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.fillna(fill_value)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model matrix from raw application rows."""
    out = df.copy()

    out["has_linked_bank_feed"] = out["has_linked_bank_feed"].astype(float)
    out["has_prior_default"] = (out["prior_loans_default_count"] > 0).astype(float)
    out["prior_default_rate"] = out["prior_loans_default_count"] / np.maximum(
        out["prior_loans_count"], 1
    )
    out["inquiry_intensity"] = (
        out["multi_lender_inquiry_count_30d"]
        + 0.5 * out["recent_inquiries_count_6mo"]
    )
    out["external_decline_recency"] = out["days_since_last_external_decline"].fillna(
        DEFAULT_NO_DECLINE_SENTINEL
    )

    draw = daily_draw(out["requested_amount"])
    out["requested_to_stated_revenue"] = _safe_ratio(
        out["requested_amount"], out["stated_annual_revenue"]
    )
    out["debt_to_stated_revenue"] = _safe_ratio(
        out["existing_debt_obligations"], out["stated_annual_revenue"]
    )
    out["debt_to_observed_revenue"] = _safe_ratio(
        out["existing_debt_obligations"], out["observed_monthly_revenue_avg_3mo"]
    )
    out["daily_draw_burden"] = _safe_ratio(draw, out["observed_monthly_revenue_avg_3mo"])
    out["stated_vs_observed_revenue_ratio"] = _safe_ratio(
        out["stated_annual_revenue"],
        out["observed_monthly_revenue_avg_3mo"] * 12.0,
    )
    out["stated_vs_vintage_time_gap"] = (
        out["stated_time_in_business"] - out["vintage_years"]
    ).abs()

    return out


def approved_matured_mask(df: pd.DataFrame) -> pd.Series:
    """Rows with observed default outcomes (prior-approved and matured)."""
    return df["default_flag"].notna() & (df["prior_decision"] == 1)
