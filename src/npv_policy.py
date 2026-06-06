"""Expected NPV and approve/decline policy for Deliverable A.

Optimization objective (portfolio level)
---------------------------------------
We choose decisions d_i in {0, 1} to maximize expected portfolio profit:

    max_{d}  E[ sum_i  d_i * NPV_i ]

Per-loan decision rule (equivalent for independent loans)
---------------------------------------------------------
Approve borrower i iff expected net present value is positive:

    decision_i = 1  if  E[NPV_i | approve, x_i] > 0
    decision_i = 0  otherwise

Expected NPV decomposition
--------------------------
Let:
    R_i  = requested_amount
    r    = 0.35  (APR)
    T    = 60    (term in days)
    F_i  = 0.03 * R_i                         (origination fee, collected upfront)
    L_i  = R_i * (1 + r*T/365)                (total scheduled repayment)
    D_i  = L_i / T                            (daily ACH draw)
    p_i  = P(default | x_i)                   (predicted PD)
    t_i  = E[days_to_default | default, x_i]  (expected default day)
    rec_i = E[recovery | default, x_i]        (expected dollars recovered)

If the loan repays in full:
    NPV_i^repay = F_i + L_i - R_i = F_i + R_i * r * T / 365

If the loan defaults on day t with recovery rec:
    NPV_i^def = F_i + t * D_i + rec - R_i

Expected NPV:
    E[NPV_i] = (1 - p_i) * NPV_i^repay + p_i * NPV_i^def

Variables used in the decision
------------------------------
From the applicant record x_i:
    - requested_amount (R_i): loan size, drives fee, draws, and loss
    - predicted PD p_i from gradient-boosted model using application features
    - expected default day t_i (mean from training defaults, optionally adjusted)
    - expected recovery rec_i from gradient-boosted recovery model on defaults

PD model inputs (see feature_engineering.py):
    payment behavior: invoice_payment_delinquency_rate
    liquidity: observed_cash_balance_p10, payroll_regularity_score
    credit stress: aggregate_credit_utilization, existing_debt_obligations
    affordability ratios: requested_amount_to_observed_revenue, daily_draw_burden
    bureau / history: owner_personal_credit_band, prior_loans_default_count, inquiries
    business context: sector, geography_region, employee_count_bucket, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import APR, ORIGINATION_FEE_RATE, TERM_DAYS
from src.feature_engineering import daily_draw


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
    t = np.asarray(default_day, dtype=float)
    rec = np.asarray(recovery_amount, dtype=float)
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


def approval_decision(expected_npv_values: pd.Series | np.ndarray) -> np.ndarray:
    """Approve iff E[NPV] > 0."""
    return (np.asarray(expected_npv_values, dtype=float) > 0.0).astype(int)
