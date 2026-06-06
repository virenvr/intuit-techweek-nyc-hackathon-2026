#!/usr/bin/env python3
"""Generate submission_A_decisions.csv using NPV-based approve/decline policy.

Run with the dobby virtual environment:

    dobby/bin/python run_deliverable_a.py
    dobby/bin/python run_deliverable_a.py --output-dir submission

Decision logic
--------------
We maximize expected portfolio profit:

    max_{d_i}  E[ sum_i  d_i * NPV_i ]

For independent loans this reduces to approving whenever expected NPV is positive:

    decision_i = 1  if  E[NPV_i | x_i] > 0
    decision_i = 0  otherwise

Expected NPV for applicant i:

    E[NPV_i] = (1 - p_i) * NPV_i^repay + p_i * NPV_i^def

where:
    p_i           = calibrated PD from application features x_i
    NPV_i^repay   = F_i + R_i * r * T / 365
    NPV_i^def     = F_i + t_i * D_i + rec_i - R_i

    R_i = requested_amount
    F_i = 0.03 * R_i
    D_i = R_i * (1 + r*T/365) / T
    r   = 0.35, T = 60 days
    t_i = mean days_to_default on training defaults
    rec_i = predicted recovery dollars from default recovery model

Variables driving the decision
------------------------------
Loan economics: requested_amount
Predicted default probability p_i from:
    invoice_payment_delinquency_rate, aggregate_credit_utilization,
    observed_cash_balance_p10, payroll_regularity_score,
    requested_amount_to_observed_revenue, daily_draw_burden,
    existing_debt_obligations, prior_loans_default_count,
    owner_personal_credit_band, sector, geography_region, and related features
Default timing: mean days_to_default (training defaults)
Recovery: final_recovered_amount / outstanding balance model on default history
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.constants import DATA_DIR
from src.models import (
    UnderwritingModels,
    pd_intervals,
    predict_pd,
    predict_recovery_amount,
    train_models,
)
from src.npv_policy import approval_decision, expected_npv


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, validation, test


def build_submission(
    models: UnderwritingModels,
    applicants: pd.DataFrame,
) -> pd.DataFrame:
    pd_hat = predict_pd(models, applicants)
    pd_lower, pd_upper = pd_intervals(models, pd_hat)

    default_day = pd.Series(models.default_day_mean, index=applicants.index)
    recovery = predict_recovery_amount(models, applicants)
    npv = expected_npv(
        applicants["requested_amount"],
        pd_hat,
        default_day,
        recovery,
    )
    decisions = approval_decision(npv)

    return pd.DataFrame(
        {
            "applicant_id": applicants["applicant_id"],
            "decision": decisions,
            "predicted_pd": pd_hat,
            "pd_lower_90": pd_lower,
            "pd_upper_90": pd_upper,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Deliverable A submission.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(DATA_DIR),
        help="Directory containing train/validation/test CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("submission"),
        help="Folder for submission_A_decisions.csv",
    )
    args = parser.parse_args()

    train, validation, test = load_data(args.data_dir)
    print("Training PD and recovery models on approved+matured train rows...")
    models = train_models(train, validation)

    scoring = pd.concat([validation, test], ignore_index=True)
    submission = build_submission(models, scoring)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "submission_A_decisions.csv"
    submission.to_csv(out_path, index=False)

    approve_rate = submission["decision"].mean()
    mean_pd = submission["predicted_pd"].mean()
    print(f"Wrote {out_path} ({len(submission):,} rows)")
    print(f"Approve rate: {approve_rate:.1%} | Mean predicted PD: {mean_pd:.3f}")
    print(f"Mean default day (training): {models.default_day_mean:.1f}")


if __name__ == "__main__":
    main()
