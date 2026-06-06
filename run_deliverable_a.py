#!/usr/bin/env python3
"""Generate submission_A_decisions.csv using NPV-based approve/decline policy.

Windows (PowerShell):
    python -m venv .venv
    .venv\\Scripts\\Activate.ps1
    pip install -r requirements.txt
    python run_deliverable_a.py --output-dir submission

Unix:
    python run_deliverable_a.py --output-dir submission
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import DATA_DIR
from src.models import (
    UnderwritingModels,
    pd_intervals,
    predict_default_day,
    predict_pd,
    predict_recovery_amount,
    train_models,
    validation_interval_coverage,
)
from src.npv_policy import approval_decision, expected_npv, portfolio_realized_profit, tune_npv_threshold


SELECTION_BIAS_NOTE = """
Selection bias (Deliverable D — Section 1)
------------------------------------------
Training labels (default_flag, days_to_default) exist only for loans the PRIOR
lender approved AND that have matured. Declined and immature applications have
blank outcomes.

Our PD model is fit on that approved+matured subset, then applied to all
validation + test applicants (including those we may decline). PD and NPV for
declined segments are therefore extrapolations, not directly observed.

Mitigations in this pipeline:
  - Train PD only on approved_matured_mask rows
  - Isotonic calibration on validation (same label population)
  - NPV threshold tuned on validation realized profit (not AUC alone)
  - Ablation: prior_underwriter_score excluded unless it improves val profit
  - Per-applicant default-day and dollar recovery models on default history

Remaining gap: no full reject-inference / IPW — document in writeup limitations.
"""


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, validation, test


def score_expected_npv(models: UnderwritingModels, applicants: pd.DataFrame) -> np.ndarray:
    pd_hat = predict_pd(models, applicants)
    default_day = predict_default_day(models, applicants)
    recovery = predict_recovery_amount(models, applicants)
    return expected_npv(
        applicants["requested_amount"],
        pd_hat,
        default_day,
        recovery,
    )


def fit_and_tune(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    include_prior_underwriter_score: bool,
) -> tuple[UnderwritingModels, float, float, float]:
    """Train models, tune NPV threshold on validation profit, return metrics."""
    models = train_models(
        train,
        validation,
        include_prior_underwriter_score=include_prior_underwriter_score,
    )
    enpv = score_expected_npv(models, validation)
    best_tau, best_profit = tune_npv_threshold(validation, enpv)
    models.npv_threshold = best_tau
    coverage = validation_interval_coverage(models, validation)
    return models, best_profit, best_tau, coverage


def build_submission(
    models: UnderwritingModels,
    applicants: pd.DataFrame,
) -> pd.DataFrame:
    pd_hat = predict_pd(models, applicants)
    pd_lower, pd_upper = pd_intervals(models, applicants, pd_hat)
    default_day = predict_default_day(models, applicants)
    recovery = predict_recovery_amount(models, applicants)
    npv = expected_npv(
        applicants["requested_amount"],
        pd_hat,
        default_day,
        recovery,
    )
    decisions = approval_decision(npv, npv_threshold=models.npv_threshold)

    return pd.DataFrame(
        {
            "applicant_id": applicants["applicant_id"],
            "decision": decisions,
            "predicted_pd": pd_hat,
            "pd_lower_90": pd_lower,
            "pd_upper_90": pd_upper,
        }
    )


def write_methodology_log(
    path: Path,
    *,
    chosen_prior_score: bool,
    profit_without: float,
    profit_with: float,
    best_tau: float,
    coverage: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = (
        f"# Deliverable A methodology log ({stamp})\n\n"
        f"{SELECTION_BIAS_NOTE.strip()}\n\n"
        f"## Ablation: prior_underwriter_score\n"
        f"- Validation profit WITHOUT prior score: ${profit_without:,.0f}\n"
        f"- Validation profit WITH prior score:    ${profit_with:,.0f}\n"
        f"- Selected config: {'WITH' if chosen_prior_score else 'WITHOUT'} "
        f"prior_underwriter_score\n\n"
        f"## Policy tuning\n"
        f"- NPV threshold (tau): ${best_tau:,.2f}  (approve if E[NPV] > tau)\n"
        f"- Validation interval coverage (binary y in [lower, upper]): {coverage:.1%}\n"
    )
    path.write_text(body, encoding="utf-8")


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
    parser.add_argument(
        "--methodology-log",
        type=Path,
        default=Path("docs/deliverable_a_methodology.md"),
        help="Write selection-bias note and ablation results for writeup Section 1.",
    )
    args = parser.parse_args()

    train, validation, test = load_data(args.data_dir)
    print("Training ablation: without prior_underwriter_score...")
    models_no_prior, profit_no, tau_no, cov_no = fit_and_tune(
        train, validation, include_prior_underwriter_score=False
    )
    print("Training ablation: with prior_underwriter_score...")
    models_yes, profit_yes, tau_yes, cov_yes = fit_and_tune(
        train, validation, include_prior_underwriter_score=True
    )

    if profit_yes > profit_no:
        models = models_yes
        chosen_prior = True
        best_profit, best_tau, coverage = profit_yes, tau_yes, cov_yes
    else:
        models = models_no_prior
        chosen_prior = False
        best_profit, best_tau, coverage = profit_no, tau_no, cov_no

    print(
        f"Ablation winner: {'WITH' if chosen_prior else 'WITHOUT'} prior_underwriter_score "
        f"(val profit ${best_profit:,.0f}, tau=${best_tau:,.2f}, coverage={coverage:.1%})"
    )

    write_methodology_log(
        args.methodology_log,
        chosen_prior_score=chosen_prior,
        profit_without=profit_no,
        profit_with=profit_yes,
        best_tau=best_tau,
        coverage=coverage,
    )
    print(f"Wrote methodology log: {args.methodology_log}")

    scoring = pd.concat([validation, test], ignore_index=True)
    submission = build_submission(models, scoring)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "submission_A_decisions.csv"
    submission.to_csv(out_path, index=False)

    val_sub = submission.iloc[: len(validation)]
    val_decisions = val_sub["decision"].to_numpy()
    realized = portfolio_realized_profit(validation, val_decisions)

    print(f"Wrote {out_path} ({len(submission):,} rows)")
    print(
        f"Approve rate: {submission['decision'].mean():.1%} | "
        f"Mean PD: {submission['predicted_pd'].mean():.3f} | "
        f"Val realized profit: ${realized:,.0f}"
    )


if __name__ == "__main__":
    main()
