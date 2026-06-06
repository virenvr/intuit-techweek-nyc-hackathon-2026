#!/usr/bin/env python3
"""Generate submission_A_decisions.csv — PRD-aligned NPV underwriting pipeline.

Pipeline (PRD technical roadmap)
-------------------------------
1. Feature engineering (application-time features)
2. Calibrated probability model (HistGradientBoosting + isotonic regression)
3. Wilson score 90% confidence intervals (deterministic, bounded in [0, 1])
4. NPV engine — E[NPV] = (1-p)*NPV_repay + p*NPV_default
5. Decision engine — approve iff E[NPV] > 0
6. Validation metrics + submission file

Hackathon submission schema (validate_submission.py):
    applicant_id, decision (0/1), predicted_pd, pd_lower_90, pd_upper_90

Run:
    dobby/bin/python run_deliverable_a.py --output-dir submission
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import DATA_DIR
from src.evaluation import evaluate_validation
from src.models import (
    UnderwritingModels,
    pd_intervals,
    predict_default_day,
    predict_pd,
    predict_recovery_amount,
    train_models,
)
from src.npv_policy import approval_decision, expected_npv, portfolio_realized_profit

RANDOM_SEED = 42

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
  - ENPV > 0 decision rule with portfolio metrics on validation
  - Wilson score 90% PD intervals (k = p-hat * n, n = approved+matured train size)
  - Ablation: prior_underwriter_score excluded unless it improves val ENPV
  - Per-applicant default-day and dollar recovery models on default history

Remaining gap: no full reject-inference / IPW — document in writeup limitations.
"""


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, validation, test


def score_applicants(
    models: UnderwritingModels,
    applicants: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict PD, intervals, and expected NPV for each applicant."""
    pd_hat = predict_pd(models, applicants)
    pd_lower, pd_upper = pd_intervals(models, applicants, pd_hat)
    default_day = predict_default_day(models, applicants)
    recovery = predict_recovery_amount(models, applicants)
    enpv = expected_npv(
        applicants["requested_amount"],
        pd_hat,
        default_day,
        recovery,
    )
    return pd_hat, pd_lower, pd_upper, enpv, default_day


def train_pipeline(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    include_prior_underwriter_score: bool,
    interval_method: str,
    n_bootstrap: int,
    npv_threshold: float,
    pd_interval_n: int | None,
) -> UnderwritingModels:
    """Phase 2–3: train calibrated PD, intervals, timing, and recovery models."""
    return train_models(
        train,
        validation,
        include_prior_underwriter_score=include_prior_underwriter_score,
        npv_threshold=npv_threshold,
        interval_method=interval_method,
        n_bootstrap=n_bootstrap,
        pd_interval_n=pd_interval_n,
        random_state=RANDOM_SEED,
    )


def build_submission(
    models: UnderwritingModels,
    applicants: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Phase 5–6: score applicants and apply ENPV decision rule."""
    pd_hat, pd_lower, pd_upper, enpv, _ = score_applicants(models, applicants)
    decisions = approval_decision(enpv, npv_threshold=models.npv_threshold)

    submission = pd.DataFrame(
        {
            "applicant_id": applicants["applicant_id"],
            "decision": decisions,
            "predicted_pd": pd_hat,
            "pd_lower_90": pd_lower,
            "pd_upper_90": pd_upper,
        }
    )
    return submission, enpv


def write_methodology_log(
    path: Path,
    *,
    chosen_prior_score: bool,
    profit_without: float,
    profit_with: float,
    enpv_without: float,
    enpv_with: float,
    npv_threshold: float,
    interval_method: str,
    n_bootstrap: int,
    pd_interval_n: int,
    coverage: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = (
        f"# Deliverable A methodology log ({stamp})\n\n"
        f"{SELECTION_BIAS_NOTE.strip()}\n\n"
        f"## Ablation: prior_underwriter_score\n"
        f"- Validation portfolio E[NPV] WITHOUT prior score: ${enpv_without:,.0f}\n"
        f"- Validation portfolio E[NPV] WITH prior score:    ${enpv_with:,.0f}\n"
        f"- Validation realized profit WITHOUT: ${profit_without:,.0f}\n"
        f"- Validation realized profit WITH:    ${profit_with:,.0f}\n"
        f"- Selected config: {'WITH' if chosen_prior_score else 'WITHOUT'} "
        f"prior_underwriter_score\n\n"
        f"## Policy (PRD FR5)\n"
        f"- Decision rule: approve iff E[NPV] > {npv_threshold:,.2f}\n"
        f"- Interval method: {interval_method}"
        + (
            f" (Wilson score, n={pd_interval_n:,}, z=1.645, k=p-hat*n)"
            if interval_method == "wilson"
            else (f" (N={n_bootstrap})" if interval_method == "bootstrap" else "")
        )
        + f"\n"
        f"- Validation interval coverage (binary y in [lower, upper]): {coverage:.1%}\n"
        f"- Random seed: {RANDOM_SEED}\n"
    )
    path.write_text(body, encoding="utf-8")


def write_evaluation_report(
    path: Path,
    metrics: dict[str, float],
    *,
    interval_method: str,
    n_bootstrap: int,
    pd_interval_n: int,
    npv_threshold: float,
) -> None:
    """PRD acceptance-criteria evaluation summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Deliverable A evaluation report ({stamp})",
        "",
        "## Classification metrics",
        f"- Log loss:    {metrics.get('log_loss', float('nan')):.4f}",
        f"- AUC-ROC:     {metrics.get('auc_roc', float('nan')):.4f}",
        f"- Brier score: {metrics.get('brier_score', float('nan')):.4f}",
        "",
        "## Calibration metrics",
        f"- Expected Calibration Error (ECE): {metrics.get('ece', float('nan')):.4f}",
        "",
        "## Interval metrics",
        f"- Method: {interval_method}"
        + (
            f" (Wilson score, n={pd_interval_n:,}, z=1.645)"
            if interval_method == "wilson"
            else (f" (N={n_bootstrap} bootstrap models)" if interval_method == "bootstrap" else "")
        ),
        f"- Coverage rate (target 88–92%): {metrics.get('interval_coverage', float('nan')):.1%}",
        "",
        "## Business metrics",
        f"- Approval rate: {metrics.get('approval_rate', float('nan')):.1%}",
        f"- Mean predicted PD: {metrics.get('mean_predicted_pd', float('nan')):.3f}",
        f"- Portfolio E[NPV] (approved): ${metrics.get('portfolio_expected_npv', float('nan')):,.0f}",
        f"- Portfolio realized profit (validation): "
        f"${metrics.get('portfolio_realized_profit', float('nan')):,.0f}",
        f"- NPV threshold: ${npv_threshold:,.2f}",
        "",
        "## Reproducibility",
        f"- Random seed: {RANDOM_SEED}",
        f"- Feature version: src/feature_engineering.py (application-time)",
        f"- Model: HistGradientBoostingClassifier + IsotonicRegression",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_config(
    models: UnderwritingModels,
    validation: pd.DataFrame,
) -> tuple[dict[str, float], float]:
    """Score validation set and compute PRD metrics."""
    pd_hat, pd_lower, pd_upper, enpv, _ = score_applicants(models, validation)
    decisions = approval_decision(enpv, npv_threshold=models.npv_threshold)
    realized = portfolio_realized_profit(validation, decisions)
    metrics = evaluate_validation(
        validation,
        pd_hat=pd_hat,
        pd_lower=pd_lower,
        pd_upper=pd_upper,
        expected_npv_values=enpv,
        decisions=decisions,
        realized_profit=realized,
    )
    return metrics, realized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Deliverable A submission (PRD-aligned NPV pipeline)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path("submission"))
    parser.add_argument(
        "--methodology-log",
        type=Path,
        default=Path("docs/deliverable_a_methodology.md"),
    )
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=Path("docs/deliverable_a_evaluation.md"),
    )
    parser.add_argument(
        "--interval-method",
        choices=("wilson", "bootstrap", "residual"),
        default="wilson",
        help="90%% interval method: Wilson score (default), bootstrap, or residual quantile.",
    )
    parser.add_argument(
        "--pd-interval-n",
        type=int,
        default=None,
        help="Effective n for Wilson intervals (default: approved+matured train count).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=25,
        help="Number of bootstrap PD models when --interval-method=bootstrap.",
    )
    parser.add_argument(
        "--npv-threshold",
        type=float,
        default=0.0,
        help="Approve iff E[NPV] > threshold (PRD default: 0).",
    )
    args = parser.parse_args()

    train, validation, test = load_data(args.data_dir)

    print(
        f"Training pipeline (interval={args.interval_method}, "
        f"tau={args.npv_threshold:,.2f})..."
    )
    print("  Ablation: without prior_underwriter_score...")
    models_no_prior = train_pipeline(
        train,
        validation,
        include_prior_underwriter_score=False,
        interval_method=args.interval_method,
        n_bootstrap=args.n_bootstrap,
        npv_threshold=args.npv_threshold,
        pd_interval_n=args.pd_interval_n,
    )
    metrics_no, profit_no = evaluate_config(models_no_prior, validation)

    print("  Ablation: with prior_underwriter_score...")
    models_yes = train_pipeline(
        train,
        validation,
        include_prior_underwriter_score=True,
        interval_method=args.interval_method,
        n_bootstrap=args.n_bootstrap,
        npv_threshold=args.npv_threshold,
        pd_interval_n=args.pd_interval_n,
    )
    metrics_yes, profit_yes = evaluate_config(models_yes, validation)

    if metrics_yes["portfolio_expected_npv"] > metrics_no["portfolio_expected_npv"]:
        models = models_yes
        chosen_prior = True
        metrics = metrics_yes
        best_profit = profit_yes
        enpv_no = metrics_no["portfolio_expected_npv"]
        enpv_yes = metrics_yes["portfolio_expected_npv"]
    else:
        models = models_no_prior
        chosen_prior = False
        metrics = metrics_no
        best_profit = profit_no
        enpv_no = metrics_no["portfolio_expected_npv"]
        enpv_yes = metrics_yes["portfolio_expected_npv"]

    print(
        f"Ablation winner: {'WITH' if chosen_prior else 'WITHOUT'} prior_underwriter_score | "
        f"val E[NPV]=${metrics['portfolio_expected_npv']:,.0f} | "
        f"coverage={metrics['interval_coverage']:.1%} | "
        f"AUC={metrics['auc_roc']:.3f}"
    )

    write_methodology_log(
        args.methodology_log,
        chosen_prior_score=chosen_prior,
        profit_without=profit_no,
        profit_with=profit_yes,
        enpv_without=enpv_no,
        enpv_with=enpv_yes,
        npv_threshold=args.npv_threshold,
        interval_method=args.interval_method,
        n_bootstrap=args.n_bootstrap,
        pd_interval_n=models.pd_interval_n,
        coverage=metrics["interval_coverage"],
    )
    write_evaluation_report(
        args.evaluation_report,
        metrics,
        interval_method=args.interval_method,
        n_bootstrap=args.n_bootstrap,
        pd_interval_n=models.pd_interval_n,
        npv_threshold=args.npv_threshold,
    )
    print(f"Wrote methodology log: {args.methodology_log}")
    print(f"Wrote evaluation report: {args.evaluation_report}")

    scoring = pd.concat([validation, test], ignore_index=True)
    submission, enpv = build_submission(models, scoring)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "submission_A_decisions.csv"
    submission.to_csv(out_path, index=False)

    print(f"Wrote {out_path} ({len(submission):,} rows)")
    print(
        f"Approve rate: {submission['decision'].mean():.1%} | "
        f"Mean PD: {submission['predicted_pd'].mean():.3f} | "
        f"Mean E[NPV]: ${enpv.mean():,.0f} | "
        f"Val realized profit: ${best_profit:,.0f}"
    )


if __name__ == "__main__":
    main()
