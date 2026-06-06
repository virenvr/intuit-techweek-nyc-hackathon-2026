#!/usr/bin/env python3
"""Generate submission_B_trajectory.csv (Deliverable B v1).

Steps covered
-------------
1. Assign cohort_week from application_timestamp
2. Load Deliverable A decisions (your approved sets)
3-4. Train discrete-time default-by-day-7a models + historical KM failsafes
5. Aggregate cohort-level cumulative default rates
6. Enforce monotonicity within each cohort
7. Attach binomial 90% intervals
8. Write submission_B_trajectory.csv and run format checks

Windows (PowerShell):
    .venv\\Scripts\\Activate.ps1
    python run_deliverable_b.py --output-dir submission
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.cohorts import attach_cohort_week, load_cohort_definitions
from src.constants import (
    B_TEMPLATE_PATH,
    COHORT_DEFINITIONS_PATH,
    DATA_DIR,
    DEFAULT_SUBMISSION_A,
    N_COHORT_WEEKS,
)
from src.trajectory import (
    TrajectoryModels,
    build_trajectory_grid,
    merge_decisions,
    train_trajectory_models,
)


def load_applicants(data_dir: Path) -> pd.DataFrame:
    """Validation + test applicants (same population as Deliverable A scoring)."""
    validation = pd.read_csv(data_dir / "validation.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return pd.concat([validation, test], ignore_index=True)


def write_b_log(
    path: Path,
    *,
    models: TrajectoryModels,
    scoring_df: pd.DataFrame,
    grid: pd.DataFrame,
) -> None:
    """Lightweight v1 log for writeup / debugging (Step 9 narrative deferred)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    approved = scoring_df.loc[scoring_df["decision"] == 1]
    cohort_counts = approved.groupby("cohort_week").size()
    thin = cohort_counts[cohort_counts < models.min_approved_cohort_size]

    lines = [
        f"# Deliverable B v1 log ({stamp})",
        "",
        "## Cohort assignment (Step 1)",
        f"- Scoring applicants: {len(scoring_df):,} (validation + test)",
        f"- Approved (from A): {len(approved):,} ({len(approved)/len(scoring_df):.1%})",
        "",
        "## Approved per cohort (Step 2)",
    ]
    for w in range(1, N_COHORT_WEEKS + 1):
        n = int(cohort_counts.get(w, 0))
        flag = " [THIN - KM blend failsafe]" if w in thin.index else ""
        lines.append(f"- Cohort {w:2d}: {n:,} approved{flag}")

    lines.extend(
        [
            "",
            "## Grid summary (Steps 5-7)",
            f"- CDR at age 13 weeks (day 91) by cohort:",
        ]
    )
    final = grid.loc[grid["loan_age_weeks"] == 13]
    for _, row in final.iterrows():
        lines.append(
            f"  - Cohort {int(row.cohort_week)}: "
            f"{row.cumulative_default_rate:.3f} "
            f"[{row.cdr_lower_90:.3f}, {row.cdr_upper_90:.3f}]"
        )

    lines.extend(
        [
            "",
            "## v1 failsafes (tune later)",
            f"- min_approved_cohort_size: {models.min_approved_cohort_size}",
            f"- blend_weight (thin cohorts): {models.blend_weight}",
            "- Empty cohort -> historical KM only",
            "- Monotonicity: cummax enforced per cohort",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Deliverable B submission (v1).")
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_DIR))
    parser.add_argument(
        "--decisions-path",
        type=Path,
        default=Path(DEFAULT_SUBMISSION_A),
        help="Deliverable A submission with decision column.",
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=Path(B_TEMPLATE_PATH),
    )
    parser.add_argument(
        "--cohort-defs-path",
        type=Path,
        default=Path(COHORT_DEFINITIONS_PATH),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("submission"))
    parser.add_argument(
        "--methodology-log",
        type=Path,
        default=Path("docs/deliverable_b_methodology.md"),
    )
    parser.add_argument(
        "--include-prior-underwriter-score",
        action="store_true",
        help="Use prior_underwriter_score in timing models (default: off, match A ablation).",
    )
    args = parser.parse_args()

    cohort_defs = load_cohort_definitions(args.cohort_defs_path)
    template = pd.read_csv(args.template_path)
    decisions = pd.read_csv(args.decisions_path)

    train = pd.read_csv(args.data_dir / "train.csv")
    validation = pd.read_csv(args.data_dir / "validation.csv")
    validation = attach_cohort_week(validation, cohort_defs, strict=True)

    print("Training timing models on approved+matured train rows...")
    models = train_trajectory_models(
        train,
        validation=validation,
        include_prior_underwriter_score=args.include_prior_underwriter_score,
    )

    applicants = load_applicants(args.data_dir)
    applicants = attach_cohort_week(applicants, cohort_defs, strict=True)
    scoring_df = merge_decisions(applicants, decisions)

    print("Building 13x13 trajectory grid from A decisions...")
    grid = build_trajectory_grid(models, scoring_df, template)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "submission_B_trajectory.csv"
    grid.to_csv(out_path, index=False)

    write_b_log(args.methodology_log, models=models, scoring_df=scoring_df, grid=grid)

    print(f"Wrote {out_path} ({len(grid)} rows)")
    print(f"Wrote log: {args.methodology_log}")
    print(
        f"CDR range: [{grid.cumulative_default_rate.min():.3f}, "
        f"{grid.cumulative_default_rate.max():.3f}]"
    )


if __name__ == "__main__":
    main()
