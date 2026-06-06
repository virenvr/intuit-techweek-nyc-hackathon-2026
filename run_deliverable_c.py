#!/usr/bin/env python3
"""Generate submission_C_counterfactuals.csv (Deliverable C v1).

See docs/DELIVERABLE_C_STEPS.md for the full step plan.

Windows (PowerShell):
    .venv\\Scripts\\Activate.ps1
    python run_deliverable_c.py --output-dir submission
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.constants import DATA_DIR, INTERVENTION_QUERIES_PATH
from src.counterfactuals import CounterfactualRunStats, predict_counterfactuals
from src.models import train_models


CAUSAL_NOTE = """
Deliverable C — causal traps (writeup Section 3)
------------------------------------------------
Queries ask for P(default | do(feature=v), rest fixed). This is NOT the same as
overwriting a column and re-predicting without adjustment — that answers an
observational counterfactual and fails when confounders exist.

v1 handling:
  - Intervenable features (data_dictionary.intervenable=True):
        apply do(v) with side effects (bank-feed link flag, derived ratios),
        then calibrated PD from the same model as Deliverable A.
  - Non-intervenable features (~174/900 queries — e.g. prior_loans_count,
        account_age_days, sector): causal do() is ill-defined; we blend toward
        observational PD and widen intervals (see constants NON_INTERVENABLE_*).

Remaining gaps for v2: double ML / causal forest; explicit confounding control;
reject-inference consistency with Deliverable A selection bias.
"""


def load_applicants(data_dir: Path) -> pd.DataFrame:
    validation = pd.read_csv(data_dir / "validation.csv")
    test = pd.read_csv(data_dir / "test.csv")
    combined = pd.concat([validation, test], ignore_index=True)
    if combined["applicant_id"].duplicated().any():
        raise ValueError("Duplicate applicant_id in validation/test")
    return combined


def write_methodology_log(
    path: Path,
    stats: CounterfactualRunStats,
    annotated: pd.DataFrame,
    *,
    include_prior_underwriter_score: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    by_group = (
        annotated.groupby("feature_group")["query_id"]
        .count()
        .sort_values(ascending=False)
    )
    non_int_features = (
        annotated.loc[~annotated["intervenable"], "feature_name"]
        .value_counts()
        .head(10)
    )

    lines = [
        f"# Deliverable C v1 log ({stamp})",
        "",
        CAUSAL_NOTE.strip(),
        "",
        "## Query breakdown",
        f"- Total queries: {stats.n_queries}",
        f"- Intervenable: {stats.n_intervenable}",
        f"- Non-intervenable (failsafe path): {stats.n_non_intervenable}",
        f"- PD model prior_underwriter_score: {'ON' if include_prior_underwriter_score else 'OFF'}",
        "",
        "## Queries by feature group",
    ]
    for grp, count in by_group.items():
        lines.append(f"- {grp}: {count}")

    lines.extend(
        [
            "",
            "## Top non-intervenable queried features",
        ]
    )
    for feat, count in non_int_features.items():
        lines.append(f"- {feat}: {count}")

    lines.extend(
        [
            "",
            "## Intervals",
            f"- Mean interval width (all): {stats.mean_interval_width:.3f}",
            f"- Mean interval width (non-intervenable): "
            f"{stats.mean_interval_width_non_intervenable:.3f}",
            "",
            "## v1 failsafes",
            "- NON_INTERVENABLE_BLEND, NON_INTERVENABLE_INTERVAL_MULTIPLIER in src/constants.py",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Deliverable C submission (v1).")
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_DIR))
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=Path(INTERVENTION_QUERIES_PATH),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("submission"))
    parser.add_argument(
        "--methodology-log",
        type=Path,
        default=Path("docs/deliverable_c_methodology.md"),
    )
    parser.add_argument(
        "--include-prior-underwriter-score",
        action="store_true",
        help="Include prior_underwriter_score in PD model (default: off for causal clarity).",
    )
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    validation = pd.read_csv(args.data_dir / "validation.csv")
    queries = pd.read_csv(args.queries_path)
    applicants = load_applicants(args.data_dir)

    missing = set(queries["applicant_id"]) - set(applicants["applicant_id"])
    if missing:
        raise ValueError(f"{len(missing)} query applicant_ids not in validation/test")

    print("Training PD model for counterfactual engine...")
    models = train_models(
        train,
        validation,
        include_prior_underwriter_score=args.include_prior_underwriter_score,
    )

    print(f"Predicting {len(queries)} counterfactuals...")
    submission, stats, annotated = predict_counterfactuals(
        models, applicants, queries
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "submission_C_counterfactuals.csv"
    submission.to_csv(out_path, index=False)

    write_methodology_log(
        args.methodology_log,
        stats,
        annotated,
        include_prior_underwriter_score=args.include_prior_underwriter_score,
    )

    print(f"Wrote {out_path} ({len(submission)} rows)")
    print(f"Wrote log: {args.methodology_log}")
    print(
        f"Intervenable: {stats.n_intervenable} | Non-intervenable: {stats.n_non_intervenable} | "
        f"Mean PD_cf: {submission.predicted_pd_cf.mean():.3f}"
    )


if __name__ == "__main__":
    main()
