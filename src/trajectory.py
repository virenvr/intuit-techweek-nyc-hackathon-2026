"""Default timing trajectories for Deliverable B (Steps 3-7).

v1 contract
-----------
- Train discrete-time default-by-day-7a classifiers on approved+matured history.
- Aggregate to cohort-level CDR using Deliverable A approve decisions.
- Failsafe to historical Kaplan-Meier curves when a cohort has too few approvals.
- Enforce monotone cumulative rates; attach simple binomial 90% intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.constants import (
    DEFAULT_WINDOW_DAYS,
    INTERVAL_Z_SCORE,
    MIN_APPROVED_COHORT_SIZE,
    MIN_INTERVAL_HALF_WIDTH,
    N_COHORT_WEEKS,
    N_LOAN_AGE_WEEKS,
)
from src.feature_engineering import (
    ENGINEERED_FEATURES,
    PRIOR_LENDER_FEATURES,
    RAW_BOOLEAN_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
    approved_matured_mask,
    build_features,
    feature_columns,
)


def loan_age_to_day(loan_age_weeks: int) -> int:
    """Convert loan age in weeks to observation day (7a)."""
    return int(loan_age_weeks) * 7


def _feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return build_features(df)[feature_cols]


def _build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    numeric_cols = [
        c
        for c in feature_cols
        if c in RAW_NUMERIC_FEATURES or c in PRIOR_LENDER_FEATURES or c in ENGINEERED_FEATURES
    ]
    cat_cols = [c for c in feature_cols if c in RAW_CATEGORICAL_FEATURES]
    bool_cols = [c for c in feature_cols if c in RAW_BOOLEAN_FEATURES]
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
            ("bool", "passthrough", bool_cols),
        ]
    )


def _default_by_day_label(df: pd.DataFrame, day: int) -> np.ndarray:
    """1 if loan defaulted on or before `day`, else 0 (approved+matured rows only)."""
    defaulted = df["default_flag"].astype(bool).to_numpy()
    days = df["days_to_default"].to_numpy(dtype=float)
    return (defaulted & (days <= day)).astype(int)


@dataclass
class TrajectoryModels:
    """Timing models + historical cohort KM failsafes."""

    age_models: list[Pipeline]
    feature_cols: list[str]
    cohort_km_cdr: dict[int, np.ndarray] = field(default_factory=dict)
    global_km_cdr: np.ndarray = field(default_factory=lambda: np.zeros(N_LOAN_AGE_WEEKS))
    min_approved_cohort_size: int = MIN_APPROVED_COHORT_SIZE
    blend_weight: float = 0.35  # weight on historical KM when cohort is small (tunable)


def _kaplan_meier_cdr(days: np.ndarray, defaulted: np.ndarray, eval_days: list[int]) -> np.ndarray:
    """Empirical KM: CDR(t) = P(default by day t) for simple v1 historical curves."""
    days = np.asarray(days, dtype=float)
    defaulted = np.asarray(defaulted, dtype=bool)
    n = len(days)
    if n == 0:
        return np.zeros(len(eval_days))

    cdrs = []
    for t in eval_days:
        cdrs.append(float(np.mean(defaulted & (days <= t))))
    return np.array(cdrs)


def _fit_historical_km(
    history: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Per-cohort and global KM curves at loan ages 1..13."""
    eval_days = [loan_age_to_day(a) for a in range(1, N_LOAN_AGE_WEEKS + 1)]
    matured = history.loc[approved_matured_mask(history)].copy()
    matured = matured.loc[matured["cohort_week"].notna()].copy()
    if matured.empty:
        zeros = np.zeros(len(eval_days))
        return {w: zeros.copy() for w in range(1, N_COHORT_WEEKS + 1)}, zeros

    defaulted = matured["default_flag"].astype(bool).to_numpy()
    days = matured["days_to_default"].fillna(DEFAULT_WINDOW_DAYS + 1).to_numpy()

    global_cdr = _kaplan_meier_cdr(days, defaulted, eval_days)
    cohort_cdr: dict[int, np.ndarray] = {}
    for w in range(1, N_COHORT_WEEKS + 1):
        sub = matured.loc[matured["cohort_week"] == w]
        if len(sub) < MIN_APPROVED_COHORT_SIZE:
            cohort_cdr[w] = global_cdr.copy()
            continue
        d = sub["days_to_default"].fillna(DEFAULT_WINDOW_DAYS + 1).to_numpy()
        e = sub["default_flag"].astype(bool).to_numpy()
        cohort_cdr[w] = _kaplan_meier_cdr(d, e, eval_days)
    return cohort_cdr, global_cdr


def train_trajectory_models(
    train: pd.DataFrame,
    validation: pd.DataFrame | None = None,
    *,
    include_prior_underwriter_score: bool = False,
) -> TrajectoryModels:
    """Train one classifier per loan-age checkpoint (Steps 3-4)."""
    feature_cols = feature_columns(
        include_prior_underwriter_score=include_prior_underwriter_score
    )
    history = train.loc[approved_matured_mask(train)].copy()
    x_train = _feature_matrix(history, feature_cols)

    age_models: list[Pipeline] = []
    for age in range(1, N_LOAN_AGE_WEEKS + 1):
        day = loan_age_to_day(age)
        y = _default_by_day_label(history, day)
        model = Pipeline(
            steps=[
                ("prep", _build_preprocessor(feature_cols)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=5,
                        learning_rate=0.08,
                        max_iter=120,
                        random_state=42 + age,
                    ),
                ),
            ]
        )
        model.fit(x_train, y)
        age_models.append(model)

    # Cohort KM from validation (falls in cohort windows); train is pre-cohort history.
    km_source = validation if validation is not None else train
    cohort_km, global_km = _fit_historical_km(km_source)
    return TrajectoryModels(
        age_models=age_models,
        feature_cols=feature_cols,
        cohort_km_cdr=cohort_km,
        global_km_cdr=global_km,
    )


def predict_individual_cdr(models: TrajectoryModels, df: pd.DataFrame, age: int) -> np.ndarray:
    """P(default by day 7*age | x) for each row."""
    idx = age - 1
    x = _feature_matrix(df, models.feature_cols)
    return models.age_models[idx].predict_proba(x)[:, 1]


def _historical_cdr(models: TrajectoryModels, cohort_week: int, age: int) -> float:
    idx = age - 1
    if cohort_week in models.cohort_km_cdr:
        return float(models.cohort_km_cdr[cohort_week][idx])
    return float(models.global_km_cdr[idx])


def _binomial_interval(rate: float, n: int) -> tuple[float, float]:
    """Simple normal-approx 90% interval for a proportion (v1; tune later)."""
    if n <= 0:
        half = 0.5
    else:
        se = np.sqrt(max(rate * (1.0 - rate), 0.0) / n)
        half = max(MIN_INTERVAL_HALF_WIDTH, INTERVAL_Z_SCORE * se)
    lower = np.clip(rate - half, 0.0, 1.0)
    upper = np.clip(rate + half, 0.0, 1.0)
    return float(lower), float(upper)


def enforce_monotone_cohort(curve: pd.DataFrame) -> pd.DataFrame:
    """Step 6: cumulative rates and interval bounds non-decrease in loan age."""
    out = curve.sort_values("loan_age_weeks").copy()
    out["cumulative_default_rate"] = out["cumulative_default_rate"].cummax()

    lowers = []
    uppers = []
    running_lower = 0.0
    running_upper = 0.0
    for _, row in out.iterrows():
        rate = float(row["cumulative_default_rate"])
        lo = min(float(row["cdr_lower_90"]), rate)
        hi = max(float(row["cdr_upper_90"]), rate)
        running_lower = max(running_lower, lo)
        running_upper = max(running_upper, hi)
        running_upper = max(running_upper, running_lower)
        lowers.append(running_lower)
        uppers.append(running_upper)

    out["cdr_lower_90"] = np.clip(lowers, 0.0, 1.0)
    out["cdr_upper_90"] = np.clip(uppers, 0.0, 1.0)
    out["cdr_lower_90"] = np.minimum(out["cdr_lower_90"], out["cumulative_default_rate"])
    out["cdr_upper_90"] = np.maximum(out["cdr_upper_90"], out["cumulative_default_rate"])
    return out


def build_trajectory_grid(
    models: TrajectoryModels,
    scoring_df: pd.DataFrame,
    template: pd.DataFrame,
) -> pd.DataFrame:
    """Steps 2, 5, 7: fill 169-row grid from A decisions + timing models."""
    grid = template.copy()
    rows: list[dict[str, float | int]] = []

    for w in range(1, N_COHORT_WEEKS + 1):
        cohort_members = scoring_df[
            (scoring_df["cohort_week"] == w) & (scoring_df["decision"] == 1)
        ]
        n_approved = len(cohort_members)

        for age in range(1, N_LOAN_AGE_WEEKS + 1):
            hist = _historical_cdr(models, w, age)

            if n_approved == 0:
                # Failsafe: no approvals in cohort -> pure historical KM
                rate = hist
            elif n_approved < models.min_approved_cohort_size:
                # Failsafe: thin cohort -> blend portfolio mean with historical KM
                indiv = predict_individual_cdr(models, cohort_members, age)
                portfolio = float(np.mean(indiv))
                bw = models.blend_weight
                rate = (1.0 - bw) * portfolio + bw * hist
            else:
                indiv = predict_individual_cdr(models, cohort_members, age)
                rate = float(np.mean(indiv))

            lower, upper = _binomial_interval(rate, n_approved if n_approved else 1)
            rows.append(
                {
                    "cohort_week": w,
                    "loan_age_weeks": age,
                    "cumulative_default_rate": rate,
                    "cdr_lower_90": lower,
                    "cdr_upper_90": upper,
                }
            )

    filled = pd.DataFrame(rows)
    out_parts = []
    for w in range(1, N_COHORT_WEEKS + 1):
        part = filled.loc[filled["cohort_week"] == w].copy()
        out_parts.append(enforce_monotone_cohort(part))
    return pd.concat(out_parts, ignore_index=True)


def merge_decisions(
    applicants: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach decision column from Deliverable A submission."""
    out = applicants.merge(
        decisions[["applicant_id", "decision"]],
        on="applicant_id",
        how="left",
        validate="one_to_one",
    )
    missing = out["decision"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} applicants missing from Deliverable A decisions — "
            "regenerate submission_A_decisions.csv first"
        )
    out["decision"] = out["decision"].astype(int)
    return out
