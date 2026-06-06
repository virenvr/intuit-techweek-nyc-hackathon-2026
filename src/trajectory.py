"""Default timing trajectories for Deliverable B (Steps 3-7).

v2 contract
-----------
- Train ONE discrete-time hazard model on approved+matured history (person-period
  expansion, loan age as a feature) instead of 13 independent per-age classifiers.
  CDR is rebuilt as 1 - prod(1 - hazard), which is smooth and monotone by
  construction and removes the "flat tail then age-13 spike" artifact.
- Aggregate to cohort-level CDR using Deliverable A approve decisions.
- Failsafe to historical Kaplan-Meier curves when a cohort has too few approvals.
- Attach intervals that actually reflect uncertainty: a binomial sampling term
  (cohort size), a Greenwood shape term (grows as the at-risk set thins with age),
  and an extrapolation inflation that grows past the observed boundary 14 - w.

NOTE: this preserves the public interface used by run_deliverable_b.py
(`train_trajectory_models`, `build_trajectory_grid`, `merge_decisions`,
`TrajectoryModels`) and the same column / constant / feature_engineering
assumptions as v1 (`default_flag`, `days_to_default`, the feature lists, etc.).
It has not been run against the real data here — run it on train/validation and
re-check validate_submission.py.
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


@dataclass
class TrajectoryModels:
    """Single discrete-time hazard model + historical cohort KM failsafes."""

    prep: ColumnTransformer
    clf: HistGradientBoostingClassifier
    feature_cols: list[str]
    greenwood_var: np.ndarray  # length N_LOAN_AGE_WEEKS: Var(CDR(a)) shape term
    cohort_km_cdr: dict[int, np.ndarray] = field(default_factory=dict)
    global_km_cdr: np.ndarray = field(default_factory=lambda: np.zeros(N_LOAN_AGE_WEEKS))
    min_approved_cohort_size: int = MIN_APPROVED_COHORT_SIZE
    blend_weight: float = 0.35  # weight on historical KM when cohort is small (tunable)
    extrap_inflation_per_week: float = 0.15  # widen bands past observed boundary (tunable)


# ----------------------------------------------------------------------------
# Hazard model (replaces the 13 independent per-age classifiers)
# ----------------------------------------------------------------------------


def _person_period(
    history: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand each matured loan into at-risk weekly intervals (discrete-time).

    For loan i and week k (1..13): the loan is *at risk* if it had not defaulted
    before the start of the interval (day 7*(k-1)); the *event* is whether it
    defaulted within (7*(k-1), 7*k]. Loans drop out of the risk set after default.

    Returns the stacked feature frame, the age vector, the event labels, and the
    per-week (at_risk_count, event_count) arrays used for the Greenwood variance.
    """
    feats = build_features(history)[feature_cols].reset_index(drop=True)
    defaulted = history["default_flag"].astype(bool).to_numpy()
    raw_days = history["days_to_default"].to_numpy(dtype=float)
    # Day of default; +inf for loans that never defaulted within the window.
    eff_day = np.where(defaulted, raw_days, np.inf)

    frames: list[pd.DataFrame] = []
    age_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    n_at_risk = np.zeros(N_LOAN_AGE_WEEKS)
    n_events = np.zeros(N_LOAN_AGE_WEEKS)

    for k in range(1, N_LOAN_AGE_WEEKS + 1):
        lo = loan_age_to_day(k - 1)
        hi = loan_age_to_day(k)
        at_risk = eff_day > lo  # survived to the start of interval k
        event = at_risk & defaulted & (eff_day <= hi)

        n_at_risk[k - 1] = int(at_risk.sum())
        n_events[k - 1] = int(event.sum())
        if not at_risk.any():
            continue

        frames.append(feats.loc[at_risk])
        age_chunks.append(np.full(int(at_risk.sum()), k, dtype=float))
        label_chunks.append(event[at_risk].astype(int))

    x_feat = pd.concat(frames, ignore_index=True)
    age_arr = np.concatenate(age_chunks)
    y = np.concatenate(label_chunks)
    return x_feat, age_arr, y, n_at_risk, n_events


def _greenwood_variance(n_at_risk: np.ndarray, n_events: np.ndarray) -> np.ndarray:
    """Greenwood-style Var(CDR(a)) from training risk-set counts.

    Var(S(a)) = S(a)^2 * sum_{k<=a} d_k / (n_k (n_k - d_k)); Var(CDR)=Var(S).
    Grows as the at-risk set thins at later ages -> wider bands at the tail.
    """
    haz = np.divide(
        n_events, n_at_risk, out=np.zeros_like(n_events), where=n_at_risk > 0
    )
    surv = np.cumprod(1.0 - haz)
    denom = n_at_risk * (n_at_risk - n_events)
    term = np.divide(n_events, denom, out=np.zeros_like(n_events), where=denom > 0)
    return (surv ** 2) * np.cumsum(term)


def _individual_cdr_matrix(models: TrajectoryModels, df: pd.DataFrame) -> np.ndarray:
    """Per-row CDR curve over ages 1..13: shape (n_rows, N_LOAN_AGE_WEEKS)."""
    feats = _feature_matrix(df, models.feature_cols)
    x_prep = models.prep.transform(feats)
    n = x_prep.shape[0]
    hazards = np.zeros((n, N_LOAN_AGE_WEEKS))
    for k in range(1, N_LOAN_AGE_WEEKS + 1):
        x_age = np.hstack([x_prep, np.full((n, 1), float(k))])
        hazards[:, k - 1] = models.clf.predict_proba(x_age)[:, 1]
    survival = np.cumprod(1.0 - hazards, axis=1)
    return 1.0 - survival


def predict_individual_cdr(models: TrajectoryModels, df: pd.DataFrame, age: int) -> np.ndarray:
    """P(default by day 7*age | x) for each row (kept for interface compatibility)."""
    return _individual_cdr_matrix(models, df)[:, age - 1]


# ----------------------------------------------------------------------------
# Historical Kaplan-Meier failsafes (unchanged from v1)
# ----------------------------------------------------------------------------


def _kaplan_meier_cdr(days: np.ndarray, defaulted: np.ndarray, eval_days: list[int]) -> np.ndarray:
    """Empirical CDR(t) = P(default by day t) for simple historical curves."""
    days = np.asarray(days, dtype=float)
    defaulted = np.asarray(defaulted, dtype=bool)
    if len(days) == 0:
        return np.zeros(len(eval_days))
    return np.array([float(np.mean(defaulted & (days <= t))) for t in eval_days])


def _fit_historical_km(history: pd.DataFrame) -> tuple[dict[int, np.ndarray], np.ndarray]:
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
    """Train a single discrete-time hazard model (Steps 3-4)."""
    feature_cols = feature_columns(
        include_prior_underwriter_score=include_prior_underwriter_score
    )
    history = train.loc[approved_matured_mask(train)].copy()

    # Person-period expansion -> one hazard model with loan age as a feature.
    x_feat, age_arr, y, n_at_risk, n_events = _person_period(history, feature_cols)
    prep = _build_preprocessor(feature_cols).fit(_feature_matrix(history, feature_cols))
    x_long = np.hstack([prep.transform(x_feat), age_arr.reshape(-1, 1)])

    clf = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.08,
        max_iter=200,
        random_state=42,
    )
    clf.fit(x_long, y)

    greenwood_var = _greenwood_variance(n_at_risk, n_events)

    # Cohort KM from validation (falls in cohort windows); train is pre-cohort history.
    km_source = validation if validation is not None else train
    cohort_km, global_km = _fit_historical_km(km_source)

    return TrajectoryModels(
        prep=prep,
        clf=clf,
        feature_cols=feature_cols,
        greenwood_var=greenwood_var,
        cohort_km_cdr=cohort_km,
        global_km_cdr=global_km,
    )


# ----------------------------------------------------------------------------
# Intervals + monotonicity
# ----------------------------------------------------------------------------


def _interval(
    rate: float,
    n: int,
    age: int,
    cohort_week: int,
    models: TrajectoryModels,
) -> tuple[float, float]:
    """90% interval: binomial sampling + Greenwood shape + extrapolation inflation."""
    n_eff = max(int(n), 1)
    var_binom = max(rate * (1.0 - rate), 0.0) / n_eff
    var_shape = float(models.greenwood_var[age - 1])
    half = INTERVAL_Z_SCORE * np.sqrt(var_binom + var_shape)

    # Weeks of forecast past the observed boundary (cohort w observed to age 14-w).
    extrap_weeks = max(0, age - (N_COHORT_WEEKS + 1 - cohort_week))
    half *= 1.0 + models.extrap_inflation_per_week * extrap_weeks

    half = max(MIN_INTERVAL_HALF_WIDTH, half)
    return float(np.clip(rate - half, 0.0, 1.0)), float(np.clip(rate + half, 0.0, 1.0))


def enforce_monotone_cohort(curve: pd.DataFrame) -> pd.DataFrame:
    """Step 6: cumulative rates and interval bounds non-decrease in loan age."""
    out = curve.sort_values("loan_age_weeks").copy()
    out["cumulative_default_rate"] = out["cumulative_default_rate"].cummax()

    lowers: list[float] = []
    uppers: list[float] = []
    running_lower = 0.0
    running_upper = 0.0
    for _, row in out.iterrows():
        rate = float(row["cumulative_default_rate"])
        lo = min(float(row["cdr_lower_90"]), rate)
        hi = max(float(row["cdr_upper_90"]), rate)
        running_lower = max(running_lower, lo)
        running_upper = max(running_upper, hi, running_lower)
        lowers.append(running_lower)
        uppers.append(running_upper)

    out["cdr_lower_90"] = np.clip(lowers, 0.0, 1.0)
    out["cdr_upper_90"] = np.clip(uppers, 0.0, 1.0)
    out["cdr_lower_90"] = np.minimum(out["cdr_lower_90"], out["cumulative_default_rate"])
    out["cdr_upper_90"] = np.maximum(out["cdr_upper_90"], out["cumulative_default_rate"])
    return out


def _cohort_curve(
    models: TrajectoryModels, members: pd.DataFrame, cohort_week: int
) -> np.ndarray:
    """Length-13 CDR curve for a cohort, with thin/empty KM failsafes."""
    n_approved = len(members)
    hist = models.cohort_km_cdr.get(cohort_week, models.global_km_cdr)

    if n_approved == 0:
        return np.asarray(hist, dtype=float)

    portfolio = _individual_cdr_matrix(models, members).mean(axis=0)
    if n_approved < models.min_approved_cohort_size:
        bw = models.blend_weight
        return (1.0 - bw) * portfolio + bw * np.asarray(hist, dtype=float)
    return portfolio


def build_trajectory_grid(
    models: TrajectoryModels,
    scoring_df: pd.DataFrame,
    template: pd.DataFrame,
) -> pd.DataFrame:
    """Steps 2, 5, 7: fill the 169-row grid from A decisions + the hazard model."""
    _ = template  # grid is rebuilt deterministically below; template kept for the API
    out_parts: list[pd.DataFrame] = []

    for w in range(1, N_COHORT_WEEKS + 1):
        members = scoring_df[
            (scoring_df["cohort_week"] == w) & (scoring_df["decision"] == 1)
        ]
        n_approved = len(members)
        curve = _cohort_curve(models, members, w)

        rows: list[dict[str, float | int]] = []
        for age in range(1, N_LOAN_AGE_WEEKS + 1):
            rate = float(curve[age - 1])
            lower, upper = _interval(rate, n_approved, age, w, models)
            rows.append(
                {
                    "cohort_week": w,
                    "loan_age_weeks": age,
                    "cumulative_default_rate": rate,
                    "cdr_lower_90": lower,
                    "cdr_upper_90": upper,
                }
            )
        out_parts.append(enforce_monotone_cohort(pd.DataFrame(rows)))

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