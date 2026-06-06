"""PD and recovery models for Deliverable A."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.constants import DEFAULT_WINDOW_DAYS
from src.wilson import wilson_score_interval
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


@dataclass
class PDCalibratedModel:
    """Single bootstrap member: raw PD classifier + isotonic calibrator."""

    pd_model: Pipeline
    calibrator: IsotonicRegression


@dataclass
class UnderwritingModels:
    pd_model: Pipeline
    pd_calibrator: IsotonicRegression
    recovery_model: HistGradientBoostingRegressor
    default_day_model: HistGradientBoostingRegressor | None
    residual_quantile_model: Pipeline | None
    npv_threshold: float
    feature_cols: list[str]
    include_prior_underwriter_score: bool
    default_day_mean: float
    min_interval_half_width: float = 0.02
    interval_method: str = "residual"
    bootstrap_ensemble: list[PDCalibratedModel] | None = None
    pd_interval_n: int = 1
    # Maps each isotonic calibration level -> number of calibration points supporting
    # it. Used to give Wilson intervals an honest *local* sample size instead of the
    # global training-set size (which collapsed every interval to ~1/20th its width).
    pd_level_n: dict[float, int] | None = None


def _feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    featured = build_features(df)
    return featured[feature_cols]


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


def _make_pd_classifier(*, random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        random_state=random_state,
    )


def _fit_pd_calibrated(
    train_pd: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    *,
    random_state: int,
    calibration_df: pd.DataFrame | None = None,
) -> tuple[Pipeline, IsotonicRegression]:
    """Fit one PD classifier + isotonic calibrator.

    The classifier is trained on `train_pd`. The calibrator is fit on
    `calibration_df` if supplied, otherwise on `validation`. Passing a calibration
    fold that is disjoint from the reporting/validation fold removes the optimistic
    bias from fitting and evaluating calibration on the same rows.
    """
    x_train = _feature_matrix(train_pd, feature_cols)
    y_pd = train_pd["default_flag"].astype(int)
    pd_model = Pipeline(
        steps=[
            ("prep", _build_preprocessor(feature_cols)),
            ("clf", _make_pd_classifier(random_state=random_state)),
        ]
    )
    pd_model.fit(x_train, y_pd)

    cal_source = calibration_df if calibration_df is not None else validation
    cal_mask = approved_matured_mask(cal_source)
    x_cal_all = _feature_matrix(cal_source, feature_cols)
    cal_pd_raw = pd_model.predict_proba(x_cal_all)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    cal_y = cal_source.loc[cal_mask, "default_flag"].astype(int)
    if int(cal_mask.sum()) >= 2 and cal_y.nunique() > 1:
        calibrator.fit(cal_pd_raw[cal_mask.to_numpy()], cal_y)
    else:
        # Degenerate calibration fold: fall back to identity so predict_pd still works.
        calibrator.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    return pd_model, calibrator


def _train_bootstrap_ensemble(
    train_pd: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_bootstrap: int,
    seed: int,
) -> list[PDCalibratedModel]:
    """Train N bootstrap PD models; 5th/95th percentiles form 90% intervals."""
    n = len(train_pd)
    rng = np.random.default_rng(seed)
    ensemble: list[PDCalibratedModel] = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot = train_pd.iloc[idx]
        pd_model, calibrator = _fit_pd_calibrated(
            boot,
            validation,
            feature_cols,
            random_state=seed + b,
        )
        ensemble.append(PDCalibratedModel(pd_model=pd_model, calibrator=calibrator))
    return ensemble


def train_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    calibration: pd.DataFrame | None = None,
    include_prior_underwriter_score: bool = False,
    npv_threshold: float = 0.0,
    interval_method: str = "residual",
    n_bootstrap: int = 25,
    pd_interval_n: int | None = None,
    random_state: int = 42,
) -> UnderwritingModels:
    """Train PD (approved+matured), timing, recovery, and uncertainty models.

    If `calibration` is provided it is used to fit the isotonic calibrator and the
    interval support, leaving `validation` purely for reporting. This avoids the
    same-fold bias where calibration is fit and evaluated on identical rows.
    """
    feature_cols = feature_columns(
        include_prior_underwriter_score=include_prior_underwriter_score
    )
    train_mask = approved_matured_mask(train)
    train_pd = train.loc[train_mask].copy()
    cal_source = calibration if calibration is not None else validation

    pd_model, calibrator = _fit_pd_calibrated(
        train_pd,
        validation,
        feature_cols,
        random_state=random_state,
        calibration_df=cal_source,
    )

    cal_mask = approved_matured_mask(cal_source)
    x_cal_all = _feature_matrix(cal_source, feature_cols)
    cal_pd_raw = pd_model.predict_proba(x_cal_all)[:, 1]
    val_mask = cal_mask
    x_val_all = x_cal_all
    val_pd_raw = cal_pd_raw

    # Honest local sample size per calibration level (isotonic step), used by Wilson.
    calibrated_levels = np.clip(calibrator.predict(cal_pd_raw[cal_mask.to_numpy()]), 0.0, 1.0)
    levels, counts = np.unique(np.round(calibrated_levels, 9), return_counts=True)
    pd_level_n = {float(lv): int(c) for lv, c in zip(levels, counts)}

    default_rows = train_pd[train_pd["default_flag"] == 1].copy()
    default_day_model: HistGradientBoostingRegressor | None
    if default_rows.empty:
        default_day_mean = float(DEFAULT_WINDOW_DAYS / 2.0)
        default_day_model = None
        recovery_model = HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.08,
            max_iter=150,
            random_state=random_state,
        )
        recovery_model.fit(
            _feature_matrix(train_pd, feature_cols),
            np.zeros(len(train_pd), dtype=float),
        )
    else:
        default_day_mean = float(default_rows["days_to_default"].mean())
        default_day_model = HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.08,
            max_iter=150,
            random_state=random_state,
        )
        default_day_model.fit(
            _feature_matrix(default_rows, feature_cols),
            default_rows["days_to_default"].astype(float),
        )
        recovery_model = HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.08,
            max_iter=150,
            random_state=random_state,
        )
        recovery_model.fit(
            _feature_matrix(default_rows, feature_cols),
            default_rows["final_recovered_amount"].astype(float),
        )

    bootstrap_ensemble: list[PDCalibratedModel] | None = None
    residual_quantile_model: Pipeline | None = None
    interval_n = int(pd_interval_n if pd_interval_n is not None else len(train_pd))

    def _fit_residual_quantile_model() -> Pipeline:
        """Per-row 0.90-quantile of |y - p_hat|, fit on the calibration fold."""
        labeled = cal_source.loc[cal_mask]
        x_labeled = _feature_matrix(labeled, feature_cols)
        calibrated = np.clip(calibrator.predict(val_pd_raw[cal_mask.to_numpy()]), 0.0, 1.0)
        y = labeled["default_flag"].astype(int).to_numpy()
        abs_residual = np.abs(y - calibrated)
        model = Pipeline(
            steps=[
                ("prep", _build_preprocessor(feature_cols)),
                (
                    "reg",
                    GradientBoostingRegressor(
                        loss="quantile",
                        alpha=0.90,
                        max_depth=4,
                        learning_rate=0.08,
                        n_estimators=120,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        model.fit(x_labeled, abs_residual)
        return model

    if interval_method == "bootstrap":
        residual_quantile_model = _fit_residual_quantile_model()
        bootstrap_ensemble = _train_bootstrap_ensemble(
            train_pd,
            cal_source,
            feature_cols,
            n_bootstrap=n_bootstrap,
            seed=random_state,
        )
    elif interval_method == "residual":
        residual_quantile_model = _fit_residual_quantile_model()
    elif interval_method != "wilson":
        raise ValueError(
            f"Unknown interval_method {interval_method!r}; "
            "expected 'residual', 'bootstrap', or 'wilson'."
        )

    return UnderwritingModels(
        pd_model=pd_model,
        pd_calibrator=calibrator,
        recovery_model=recovery_model,
        default_day_model=default_day_model,
        residual_quantile_model=residual_quantile_model,
        npv_threshold=npv_threshold,
        feature_cols=feature_cols,
        include_prior_underwriter_score=include_prior_underwriter_score,
        default_day_mean=default_day_mean,
        interval_method=interval_method,
        bootstrap_ensemble=bootstrap_ensemble,
        pd_interval_n=interval_n,
        pd_level_n=pd_level_n,
    )


def predict_pd(models: UnderwritingModels, df: pd.DataFrame) -> np.ndarray:
    raw = models.pd_model.predict_proba(_feature_matrix(df, models.feature_cols))[:, 1]
    return np.clip(models.pd_calibrator.predict(raw), 0.0, 1.0)


def predict_default_day(models: UnderwritingModels, df: pd.DataFrame) -> np.ndarray:
    if models.default_day_model is None:
        n = len(df)
        return np.clip(
            np.full(n, models.default_day_mean), 1.0, float(DEFAULT_WINDOW_DAYS)
        )
    pred = models.default_day_model.predict(_feature_matrix(df, models.feature_cols))
    return np.clip(pred, 1.0, float(DEFAULT_WINDOW_DAYS))


def predict_recovery_amount(models: UnderwritingModels, df: pd.DataFrame) -> np.ndarray:
    """Predict recovery in dollars (final_recovered_amount), clipped to non-negative."""
    return np.maximum(
        models.recovery_model.predict(_feature_matrix(df, models.feature_cols)),
        0.0,
    )


def _wilson_pd_intervals(
    models: UnderwritingModels,
    pd_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """90% Wilson score intervals using an honest *local* effective sample size.

    The previous version used a single global n (the full training-set size), which
    made every interval ~1/sqrt(N) too tight. Here each row's n is the number of
    calibration points sharing its isotonic level; rows whose level is unseen fall
    back to the median level count (a conservative, non-degenerate floor).
    """
    pd_hat = np.asarray(pd_hat, dtype=float)
    level_n = models.pd_level_n
    if level_n:
        fallback = float(np.median(list(level_n.values())))
        keys = np.round(pd_hat, 9)
        n_eff = np.array([level_n.get(float(k), fallback) for k in keys], dtype=float)
    else:
        # No calibration support map: fall back to the configured interval n.
        n_eff = np.full(pd_hat.shape, float(max(models.pd_interval_n, 1)))
    n_eff = np.maximum(n_eff, 1.0)
    lower, upper = wilson_score_interval(pd_hat, n_eff)
    lower = np.minimum(lower, pd_hat)
    upper = np.maximum(upper, pd_hat)
    return lower, upper


def _bootstrap_pd_intervals(
    models: UnderwritingModels,
    df: pd.DataFrame,
    pd_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """90% intervals from bootstrap ensemble (5th / 95th percentiles)."""
    if not models.bootstrap_ensemble:
        raise ValueError("bootstrap_ensemble is required for bootstrap intervals")

    x = _feature_matrix(df, models.feature_cols)
    preds = []
    for member in models.bootstrap_ensemble:
        raw = member.pd_model.predict_proba(x)[:, 1]
        preds.append(np.clip(member.calibrator.predict(raw), 0.0, 1.0))
    stack = np.vstack(preds)
    lower = np.clip(np.percentile(stack, 5, axis=0), 0.0, 1.0)
    upper = np.clip(np.percentile(stack, 95, axis=0), 0.0, 1.0)
    pd_hat = np.asarray(pd_hat, dtype=float)
    lower = np.minimum(lower, pd_hat)
    upper = np.maximum(upper, pd_hat)

    # Widen with residual quantile floor so binary outcomes meet ~90% coverage.
    if models.residual_quantile_model is not None:
        res_lower, res_upper = _residual_pd_intervals(models, df, pd_hat)
        lower = np.minimum(lower, res_lower)
        upper = np.maximum(upper, res_upper)

    return lower, upper


def _residual_pd_intervals(
    models: UnderwritingModels,
    df: pd.DataFrame,
    pd_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row 90% intervals from feature-dependent calibration residuals."""
    if models.residual_quantile_model is None:
        raise ValueError("residual_quantile_model is required for residual intervals")

    half_width = models.residual_quantile_model.predict(
        _feature_matrix(df, models.feature_cols)
    )
    half_width = np.maximum(half_width, models.min_interval_half_width)
    pd_hat = np.asarray(pd_hat, dtype=float)
    lower = np.clip(pd_hat - half_width, 0.0, 1.0)
    upper = np.clip(pd_hat + half_width, 0.0, 1.0)
    lower = np.minimum(lower, pd_hat)
    upper = np.maximum(upper, pd_hat)
    return lower, upper


def pd_intervals(
    models: UnderwritingModels,
    df: pd.DataFrame,
    pd_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """90% prediction intervals (Wilson, bootstrap, or residual quantile)."""
    if models.interval_method == "wilson":
        return _wilson_pd_intervals(models, pd_hat)
    if models.interval_method == "bootstrap":
        return _bootstrap_pd_intervals(models, df, pd_hat)
    return _residual_pd_intervals(models, df, pd_hat)


def validation_interval_coverage(
    models: UnderwritingModels,
    validation: pd.DataFrame,
) -> float:
    """Fraction of labeled validation rows where binary outcome lies in [lower, upper]."""
    mask = approved_matured_mask(validation)
    labeled = validation.loc[mask]
    pd_hat = predict_pd(models, labeled)
    lower, upper = pd_intervals(models, labeled, pd_hat)
    y = labeled["default_flag"].astype(float).to_numpy()
    return float(np.mean((y >= lower) & (y <= upper)))