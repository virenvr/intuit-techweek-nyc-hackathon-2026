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
    default_day_model: HistGradientBoostingRegressor
    residual_quantile_model: Pipeline | None
    npv_threshold: float
    feature_cols: list[str]
    include_prior_underwriter_score: bool
    default_day_mean: float
    min_interval_half_width: float = 0.02
    interval_method: str = "wilson"
    bootstrap_ensemble: list[PDCalibratedModel] | None = None
    pd_interval_n: int = 1


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


def _make_pd_classifier(*, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.08,
                    max_iter=200,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _fit_pd_calibrated(
    train_pd: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    *,
    random_state: int,
) -> tuple[Pipeline, IsotonicRegression]:
    """Fit one PD classifier + isotonic calibrator on validation labels."""
    x_train = _feature_matrix(train_pd, feature_cols)
    y_pd = train_pd["default_flag"].astype(int)
    pd_model = Pipeline(
        steps=[
            ("prep", _build_preprocessor(feature_cols)),
            ("clf", _make_pd_classifier(random_state=random_state).named_steps["clf"]),
        ]
    )
    pd_model.fit(x_train, y_pd)

    val_mask = approved_matured_mask(validation)
    x_val_all = _feature_matrix(validation, feature_cols)
    val_pd_raw = pd_model.predict_proba(x_val_all)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(
        val_pd_raw[val_mask],
        validation.loc[val_mask, "default_flag"].astype(int),
    )
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
    include_prior_underwriter_score: bool = False,
    npv_threshold: float = 0.0,
    interval_method: str = "wilson",
    n_bootstrap: int = 25,
    pd_interval_n: int | None = None,
    random_state: int = 42,
) -> UnderwritingModels:
    """Train PD (approved+matured), timing, recovery, and uncertainty models."""
    feature_cols = feature_columns(
        include_prior_underwriter_score=include_prior_underwriter_score
    )
    train_mask = approved_matured_mask(train)
    train_pd = train.loc[train_mask].copy()

    pd_model, calibrator = _fit_pd_calibrated(
        train_pd,
        validation,
        feature_cols,
        random_state=random_state,
    )

    val_mask = approved_matured_mask(validation)
    x_val_all = _feature_matrix(validation, feature_cols)
    val_pd_raw = pd_model.predict_proba(x_val_all)[:, 1]

    default_rows = train_pd[train_pd["default_flag"] == 1].copy()
    default_day_mean = float(default_rows["days_to_default"].mean())

    default_day_model = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.08,
        max_iter=150,
        random_state=42,
    )
    default_day_model.fit(
        _feature_matrix(default_rows, feature_cols),
        default_rows["days_to_default"].astype(float),
    )

    recovery_model = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.08,
        max_iter=150,
        random_state=42,
    )
    recovery_model.fit(
        _feature_matrix(default_rows, feature_cols),
        default_rows["final_recovered_amount"].astype(float),
    )

    bootstrap_ensemble: list[PDCalibratedModel] | None = None
    residual_quantile_model: Pipeline | None = None
    interval_n = int(pd_interval_n if pd_interval_n is not None else len(train_pd))

    if interval_method == "bootstrap":
        x_val_labeled = _feature_matrix(validation.loc[val_mask], feature_cols)
        calibrated_val = calibrator.predict(val_pd_raw[val_mask])
        y_val = validation.loc[val_mask, "default_flag"].astype(int).to_numpy()
        abs_residual = np.abs(y_val - calibrated_val)
        residual_quantile_model = Pipeline(
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
        residual_quantile_model.fit(x_val_labeled, abs_residual)
        bootstrap_ensemble = _train_bootstrap_ensemble(
            train_pd,
            validation,
            feature_cols,
            n_bootstrap=n_bootstrap,
            seed=random_state,
        )
    elif interval_method == "residual":
        x_val_labeled = _feature_matrix(validation.loc[val_mask], feature_cols)
        calibrated_val = calibrator.predict(val_pd_raw[val_mask])
        y_val = validation.loc[val_mask, "default_flag"].astype(int).to_numpy()
        abs_residual = np.abs(y_val - calibrated_val)
        residual_quantile_model = Pipeline(
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
        residual_quantile_model.fit(x_val_labeled, abs_residual)

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
    )


def predict_pd(models: UnderwritingModels, df: pd.DataFrame) -> np.ndarray:
    raw = models.pd_model.predict_proba(_feature_matrix(df, models.feature_cols))[:, 1]
    return np.clip(models.pd_calibrator.predict(raw), 0.0, 1.0)


def predict_default_day(models: UnderwritingModels, df: pd.DataFrame) -> np.ndarray:
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
    """90% Wilson score intervals from p-hat and effective group size n."""
    pd_hat = np.asarray(pd_hat, dtype=float)
    lower, upper = wilson_score_interval(pd_hat, models.pd_interval_n)
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
