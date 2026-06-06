"""PD and recovery models for Deliverable A."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.feature_engineering import (
    FEATURE_COLUMNS,
    ENGINEERED_FEATURES,
    RAW_BOOLEAN_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
    approved_matured_mask,
    build_features,
    daily_draw,
)


@dataclass
class UnderwritingModels:
    pd_model: Pipeline
    pd_calibrator: IsotonicRegression
    recovery_model: HistGradientBoostingRegressor
    default_day_mean: float
    interval_half_width: float


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    featured = build_features(df)
    return featured[FEATURE_COLUMNS]


def _build_preprocessor() -> ColumnTransformer:
    numeric_cols = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
    return ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_cols,
            ),
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
                RAW_CATEGORICAL_FEATURES,
            ),
            ("bool", "passthrough", RAW_BOOLEAN_FEATURES),
        ]
    )


def train_models(train: pd.DataFrame, validation: pd.DataFrame) -> UnderwritingModels:
    """Train PD (approved+matured) and recovery (defaults) models."""
    train_mask = approved_matured_mask(train)
    train_pd = train.loc[train_mask].copy()
    y_pd = train_pd["default_flag"].astype(int)

    x_train = _feature_matrix(train_pd)
    x_val = _feature_matrix(validation.loc[approved_matured_mask(validation)])

    pd_model = Pipeline(
        steps=[
            ("prep", _build_preprocessor()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.08,
                    max_iter=200,
                    random_state=42,
                ),
            ),
        ]
    )
    pd_model.fit(x_train, y_pd)

    val_mask = approved_matured_mask(validation)
    val_pd_raw = pd_model.predict_proba(_feature_matrix(validation))[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_pd_raw[val_mask], validation.loc[val_mask, "default_flag"].astype(int))

    default_rows = train_pd[train_pd["default_flag"] == 1]
    default_day_mean = float(default_rows["days_to_default"].mean())

    recovery_target = (
        default_rows["final_recovered_amount"] / default_rows["requested_amount"]
    ).clip(0.0, 1.0)
    recovery_model = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.08,
        max_iter=150,
        random_state=42,
    )
    recovery_model.fit(_feature_matrix(default_rows), recovery_target)

    calibrated_val = calibrator.predict(val_pd_raw[val_mask])
    interval_half_width = float(
        np.quantile(np.abs(calibrated_val - validation.loc[val_mask, "default_flag"]), 0.90)
    )
    interval_half_width = max(interval_half_width, 0.05)

    return UnderwritingModels(
        pd_model=pd_model,
        pd_calibrator=calibrator,
        recovery_model=recovery_model,
        default_day_mean=default_day_mean,
        interval_half_width=interval_half_width,
    )


def predict_pd(models: UnderwritingModels, df: pd.DataFrame) -> np.ndarray:
    raw = models.pd_model.predict_proba(_feature_matrix(df))[:, 1]
    return np.clip(models.pd_calibrator.predict(raw), 0.0, 1.0)


def predict_recovery_amount(
    models: UnderwritingModels, df: pd.DataFrame
) -> np.ndarray:
    rate = np.clip(models.recovery_model.predict(_feature_matrix(df)), 0.0, 1.0)
    r = df["requested_amount"].to_numpy(dtype=float)
    total_repayment = r * (1.0 + 0.35 * 60.0 / 365.0)
    outstanding = np.maximum(
        0.0,
        total_repayment - models.default_day_mean * daily_draw(df["requested_amount"]),
    )
    return rate * outstanding


def pd_intervals(
    models: UnderwritingModels, pd_hat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.clip(pd_hat - models.interval_half_width, 0.0, 1.0)
    upper = np.clip(pd_hat + models.interval_half_width, 0.0, 1.0)
    return lower, upper
