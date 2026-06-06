"""Counterfactual PD predictions for Deliverable C."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constants import (
    MIN_CF_INTERVAL_HALF_WIDTH,
    NON_INTERVENABLE_BLEND,
    NON_INTERVENABLE_INTERVAL_MULTIPLIER,
)
from src.interventions import apply_do, classify_queries, load_feature_registry
from src.models import UnderwritingModels, pd_intervals, predict_pd


@dataclass
class CounterfactualRunStats:
    n_queries: int
    n_intervenable: int
    n_non_intervenable: int
    mean_interval_width: float
    mean_interval_width_non_intervenable: float


def _enforce_interval(
    pd_hat: float, lower: float, upper: float
) -> tuple[float, float, float]:
    pd_hat = float(np.clip(pd_hat, 0.0, 1.0))
    lower = float(np.clip(min(lower, pd_hat), 0.0, 1.0))
    upper = float(np.clip(max(upper, pd_hat), 0.0, 1.0))
    return pd_hat, lower, upper


def _widen_interval(
    pd_hat: float,
    lower: float,
    upper: float,
    *,
    multiplier: float,
) -> tuple[float, float, float]:
    half = max((upper - lower) / 2.0 * multiplier, MIN_CF_INTERVAL_HALF_WIDTH)
    lower = np.clip(pd_hat - half, 0.0, 1.0)
    upper = np.clip(pd_hat + half, 0.0, 1.0)
    return _enforce_interval(pd_hat, lower, upper)


def predict_counterfactuals(
    models: UnderwritingModels,
    applicants: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    non_intervenable_blend: float = NON_INTERVENABLE_BLEND,
    non_intervenable_interval_multiplier: float = NON_INTERVENABLE_INTERVAL_MULTIPLIER,
) -> tuple[pd.DataFrame, CounterfactualRunStats, pd.DataFrame]:
    """Build submission_C rows for all queries."""
    registry = load_feature_registry()
    annotated = classify_queries(queries, registry)

    lookup = applicants.set_index("applicant_id", drop=False)
    rows: list[dict[str, float | str]] = []
    widths: list[float] = []
    widths_non: list[float] = []

    for q in annotated.itertuples(index=False):
        app = lookup.loc[q.applicant_id]
        cf_df = apply_do(app, q.feature_name, q.intervention_value, registry)
        obs_df = pd.DataFrame([app])

        pd_cf_raw = float(predict_pd(models, cf_df)[0])
        pd_obs = float(predict_pd(models, obs_df)[0])

        if q.intervenable:
            pd_hat = pd_cf_raw
            lower, upper = pd_intervals(models, cf_df, np.array([pd_hat]))
            lower, upper = float(lower[0]), float(upper[0])
        else:
            # Non-intervenable trap: do() is ill-defined — blend toward observational PD.
            alpha = float(np.clip(non_intervenable_blend, 0.0, 1.0))
            pd_hat = (1.0 - alpha) * pd_cf_raw + alpha * pd_obs
            lower, upper = pd_intervals(models, cf_df, np.array([pd_hat]))
            pd_hat, lower, upper = _widen_interval(
                pd_hat,
                float(lower[0]),
                float(upper[0]),
                multiplier=non_intervenable_interval_multiplier,
            )

        pd_hat, lower, upper = _enforce_interval(pd_hat, lower, upper)
        width = upper - lower
        widths.append(width)
        if not q.intervenable:
            widths_non.append(width)

        rows.append(
            {
                "query_id": q.query_id,
                "predicted_pd_cf": pd_hat,
                "pd_cf_lower_90": lower,
                "pd_cf_upper_90": upper,
            }
        )

    submission = pd.DataFrame(rows)
    stats = CounterfactualRunStats(
        n_queries=len(submission),
        n_intervenable=int(annotated["intervenable"].sum()),
        n_non_intervenable=int((~annotated["intervenable"]).sum()),
        mean_interval_width=float(np.mean(widths)),
        mean_interval_width_non_intervenable=float(np.mean(widths_non))
        if widths_non
        else float("nan"),
    )
    return submission, stats, annotated
