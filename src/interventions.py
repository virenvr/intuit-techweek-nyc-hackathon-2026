"""Intervention registry and do(feature=v) application for Deliverable C."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import DATA_DICTIONARY_PATH
from src.feature_engineering import requested_to_observed_revenue

BANK_FEED_FEATURES = frozenset(
    {
        "observed_monthly_revenue_avg_3mo",
        "observed_revenue_trend_3mo",
        "observed_revenue_volatility",
        "observed_cash_balance_p10",
        "observed_overdraft_count_3mo",
        "payroll_regularity_score",
    }
)

DERIVED_CONTEXT_FEATURES = frozenset({"requested_amount_to_observed_revenue"})


@dataclass(frozen=True)
class FeatureMeta:
    field: str
    intervenable: bool
    group: str


def load_feature_registry(
    path: str | Path = DATA_DICTIONARY_PATH,
) -> dict[str, FeatureMeta]:
    """Load intervenable flag and group from data_dictionary.csv."""
    dd = pd.read_csv(path)
    registry: dict[str, FeatureMeta] = {}
    for row in dd.itertuples(index=False):
        intervenable = str(row.intervenable).strip().lower() in {"true", "1", "yes"}
        registry[row.field] = FeatureMeta(
            field=row.field,
            intervenable=intervenable,
            group=str(row.group),
        )
    return registry


def get_feature_meta(feature_name: str, registry: dict[str, FeatureMeta]) -> FeatureMeta:
    if feature_name not in registry:
        raise KeyError(f"Unknown feature '{feature_name}' — not in data dictionary")
    return registry[feature_name]


def recompute_derived_context_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute application_context ratios affected by interventions."""
    out = df.copy()
    if (
        "requested_amount" in out.columns
        and "observed_monthly_revenue_avg_3mo" in out.columns
    ):
        out["requested_amount_to_observed_revenue"] = requested_to_observed_revenue(out)
    return out


def apply_do(
    applicant: pd.Series | pd.DataFrame,
    feature_name: str,
    intervention_value: float | int | bool,
    registry: dict[str, FeatureMeta],
) -> pd.DataFrame:
    """Apply do(feature = v) holding other applicant features fixed.

    Side effects (v1 contract logic):
      - Bank-feed intervention implies feed is linked.
      - Derived context ratios recomputed when inputs change.
    """
    if isinstance(applicant, pd.Series):
        row = applicant.copy()
    else:
        if len(applicant) != 1:
            raise ValueError("apply_do expects a single applicant row")
        row = applicant.iloc[0].copy()

    meta = get_feature_meta(feature_name, registry)
    row[feature_name] = intervention_value

    if feature_name in BANK_FEED_FEATURES or meta.group == "bank_feed":
        row["has_linked_bank_feed"] = True

    out = pd.DataFrame([row])
    out = recompute_derived_context_fields(out)
    return out


def classify_queries(
    queries: pd.DataFrame, registry: dict[str, FeatureMeta]
) -> pd.DataFrame:
    """Annotate queries with intervenable flag and group."""
    out = queries.copy()
    out["intervenable"] = out["feature_name"].map(
        lambda f: registry[f].intervenable if f in registry else False
    )
    out["feature_group"] = out["feature_name"].map(
        lambda f: registry[f].group if f in registry else "unknown"
    )
    return out