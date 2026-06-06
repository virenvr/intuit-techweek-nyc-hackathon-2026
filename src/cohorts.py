"""Cohort-week assignment for Deliverable B (Step 1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import COHORT_DEFINITIONS_PATH


def load_cohort_definitions(
    path: str | Path = COHORT_DEFINITIONS_PATH,
) -> pd.DataFrame:
    """Load cohort_week -> [start_date, end_date] mapping."""
    defs = pd.read_csv(path)
    defs["start_date"] = pd.to_datetime(defs["start_date"])
    defs["end_date"] = pd.to_datetime(defs["end_date"])
    return defs.sort_values("cohort_week").reset_index(drop=True)


def assign_cohort_week(
    df: pd.DataFrame,
    cohort_defs: pd.DataFrame,
    *,
    timestamp_col: str = "application_timestamp",
    strict: bool = True,
) -> pd.Series:
    """Map each application timestamp to cohort_week 1-13 (inclusive end dates).

    Rows outside the 13 scoring cohort windows receive NA. Set strict=False to allow
    that (historical train data). Set strict=True for validation/test scoring rows.
    """
    # Normalize to calendar dates so midnight boundaries do not spill across weeks.
    ts = pd.to_datetime(df[timestamp_col]).dt.normalize()
    cohort = pd.Series(pd.NA, index=df.index, dtype="Int64")

    for row in cohort_defs.itertuples(index=False):
        start = pd.Timestamp(row.start_date).normalize()
        end = pd.Timestamp(row.end_date).normalize()
        in_range = (ts >= start) & (ts <= end)
        # First match wins — prevents silent overwrite if definition windows overlap.
        cohort.loc[in_range & cohort.isna()] = int(row.cohort_week)

    missing = cohort.isna().sum()
    if strict and missing:
        raise ValueError(
            f"{missing} rows could not be mapped to a cohort week — check timestamps "
            f"against {COHORT_DEFINITIONS_PATH}"
        )
    if not strict and missing:
        # Historical train rows predate cohort windows; expected for Deliverable B v1.
        pass
    return cohort


def attach_cohort_week(
    df: pd.DataFrame,
    cohort_defs: pd.DataFrame | None = None,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Return copy of df with cohort_week column attached."""
    out = df.copy()
    defs = cohort_defs if cohort_defs is not None else load_cohort_definitions()
    out["cohort_week"] = assign_cohort_week(out, defs, strict=strict)
    return out
