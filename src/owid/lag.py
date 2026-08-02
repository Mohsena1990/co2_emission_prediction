"""
One-year-lag alignment for OWID annual shares (spec section 8.2).

OWID shares are annual, not quarterly. Interpolating a completed annual
value across its own year (and using it to forecast quarters within that
same year) would expose future annual information - e.g. using 2021's full
renewable share, only knowable after 2021 ends, to forecast 2021Q2. Instead:
for every quarter in year y, use no annual value later than year y-1's -
the primary leakage-safe rule (spec 8.2 permits a stricter release-date-aware
merge_asof if reliable OWID release dates can be reconstructed; out of scope
here, documented rather than silently approximated).
"""
from __future__ import annotations

import pandas as pd

from .fetch import resolve_value_column


def build_owid_lagged_feature(
    uk_df: pd.DataFrame,
    metadata: dict,
    feature_name: str,
    full_quarterly_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Build a quarterly Series where every quarter in year y is assigned
    year (y-1)'s OWID annual share value (spec 8.2's one-year-lag rule).
    Quarters in years before OWID's earliest year (y-1 unavailable) are
    left as NaN - this is genuine unavailability, not a "shock absent from
    a shock indicator" situation like mobility, so it is NOT neutral-zeroed;
    downstream configuration-matrix building drops NaN rows exactly as it
    already does for other externally-sourced blocks (e.g. grid).

    Args:
        uk_df, metadata: output of fetch.load_owid_uk_series() /
            load_cached_owid_uk_series().
        feature_name: output Series name, e.g. 'OWID_RenewableShare_L1Y'.
        full_quarterly_index: the quarterly sample index to align onto
            (e.g. the main engineered feature matrix's DatetimeIndex).

    Returns:
        pd.Series indexed by full_quarterly_index.
    """
    value_col = resolve_value_column(uk_df, metadata)
    year_to_value = uk_df.set_index("Year")[value_col].to_dict()

    lagged_values = [year_to_value.get(q_start.year - 1) for q_start in full_quarterly_index]
    return pd.Series(lagged_values, index=full_quarterly_index, name=feature_name, dtype=float)
