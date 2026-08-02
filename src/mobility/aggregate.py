"""
Quarterly aggregation of Google COVID-19 mobility-shock data (spec section 4).

IMPORTANT (spec 4.2): the six `Mobility_*` columns are COVID mobility-SHOCK
indicators - daily percentage deviations from a pre-COVID baseline - not a
complete or continuously observed UK mobility series. They are only
meaningful during the official Google reporting window
(~2020-02-15 to mid-October 2022).

Two-step pipeline, mirroring src/grid/aggregate.py's completeness-threshold
pattern:
  1. `aggregate_to_quarterly()` - daily -> quarterly mean per category,
     with completeness-threshold exclusion AND a primary-vs-sensitivity
     split at `primary_cutoff_quarter` (spec 4.3: official data end during
     2022Q4, so complete quarters through 2022Q3 are primary; partial
     2022Q4 is a sensitivity-only observation, not silently included).
  2. `apply_neutral_zero_convention()` - reindex onto the FULL quarterly
     sample index and fill non-reporting-window quarters with 0.0 (spec
     4.4: "no observed shock represented", not "measured baseline" - never
     interpolated).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..core.logging_utils import get_logger
from .fetch import CATEGORY_COLUMN_MAP


def aggregate_to_quarterly(
    df_daily: pd.DataFrame,
    minimum_quarter_completeness: float = 0.80,
    primary_cutoff_quarter: Optional[str] = "2022Q3",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate UK-national daily mobility data (as returned by
    fetch.load_cached_uk_mobility) to quarterly features.

    Args:
        df_daily: must have a 'date' column plus the six raw Google
            percent-change columns (CATEGORY_COLUMN_MAP keys).
        minimum_quarter_completeness: minimum fraction of expected daily
            observations (averaged across the six categories) required for
            a quarter to be included in the primary series.
        primary_cutoff_quarter: last quarter eligible for the PRIMARY
            aggregation (e.g. "2022Q3"); any later quarter is excluded here
            regardless of completeness, since the reporting window ends
            mid-quarter. Pass None to disable the cutoff (used for the
            explicit partial-2022Q4 sensitivity variant, spec 4.3).

    Returns:
        (quarterly_features, quality_report). quality_report has one row
        per observed quarter with per-category observed/expected days,
        coverage ratio, missing-day count, first/last observed date, an
        overall completeness_ratio, and inclusion_status - independent of
        whether the quarter passed the threshold, so Table 10's coverage
        reporting can be built directly from it.
    """
    logger = get_logger()
    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["_quarter"] = df["date"].dt.to_period("Q")

    cutoff_period = pd.Period(primary_cutoff_quarter, freq="Q") if primary_cutoff_quarter else None

    feature_rows = []
    quality_rows = []

    for period, g in df.groupby("_quarter"):
        n_days_expected = (period.end_time.normalize() - period.start_time.normalize()).days + 1
        quality_row = {"quarter": period.start_time}
        cat_coverages = []

        for raw_col, feat_name in CATEGORY_COLUMN_MAP.items():
            observed_mask = g[raw_col].notna() if raw_col in g.columns else pd.Series(False, index=g.index)
            n_observed = int(observed_mask.sum())
            coverage = n_observed / n_days_expected if n_days_expected > 0 else 0.0
            cat_coverages.append(coverage)

            observed_dates = g.loc[observed_mask, "date"]
            quality_row[f"{feat_name}_observed_days"] = n_observed
            quality_row[f"{feat_name}_expected_days"] = n_days_expected
            quality_row[f"{feat_name}_coverage_ratio"] = coverage
            quality_row[f"{feat_name}_missing_days"] = n_days_expected - n_observed
            quality_row[f"{feat_name}_first_observed"] = (
                str(observed_dates.min().date()) if not observed_dates.empty else None
            )
            quality_row[f"{feat_name}_last_observed"] = (
                str(observed_dates.max().date()) if not observed_dates.empty else None
            )

        overall_completeness = float(np.mean(cat_coverages)) if cat_coverages else 0.0
        is_post_cutoff = cutoff_period is not None and period > cutoff_period
        meets_threshold = overall_completeness >= minimum_quarter_completeness

        if is_post_cutoff:
            inclusion_status = "excluded_post_primary_window"
        elif not meets_threshold:
            inclusion_status = "excluded_incomplete"
        else:
            inclusion_status = "included"

        quality_row["completeness_ratio"] = overall_completeness
        quality_row["min_completeness_threshold"] = minimum_quarter_completeness
        quality_row["inclusion_status"] = inclusion_status
        quality_rows.append(quality_row)

        if inclusion_status != "included":
            continue

        row = {"quarter": period.start_time}
        for raw_col, feat_name in CATEGORY_COLUMN_MAP.items():
            row[feat_name] = g[raw_col].mean() if raw_col in g.columns else np.nan
        feature_rows.append(row)

    feature_columns = list(CATEGORY_COLUMN_MAP.values())
    if feature_rows:
        quarterly_features = pd.DataFrame(feature_rows).set_index("quarter").sort_index()
    else:
        quarterly_features = pd.DataFrame(columns=feature_columns).rename_axis("quarter")
    quality_report = pd.DataFrame(quality_rows).set_index("quarter").sort_index()

    n_excluded = int((quality_report["inclusion_status"] != "included").sum())
    logger.info(
        f"Aggregated mobility data to {len(quarterly_features)} quarters "
        f"(min_completeness={minimum_quarter_completeness}, "
        f"primary_cutoff={primary_cutoff_quarter}); {n_excluded} quarter(s) excluded"
    )

    return quarterly_features, quality_report


def apply_neutral_zero_convention(
    quarterly_features: pd.DataFrame,
    full_quarterly_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    spec 4.4: outside the official Google mobility reporting window (and
    for any quarter aggregate_to_quarterly() excluded for incompleteness or
    the primary-cutoff rule), MobilityShock_{j,q} = 0 - "no observed Google
    COVID mobility deviation is represented", NOT "measured mobility equal
    to its baseline". Never interpolated backward into the pre-COVID period
    or forward beyond the reporting window.

    Args:
        quarterly_features: output of aggregate_to_quarterly() (index:
            quarter-start Timestamps, may be a strict subset of
            full_quarterly_index).
        full_quarterly_index: the complete quarterly sample index the
            forecasting pipeline uses (e.g. the main engineered feature
            matrix's DatetimeIndex).

    Returns:
        DataFrame reindexed onto full_quarterly_index with all
        outside-window/excluded quarters filled with 0.0.
    """
    feature_columns = list(CATEGORY_COLUMN_MAP.values())
    if quarterly_features.empty:
        quarterly_features = pd.DataFrame(columns=feature_columns, dtype=float)
    return quarterly_features.reindex(full_quarterly_index).fillna(0.0)
