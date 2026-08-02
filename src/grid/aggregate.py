"""
Quarterly aggregation of half-hourly GB grid data (spec section 6).

IMPORTANT: `Grid_CI_*` and the generation-mix shares describe the GB
electricity SYSTEM's carbon intensity and fuel mix - they are external
predictors, not a re-measurement of total UK CO2e (which covers all
sectors, not just electricity). Never substitute one for the other.

Compact quarterly feature block (10 features - spec section 7's exact list;
the final 12-item grid block, spec section 9, adds the 2 OWID_*_L1Y columns
built separately in src/owid/ - deliberately not one column per fuel, since
generation shares are compositional/collinear and the quarterly sample is
small):

  Grid_CI_mean, Grid_CI_p90, Grid_CI_std,
  Grid_CI_high_share, Grid_CI_low_share,
  Grid_renewable_share, Grid_low_carbon_share, Grid_fossil_share,
  Grid_gas_share, Grid_wind_share

Grid_low_carbon_share = renewables + nuclear (spec 7.2: nuclear is
low-carbon but NOT renewable - kept as a distinct column from
Grid_renewable_share, never used interchangeably).

High/low intensity classification uses the Carbon Intensity API's own
official half-hourly `index` band (very low/low/moderate/high/very high)
rather than an invented numeric gCO2/kWh cutoff - this is the documented,
reproducible convention (see spec 6.2's "fixed physical thresholds" option).
Training-relative (P90-of-training) thresholds are NOT implemented here:
doing so correctly requires recomputing the threshold per outer CV fold
(spec 6.2), which would mean the grid feature block could no longer be
precomputed once and reused across folds like every other feature family in
this pipeline. This is a scoped, documented simplification - see the final
report - not an oversight.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.logging_utils import get_logger

RENEWABLE_FUELS = ["biomass", "hydro", "solar", "wind"]
FOSSIL_FUELS = ["gas", "coal"]
HIGH_INDEX_BANDS = {"high", "very high"}
LOW_INDEX_BANDS = {"very low", "low"}


def load_raw_halfhourly(cache_dir: Path) -> pd.DataFrame:
    """
    Parse every cached intensity_*.json / generation_*.json file in
    `cache_dir` into a single half-hourly DataFrame indexed by the interval
    start time (UTC), deduplicated on overlapping chunk boundaries.
    """
    logger = get_logger()
    cache_dir = Path(cache_dir)

    intensity_records = {}
    for path in sorted(cache_dir.glob("intensity_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload.get("data", []):
            ts = pd.Timestamp(row["from"])
            intensity_records[ts] = {
                "intensity_actual": row["intensity"].get("actual"),
                "intensity_forecast": row["intensity"].get("forecast"),
                "intensity_index": row["intensity"].get("index"),
            }

    generation_records = {}
    for path in sorted(cache_dir.glob("generation_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload.get("data", []):
            ts = pd.Timestamp(row["from"])
            mix = {f"gen_{item['fuel']}": item["perc"] for item in row.get("generationmix", [])}
            generation_records[ts] = mix

    if not intensity_records:
        raise FileNotFoundError(
            f"No cached intensity_*.json files found in {cache_dir}. "
            f"Run fetch_and_cache_range() first."
        )

    df_intensity = pd.DataFrame.from_dict(intensity_records, orient="index").sort_index()
    df_gen = pd.DataFrame.from_dict(generation_records, orient="index").sort_index()

    df = df_intensity.join(df_gen, how="left")
    df.index.name = "interval_start"

    logger.info(
        f"Loaded {len(df)} half-hourly grid intervals from {cache_dir} "
        f"({df.index.min()} to {df.index.max()})"
    )
    return df


def _longest_true_streak(mask: pd.Series) -> int:
    """Longest run of consecutive True values in a boolean Series."""
    if mask.empty:
        return 0
    groups = (mask != mask.shift()).cumsum()
    streak_lengths = mask.groupby(groups).sum()
    return int(streak_lengths[mask.groupby(groups).first().astype(bool)].max()) if mask.any() else 0


def _impute_intensity(df: pd.DataFrame) -> Tuple[pd.Series, int]:
    """
    Missing-value rule (spec section 6.3): prefer the `actual` intensity
    reading; where it is null, fall back to `forecast` for that same
    half-hour (the API occasionally omits `actual` for the most recent
    settlement periods before outturn data is finalised); remaining gaps are
    left as NaN and excluded from the quarter's completeness count.

    Returns (imputed series, n_forecast_fallback_used).
    """
    actual = df["intensity_actual"]
    forecast = df["intensity_forecast"]
    used_fallback = actual.isna() & forecast.notna()
    imputed = actual.where(~used_fallback, forecast)
    return imputed, int(used_fallback.sum())


def aggregate_to_quarterly(
    df_halfhourly: pd.DataFrame,
    min_completeness: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate half-hourly grid data to quarterly features.

    Args:
        df_halfhourly: output of load_raw_halfhourly().
        min_completeness: minimum fraction of expected half-hourly
            observations required for a quarter to be included
            (configurable per spec section 6.3 / grid.minimum_quarter_completeness).

    Returns:
        (quarterly_features, quality_report) - quality_report has one row
        per quarter with expected/observed counts, completeness ratio,
        imputation count, and inclusion_status, independent of whether the
        quarter passed the threshold (Table 4 is built directly from this).
    """
    logger = get_logger()
    df = df_halfhourly.copy()
    df["intensity_imputed"], df["_fallback_used"] = _impute_intensity(df)
    df["_is_high"] = df["intensity_index"].isin(HIGH_INDEX_BANDS)
    df["_is_low"] = df["intensity_index"].isin(LOW_INDEX_BANDS)

    gen_cols = [c for c in df.columns if c.startswith("gen_")]
    for col in gen_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["_renewable_share_row"] = df[[f"gen_{f}" for f in RENEWABLE_FUELS if f"gen_{f}" in df.columns]].sum(axis=1, min_count=1)
    low_carbon_fuels = RENEWABLE_FUELS + ["nuclear"]
    df["_low_carbon_share_row"] = df[[f"gen_{f}" for f in low_carbon_fuels if f"gen_{f}" in df.columns]].sum(axis=1, min_count=1)
    df["_fossil_share_row"] = df[[f"gen_{f}" for f in FOSSIL_FUELS if f"gen_{f}" in df.columns]].sum(axis=1, min_count=1)

    df["_quarter"] = df.index.to_period("Q")

    feature_rows = []
    quality_rows = []

    for period, g in df.groupby("_quarter"):
        q_start = period.start_time
        n_days = (period.end_time.normalize() - period.start_time.normalize()).days + 1
        expected_obs = n_days * 48
        observed_obs = int(g["intensity_imputed"].notna().sum())
        completeness = observed_obs / expected_obs if expected_obs > 0 else 0.0
        n_fallback = int(g["_fallback_used"].sum())
        n_missing = expected_obs - observed_obs
        included = completeness >= min_completeness

        quality_rows.append({
            "quarter": q_start,
            "expected_obs": expected_obs,
            "observed_obs": observed_obs,
            "completeness_ratio": completeness,
            "n_missing_intervals": n_missing,
            "n_forecast_fallback_imputed": n_fallback,
            "imputation_rule": "actual; fallback to forecast if actual is null; else NaN",
            "min_completeness_threshold": min_completeness,
            "inclusion_status": "included" if included else "excluded_incomplete",
        })

        if not included:
            continue

        ci = g["intensity_imputed"].dropna()
        if ci.empty:
            continue

        row = {
            "quarter": q_start,
            "Grid_CI_mean": ci.mean(),
            "Grid_CI_p90": ci.quantile(0.90),
            "Grid_CI_std": ci.std(),
            "Grid_CI_high_share": g["_is_high"].mean(),
            "Grid_CI_low_share": g["_is_low"].mean(),
            "Grid_renewable_share": g["_renewable_share_row"].mean(),
            "Grid_low_carbon_share": g["_low_carbon_share_row"].mean(),
            "Grid_fossil_share": g["_fossil_share_row"].mean(),
            "Grid_gas_share": g["gen_gas"].mean() if "gen_gas" in g.columns else np.nan,
            "Grid_wind_share": g["gen_wind"].mean() if "gen_wind" in g.columns else np.nan,
        }
        feature_rows.append(row)

    quarterly_feature_columns = [
        "Grid_CI_mean", "Grid_CI_p90", "Grid_CI_std",
        "Grid_CI_high_share", "Grid_CI_low_share",
        "Grid_renewable_share", "Grid_low_carbon_share", "Grid_fossil_share",
        "Grid_gas_share", "Grid_wind_share",
    ]
    if feature_rows:
        quarterly_features = pd.DataFrame(feature_rows).set_index("quarter").sort_index()
    else:
        # No quarter passed the completeness threshold - pd.DataFrame([])
        # has no columns at all, so .set_index("quarter") would raise
        # KeyError. Return a well-formed empty frame instead of crashing.
        quarterly_features = pd.DataFrame(
            columns=quarterly_feature_columns
        ).rename_axis("quarter")
    quality_report = pd.DataFrame(quality_rows).set_index("quarter").sort_index()

    n_excluded = (quality_report["inclusion_status"] != "included").sum()
    logger.info(
        f"Aggregated grid data to {len(quarterly_features)} quarters "
        f"(min_completeness={min_completeness}); {n_excluded} quarter(s) "
        f"excluded for incompleteness"
    )

    return quarterly_features, quality_report
