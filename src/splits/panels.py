"""
Panel construction (originally spec section 9 of the earlier A1-A4-only
spec; now serves the revised spec's section 14 "fair sample period"
requirement for the Stream A/B architecture).

**PRIMARY_PERIOD_PANEL ('panel2_common_period') is the PRIMARY comparison
period for the whole Stream A/B study** (spec section 14): identical outer
folds/horizons/target dates across ALL of A1-A4/B1-B4, bounded by grid-data
availability (2018Q1 in practice - see src/grid/fetch.py::
EARLIEST_AVAILABLE_DATE for the spec's 2009Q1 assumption vs. the verified
actual start). Every candidate in the global Stream A (20) and Stream B
(100) MCDM rankings must be evaluated on this identical period.

`SENSITIVITY_LONG_PERIOD_PANEL` ('panel1_full_period', the longest valid
window for A1/A2 alone, no grid data required) is now a documented
SENSITIVITY-ONLY comparison (spec section 14 explicitly permits retaining
it, but bans it from the main global MCDM ranking since it uses a different
sample than A3/A4/B3/B4 can reach) - this is a demotion from the earlier
A1-A4-only spec, where the full-period comparison was A1/A2's primary
reporting basis.
"""
from dataclasses import replace
from typing import Dict, Optional

import pandas as pd

from ..core.config import SplitConfig
from ..core.logging_utils import get_logger
from .walk_forward import CVPlan, create_walk_forward_splits

PRIMARY_PERIOD_PANEL = 'panel2_common_period'
SENSITIVITY_LONG_PERIOD_PANEL = 'panel1_full_period'


def panel2_split_config(base: SplitConfig) -> SplitConfig:
    """
    Build the SplitConfig used for Panel 2 (common period) outer AND inner
    folds - both must use Panel 2's own, smaller sizing
    (panel2_min_train_size/panel2_test_size/panel2_inner_min_train_size/
    panel2_inner_test_size), since Panel 1's defaults (sized for the ~101
    quarter full period) leave zero valid folds over the ~29-quarter
    grid-covered common period. Centralised here so scripts/00
    (Panel 2 outer CVPlan) and scripts/10 (Panel 2 tuning/nested plans)
    never drift apart.
    """
    return replace(
        base,
        min_train_size=base.panel2_min_train_size,
        test_size=base.panel2_test_size,
        inner_min_train_size=base.panel2_inner_min_train_size,
        inner_test_size=base.panel2_inner_test_size,
    )


def build_common_period_cv_plan(
    X: pd.DataFrame,
    y: pd.Series,
    split_config: SplitConfig,
    common_period_start: pd.Timestamp,
    common_period_end: Optional[pd.Timestamp] = None,
) -> CVPlan:
    """
    Build Panel 2's outer CVPlan: identical forecast origins/target
    dates/horizons for all of A1-A4, restricted to
    [common_period_start, common_period_end].

    Callers should build ONE common-period plan (from any configuration's
    matrix - only its DatetimeIndex matters) and reuse the same CVPlan
    object for all four configurations, to guarantee identical folds per
    spec section 9 ("All four configurations must use identical forecast
    origins/target dates/outer folds/horizons").

    Args:
        X: Feature matrix with a DatetimeIndex spanning at least
            [common_period_start, common_period_end].
        y: Target Series aligned to X's index.
        split_config: SplitConfig (outer min_train_size/test_size/horizons).
        common_period_start: Earliest date every configuration has data for
            (in practice, bounded by grid-data availability - A1/A2 cover
            the full raw-data period but A3/A4 only exist where grid data
            does).
        common_period_end: Latest date every configuration has data for
            (default: X's last date).

    Returns:
        CVPlan restricted to the common period.
    """
    logger = get_logger()
    if common_period_end is None:
        common_period_end = X.index.max()

    mask = (X.index >= common_period_start) & (X.index <= common_period_end)
    X_common = X.loc[mask]
    y_common = y.loc[X_common.index]

    if len(X_common) == 0:
        raise ValueError(
            f"build_common_period_cv_plan: no rows in "
            f"[{common_period_start}, {common_period_end}]"
        )

    plan = create_walk_forward_splits(X_common, y_common, split_config)
    logger.info(
        f"Built Panel 2 (common period) CV plan: {plan.n_folds} folds over "
        f"{len(X_common)} quarters [{common_period_start} - {common_period_end}]"
    )
    return plan


def slice_configurations_to_common_period(
    X_by_config: Dict[str, pd.DataFrame],
    y: pd.Series,
    reference_tag: str = 'A3',
) -> Dict[str, pd.DataFrame]:
    """
    Reslice every A1-A4 (or B-candidate-pool) matrix to the exact
    [common_start, common_end] window `build_common_period_cv_plan` used
    when it built the shared PRIMARY_PERIOD_PANEL CVPlan.

    THIS IS NOT OPTIONAL. `generate_cv_folds` (src/splits/walk_forward.py)
    indexes X/y POSITIONALLY (`X.iloc[fold.train_indices]`) - it has no
    idea what calendar dates those integer positions refer to. Those
    positions were computed relative to a matrix already sliced to the
    common window (see `build_common_period_cv_plan` above). A1/A2's
    on-disk matrices are NOT pre-restricted to that window (they retain
    the full raw-data history back to ~2000; only A3/A4 are naturally
    bounded by grid-data availability) - passing them to
    `generate_cv_folds` unsliced silently walks the SAME integer positions
    into a completely different, wrong calendar window. Caught during
    Phase 6 smoke-testing (Stream A/B rebuild): naive-baseline predictions
    landed on 2004-2006 dates instead of the intended ~2018-2024 common
    period.

    Args:
        X_by_config: {tag: DataFrame} for every configuration to reslice.
        y: Full-length target Series (any configuration's index is a
            subset of this).
        reference_tag: Which configuration's own [min, max] index defines
            the common window - must match exactly what
            scripts/00_make_dataset.py used when building
            data/processed/panel2_cv_plan.pkl (A3's grid-availability
            bound, spec section 14).

    Returns:
        {tag: DataFrame} - every matrix restricted to the identical
        [common_start, common_end] window, row-count-aligned so
        `generate_cv_folds`'s positional indices resolve to the correct
        calendar dates regardless of which configuration is passed.
    """
    logger = get_logger()
    reference = X_by_config[reference_tag]
    common_start, common_end = reference.index.min(), reference.index.max()
    logger.info(
        f"Slicing all configurations to the primary common period "
        f"[{common_start} - {common_end}] (reference: {reference_tag})"
    )

    sliced = {}
    for tag, X in X_by_config.items():
        mask = (X.index >= common_start) & (X.index <= common_end)
        X_c = X.loc[mask]
        if len(X_c) != len(reference):
            logger.warning(
                f"{tag}: {len(X_c)} rows in the common period, vs "
                f"{reference_tag}'s {len(reference)} - positional fold "
                f"indices require IDENTICAL row counts/dates across all configurations."
            )
        sliced[tag] = X_c
    return sliced


def assert_cv_plan_fits_matrix(cv_plan: CVPlan, X: pd.DataFrame, tag: str = '') -> None:
    """
    Fail fast if `cv_plan`'s fold indices (positional, per
    `generate_cv_folds`) reference rows beyond `X`'s length - the
    unambiguous symptom of the common-period misalignment
    `slice_configurations_to_common_period` exists to prevent, e.g. if a
    caller forgets to reslice a newly-loaded matrix before use.
    """
    if cv_plan.n_folds == 0:
        return
    max_index_referenced = max(
        max(f.train_indices.max(), f.test_indices.max()) for f in cv_plan.folds
    )
    if len(X) <= max_index_referenced:
        raise ValueError(
            f"{tag}: {len(X)} rows but the CV plan's folds reference "
            f"position {max_index_referenced} - this matrix and the CV "
            "plan are out of sync (generate_cv_folds indexes X "
            "positionally). Call slice_configurations_to_common_period "
            "first, or regenerate the CV plan from this exact matrix."
        )
