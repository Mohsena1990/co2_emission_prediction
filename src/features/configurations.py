"""
A1-A4 data-configuration matrix builder (spec section 8).

A1: audited raw quarterly predictors only (11 = 5 retained raw + 6 Google
    mobility-shock predictors; TEC/CEI hard-removed, spec section 2).
A2: A1 + 14 safely-constructed engineered predictors (nominal max 25).
A3: A1 + audited quarterly grid features.
A4: A2 + audited quarterly grid features.

Membership is driven entirely by `config/feature_registry.yaml`'s
`configuration_membership` column (spec section 7/8: "the model matrix must
be generated through this registry") - this module does not hard-code
feature lists, so a registry change (e.g. adding grid rows in a later
pipeline stage) is picked up automatically without touching this code.
"""
from typing import Dict, List, Optional

import pandas as pd

from ..core.logging_utils import get_logger
from .registry import FeatureRegistry, CONFIGURATION_TAGS

# Spec section 19.5: the named target-derived features to exclude in the
# primary sensitivity comparison (narrower than "all target-derived", which
# would also drop the four CO2e lags). CEI_lag1/CO2e_per_TEC are gone
# entirely now that TEC/CEI are hard-removed per spec section 2 - only the
# remaining named target-derived features are listed here.
NAMED_TARGET_DERIVED_FEATURES = [
    'CO2e_per_Population', 'CO2e_dlog',
]


def build_configuration_matrices(
    X: pd.DataFrame,
    registry: FeatureRegistry,
    grid_df: Optional[pd.DataFrame] = None,
    mobility_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Slice the full engineered feature pool `X` into the four data
    configurations using each feature's `configuration_membership` tag.

    Args:
        X: Full engineered candidate-feature matrix (must have been built
            with `include_intensity_features=True` for A2/A4 to reach their
            nominal 25-feature pool - see scripts/00_make_dataset.py).
        registry: Loaded FeatureRegistry (config/feature_registry.yaml).
        grid_df: Quarterly grid feature matrix, already aligned to (a
            subset of) X's index, or None if grid data has not been fetched
            yet. When None, A3/A4 naturally reduce to A1/A2 respectively,
            since no registry entries with configuration_membership
            containing 'A3'/'A4' exist in X (grid rows are absent from X by
            construction - they come only from `grid_df`).
        mobility_df: Quarterly mobility-shock feature matrix (spec section
            4), or None if not yet computed. Unlike grid_df, this is
            expected to already be reindexed onto X's FULL index with the
            neutral-zero convention applied outside the reporting window
            (src/mobility/aggregate.py::apply_neutral_zero_convention) -
            so merging it never truncates any configuration's sample
            period, and it is merged into ALL FOUR configurations (spec
            section 3: mobility is part of the 11-feature raw pool present
            in A1-A4), not just A3/A4 like grid.

    Returns:
        Dict with keys 'A1', 'A2', 'A3', 'A4', each a DataFrame subset of X
        (plus grid_df columns for A3/A4, and mobility_df columns for all
        four, when provided).
    """
    logger = get_logger()
    matrices: Dict[str, pd.DataFrame] = {}

    for tag in CONFIGURATION_TAGS:
        cols = [
            name for name in registry.configuration_members(tag)
            if name in X.columns
        ]
        matrices[tag] = X[cols].copy()

    if mobility_df is not None and len(mobility_df.columns) > 0:
        merged_any = False
        for tag in CONFIGURATION_TAGS:
            cols = [c for c in mobility_df.columns if c in registry.configuration_members(tag)]
            if not cols:
                continue
            common_idx = matrices[tag].index.intersection(mobility_df.index)
            matrices[tag] = pd.concat(
                [matrices[tag].loc[common_idx], mobility_df.loc[common_idx, cols]], axis=1
            )
            merged_any = True
        if merged_any:
            logger.info(
                f"Mobility features merged into all four configurations "
                f"(A1: {len(matrices['A1'].columns)} cols)"
            )
    else:
        logger.warning(
            "build_configuration_matrices: no mobility_df supplied - A1-A4 "
            "omit the six Mobility_* predictors (mobility fusion not yet "
            "wired for this run)."
        )

    if grid_df is not None and len(grid_df.columns) > 0:
        for tag in ('A3', 'A4'):
            common_idx = matrices[tag].index.intersection(grid_df.index)
            merged = pd.concat(
                [matrices[tag].loc[common_idx], grid_df.loc[common_idx]], axis=1
            )
            # A quarter can pass the overall half-hourly completeness check
            # (based on carbon-intensity readings) while still having a
            # specific generation-mix sub-field completely absent for that
            # quarter (observed in practice: 2018Q1, the very first
            # grid-covered quarter, has complete intensity data but NaN
            # renewable/fossil/gas/wind/import shares). Rather than silently
            # feeding NaN into a model that can't handle it (or worse,
            # imputing a fabricated value), drop the affected quarter here -
            # consistent with how the rest of the pipeline already handles
            # incomplete rows (spec 6.3: exclude, do not impute, incomplete
            # quarters; documented in Table 4 via grid_quality_report.csv).
            n_before = len(merged)
            merged = merged.dropna()
            if len(merged) < n_before:
                logger.warning(
                    f"{tag}: dropped {n_before - len(merged)} quarter(s) with "
                    f"partially-missing grid features after merge "
                    f"(e.g. a generation-mix sub-field fully absent for that "
                    f"quarter despite passing the overall completeness check)"
                )
            matrices[tag] = merged
        logger.info(
            f"Grid features merged into A3 ({len(matrices['A3'].columns)} cols) "
            f"and A4 ({len(matrices['A4'].columns)} cols) over "
            f"{len(matrices['A3'].index)} common quarters"
        )
    else:
        logger.warning(
            "build_configuration_matrices: no grid_df supplied - A3 reduces "
            "to A1's columns and A4 reduces to A2's columns (grid fusion not "
            "yet wired for this run)."
        )

    for tag, mat in matrices.items():
        logger.info(f"Configuration {tag}: {len(mat.columns)} features, {len(mat)} rows")

    return matrices


def exclude_target_derived_features(
    X: pd.DataFrame,
    registry: FeatureRegistry,
    mode: str = 'named',
) -> pd.DataFrame:
    """
    Target-derived-feature sensitivity (spec section 19.5): repeat key
    comparisons after excluding features whose formula involves the target
    (CO2e), to check whether forecasting gains persist without transformed
    target information.

    Args:
        X: A configuration matrix (e.g. from build_configuration_matrices).
        registry: Loaded FeatureRegistry.
        mode: 'named' - drop only CEI_lag1/CO2e_per_Population/CO2e_per_TEC/
            CO2e_dlog (spec's primary sensitivity variant); 'all' - drop
            every column the registry marks target_derived=True (spec's
            broader "optionally all historical target-derived features"
            variant, which additionally drops the four CO2e lags).

    Returns:
        X with the relevant target-derived columns dropped (columns not
        present in X are silently ignored - a configuration may not contain
        all of them, e.g. A1 has no intensity ratios).
    """
    logger = get_logger()
    if mode == 'named':
        to_drop = [c for c in NAMED_TARGET_DERIVED_FEATURES if c in X.columns]
    elif mode == 'all':
        to_drop = [
            c for c in X.columns
            if c in registry.entries and registry.entries[c]['target_derived']
        ]
    else:
        raise ValueError(f"exclude_target_derived_features: unknown mode '{mode}', expected 'named' or 'all'")

    logger.info(f"Target-derived sensitivity (mode={mode}): dropping {len(to_drop)} feature(s): {to_drop}")
    return X.drop(columns=to_drop)


def configuration_manifest(
    matrices: Dict[str, pd.DataFrame],
    registry: FeatureRegistry,
) -> Dict[str, Dict]:
    """
    Table 3 (configuration definitions) as a machine-readable dict: per
    configuration, raw/engineered/grid feature counts, nominal vs audited
    totals, and the exact column list.
    """
    # 5 raw + 6 mobility-shock predictors (spec section 3) = 11;
    # + 14 engineered = 25 (spec section 5).
    nominal = {'A1': 11, 'A2': 25, 'A3': None, 'A4': None}  # A3/A4 nominal = 11+K / 25+K, K grid-dependent
    manifest = {}
    for tag, mat in matrices.items():
        cols = list(mat.columns)
        # Grid features are registered with kind='raw' (they ARE raw
        # electricity-system measurements, not engineered from other
        # columns) but must be counted separately here - use the
        # 'grid_exogenous'/'owid_exogenous' family tags (or, for a column
        # not in the registry at all - shouldn't happen once grid rows
        # exist, but defensive) rather than `kind` to distinguish them from
        # the raw macro/mobility predictors. OWID_*_L1Y is counted as grid
        # here (not a separate bucket) since spec section 9 defines it as
        # part of the same unified 12-item grid block, not a standalone
        # source category for reporting purposes.
        is_grid = lambda c: (c not in registry.entries) or (
            registry.entries[c].get('family') in ('grid_exogenous', 'owid_exogenous')
        )
        raw_count = sum(
            1 for c in cols
            if c in registry.entries and registry.entries[c]['kind'] == 'raw' and not is_grid(c)
        )
        engineered_count = sum(
            1 for c in cols
            if c in registry.entries and registry.entries[c]['kind'] == 'engineered' and not is_grid(c)
        )
        grid_count = sum(1 for c in cols if is_grid(c))
        manifest[tag] = {
            'n_features_total': len(cols),
            'n_raw': raw_count,
            'n_engineered': engineered_count,
            'n_grid': grid_count,
            'nominal_max': nominal.get(tag),
            'columns': cols,
        }
    return manifest
