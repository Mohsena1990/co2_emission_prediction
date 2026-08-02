#!/usr/bin/env python
"""
Script 00: Make Dataset
=======================
Load raw data, perform quality checks, and create feature-engineered dataset.

Usage:
    python scripts/00_make_dataset.py [--config CONFIG_PATH] [--input INPUT_PATH] [--run-id RUN_ID]

Outputs:
    - data/processed/df_clean.parquet
    - data/processed/X_full.parquet
    - data/processed/y.parquet
    - outputs/runs/<run_id>/tables/data_quality_report.csv
    - outputs/runs/<run_id>/tables/feature_dictionary.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (
    Config, create_run_directories, setup_logging, get_logger,
    set_seed, save_json_numpy
)
from src.data_io import (
    load_and_prepare_data, save_processed_data, create_default_schema
)
from src.quality import generate_quality_report, clean_data
from src.features import (
    engineer_features, create_feature_dictionary,
    FeatureRegistry, build_configuration_matrices, configuration_manifest
)
from src.splits import (
    create_walk_forward_splits, save_cv_plan, build_common_period_cv_plan,
    panel2_split_config as make_panel2_split_config
)
from src.grid import load_raw_halfhourly, aggregate_to_quarterly
from src.mobility import (
    load_cached_uk_mobility, apply_neutral_zero_convention,
    aggregate_to_quarterly as aggregate_mobility_to_quarterly,
)
from src.owid import load_cached_owid_uk_series, build_owid_lagged_feature


def parse_args():
    parser = argparse.ArgumentParser(description='Make dataset for CO2 forecasting')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input Excel file')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Run ID to use (for pipeline consistency)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load or create configuration
    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config()

    # Override run_id if provided (for pipeline consistency)
    if args.run_id:
        config.run_id = args.run_id

    # Override input path if provided
    if args.input:
        config.data.input_path = args.input

    # Set seed
    set_seed(config.seed)

    # Create run directories
    dirs = create_run_directories(config)

    # Setup logging
    logger = setup_logging(
        log_dir=dirs['logs'],
        run_id=config.run_id
    )

    logger.info("=" * 60)
    logger.info("Script 00: Make Dataset")
    logger.info("=" * 60)
    logger.info(f"Run ID: {config.run_id}")
    logger.info(f"Input: {config.data.input_path}")

    # Save config
    config.save(dirs['configs_snapshot'] / 'config.yaml')

    # =========================================
    # Step 1: Load Raw Data
    # =========================================
    logger.info("-" * 40)
    logger.info("Step 1: Loading raw data...")

    schema = create_default_schema()
    df_raw, metadata = load_and_prepare_data(config, schema)

    save_json_numpy(metadata, dirs['tables'] / 'data_metadata.json')
    logger.info(f"Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # =========================================
    # Step 2: Data Quality Audit
    # =========================================
    logger.info("-" * 40)
    logger.info("Step 2: Data quality audit...")

    quality_report = generate_quality_report(df_raw, dirs['tables'])
    logger.info(f"Quality report saved to {dirs['tables']}")

    # =========================================
    # Step 3: Clean Data
    # =========================================
    logger.info("-" * 40)
    logger.info("Step 3: Cleaning data...")

    df_clean, cleaning_log = clean_data(df_raw)
    save_json_numpy(cleaning_log, dirs['tables'] / 'cleaning_log.json')

    # Save clean data
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    save_processed_data(df_clean, processed_dir, 'df_clean')
    logger.info(f"Cleaned data: {len(df_clean)} rows, {len(df_clean.columns)} columns")

    # =========================================
    # Step 4: Feature Engineering
    # =========================================
    logger.info("-" * 40)
    logger.info("Step 4: Feature engineering...")

    # Spec section 5/8: A2 (and A4) are defined as the 7 audited raw
    # predictors plus 16 engineered candidates, INCLUDING the 2 intensity
    # ratios (CO2e_per_Population, CO2e_per_TEC) - i.e. the intensity family
    # is a full member of the A2/A4 candidate pool by design, not merely an
    # optional extra. `include_intensity_features` still defaults to False in
    # FeatureConfig (bug 3.1: target-derived groups must never be silently
    # enabled) - this is an explicit, documented override made here, at the
    # point the A1-A4 candidate pool is generated, not a change to the
    # dataclass default itself. A1's registry membership does not include
    # the intensity features regardless of this flag, so A1 is unaffected.
    config.features.include_intensity_features = True

    X, y, feature_metadata = engineer_features(df_clean, config)

    # Drop rows with NaN (from lag features)
    valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[valid_idx]
    y = y[valid_idx]

    logger.info(f"Features: {len(X.columns)} columns, {len(X)} valid rows")

    # Save feature data
    save_processed_data(X, processed_dir, 'X_full')
    save_processed_data(y.to_frame('target'), processed_dir, 'y')

    # Create feature dictionary
    feature_dict = create_feature_dictionary(
        X, feature_metadata,
        output_path=dirs['tables'] / 'feature_dictionary.csv'
    )
    save_json_numpy(feature_metadata, dirs['tables'] / 'feature_metadata.json')

    # =========================================
    # Step 4b: A1-A4 data-configuration matrices (spec section 8)
    # =========================================
    logger.info("-" * 40)
    logger.info("Step 4b: Building A1-A4 configuration matrices...")

    registry = FeatureRegistry.load(config.feature_registry_path)
    registry_csv_path = registry.export_csv('metadata/feature_registry.csv')
    logger.info(f"Exported registry to {registry_csv_path} (spec section 15)")

    # Mobility fusion (spec section 4): aggregate cached Google COVID-19
    # daily UK mobility data to quarterly shock indicators for A1-A4.
    # Requires scripts/fetch_mobility_data.py to have been run first
    # (network-bound, cached to disk once - the historical Google archive
    # is fixed/discontinued, so unlike grid there's no "new data" case).
    # Unlike grid_df, mobility_df is reindexed onto X's FULL index with the
    # neutral-zero convention BEFORE being passed to
    # build_configuration_matrices, so it never truncates any
    # configuration's sample period (see apply_neutral_zero_convention).
    mobility_df = None
    mobility_cache_dir = Path(config.mobility.cache_dir)
    mobility_cache_file = mobility_cache_dir / 'google_uk_national_mobility.csv'
    if config.mobility.exclude_mobility_block:
        logger.info(
            "Mobility sensitivity run: mobility.exclude_mobility_block=true - "
            "A1-A4 will omit the six Mobility_* predictors entirely (spec "
            "section 29 mandatory sensitivity)."
        )
    elif config.mobility.enabled and mobility_cache_file.exists():
        logger.info(f"Aggregating cached mobility data from {mobility_cache_dir}...")
        df_daily = load_cached_uk_mobility(mobility_cache_dir)
        primary_cutoff = None if config.mobility.include_partial_2022q4 else config.mobility.primary_cutoff_quarter
        mobility_quarterly, mobility_quality = aggregate_mobility_to_quarterly(
            df_daily,
            minimum_quarter_completeness=config.mobility.minimum_quarter_completeness,
            primary_cutoff_quarter=primary_cutoff,
        )
        mobility_df = apply_neutral_zero_convention(mobility_quarterly, X.index)
        save_processed_data(mobility_df, processed_dir, 'mobility_quarterly_features')
        mobility_quality.to_csv(dirs['tables'] / 'mobility_quality_report.csv')
        mobility_quality.to_csv(processed_dir / 'mobility_quality_report.csv')
        n_observed = int((mobility_quality['inclusion_status'] == 'included').sum())
        logger.info(
            f"Mobility quarterly features: {n_observed} quarter(s) with genuine "
            f"reporting-window data, {len(mobility_df) - n_observed} quarter(s) "
            f"neutral-zeroed outside the window"
        )
    else:
        logger.warning(
            f"No cached mobility data found in {mobility_cache_dir} (or "
            f"mobility.enabled=False) - run scripts/fetch_mobility_data.py first. "
            f"A1-A4 will omit the six Mobility_* predictors for this run."
        )

    # Grid fusion (spec section 6): aggregate cached half-hourly GB grid
    # data to quarterly features for A3/A4. Requires scripts/fetch_grid_data.py
    # to have been run first (network-bound, cached to disk, not repeated
    # here) - if the cache is empty, A3/A4 correctly fall back to A1/A2 (see
    # build_configuration_matrices docstring) rather than failing the run.
    grid_df = None
    cache_dir = Path(config.grid.cache_dir)
    if config.grid.enabled and cache_dir.exists() and any(cache_dir.glob('intensity_*.json')):
        logger.info(f"Aggregating cached grid data from {cache_dir}...")
        df_halfhourly = load_raw_halfhourly(cache_dir)
        grid_df, grid_quality = aggregate_to_quarterly(
            df_halfhourly, min_completeness=config.grid.minimum_quarter_completeness
        )
        save_processed_data(grid_df, processed_dir, 'grid_quarterly_features')
        grid_quality.to_csv(dirs['tables'] / 'grid_quality_report.csv')
        grid_quality.to_csv(processed_dir / 'grid_quality_report.csv')
        logger.info(
            f"Grid quarterly features: {len(grid_df)} quarters "
            f"[{grid_df.index.min()} - {grid_df.index.max()}], "
            f"{(grid_quality['inclusion_status'] != 'included').sum()} quarter(s) "
            f"excluded for incompleteness"
        )

        # OWID annual renewable/low-carbon shares (spec section 8): two
        # one-year-lagged structural predictors, registered as [A3, A4]
        # membership like the NESO grid block - merged onto grid_df so they
        # ride the same A3/A4-only, index-intersecting merge path in
        # build_configuration_matrices (they are not primary/high-frequency
        # like the NESO Grid_* columns, so they don't get their own
        # all-four-configurations merge like mobility does).
        owid_cache_dir = Path(config.owid.cache_dir)
        if config.owid.enabled:
            try:
                owid_cols = {}
                for kind, feature_name in (
                    ('renewable', 'OWID_RenewableShare_L1Y'),
                    ('low_carbon', 'OWID_LowCarbonShare_L1Y'),
                ):
                    uk_df, owid_metadata = load_cached_owid_uk_series(owid_cache_dir, kind)
                    owid_cols[feature_name] = build_owid_lagged_feature(
                        uk_df, owid_metadata, feature_name, grid_df.index
                    )
                grid_df = pd.concat([grid_df, pd.DataFrame(owid_cols)], axis=1)
                logger.info(
                    f"OWID one-year-lagged shares merged into grid block "
                    f"({sum(v.notna().sum() for v in owid_cols.values())} non-NaN cells "
                    f"across {len(owid_cols)} columns)"
                )
            except FileNotFoundError:
                logger.warning(
                    f"No cached OWID data found in {owid_cache_dir} (or "
                    f"owid.enabled=False) - run scripts/fetch_owid_data.py first. "
                    f"A3/A4 will omit OWID_RenewableShare_L1Y/OWID_LowCarbonShare_L1Y "
                    f"for this run."
                )
    else:
        logger.warning(
            f"No cached grid data found in {cache_dir} (or grid.enabled=False) - "
            f"run scripts/fetch_grid_data.py first. A3/A4 will fall back to A1/A2 "
            f"for this run."
        )

    config_matrices = build_configuration_matrices(X, registry, grid_df=grid_df, mobility_df=mobility_df)

    for tag, mat in config_matrices.items():
        save_processed_data(mat, processed_dir, f'X_{tag}')

    manifest = configuration_manifest(config_matrices, registry)
    save_json_numpy(manifest, processed_dir / 'configurations_manifest.json')
    save_json_numpy(manifest, dirs['tables'] / 'configurations_manifest.json')

    for tag, info in manifest.items():
        logger.info(
            f"  {tag}: {info['n_features_total']} features "
            f"(raw={info['n_raw']}, engineered={info['n_engineered']}, "
            f"grid={info['n_grid']}, nominal_max={info['nominal_max']})"
        )

    # =========================================
    # Step 5: Create CV Splits
    # =========================================
    logger.info("-" * 40)
    logger.info("Step 5: Creating walk-forward CV splits (sensitivity-only: full period, A1/A2)...")

    # NOTE (spec section 14): this full-period plan is now a documented
    # SENSITIVITY-ONLY comparison (src/splits/panels.py::
    # SENSITIVITY_LONG_PERIOD_PANEL) - the PRIMARY comparison period for the
    # whole Stream A/B study is the common (grid-covered) period built below
    # (PRIMARY_PERIOD_PANEL). Kept for the A1/A2-only long-period sensitivity
    # analysis (spec section 14 permits this, but bans it from the main
    # global MCDM ranking).
    cv_plan = create_walk_forward_splits(X, y, config.splits)
    save_cv_plan(cv_plan, processed_dir / 'cv_plan')

    logger.info(f"Created {cv_plan.n_folds} CV folds (sensitivity-only)")

    # Save CV plan summary
    cv_plan.to_dataframe().to_csv(dirs['tables'] / 'cv_plan.csv', index=False)

    # PRIMARY comparison period (spec section 14): common period across
    # A1-A4, bounded by grid-data availability. Only meaningful once grid
    # features exist in A3/A4 - build_configuration_matrices already
    # restricts A3/A4 to the intersection of the raw-data index and
    # grid_df's index, so that intersection's [min, max] IS the common
    # period every configuration can be fairly compared over. This is the
    # ONE plan every Stream A (20) and Stream B (100) candidate must share
    # (identical outer folds/horizons/target dates) for the global MCDM
    # ranking - see src/splits/panels.py::PRIMARY_PERIOD_PANEL.
    if config.grid.enabled and manifest['A3']['n_grid'] > 0:
        common_start = config_matrices['A3'].index.min()
        common_end = config_matrices['A3'].index.max()
        # The common period is much shorter than the full period
        # (grid-availability-bounded, ~2018 onward - see
        # src/grid/fetch.py::EARLIEST_AVAILABLE_DATE for why this is 2018,
        # not the spec's stated 2009) - reuses config.splits' horizons/
        # weights but with its own, smaller outer/inner fold sizing (see
        # src/splits/panels.py::panel2_split_config), since the full
        # period's sizing would yield zero folds over ~29 quarters.
        panel2_cv_plan = build_common_period_cv_plan(
            X, y, make_panel2_split_config(config.splits),
            common_period_start=common_start, common_period_end=common_end
        )
        save_cv_plan(panel2_cv_plan, processed_dir / 'panel2_cv_plan')
        panel2_cv_plan.to_dataframe().to_csv(dirs['tables'] / 'panel2_cv_plan.csv', index=False)
        logger.info(f"PRIMARY common-period CV plan: {panel2_cv_plan.n_folds} folds")
        if panel2_cv_plan.n_folds == 0:
            logger.warning(
                "The primary common-period plan produced 0 CV folds even "
                f"with panel2_min_train_size={config.splits.panel2_min_train_size}/"
                f"panel2_test_size={config.splits.panel2_test_size} - the "
                "common grid-covered period may be too short; consider "
                "lowering these further."
            )
    else:
        logger.info(
            "PRIMARY common-period CV plan skipped: no grid features in "
            "A3/A4 yet for this run."
        )

    # =========================================
    # Summary
    # =========================================
    logger.info("=" * 60)
    logger.info("Dataset creation complete!")
    logger.info(f"  Raw data: {len(df_raw)} rows")
    logger.info(f"  Clean data: {len(df_clean)} rows")
    logger.info(f"  Features: {len(X.columns)} columns")
    logger.info(f"  Valid samples: {len(X)}")
    logger.info(f"  CV folds: {cv_plan.n_folds}")
    logger.info(f"  Outputs saved to: {dirs['root']}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
