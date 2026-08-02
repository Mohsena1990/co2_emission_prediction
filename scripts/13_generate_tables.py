#!/usr/bin/env python
"""
Script 13: Generate all required tables (spec section 24, Tables 1-10) from
a Stream A/B run's saved machine-readable outputs.

Writes to the top-level outputs/tables/ directory, reading from
outputs/runs/<run_id>/ (this repo's existing run-versioned convention -
each table's source run_id is recorded in outputs/tables/MANIFEST.json).

Usage:
    python scripts/13_generate_tables.py [--config CONFIG_PATH] [--run-id RUN_ID]

Outputs (outputs/tables/):
    table1_revised_raw_data.csv
    table2_feature_registry.csv
    table3_ab_configuration_definitions.csv
    table4_stream_A_comparison.csv
    table5_stream_B_comparison.csv
    table6_global_factor_summary.csv
    table7_stream_winners_and_final.csv
    table8_fs_outputs.csv
    table9_covid_regime_interpretation.csv
    table10_source_data_quality.csv
    MANIFEST.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.core import Config, get_latest_run_id, setup_logging
from src.features import FeatureRegistry
from src.decision import build_pareto_mcda_table
from src.reporting import (
    build_table1_raw_variable_inventory, build_table2_feature_registry,
    build_table3_ab_configuration_definitions, build_table6_global_factor_summary,
    build_table8_fs_outputs, build_table9_covid_regime_interpretation,
    build_table10_source_data_quality,
)

CONFIGURATIONS = ['A1', 'A2', 'A3', 'A4']
STREAM_B_TAGS = ['B1', 'B2', 'B3', 'B4']


def parse_args():
    parser = argparse.ArgumentParser(description='Generate all Tables 1-10 from a Stream A/B run')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='outputs/tables')
    return parser.parse_args()


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_csv(path: Path, **kwargs):
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def main():
    args = parse_args()

    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config()

    run_id = args.run_id or get_latest_run_id(config.output.base_dir)
    if not run_id:
        print("ERROR: no run_id given and no runs found under outputs/runs/")
        return 1

    logger = setup_logging(log_dir=Path('outputs/logs'), run_id='generate_tables')
    logger.info("=" * 60)
    logger.info(f"Script 13: Generate Tables 1-10 (source run_id={run_id})")
    logger.info("=" * 60)

    run_dir = Path(config.output.base_dir) / 'runs' / run_id
    processed_dir = Path('data/processed')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = FeatureRegistry.load(config.feature_registry_path)
    manifest = {'source_run_id': run_id, 'tables': {}}

    def save(name: str, df: pd.DataFrame):
        if df is None or df.empty:
            logger.warning(f"  {name}: EMPTY - not written (source data missing for this run)")
            manifest['tables'][name] = {'status': 'empty', 'n_rows': 0}
            return
        path = output_dir / f'{name}.csv'
        df.to_csv(path, index=False)
        manifest['tables'][name] = {'status': 'written', 'n_rows': len(df)}
        logger.info(f"  {name}: {len(df)} rows -> {path}")

    # ---- Table 1: revised raw data (5 raw + 6 mobility + target) ----
    quality_report = _load_json(run_dir / 'tables' / 'data_quality_report.json') or {}
    save('table1_revised_raw_data', build_table1_raw_variable_inventory(registry, quality_report))

    # ---- Table 2: complete feature registry ----
    save('table2_feature_registry', build_table2_feature_registry(registry))

    # ---- Table 3: A1-A4 + B1-B4 configuration definitions ----
    config_manifest = _load_json(processed_dir / 'configurations_manifest.json') or {}
    save('table3_ab_configuration_definitions',
         build_table3_ab_configuration_definitions(config_manifest, registry))

    # ---- Load Stream A/B raw metrics + FS results (shared by Tables 4-8) ----
    stream_a_metrics = _load_csv(run_dir / 'metrics' / 'stream_A' / 'metrics.csv')
    stream_b_metrics = _load_csv(run_dir / 'metrics' / 'stream_B' / 'metrics.csv')
    for df in (stream_a_metrics, stream_b_metrics):
        if not df.empty and 'configuration' in df.columns:
            df.drop(df[df['configuration'] == 'naive_reference'].index, inplace=True)

    stream_a_ranked = build_pareto_mcda_table(stream_a_metrics) if not stream_a_metrics.empty else pd.DataFrame()
    stream_b_ranked = build_pareto_mcda_table(stream_b_metrics) if not stream_b_metrics.empty else pd.DataFrame()

    # ---- Table 4: full Stream A comparison (20 rows) ----
    save('table4_stream_A_comparison', stream_a_ranked)

    # ---- Table 5: full Stream B comparison (100 rows), + selected-feature list ----
    fs_results_by_config = {}
    feature_pool_by_config = {}
    for tag in CONFIGURATIONS:
        fs_results = _load_json(run_dir / 'selected_features' / f'streamB_fs_results_{tag}.json')
        if fs_results:
            b_tag = f'B{tag[1]}'
            fs_results_by_config[b_tag] = fs_results
            feature_pool_by_config[b_tag] = list(registry.configuration_members(tag))

    if not stream_b_ranked.empty and fs_results_by_config:
        def _feature_list(row):
            b_tag = f'B{row["configuration"][1]}'
            fs_res = fs_results_by_config.get(b_tag, {}).get(row['fs_option'])
            return ', '.join(fs_res['selected_features']) if fs_res else ''
        stream_b_ranked = stream_b_ranked.copy()
        stream_b_ranked['selected_features'] = stream_b_ranked.apply(_feature_list, axis=1)
    save('table5_stream_B_comparison', stream_b_ranked)

    # ---- Table 6: global factor summary ----
    save('table6_global_factor_summary',
         build_table6_global_factor_summary(stream_a_ranked, stream_b_ranked, models=config.model.models))

    # ---- Table 7: stream winners + final winner (already produced by script 11) ----
    table7 = _load_csv(run_dir / 'pareto_mcda' / 'final' / 'table7_stream_winners_and_final.csv')
    save('table7_stream_winners_and_final', table7)

    # ---- Table 8: FS outputs (candidate feature x B-config x FS1-5) ----
    save('table8_fs_outputs', build_table8_fs_outputs(fs_results_by_config, feature_pool_by_config))

    # ---- Table 9: COVID regime interpretation for Best_A/Best_B/Best_Overall ----
    winner_regime_data = {}
    for label, subdir in (('Best_A', 'best_A'), ('Best_B', 'best_B'), ('Best_Overall', 'best_overall')):
        winner_regime_data[label] = _load_csv(run_dir / 'interpretability' / subdir / 'regime_importance.csv')
    save('table9_covid_regime_interpretation', build_table9_covid_regime_interpretation(winner_regime_data))

    # ---- Table 10: source and data-quality validation ----
    mobility_metadata = _load_json(Path('data/raw/mobility/google_uk_national_mobility.metadata.json'))
    owid_renewable_metadata = _load_json(Path('data/raw/owid/renewable_metadata.json'))
    owid_low_carbon_metadata = _load_json(Path('data/raw/owid/low_carbon_metadata.json'))
    grid_quality = _load_csv(processed_dir / 'grid_quality_report.csv')
    save('table10_source_data_quality', build_table10_source_data_quality(
        mobility_metadata=mobility_metadata, grid_quality_report=grid_quality,
        owid_renewable_metadata=owid_renewable_metadata, owid_low_carbon_metadata=owid_low_carbon_metadata,
    ))

    with open(output_dir / 'MANIFEST.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    n_written = sum(1 for t in manifest['tables'].values() if t['status'] == 'written')
    logger.info("=" * 60)
    logger.info(f"Script 13 complete! {n_written}/{len(manifest['tables'])} tables written to {output_dir}")
    logger.info("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
