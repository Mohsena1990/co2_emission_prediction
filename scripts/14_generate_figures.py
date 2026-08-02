#!/usr/bin/env python
"""
Script 14: Generate PDF figures (spec sections 25-28) from a Stream A/B
run's saved machine-readable outputs.

Writes to the spec-mandated outputs/figures/pdf/main/ (+ plot_data/)
layout. Covers the figures with a direct, unambiguous mapping from
already-existing saved tables (see src/reporting/pdf_figure_builders.py
docstring for the rest, which require additional upstream data not yet
persisted by this pipeline - e.g. the 36-PDF forecast atlas needs
per-fold prediction tracking already saved under fold_predictions/, but
its own builder/CLI wiring is still pending).

Usage:
    python scripts/14_generate_figures.py [--config CONFIG_PATH] [--run-id RUN_ID]
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
    fig01_revised_framework,
    fig02_predictor_governance_map,
    fig03_grid_aggregation_quality,
    fig05_incremental_configuration_value,
    fig08_fs_membership_heatmap,
    fig13_pareto_frontier,
    fig14_mcda_rank_sensitivity,
    fig15_cross_model_feature_importance,
    fig17_regime_specific_importance,
    fig18_target_derived_feature_sensitivity,
    fig03a_stream_A_global_comparison,
    fig03b_stream_B_global_overview,
    fig03c_stream_B_shortlist_magnified,
    fig03d_best_A_vs_best_B,
    build_table2_feature_registry,
)

CONFIGURATIONS = ['A1', 'A2', 'A3', 'A4']
WINNER_SUBDIRS = {'Best_A': 'best_A', 'Best_B': 'best_B', 'Best_Overall': 'best_overall'}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate PDF figures from a Stream A/B run')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='outputs/figures/pdf/main')
    return parser.parse_args()


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


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

    logger = setup_logging(log_dir=Path('outputs/logs'), run_id='generate_figures')
    logger.info("=" * 60)
    logger.info(f"Script 14: Generate PDF figures (source run_id={run_id})")
    logger.info("=" * 60)

    run_dir = Path(config.output.base_dir) / 'runs' / run_id
    processed_dir = Path('data/processed')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # ---- fig01/02: framework overview + predictor governance (data-independent / registry-only) ----
    fig01_revised_framework(output_dir / 'fig01_revised_qdecem_framework.pdf')
    generated.append('fig01')

    registry = FeatureRegistry.load(config.feature_registry_path)
    table2 = build_table2_feature_registry(registry)
    fig02_predictor_governance_map(table2, output_dir / 'fig02_predictor_governance_map.pdf')
    generated.append('fig02')

    grid_quality = _load_csv(processed_dir / 'grid_quality_report.csv')
    if not grid_quality.empty:
        fig03_grid_aggregation_quality(grid_quality, output_dir / 'fig03_grid_aggregation_quality.pdf')
        generated.append('fig03')

    # ---- Stream A/B metrics -> the two-stream Pareto/MCDA rankings (spec section 26) ----
    stream_a_metrics = _load_csv(run_dir / 'metrics' / 'stream_A' / 'metrics.csv')
    stream_b_metrics = _load_csv(run_dir / 'metrics' / 'stream_B' / 'metrics.csv')
    for df in (stream_a_metrics, stream_b_metrics):
        if not df.empty and 'configuration' in df.columns:
            df.drop(df[df['configuration'] == 'naive_reference'].index, inplace=True)

    stream_a_ranked = build_pareto_mcda_table(stream_a_metrics) if not stream_a_metrics.empty else pd.DataFrame()
    stream_b_ranked = build_pareto_mcda_table(stream_b_metrics) if not stream_b_metrics.empty else pd.DataFrame()

    if not stream_a_ranked.empty:
        fig03a_stream_A_global_comparison(stream_a_ranked, output_dir / 'fig03a_stream_A_global_comparison.pdf')
        generated.append('fig03a')
        fig13_pareto_frontier(stream_a_ranked, output_dir / 'fig11a_stream_A_pareto_frontier.pdf')
        generated.append('fig11a')

    if not stream_b_ranked.empty:
        fig03b_stream_B_global_overview(stream_b_ranked, output_dir / 'fig03b_stream_B_global_overview.pdf')
        generated.append('fig03b')
        fig03c_stream_B_shortlist_magnified(stream_b_ranked, output_dir / 'fig03c_stream_B_shortlist_magnified.pdf')
        generated.append('fig03c')
        fig13_pareto_frontier(stream_b_ranked, output_dir / 'fig11b_stream_B_pareto_frontier.pdf')
        generated.append('fig11b')

    # ---- Table 7 (stream winners + final) -> fig03d Best_A vs Best_B ----
    table7 = _load_csv(run_dir / 'pareto_mcda' / 'final' / 'table7_stream_winners_and_final.csv')
    if not table7.empty:
        fig03d_best_A_vs_best_B(table7, output_dir / 'fig03d_best_A_vs_best_B.pdf')
        generated.append('fig03d')

    # ---- supplementary Table 6 (incremental value) - not the primary decision structure ----
    table6 = _load_csv(run_dir / 'statistical_tests' / 'table6_incremental_value.csv')
    if not table6.empty:
        fig05_incremental_configuration_value(table6, output_dir / 'fig05_incremental_configuration_value.pdf')
        generated.append('fig05')

    # ---- MCDM rank-correlation sensitivity, one figure per stream ----
    for stream_key, suffix in (('stream_A', 'a'), ('stream_B', 'b')):
        rank_corr = _load_csv(run_dir / 'pareto_mcda' / stream_key / 'rank_correlation.csv')
        if not rank_corr.empty:
            fig14_mcda_rank_sensitivity(rank_corr, output_dir / f'fig14{suffix}_mcda_rank_sensitivity_{stream_key}.pdf')
            generated.append(f'fig14{suffix}')

    # ---- FS membership heatmap, one per B-configuration ----
    for tag in CONFIGURATIONS:
        fs_results = _load_json(run_dir / 'selected_features' / f'streamB_fs_results_{tag}.json')
        if fs_results:
            feature_pool = list(registry.configuration_members(tag))
            b_tag = f'B{tag[1]}'
            fig08_fs_membership_heatmap(
                fs_results, feature_pool, b_tag,
                output_dir / f'fig08_{b_tag.lower()}_feature_selection_membership.pdf'
            )
            generated.append(f'fig08_{b_tag.lower()}')

    # ---- Per-winner interpretability figures (Best_A / Best_B / Best_Overall) ----
    for label, subdir in WINNER_SUBDIRS.items():
        interp_dir = run_dir / 'interpretability' / subdir
        table11 = _load_csv(interp_dir / 'table11_cross_model_importance.csv')
        if not table11.empty:
            fig15_cross_model_feature_importance(
                table11, output_dir / f'fig15_{subdir}_cross_model_feature_importance.pdf'
            )
            generated.append(f'fig15_{subdir}')

        regime_importance = _load_csv(interp_dir / 'regime_importance.csv')
        if not regime_importance.empty:
            fig17_regime_specific_importance(
                regime_importance, output_dir / f'fig17_{subdir}_regime_specific_importance.pdf'
            )
            generated.append(f'fig17_{subdir}')

        target_derived_sensitivity = _load_csv(interp_dir / 'target_derived_sensitivity.csv')
        if not target_derived_sensitivity.empty:
            fig18_target_derived_feature_sensitivity(
                target_derived_sensitivity, output_dir / f'fig18_{subdir}_target_derived_feature_sensitivity.pdf'
            )
            generated.append(f'fig18_{subdir}')

    logger.info("=" * 60)
    logger.info(f"Script 14 complete! {len(generated)} figures generated: {generated}")
    logger.info("  Still requiring dedicated builder/CLI wiring (spec sections 25/27): the 36-PDF "
                "forecast atlas (family/stream/overall/A-vs-B out-of-sample forecasts - data already "
                "saved under fold_predictions/, builder not yet written), mobility/OWID/NESO data-quality "
                "figures, PSO convergence, forecast-error distributions, cross-model SHAP agreement.")
    logger.info("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
