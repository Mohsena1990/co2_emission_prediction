#!/usr/bin/env python
"""
Script 11: Two-stage Pareto/MCDM decision structure (spec section 21).

Reads Stream A / Stream B metrics produced by scripts/10_run_experiment_grid.py
for a given run and computes, independently:

  1. Stream A global comparison: Pareto-filter + VIKOR-rank all 20
     candidates jointly -> Best_A.
  2. Stream B global comparison: Pareto-filter + VIKOR-rank all 100
     candidates jointly -> Best_B.
  3. Final cross-stream decision: build the 2-row (Best_A, Best_B)
     criterion table (same normalized criteria/weights as each stream),
     explicitly check Pareto dominance between the two finalists, then run
     VIKOR to select Best_Overall.
  4. MCDM sensitivity (section 21.4): rank correlation + top-choice
     agreement across the six weight-emphasis schemes and VIKOR/TOPSIS/
     weighted-sum, computed independently within each stream's
     non-dominated set.

Do NOT pool Stream A and Stream B metrics into one ranking before step 3 -
spec section 1's decision process and section 21 require two independent
global rankings feeding a final two-candidate comparison, never a single
flat 120-candidate ranking.

Also retains the incremental-value comparison (old Table 6: A2-A1, A3-A1,
A4-A2, A4-A3) as a SUPPLEMENTARY factor-level diagnostic, computed only
over Stream A (all_features) cells - per spec section 20, this pairwise
view must never be the main analytical structure, so it is written under
statistical_tests/, not pareto_mcda/.

Usage:
    python scripts/11_pareto_mcda_and_incremental.py [--config CONFIG_PATH] [--run-id RUN_ID]

Outputs (spec section 30):
    - outputs/runs/<run_id>/pareto_mcda/stream_A/table4_stream_A_ranking.csv
    - outputs/runs/<run_id>/pareto_mcda/stream_A/rank_correlation.csv
    - outputs/runs/<run_id>/pareto_mcda/stream_B/table5_stream_B_ranking.csv
    - outputs/runs/<run_id>/pareto_mcda/stream_B/rank_correlation.csv
    - outputs/runs/<run_id>/pareto_mcda/final/table7_stream_winners_and_final.csv
    - outputs/runs/<run_id>/pareto_mcda/final/dominance_check.json
    - outputs/runs/<run_id>/statistical_tests/table6_incremental_value.csv (supplementary)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.core import Config, create_run_directories, setup_logging, get_latest_run_id, get_logger
from src.evaluation import compute_incremental_value_table
from src.decision import (
    build_pareto_mcda_table, rank_correlation_across_weight_sets, select_best_overall,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Two-stage Stream A / Stream B Pareto/MCDM decision structure')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run-id', type=str, default=None)
    return parser.parse_args()


def _load_stream(dirs, stream_key):
    pred_path = dirs[f'fold_predictions_stream_{stream_key}'] / 'predictions.csv'
    metrics_path = dirs[f'metrics_stream_{stream_key}'] / 'metrics.csv'
    predictions_df = pd.read_csv(pred_path, parse_dates=['target_date']) if pred_path.exists() else pd.DataFrame()
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    # Naive reference rows ride along in both streams' files (spec section
    # 18) but must never enter the Stream A (20) / Stream B (100) MCDM
    # candidate ranking - they carry no configuration/FS identity to rank.
    if not metrics_df.empty:
        metrics_df = metrics_df[metrics_df['configuration'] != 'naive_reference'].copy()
    if not predictions_df.empty:
        predictions_df = predictions_df[predictions_df['configuration'] != 'naive_reference'].copy()
    return predictions_df, metrics_df


def _best_row(ranked_table: pd.DataFrame):
    if ranked_table.empty:
        return None
    selected = ranked_table[ranked_table['final_decision'].astype(str).str.startswith('selected')]
    if selected.empty:
        return None
    return selected.iloc[0]


def _global_stream_ranking(metrics_df, expected_n, stream_name, out_dir, logger):
    logger.info("=" * 60)
    logger.info(f"{stream_name} global comparison: {len(metrics_df)}/{expected_n} candidates with results")
    logger.info("=" * 60)

    ranked = build_pareto_mcda_table(metrics_df)
    ranked_path = out_dir / f'{stream_name.lower()}_global_ranking.csv'
    ranked.to_csv(ranked_path, index=False)
    logger.info(f"{stream_name} ranking: {len(ranked)} rows -> {ranked_path}")

    rank_corr = rank_correlation_across_weight_sets(metrics_df)
    rank_corr_path = out_dir / 'rank_correlation.csv'
    rank_corr.to_csv(rank_corr_path, index=False)
    if not rank_corr.empty:
        winner_changes = (~rank_corr['top_choice_agrees']).sum()
        logger.info(
            f"{stream_name} MCDM sensitivity: {len(rank_corr)} scheme pairs, "
            f"{winner_changes} disagree on the top choice"
        )

    best = _best_row(ranked)
    if best is not None:
        logger.info(
            f"Best_{stream_name[-1]}: {best['configuration']}/{best.get('fs_option')}/{best['model']} "
            f"(weighted_mase={best['weighted_mase']:.4f}, n_features={best['n_features']:.0f})"
        )
    else:
        logger.warning(f"{stream_name}: no candidate selected (empty or degenerate ranking)")

    return ranked, best


def _final_cross_stream_decision(best_a, best_b, out_dir, logger):
    """Spec section 21.3: compare Best_A/Best_B with the same normalized
    criteria/weights, check Pareto dominance explicitly, then VIKOR-select
    Best_Overall from just the two finalists (src.decision.experiment_ranking
    ::select_best_overall - unit-tested there)."""
    logger.info("=" * 60)
    logger.info("Final cross-stream decision: Best_A vs Best_B")
    logger.info("=" * 60)

    overall_label, table7, dominance_result = select_best_overall(best_a, best_b)

    (out_dir / 'dominance_check.json').write_text(json.dumps(dominance_result, indent=2))
    logger.info(f"Dominance check: {dominance_result}")

    winner_row = table7[table7['stream_winner'] == overall_label].iloc[0]
    logger.info(
        f"Best_Overall: {overall_label} "
        f"({winner_row['configuration']}/{winner_row.get('fs_option')}/{winner_row['model']})"
    )

    table7_path = out_dir / 'table7_stream_winners_and_final.csv'
    table7.to_csv(table7_path, index=False)
    logger.info(f"Table 7 (stream winners + final winner) -> {table7_path}")

    return overall_label, table7


def main():
    args = parse_args()

    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config()

    if args.run_id:
        config.run_id = args.run_id
    else:
        latest = get_latest_run_id(config.output.base_dir)
        if latest:
            config.run_id = latest

    dirs = create_run_directories(config)
    logger = setup_logging(log_dir=dirs['logs'], run_id=config.run_id)
    logger.info("=" * 60)
    logger.info("Script 11: Two-stage Stream A / Stream B Pareto/MCDM decision structure")
    logger.info("=" * 60)

    pred_a, metrics_a = _load_stream(dirs, 'A')
    pred_b, metrics_b = _load_stream(dirs, 'B')

    if metrics_a.empty and metrics_b.empty:
        logger.error("No Stream A or Stream B metrics found - run scripts/10_run_experiment_grid.py first.")
        return 1

    ranked_a, best_a = _global_stream_ranking(metrics_a, 20, 'Stream_A', dirs['pareto_mcda_stream_A'], logger)
    ranked_b, best_b = _global_stream_ranking(metrics_b, 100, 'Stream_B', dirs['pareto_mcda_stream_B'], logger)

    if best_a is None or best_b is None:
        logger.error("Cannot run the final cross-stream decision - Best_A or Best_B is missing.")
        return 1

    overall_label, table7 = _final_cross_stream_decision(best_a, best_b, dirs['pareto_mcda_final'], logger)

    # ---- Supplementary factor-level diagnostic (spec section 20: not the
    # main decision structure) - Stream A's all_features cells only, the
    # closest analogue to the old A1-A4 incremental-value comparison. ----
    logger.info("-" * 40)
    logger.info("Computing supplementary Table 6 (incremental value, Stream A all_features)...")
    if not pred_a.empty:
        models = sorted(pred_a['model'].unique())
        table6 = compute_incremental_value_table(pred_a, models=models, fs_option='all_features')
        table6_path = dirs['statistical_tests'] / 'table6_incremental_value.csv'
        table6.to_csv(table6_path, index=False)
        logger.info(f"Table 6 (supplementary): {len(table6)} rows -> {table6_path}")

    logger.info("=" * 60)
    logger.info("Script 11 complete!")
    logger.info(f"  Best_A selected, Best_B selected, Best_Overall = {overall_label}")
    logger.info("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
