#!/usr/bin/env python
"""
Script 15: Generate the forecast atlas (spec section 25) - 36 genuine
out-of-sample forecast PDFs, all built from saved fold-level predictions:

  - 24 family-winner figures: A1-A4 (Stream A) + B1-B4 (Stream B), each
    x H1/H2/H4 - the highest-weighted_mase-ranked candidate WITHIN each
    family (spec 25.1: for visual reporting only, never in place of the
    global Stream A/B MCDM ranking from script 11).
  - 6 stream-winner figures: Best_A/Best_B x H1/H2/H4.
  - 3 final-winner figures: Best_Overall x H1/H2/H4.
  - 3 Best_A-vs-Best_B comparison figures, one per horizon.

Usage:
    python scripts/15_generate_forecast_atlas.py [--config CONFIG_PATH] [--run-id RUN_ID]

Outputs:
    outputs/figures/pdf/forecasting/fig_family_<A|B>{1-4}_H{1,2,4}.pdf
    outputs/figures/pdf/forecasting/fig_best_<A|B>_forecast_H{1,2,4}.pdf
    outputs/figures/pdf/forecasting/fig_best_overall_forecast_H{1,2,4}.pdf
    outputs/figures/pdf/forecasting/fig_best_A_vs_B_forecast_H{1,2,4}.pdf
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.core import Config, get_latest_run_id, setup_logging, get_logger
from src.core.utils import to_original_scale
from src.data_io import load_processed_data
from src.reporting import select_family_winner, fig_forecast_single, fig_forecast_comparison

CONFIGURATIONS = ['A1', 'A2', 'A3', 'A4']
HORIZONS = [1, 2, 4]


def parse_args():
    parser = argparse.ArgumentParser(description='Generate the 36-PDF forecast atlas')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='outputs/figures/pdf/forecasting')
    return parser.parse_args()


def _cell_predictions(predictions_df, configuration, fs_option, model, horizon):
    if predictions_df.empty:
        return pd.DataFrame()
    mask = (
        (predictions_df['configuration'] == configuration) & (predictions_df['fs_option'] == fs_option) &
        (predictions_df['model'] == model) & (predictions_df['horizon'] == horizon)
    )
    return predictions_df[mask].sort_values('target_date')


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

    logger = setup_logging(log_dir=Path('outputs/logs'), run_id='generate_forecast_atlas')
    logger.info("=" * 60)
    logger.info(f"Script 15: Generate forecast atlas (source run_id={run_id})")
    logger.info("=" * 60)

    run_dir = Path(config.output.base_dir) / 'runs' / run_id
    processed_dir = Path('data/processed')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full actual history, inverted to original-scale CO2e (spec section 2/17
    # - predictions_df's 'actual'/'predicted' columns are ALREADY
    # original-scale via run_configuration_model's to_original_scale step;
    # y.parquet is still on config.data.target_transform's scale, so it
    # must be inverted here too or the history line silently plots on the
    # wrong scale - caught visually during this script's own smoke test).
    y_raw = load_processed_data(processed_dir / 'y')['target']
    y_raw.index = pd.to_datetime(y_raw.index)
    y_original = pd.Series(to_original_scale(y_raw.values, config.data.target_transform), index=y_raw.index)

    stream_a_metrics = pd.read_csv(run_dir / 'metrics' / 'stream_A' / 'metrics.csv') \
        if (run_dir / 'metrics' / 'stream_A' / 'metrics.csv').exists() else pd.DataFrame()
    stream_b_metrics = pd.read_csv(run_dir / 'metrics' / 'stream_B' / 'metrics.csv') \
        if (run_dir / 'metrics' / 'stream_B' / 'metrics.csv').exists() else pd.DataFrame()
    for df in (stream_a_metrics, stream_b_metrics):
        if not df.empty and 'configuration' in df.columns:
            df.drop(df[df['configuration'] == 'naive_reference'].index, inplace=True)

    stream_a_predictions = pd.read_csv(run_dir / 'fold_predictions' / 'stream_A' / 'predictions.csv',
                                        parse_dates=['target_date']) \
        if (run_dir / 'fold_predictions' / 'stream_A' / 'predictions.csv').exists() else pd.DataFrame()
    stream_b_predictions = pd.read_csv(run_dir / 'fold_predictions' / 'stream_B' / 'predictions.csv',
                                        parse_dates=['target_date']) \
        if (run_dir / 'fold_predictions' / 'stream_B' / 'predictions.csv').exists() else pd.DataFrame()

    n_generated = 0

    # ---- 24 family-winner figures ----
    for tag in CONFIGURATIONS:
        winner = select_family_winner(stream_a_metrics, tag)
        if winner is not None:
            for h in HORIZONS:
                cell_preds = _cell_predictions(stream_a_predictions, tag, 'all_features', winner['model'], h)
                meta = {'label': f'{tag} family winner (Stream A)', 'configuration': tag,
                        'fs_option': 'all_features', 'model': winner['model'], 'horizon': h,
                        'mase': winner.get(f'mase_h{h}', float('nan')), 'n_features': winner['n_features']}
                fig_forecast_single(y_original, cell_preds, meta,
                                     output_dir / f'fig_family_{tag}_H{h}.pdf')
                n_generated += 1
        else:
            logger.warning(f"No Stream A family winner found for {tag} - skipping its 3 figures")

        b_tag = f'B{tag[1]}'
        winner = select_family_winner(stream_b_metrics, tag)
        if winner is not None:
            for h in HORIZONS:
                cell_preds = _cell_predictions(stream_b_predictions, tag, winner['fs_option'], winner['model'], h)
                meta = {'label': f'{b_tag} family winner (Stream B)', 'configuration': tag,
                        'fs_option': winner['fs_option'], 'model': winner['model'], 'horizon': h,
                        'mase': winner.get(f'mase_h{h}', float('nan')), 'n_features': winner['n_features']}
                fig_forecast_single(y_original, cell_preds, meta,
                                     output_dir / f'fig_family_{b_tag}_H{h}.pdf')
                n_generated += 1
        else:
            logger.warning(f"No Stream B family winner found for {b_tag} - skipping its 3 figures")

    # ---- 6 stream-winner + 3 final-winner + 3 A-vs-B comparison figures ----
    table7_path = run_dir / 'pareto_mcda' / 'final' / 'table7_stream_winners_and_final.csv'
    if not table7_path.exists():
        logger.error("No Table 7 found - run scripts/11_pareto_mcda_and_incremental.py first. "
                      "Skipping stream/overall/comparison figures.")
    else:
        table7 = pd.read_csv(table7_path)
        best_a_row = table7[table7['stream_winner'] == 'Best_A'].iloc[0]
        best_b_row = table7[table7['stream_winner'] == 'Best_B'].iloc[0]
        overall_label = table7[table7['is_best_overall']]['stream_winner'].iloc[0]

        streams = {'Best_A': (best_a_row, stream_a_predictions), 'Best_B': (best_b_row, stream_b_predictions)}
        for label, (row, predictions_df) in streams.items():
            for h in HORIZONS:
                cell_preds = _cell_predictions(predictions_df, row['configuration'], row.get('fs_option', 'all_features'),
                                                row['model'], h)
                meta = {'label': label, 'configuration': row['configuration'],
                        'fs_option': row.get('fs_option', 'all_features'), 'model': row['model'], 'horizon': h,
                        'mase': row.get(f'mase_h{h}', float('nan')), 'n_features': row['n_features']}
                fig_forecast_single(y_original, cell_preds, meta,
                                     output_dir / f'fig_{label.lower()}_forecast_H{h}.pdf')
                n_generated += 1

        overall_row, overall_predictions = streams[overall_label]
        for h in HORIZONS:
            cell_preds = _cell_predictions(overall_predictions, overall_row['configuration'],
                                            overall_row.get('fs_option', 'all_features'), overall_row['model'], h)
            meta = {'label': 'Best_Overall', 'configuration': overall_row['configuration'],
                    'fs_option': overall_row.get('fs_option', 'all_features'), 'model': overall_row['model'],
                    'horizon': h, 'mase': overall_row.get(f'mase_h{h}', float('nan')),
                    'n_features': overall_row['n_features']}
            fig_forecast_single(y_original, cell_preds, meta, output_dir / f'fig_best_overall_forecast_H{h}.pdf')
            n_generated += 1

        for h in HORIZONS:
            preds_a = _cell_predictions(stream_a_predictions, best_a_row['configuration'],
                                         best_a_row.get('fs_option', 'all_features'), best_a_row['model'], h)
            preds_b = _cell_predictions(stream_b_predictions, best_b_row['configuration'],
                                         best_b_row.get('fs_option', 'all_features'), best_b_row['model'], h)
            meta_a = {'label': 'Best_A', 'configuration': best_a_row['configuration'],
                      'fs_option': best_a_row.get('fs_option', 'all_features'), 'model': best_a_row['model'],
                      'horizon': h, 'mase': best_a_row.get(f'mase_h{h}', float('nan'))}
            meta_b = {'label': 'Best_B', 'configuration': best_b_row['configuration'],
                      'fs_option': best_b_row.get('fs_option', 'all_features'), 'model': best_b_row['model'],
                      'horizon': h, 'mase': best_b_row.get(f'mase_h{h}', float('nan'))}
            fig_forecast_comparison(y_original, preds_a, preds_b, meta_a, meta_b,
                                     output_dir / f'fig_best_A_vs_B_forecast_H{h}.pdf')
            n_generated += 1

    logger.info("=" * 60)
    logger.info(f"Script 15 complete! {n_generated}/36 forecast figures generated -> {output_dir}")
    logger.info("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
