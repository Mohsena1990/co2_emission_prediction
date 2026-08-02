#!/usr/bin/env python
"""
Script 10: Run the Stream A / Stream B experimental grid (spec sections
11/12/17/18).

Stream A (no feature selection): all 4 audited configurations (A1-A4) x all
5 forecasting models, on the single primary common period
(src/splits/panels.py::PRIMARY_PERIOD_PANEL, spec section 14) - 20
candidates, evaluated jointly (spec section 20: "do not select a winner
separately within each configuration before MCDM").

Stream B (with feature selection): each configuration's own audited pool
(B{n} <- A{n}, spec section 12) x FS1-FS5 (src/fs/consensus.py::
run_all_fs_options, run inside the outer-training-only inner expanding
window) x all 5 forecasting models - 100 candidates.

Two naive baselines (spec section 18: persistence 'naive_lag1' and
seasonal-naive 'naive_lag4') are run once each on the primary period and
attached to BOTH streams' predictions/metrics tables as reference rows
(configuration='naive_reference', fs_option='none') - they are NOT part of
the 20/100 candidate counts.

Usage:
    python scripts/10_run_experiment_grid.py [--config CONFIG_PATH]
        [--run-id RUN_ID] [--stream {A,B,both}] [--models MODEL [MODEL ...]]

Outputs (spec section 30):
    - outputs/runs/<run_id>/fold_predictions/stream_A/predictions.csv
    - outputs/runs/<run_id>/fold_predictions/stream_A/provenance.json
    - outputs/runs/<run_id>/metrics/stream_A/metrics.csv
    - outputs/runs/<run_id>/fold_predictions/stream_B/predictions.csv
    - outputs/runs/<run_id>/fold_predictions/stream_B/provenance.json
    - outputs/runs/<run_id>/metrics/stream_B/metrics.csv
    - outputs/runs/<run_id>/selected_features/streamB_fs_results_<config>.json
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.core import (
    Config, create_run_directories, setup_logging, get_logger,
    set_seed, save_json_numpy
)
from src.data_io import load_processed_data
from src.splits import (
    load_cv_plan, build_tuning_cv_plan,
    slice_configurations_to_common_period, assert_cv_plan_fits_matrix,
)
from src.splits.panels import PRIMARY_PERIOD_PANEL, panel2_split_config
from src.fs import run_all_fs_options
from src.pipeline import run_configuration_model, run_naive_baseline, ExperimentCell

CONFIGURATIONS = ['A1', 'A2', 'A3', 'A4']
NAIVE_MODELS = ['naive_lag1', 'naive_lag4']


def parse_args():
    parser = argparse.ArgumentParser(description='Run the Stream A / Stream B experimental grid')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--stream', type=str, default='both', choices=['A', 'B', 'both'])
    parser.add_argument('--models', type=str, nargs='+', default=None,
                         help='Restrict to specific models (default: config.model.models)')
    parser.add_argument('--skip-naive', action='store_true',
                         help='Skip the naive_lag1/naive_lag4 reference baselines')
    parser.add_argument('--shortlist-csv', type=str, default=None,
                         help="Restrict to specific (configuration, fs_option, model) cells, read from a "
                              "CSV with those columns (e.g. a Pareto-non-dominated shortlist from script 11's "
                              "output) - spec section 18's 'full budget + multiple seeds on the Pareto-"
                              "shortlisted configurations only' re-run stage. Stream B still runs FS1-5 for "
                              "every shortlisted configuration (a shortlisted cell's fs_option pins the MODEL "
                              "cell to re-run, not which FS strategies get recomputed).")
    parser.add_argument('--seed', type=int, default=None,
                         help='Override config.seed (for multi-seed PSO-stability reruns of a shortlist)')
    return parser.parse_args()


def _load_shortlist(path):
    """{stream: {'A' cells: set of (configuration, model)}, 'B': set of
    (configuration, fs_option, model)}} - Stream A ignores fs_option (it's
    always 'all_features'); Stream B matches on the full triple."""
    df = pd.read_csv(path)
    required = {'configuration', 'model'}
    if not required.issubset(df.columns):
        raise ValueError(f"--shortlist-csv must have at least columns {required}, got {list(df.columns)}")
    if 'fs_option' not in df.columns:
        df['fs_option'] = 'all_features'
    a_cells = {(r.configuration, r.model) for r in df.itertuples() if r.fs_option == 'all_features'}
    b_cells = {(r.configuration, r.fs_option, r.model) for r in df.itertuples() if r.fs_option != 'all_features'}
    return a_cells, b_cells


def _load_matrices(processed_dir: Path):
    X = {tag: load_processed_data(processed_dir / f'X_{tag}') for tag in CONFIGURATIONS}
    y = load_processed_data(processed_dir / 'y')['target']
    return X, y


def run_stream_a(X_by_config, y, eval_plan, split_config, config, run_id, logger, models, shortlist=None):
    logger.info("=" * 60)
    if shortlist is not None:
        logger.info(f"Stream A: shortlist rerun ({len(shortlist)} cells)")
    else:
        logger.info("Stream A: 4 configurations x 5 models, NO feature selection (20 candidates)")
    logger.info("=" * 60)

    all_predictions, all_provenance, all_summaries = [], [], []

    for tag in CONFIGURATIONS:
        X = X_by_config[tag]
        y_slice = y.loc[X.index]

        for model_name in models:
            if shortlist is not None and (tag, model_name) not in shortlist:
                continue
            t0 = time.time()
            cell = ExperimentCell(configuration=tag, panel=PRIMARY_PERIOD_PANEL,
                                   fs_option='all_features', model=model_name, seed=config.seed)
            logger.info(f"Stream A cell: {tag}/{model_name} ({len(X.columns)} features, {len(X)} rows)")
            try:
                result = run_configuration_model(
                    X, y_slice, eval_plan, cell, config, run_id, outer_split_config=split_config
                )
            except Exception as e:
                logger.error(f"Stream A cell {tag}/{model_name} failed: {e}")
                continue

            all_predictions.extend(result['predictions'])
            all_provenance.extend(result['fold_provenance'])
            all_summaries.append(result['summary'])
            logger.info(f"  -> weighted_mase={result['summary'].get('weighted_mase')}, "
                        f"runtime={time.time() - t0:.1f}s")

    logger.info(f"Stream A complete: {len(all_summaries)}/20 candidates produced a result")
    return all_predictions, all_provenance, all_summaries


def run_stream_b(X_by_config, y, eval_plan, split_config, config, run_id, logger, models, dirs, shortlist=None):
    logger.info("=" * 60)
    if shortlist is not None:
        logger.info(f"Stream B: shortlist rerun ({len(shortlist)} cells) - FS1-5 still run per configuration")
    else:
        logger.info("Stream B: 4 configurations x 5 FS strategies x 5 models (100 candidates)")
    logger.info("=" * 60)

    all_predictions, all_provenance, all_summaries = [], [], []
    shortlist_configs = {c for c, _, _ in shortlist} if shortlist is not None else None

    for tag in CONFIGURATIONS:
        if shortlist_configs is not None and tag not in shortlist_configs:
            continue
        # B{n} <- A{n} (spec section 12): the FS candidate pool is this
        # configuration's own audited feature set, restricted to the
        # primary common period exactly like Stream A.
        X = X_by_config[tag]
        y_slice = y.loc[X.index]

        logger.info(f"Running FS1-FS5 on {tag} ({len(X.columns)} candidate features)...")
        tuning_cv_plan = build_tuning_cv_plan(X, y_slice, eval_plan, split_config)
        fs_results = run_all_fs_options(X, y_slice, tuning_cv_plan, config)

        save_json_numpy(
            {k: {'selected_features': v['selected_features'], 'n_selected': v['n_selected']}
             for k, v in fs_results.items()},
            dirs['selected_features'] / f'streamB_fs_results_{tag}.json'
        )

        for fs_name, fs_result in fs_results.items():
            selected = [f for f in fs_result['selected_features'] if f in X.columns]
            if not selected:
                logger.warning(f"{tag}/{fs_name}: 0 usable selected features, skipping")
                continue
            X_fs = X[selected]

            for model_name in models:
                if shortlist is not None and (tag, fs_name, model_name) not in shortlist:
                    continue
                t0 = time.time()
                cell = ExperimentCell(configuration=tag, panel=PRIMARY_PERIOD_PANEL,
                                       fs_option=fs_name, model=model_name, seed=config.seed)
                logger.info(f"Stream B cell: {tag}/{fs_name}/{model_name} ({len(selected)} features)")
                try:
                    result = run_configuration_model(
                        X_fs, y_slice, eval_plan, cell, config, run_id, outer_split_config=split_config
                    )
                except Exception as e:
                    logger.error(f"Stream B cell {tag}/{fs_name}/{model_name} failed: {e}")
                    continue

                all_predictions.extend(result['predictions'])
                all_provenance.extend(result['fold_provenance'])
                all_summaries.append(result['summary'])
                logger.info(f"  -> weighted_mase={result['summary'].get('weighted_mase')}, "
                            f"runtime={time.time() - t0:.1f}s")

    logger.info(f"Stream B complete: {len(all_summaries)}/100 candidates produced a result")
    return all_predictions, all_provenance, all_summaries


def run_naive_baselines(X_by_config, y, eval_plan, split_config, config, run_id, logger):
    logger.info("=" * 60)
    logger.info("Naive reference baselines (naive_lag1, naive_lag4) - not part of Stream A/B counts")
    logger.info("=" * 60)

    # Naive forecasters ignore feature columns entirely - any configuration's
    # matrix works, its DatetimeIndex is all that's used. A1 has the widest
    # feature-availability guarantees so is used as the reference index.
    X = X_by_config['A1']
    y_slice = y.loc[X.index]

    predictions, provenance, summaries = [], [], []
    for model_name in NAIVE_MODELS:
        result = run_naive_baseline(
            X, y_slice, eval_plan, model_name, config, run_id,
            outer_split_config=split_config, panel=PRIMARY_PERIOD_PANEL,
        )
        predictions.extend(result['predictions'])
        provenance.extend(result['fold_provenance'])
        summaries.append(result['summary'])
        logger.info(f"  {model_name} -> weighted_mase={result['summary'].get('weighted_mase')}")

    return predictions, provenance, summaries


def main():
    args = parse_args()

    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config()

    if args.run_id:
        config.run_id = args.run_id
    if args.models:
        config.model.models = args.models
    if args.seed is not None:
        config.seed = args.seed

    set_seed(config.seed)
    dirs = create_run_directories(config)

    logger = setup_logging(log_dir=dirs['logs'], run_id=config.run_id)
    logger.info("=" * 60)
    logger.info("Script 10: Stream A / Stream B experimental grid")
    logger.info("=" * 60)
    logger.info(f"Models: {config.model.models}")
    logger.info(f"seed: {config.seed}")
    logger.info(f"nested_retuning: {config.optimization.nested_retuning}")

    a_shortlist = b_shortlist = None
    if args.shortlist_csv:
        a_shortlist, b_shortlist = _load_shortlist(args.shortlist_csv)
        logger.info(f"Shortlist loaded: {len(a_shortlist)} Stream A cells, {len(b_shortlist)} Stream B cells")

    processed_dir = Path('data/processed')
    X_by_config, y = _load_matrices(processed_dir)
    X_by_config = slice_configurations_to_common_period(X_by_config, y, reference_tag='A3')

    eval_plan = load_cv_plan(processed_dir / 'panel2_cv_plan.pkl')
    if eval_plan is None or eval_plan.n_folds == 0:
        raise RuntimeError(
            "The primary common-period CV plan (data/processed/panel2_cv_plan.pkl) "
            "is missing or has 0 folds - run scripts/00_make_dataset.py first."
        )
    for tag, X in X_by_config.items():
        assert_cv_plan_fits_matrix(eval_plan, X, tag=tag)
    split_config = panel2_split_config(config.splits)

    models = config.model.models

    all_predictions, all_provenance, all_summaries = [], [], []

    if not args.skip_naive:
        p, prov, s = run_naive_baselines(X_by_config, y, eval_plan, split_config, config, config.run_id, logger)
        all_predictions += p
        all_provenance += prov
        all_summaries += s

    if args.stream in ('A', 'both'):
        p, prov, s = run_stream_a(X_by_config, y, eval_plan, split_config, config, config.run_id, logger, models,
                                   shortlist=a_shortlist)
        pd.DataFrame(p).to_csv(dirs['fold_predictions_stream_A'] / 'predictions.csv', index=False)
        save_json_numpy(prov, dirs['fold_predictions_stream_A'] / 'provenance.json')
        pd.DataFrame(s).to_csv(dirs['metrics_stream_A'] / 'metrics.csv', index=False)
        logger.info(f"Stream A saved: {len(s)} cells, {len(p)} predictions")
        all_predictions += p
        all_provenance += prov
        all_summaries += s

    if args.stream in ('B', 'both'):
        p, prov, s = run_stream_b(
            X_by_config, y, eval_plan, split_config, config, config.run_id, logger, models, dirs,
            shortlist=b_shortlist
        )
        pd.DataFrame(p).to_csv(dirs['fold_predictions_stream_B'] / 'predictions.csv', index=False)
        save_json_numpy(prov, dirs['fold_predictions_stream_B'] / 'provenance.json')
        pd.DataFrame(s).to_csv(dirs['metrics_stream_B'] / 'metrics.csv', index=False)
        logger.info(f"Stream B saved: {len(s)} cells, {len(p)} predictions")
        all_predictions += p
        all_provenance += prov
        all_summaries += s

    # Naive-baseline rows appear in BOTH streams' predictions/metrics tables
    # (spec section 18 - a shared reference, not a third stream) - re-save
    # each stream's files with the naive rows appended, without double
    # counting them in the 20/100 totals reported above.
    if not args.skip_naive:
        naive_pred_df = pd.DataFrame([r for r in all_predictions if r['configuration'] == 'naive_reference'])
        naive_summ = [s for s in all_summaries if s.get('configuration') == 'naive_reference']
        for stream_key in ('A', 'B'):
            if args.stream not in (stream_key, 'both'):
                continue
            pred_path = dirs[f'fold_predictions_stream_{stream_key}'] / 'predictions.csv'
            metrics_path = dirs[f'metrics_stream_{stream_key}'] / 'metrics.csv'
            if pred_path.exists():
                stream_pred = pd.read_csv(pred_path)
                pd.concat([stream_pred, naive_pred_df], ignore_index=True).to_csv(pred_path, index=False)
            if metrics_path.exists():
                stream_metrics = pd.read_csv(metrics_path)
                pd.concat([stream_metrics, pd.DataFrame(naive_summ)], ignore_index=True).to_csv(
                    metrics_path, index=False
                )

    logger.info("=" * 60)
    logger.info("Stream A/B experimental grid complete!")
    logger.info(f"  Total cells (incl. naive): {len(all_summaries)}")
    logger.info(f"  Total predictions: {len(all_predictions)}")
    logger.info(f"  Results saved under: {dirs['root']}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
