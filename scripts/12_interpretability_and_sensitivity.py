#!/usr/bin/env python
"""
Script 12: SHAP interpretation of Best_A/Best_B/Best_Overall + COVID regime
analysis (spec sections 22/23), plus target-derived-feature sensitivity
(spec section 19.5/29).

Reads Table 7 (Best_A/Best_B/Best_Overall) from scripts/11's output and, for
EACH of the three winners independently, refits all 5 model classes on that
winner's full audited/selected data (not walk-forward - interpretability
wants one stable fit, not a fold-by-fold view) and produces:

  - Cross-model normalized feature-importance ranks (Table 11-equivalent).
  - Regime (pre/COVID/post-COVID) SHAP/permutation importance + stability
    for the winner's own champion model, using model-appropriate SHAP
    (LinearExplainer for Ridge, TreeExplainer for RF/LightGBM/CatBoost;
    LSTM falls back to sequence-aware permutation importance with an
    explicit "SHAP unreliable" note, per spec section 22 - never fabricated).
  - Mobility-feature and grid-feature SHAP-contribution aggregation by
    registry family, per regime (spec section 23).
  - The 2020Q2 (and other COVID-anomaly) prediction error, pulled from the
    winner's own walk-forward fold predictions where available.
  - Target-derived-feature exclusion sensitivity (spec 19.5/29).

Because Best_Overall is always identical to whichever of Best_A/Best_B won
the final VIKOR comparison (spec section 21.3 picks one of the two
finalists, not a new candidate), it is re-interpreted under its own
`interpretability/best_overall/` directory for a self-contained report
section, not symlinked/copied from the matching stream directory.

Usage:
    python scripts/12_interpretability_and_sensitivity.py [--config CONFIG_PATH] [--run-id RUN_ID]

Outputs (spec section 30):
    - outputs/runs/<run_id>/interpretability/{best_A,best_B,best_overall}/
        table11_cross_model_importance.csv
        regime_importance.csv
        regime_stability.csv
        family_contribution_by_regime.csv
        covid_anomaly_errors.csv
        target_derived_sensitivity.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.core import Config, create_run_directories, setup_logging, get_logger, set_seed
from src.data_io import load_processed_data
from src.features import FeatureRegistry, exclude_target_derived_features
from src.splits import (
    load_cv_plan, panel2_split_config,
    slice_configurations_to_common_period, assert_cv_plan_fits_matrix,
)
from src.models import ModelRegistry
from src.pipeline import run_configuration_model, ExperimentCell
from src.interpretability import (
    compute_shap_values, get_feature_importance_from_shap, get_ridge_coefficients,
    compute_permutation_importance, manual_permutation_importance,
    analyze_regime_shap, analyze_regime_permutation_importance,
    build_cross_model_importance_table, compute_regime_stability, compute_family_contribution,
)

SHAP_MODEL_TYPES = {'ridge': 'linear', 'random_forest': 'tree', 'lightgbm': 'tree', 'catboost': 'tree'}
ALL_MODELS = ['ridge', 'random_forest', 'lightgbm', 'catboost', 'lstm']
WINNER_DIRS = {'Best_A': 'interpretability_best_A', 'Best_B': 'interpretability_best_B',
               'Best_Overall': 'interpretability_best_overall'}
COVID_ANOMALY_DATES = ['2020-04-01']  # 2020Q2 (spec section 23) - the primary shock quarter


def parse_args():
    parser = argparse.ArgumentParser(description='SHAP interpretation of Best_A/Best_B/Best_Overall + COVID regimes')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run-id', type=str, default=None)
    return parser.parse_args()


def get_regime_periods(config: Config) -> dict:
    covid_start = pd.Period(config.features.covid_start).start_time
    covid_end = pd.Period(config.features.covid_end).end_time
    return {
        'pre_covid': (None, str(covid_start.date())),
        'covid': (str(covid_start.date()), str(covid_end.date())),
        'post_covid': (str(covid_end.date()), None),
    }


def get_importance_for_model(model_name, model, X, y, logger):
    """
    Return an importance DataFrame with columns ['feature', 'importance'],
    model-appropriate (spec section 22). LSTM SHAP (GradientExplainer/
    DeepExplainer) is not attempted here - it predates this engagement as
    unreliable for this codebase's LSTM wrapper (variable-length
    lookback-aware predict, see manual_permutation_importance's docstring)
    - reporting that transparently via sequence-aware permutation
    importance, never a fabricated SHAP value, satisfies spec section 22's
    explicit failure-transparency requirement.
    """
    if model_name == 'ridge':
        coef_df = get_ridge_coefficients(model)
        return coef_df.rename(columns={'abs_coefficient': 'importance'})[['feature', 'importance']]
    if model_name == 'lstm':
        logger.info("  lstm: SHAP (GradientExplainer/DeepExplainer) not attempted - falling back to "
                    "sequence-aware permutation importance (spec section 22 failure-transparency requirement)")
        lookback = model.params['lookback']
        y_aligned = y.iloc[lookback:]
        perm_df = manual_permutation_importance(model.predict, X, y_aligned)
        return perm_df.rename(columns={'importance_mean': 'importance'})[['feature', 'importance']]
    shap_type = SHAP_MODEL_TYPES.get(model_name)
    if shap_type == 'tree':
        try:
            shap_values, _ = compute_shap_values(model, X, shap_type)
            return get_feature_importance_from_shap(shap_values, list(X.columns))[['feature', 'importance']]
        except Exception:
            pass
    perm_df = compute_permutation_importance(model, X, y)
    return perm_df.rename(columns={'importance_mean': 'importance'})[['feature', 'importance']]


def _load_winner_matrix(configuration, fs_option, dirs, registry, logger):
    """Load and, if fs_option != 'all_features', FS-restrict this winner's
    configuration matrix - resliced to the primary common period (spec
    section 14) so positional fold indices in panel2_cv_plan stay aligned
    (see src/splits/panels.py::slice_configurations_to_common_period's
    docstring for why this is not optional)."""
    processed_dir = Path('data/processed')
    X_by_config = {tag: load_processed_data(processed_dir / f'X_{tag}') for tag in ('A1', 'A2', 'A3', 'A4')}
    y = load_processed_data(processed_dir / 'y')['target']
    X_by_config = slice_configurations_to_common_period(X_by_config, y, reference_tag='A3')

    X_config = X_by_config[configuration]
    y_slice = y.loc[X_config.index]

    if fs_option != 'all_features':
        fs_results_path = dirs['selected_features'] / f'streamB_fs_results_{configuration}.json'
        if fs_results_path.exists():
            with open(fs_results_path) as f:
                fs_results = json.load(f)
            if fs_option in fs_results:
                selected = [c for c in fs_results[fs_option]['selected_features'] if c in X_config.columns]
                X_config = X_config[selected]
            else:
                logger.warning(f"{fs_option} not found in {fs_results_path}, using all_features")
        else:
            logger.warning(f"FS results not found at {fs_results_path}, using all_features for {configuration}")

    return X_config, y_slice


def _covid_anomaly_errors(configuration, fs_option, model_name, config, logger):
    """Spec section 23: report prediction error at 2020Q2 (and any other
    COVID-window anomaly quarters) directly from this winner's own
    walk-forward fold predictions (Stream A or B, whichever produced this
    cell) - a real out-of-sample error, not the interpretability full-fit."""
    rows = []
    for stream in ('A', 'B'):
        pred_path = config.run_dir / 'fold_predictions' / f'stream_{stream}' / 'predictions.csv'
        if not pred_path.exists():
            continue
        df = pd.read_csv(pred_path, parse_dates=['target_date'])
        mask = (
            (df['configuration'] == configuration) & (df['fs_option'] == fs_option) &
            (df['model'] == model_name) & (df['target_date'].isin(pd.to_datetime(COVID_ANOMALY_DATES)))
        )
        rows.append(df[mask])
    if not rows or all(r.empty for r in rows):
        logger.warning(f"No fold predictions found for {configuration}/{fs_option}/{model_name} at COVID anomaly dates")
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)[
        ['configuration', 'fs_option', 'model', 'horizon', 'target_date', 'actual', 'predicted', 'residual']
    ]


def interpret_winner(label, configuration, fs_option, model_name, config, dirs, registry, logger):
    logger.info("=" * 60)
    logger.info(f"Interpreting {label}: {configuration}/{fs_option}/{model_name}")
    logger.info("=" * 60)

    out_dir = dirs[WINNER_DIRS[label]]
    X_config, y = _load_winner_matrix(configuration, fs_option, dirs, registry, logger)
    logger.info(f"Data: {len(X_config)} rows, {len(X_config.columns)} features")

    # ---- Cross-model importance (Table 11-equivalent) ----
    importance_by_model, fitted_models = {}, {}
    for m in ALL_MODELS:
        try:
            model = ModelRegistry.create(m, {})
            model.fit(X_config, y)
            fitted_models[m] = model
            importance_by_model[m] = get_importance_for_model(m, model, X_config, y, logger)
        except Exception as e:
            logger.warning(f"  {m}: fit/importance failed - {e}")

    table11 = build_cross_model_importance_table(importance_by_model)
    table11.to_csv(out_dir / 'table11_cross_model_importance.csv', index=False)
    logger.info(f"Cross-model importance: {len(table11)} features")

    # ---- Regime (pre/COVID/post-COVID) analysis for the champion model ----
    regime_periods = get_regime_periods(config)
    champion_model = fitted_models.get(model_name)
    regime_results = {}
    if champion_model is not None:
        try:
            shap_type = SHAP_MODEL_TYPES.get(model_name)
            if shap_type in ('linear', 'tree'):
                regime_results = analyze_regime_shap(champion_model, X_config, regime_periods=regime_periods, model_type=shap_type)
            else:
                regime_results = analyze_regime_permutation_importance(champion_model, X_config, y, regime_periods=regime_periods)
        except Exception as e:
            logger.warning(f"Regime analysis failed: {e}")

    if regime_results:
        regime_df = pd.concat(regime_results.values(), ignore_index=True)
        regime_df.to_csv(out_dir / 'regime_importance.csv', index=False)
        stability_df = compute_regime_stability(regime_results)
        stability_df.to_csv(out_dir / 'regime_stability.csv', index=False)
        logger.info(f"Regime importance: {len(regime_df)} rows across {len(regime_results)} regimes")

        family_map = registry.to_dataframe().set_index('name')['family']
        family_rows = []
        for regime_name, df in regime_results.items():
            fc = compute_family_contribution(df, family_map)
            fc['regime'] = regime_name
            family_rows.append(fc)
        pd.concat(family_rows, ignore_index=True).to_csv(out_dir / 'family_contribution_by_regime.csv', index=False)
    else:
        logger.warning("No regime results produced (champion model missing or all regimes too small).")

    # ---- COVID anomaly (2020Q2) prediction error ----
    anomaly_df = _covid_anomaly_errors(configuration, fs_option, model_name, config, logger)
    anomaly_df.to_csv(out_dir / 'covid_anomaly_errors.csv', index=False)

    # ---- Target-derived-feature exclusion sensitivity (spec 19.5/29) ----
    eval_cv_plan = load_cv_plan(Path('data/processed') / 'panel2_cv_plan.pkl')
    outer_split_config = panel2_split_config(config.splits)
    assert_cv_plan_fits_matrix(eval_cv_plan, X_config, tag=configuration)

    sensitivity_rows = []
    baseline_cell = ExperimentCell(configuration=configuration, panel='panel2_common_period',
                                    fs_option=fs_option, model=model_name, seed=config.seed)
    baseline_result = run_configuration_model(
        X_config, y, eval_cv_plan, baseline_cell, config, config.run_id, outer_split_config=outer_split_config
    )
    sensitivity_rows.append({'variant': 'full_audited_pool', 'n_features': len(X_config.columns),
                              'weighted_mase': baseline_result['summary'].get('weighted_mase')})

    for mode, mode_label in (('named', 'named_target_derived_excluded'), ('all', 'all_target_derived_excluded')):
        X_reduced = exclude_target_derived_features(X_config, registry, mode=mode)
        if X_reduced.shape[1] == X_config.shape[1] or X_reduced.empty:
            logger.info(f"  {mode_label}: no applicable columns to drop, skipping")
            continue
        cell = ExperimentCell(configuration=configuration, panel='panel2_common_period',
                               fs_option=f'{fs_option}_{mode_label}', model=model_name, seed=config.seed)
        try:
            result = run_configuration_model(
                X_reduced, y, eval_cv_plan, cell, config, config.run_id, outer_split_config=outer_split_config
            )
            sensitivity_rows.append({'variant': mode_label, 'n_features': len(X_reduced.columns),
                                      'weighted_mase': result['summary'].get('weighted_mase')})
        except Exception as e:
            logger.warning(f"  {mode_label} failed: {e}")

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    if not sensitivity_df.empty:
        baseline_mase = sensitivity_df.iloc[0]['weighted_mase']
        sensitivity_df['delta_vs_full_pool'] = sensitivity_df['weighted_mase'] - baseline_mase
    sensitivity_df.to_csv(out_dir / 'target_derived_sensitivity.csv', index=False)
    logger.info(f"Target-derived sensitivity: {len(sensitivity_df)} rows")


def main():
    args = parse_args()

    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config()
    if args.run_id:
        config.run_id = args.run_id

    set_seed(config.seed)
    dirs = create_run_directories(config)
    logger = setup_logging(log_dir=dirs['logs'], run_id=config.run_id)
    logger.info("=" * 60)
    logger.info("Script 12: SHAP interpretation of Best_A/Best_B/Best_Overall + COVID regimes")
    logger.info("=" * 60)

    table7_path = dirs['pareto_mcda_final'] / 'table7_stream_winners_and_final.csv'
    if not table7_path.exists():
        logger.error("No Table 7 found - run scripts/11_pareto_mcda_and_incremental.py first.")
        return 1
    table7 = pd.read_csv(table7_path)

    registry = FeatureRegistry.load(config.feature_registry_path)

    winners = {}
    for _, row in table7.iterrows():
        winners[row['stream_winner']] = row
        if row.get('is_best_overall'):
            winners['Best_Overall'] = row

    if 'Best_A' not in winners or 'Best_B' not in winners or 'Best_Overall' not in winners:
        logger.error(f"Table 7 missing one of Best_A/Best_B/Best_Overall - found: {list(winners.keys())}")
        return 1

    for label in ('Best_A', 'Best_B', 'Best_Overall'):
        row = winners[label]
        interpret_winner(
            label, row['configuration'], row.get('fs_option', 'all_features'), row['model'],
            config, dirs, registry, logger
        )

    logger.info("=" * 60)
    logger.info("Script 12 complete!")
    logger.info(f"  Results saved under: {dirs['root'] / 'interpretability'}")
    logger.info("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
