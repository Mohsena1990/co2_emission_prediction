"""
Single-cell experiment runner (spec sections 12/15): for one
(configuration, panel, FS option, model) combination, tunes hyperparameters
(sweep or nested per config.optimization.nested_retuning), walk-forward
refits/predicts every outer fold, and returns fold-level predictions +
provenance + summary metrics - the machine-readable unit every table and
figure in outputs/ must trace back to.
"""
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.config import Config, SplitConfig
from ..core.logging_utils import get_logger
from ..core.utils import assert_prediction_alignment, to_original_scale as _to_original_scale
from ..splits.walk_forward import CVPlan, generate_cv_folds
from ..splits.nested_walk_forward import create_nested_walk_forward_splits, build_tuning_cv_plan
from ..models.base import ModelRegistry
from ..optimization.model_optimizer import optimize_model, optimize_model_nested
from ..evaluation.metrics import evaluate_by_horizon, evaluate_stability, create_evaluation_summary


@dataclass
class ExperimentCell:
    """Identifies one cell of the config x panel x FS x model experimental grid."""
    configuration: str  # 'A1' | 'A2' | 'A3' | 'A4'
    panel: str           # 'panel1_full_period' | 'panel2_common_period'
    fs_option: str        # 'all_features' | 'fs_linear' | ... | 'fs_consensus'
    model: str             # 'ridge' | 'random_forest' | 'lightgbm' | 'catboost' | 'lstm'
    seed: int = 42


def run_configuration_model(
    X: pd.DataFrame,
    y: pd.Series,
    eval_cv_plan: CVPlan,
    cell: ExperimentCell,
    config: Config,
    run_id: str,
    outer_split_config: Optional[SplitConfig] = None,
) -> Dict[str, Any]:
    """
    Run one experimental-grid cell end-to-end (spec section 15 pseudocode):

        outer training data
            -> (feature selection already applied upstream - X is pre-sliced
               to cell.fs_option's selected features, or a configuration's
               full audited pool for Stage 1 / fs_option='all_features')
            -> inner expanding-window hyperparameter tuning
            -> refit on full outer training data
            -> predict untouched outer target

    Args:
        X: Feature matrix for this cell.
        y: Target Series aligned to X's index.
        eval_cv_plan: Outer CVPlan (Panel 1 or Panel 2) to walk forward over
            for FINAL reported performance - never used for tuning.
        cell: Identifies this grid cell (configuration/panel/fs_option/model/seed).
        config: Configuration (models, optimization, splits, evaluation).
        run_id: Run identifier, stored in every provenance row.
        outer_split_config: The SplitConfig that produced `eval_cv_plan`
            (config.splits for Panel 1, the panel2-sized copy for Panel 2).
            Required when config.optimization.nested_retuning=True, so the
            per-outer-fold NestedFold list is built from the exact same
            outer folds `eval_cv_plan` uses. Defaults to config.splits.

    Returns:
        Dict with 'predictions' (long-format list of per-observation dicts),
        'fold_provenance' (one row per outer fold, spec section 15 columns),
        and 'summary' (create_evaluation_summary output plus grid-cell tags).
    """
    logger = get_logger()
    outer_split_config = outer_split_config or config.splits
    nested_retuning = getattr(config.optimization, 'nested_retuning', False)
    tag = f"[{cell.configuration}/{cell.panel}/{cell.fs_option}/{cell.model}]"

    # Validate the target transform BEFORE any tuning/fold work, not inside
    # the per-fold try/except below - an unsupported transform must fail
    # fast and loud, not be silently swallowed into "every fold failed,
    # zero predictions" (a warning easy to miss amid normal per-fold noise).
    _to_original_scale(np.array([1.0]), config.data.target_transform)

    predictions: List[Dict[str, Any]] = []
    fold_provenance: List[Dict[str, Any]] = []
    train_history_by_fold: Dict[Any, np.ndarray] = {}

    shared_best_params: Dict[str, Any] = {}
    nested_folds_by_id: Dict[Any, Any] = {}

    if nested_retuning:
        nested_folds = create_nested_walk_forward_splits(X, y, outer_split_config)
        nested_folds_by_id = {nf.outer_fold_id: nf for nf in nested_folds}
        logger.info(f"{tag} nested_retuning=True: {len(nested_folds)} outer folds with valid inner plans")
    else:
        tuning_cv_plan = build_tuning_cv_plan(X, y, eval_cv_plan, outer_split_config)
        opt_result = optimize_model(X, y, tuning_cv_plan, cell.model, config)
        shared_best_params = opt_result.get('best_params', {})
        logger.info(f"{tag} sweep tuning: best_params={shared_best_params}")

    for X_train, y_train, X_test, y_test, fold in generate_cv_folds(X, y, eval_cv_plan):
        t0 = time.time()

        if nested_retuning:
            nested_fold = nested_folds_by_id.get(fold.fold_id)
            if nested_fold is None:
                logger.warning(
                    f"{tag} no inner plan for outer fold {fold.fold_id} (h={fold.horizon}) - "
                    f"outer training window too short for nested retuning; skipping this fold."
                )
                continue
            fold_opt_result = optimize_model_nested(X, y, nested_fold, cell.model, config)
            params = fold_opt_result.get('best_params', {})
        else:
            params = shared_best_params

        try:
            model = ModelRegistry.create(cell.model, params)
            model.fit(X_train, y_train)

            if cell.model == 'lstm' and hasattr(model, 'build_predict_input'):
                predict_input = model.build_predict_input(X_train, X_test)
                y_pred = model.predict(predict_input, n_targets=len(X_test))
            else:
                y_pred = model.predict(X_test)

            # Alignment check first, on whatever scale the model natively
            # predicts in (length/index only - the check doesn't care about
            # units) - THEN invert to original-scale CO2e for every
            # downstream use (predictions, MASE seasonal-naive scale,
            # metrics). Models may still be trained on the transformed
            # target (y_train, above) - only reporting must be original-scale.
            assert_prediction_alignment(
                y_test, y_pred,
                context=f"{tag} fold={fold.fold_id}, h={fold.horizon}"
            )
            target_transform = config.data.target_transform
            y_train_original = _to_original_scale(y_train.values, target_transform)
            y_test_original = _to_original_scale(y_test.values, target_transform)
            y_pred_original = _to_original_scale(np.asarray(y_pred), target_transform)
            train_history_by_fold[fold.fold_id] = y_train_original
        except Exception as e:
            logger.warning(f"{tag} fold {fold.fold_id} (h={fold.horizon}) failed: {e}")
            continue

        runtime = time.time() - t0

        for date, actual, predicted in zip(y_test.index, y_test_original, y_pred_original):
            predictions.append({
                'run_id': run_id,
                'configuration': cell.configuration,
                'panel': cell.panel,
                'fs_option': cell.fs_option,
                'model': cell.model,
                'horizon': fold.horizon,
                'fold_id': fold.fold_id,
                'target_date': date,
                'actual': float(actual),
                'predicted': float(predicted),
                'residual': float(predicted) - float(actual),
            })

        fold_provenance.append({
            'run_id': run_id,
            'configuration': cell.configuration,
            'panel': cell.panel,
            'fs_option': cell.fs_option,
            'model': cell.model,
            'horizon': fold.horizon,
            'fold_id': fold.fold_id,
            'train_start': fold.train_start,
            'train_end': fold.train_end,
            'forecast_origin': fold.train_end,
            'test_start': fold.test_start,
            'test_end': fold.test_end,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'selected_features': list(X.columns),
            'n_features': len(X.columns),
            'hyperparameters': params,
            'seed': cell.seed,
            'runtime_seconds': runtime,
        })

    predictions_df = pd.DataFrame(predictions)

    if predictions_df.empty:
        logger.warning(f"{tag} produced zero predictions - every outer fold failed or was skipped.")
        summary = {
            'model': cell.model, 'configuration': cell.configuration, 'panel': cell.panel,
            'fs_option': cell.fs_option, 'n_features': len(X.columns),
            'weighted_mae': float('nan'), 'weighted_mase': float('nan'),
            'worst_horizon_mase': float('nan'), 'stability_score': 0.0, 'n_folds': 0,
        }
        return {'predictions': [], 'fold_provenance': fold_provenance, 'summary': summary}

    horizon_metrics = evaluate_by_horizon(
        predictions_df,
        outer_split_config.horizons,
        train_history_by_fold=train_history_by_fold,
        seasonal_period=config.evaluation.seasonal_period,
    )
    stability_metrics = evaluate_stability(predictions_df)
    summary = create_evaluation_summary(
        cell.model, horizon_metrics, stability_metrics, outer_split_config.horizon_weights
    )
    summary.update({
        'configuration': cell.configuration,
        'panel': cell.panel,
        'fs_option': cell.fs_option,
        'n_features': len(X.columns),
        'n_folds': len(fold_provenance),
        'total_runtime_seconds': sum(fp['runtime_seconds'] for fp in fold_provenance),
    })

    return {
        'predictions': predictions_df.to_dict(orient='records'),
        'fold_provenance': fold_provenance,
        'summary': summary,
    }


def run_naive_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    eval_cv_plan: CVPlan,
    model_name: str,
    config: Config,
    run_id: str,
    outer_split_config: Optional[SplitConfig] = None,
    panel: str = 'common_period',
) -> Dict[str, Any]:
    """
    Run one naive baseline (spec section 18: 'naive_lag1'/'naive_lag4') over
    the same outer folds as every Stream A/B cell. Unlike
    `run_configuration_model`, there is no inner-loop hyperparameter tuning
    (naive forecasters have no hyperparameters) - this is a lighter walk
    forward-only path, sharing the same predictions/fold_provenance/summary
    schema so naive rows drop into the same metrics tables as every tuned
    model. `X` is only used for its DatetimeIndex (naive forecasters ignore
    feature columns entirely) - pass any configuration's matrix restricted
    to the primary common period.

    Returns the same dict shape as `run_configuration_model`, with
    `configuration='naive_reference'` and `fs_option='none'` so naive rows
    are identifiable and excluded from the Stream A (20) / Stream B (100)
    candidate counts while still appearing in the joint comparison tables.
    """
    logger = get_logger()
    outer_split_config = outer_split_config or config.splits
    tag = f"[naive_reference/{model_name}]"

    predictions: List[Dict[str, Any]] = []
    fold_provenance: List[Dict[str, Any]] = []
    train_history_by_fold: Dict[Any, np.ndarray] = {}

    for X_train, y_train, X_test, y_test, fold in generate_cv_folds(X, y, eval_cv_plan):
        t0 = time.time()
        try:
            model = ModelRegistry.create(model_name, {})
            model.fit(X_train, y_train)
            # Naive forecasters are feature-free: NaiveLag4Forecaster needs
            # the TARGET date (origin + horizon) to resolve its t-4 seasonal
            # lookup, not the forecast ORIGIN date that X_test is indexed by
            # under this framework's direct-horizon construction (X_test's
            # rows are origins; y_test's rows are the corresponding targets).
            # Passing X_test here would silently degenerate naive_lag4 to
            # naive_lag1 for every horizon except h=4.
            y_pred = model.predict(y_test.to_frame())

            assert_prediction_alignment(
                y_test, y_pred, context=f"{tag} fold={fold.fold_id}, h={fold.horizon}"
            )
            target_transform = config.data.target_transform
            y_train_original = _to_original_scale(y_train.values, target_transform)
            y_test_original = _to_original_scale(y_test.values, target_transform)
            y_pred_original = _to_original_scale(np.asarray(y_pred), target_transform)
            train_history_by_fold[fold.fold_id] = y_train_original
        except Exception as e:
            logger.warning(f"{tag} fold {fold.fold_id} (h={fold.horizon}) failed: {e}")
            continue

        runtime = time.time() - t0

        for date, actual, predicted in zip(y_test.index, y_test_original, y_pred_original):
            predictions.append({
                'run_id': run_id, 'configuration': 'naive_reference', 'panel': panel,
                'fs_option': 'none', 'model': model_name, 'horizon': fold.horizon,
                'fold_id': fold.fold_id, 'target_date': date,
                'actual': float(actual), 'predicted': float(predicted),
                'residual': float(predicted) - float(actual),
            })

        fold_provenance.append({
            'run_id': run_id, 'configuration': 'naive_reference', 'panel': panel,
            'fs_option': 'none', 'model': model_name, 'horizon': fold.horizon,
            'fold_id': fold.fold_id, 'train_start': fold.train_start, 'train_end': fold.train_end,
            'forecast_origin': fold.train_end, 'test_start': fold.test_start,
            'test_end': fold.test_end, 'n_train': len(X_train), 'n_test': len(X_test),
            'selected_features': [], 'n_features': 0, 'hyperparameters': {},
            'seed': config.seed, 'runtime_seconds': runtime,
        })

    predictions_df = pd.DataFrame(predictions)
    if predictions_df.empty:
        logger.warning(f"{tag} produced zero predictions.")
        summary = {
            'model': model_name, 'configuration': 'naive_reference', 'panel': panel,
            'fs_option': 'none', 'n_features': 0, 'weighted_mae': float('nan'),
            'weighted_mase': float('nan'), 'worst_horizon_mase': float('nan'),
            'stability_score': 0.0, 'n_folds': 0,
        }
        return {'predictions': [], 'fold_provenance': fold_provenance, 'summary': summary}

    horizon_metrics = evaluate_by_horizon(
        predictions_df, outer_split_config.horizons,
        train_history_by_fold=train_history_by_fold,
        seasonal_period=config.evaluation.seasonal_period,
    )
    stability_metrics = evaluate_stability(predictions_df)
    summary = create_evaluation_summary(
        model_name, horizon_metrics, stability_metrics, outer_split_config.horizon_weights
    )
    summary.update({
        'configuration': 'naive_reference', 'panel': panel, 'fs_option': 'none', 'n_features': 0,
        'n_folds': len(fold_provenance),
        'total_runtime_seconds': sum(fp['runtime_seconds'] for fp in fold_provenance),
    })

    return {
        'predictions': predictions_df.to_dict(orient='records'),
        'fold_provenance': fold_provenance,
        'summary': summary,
    }
