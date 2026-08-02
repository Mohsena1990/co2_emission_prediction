"""
Model hyperparameter optimization using swarm optimization.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from ..core.logging_utils import get_logger
from ..core.config import Config
from ..core.utils import calculate_weighted_mae, save_json_numpy
from ..splits.walk_forward import CVPlan, generate_cv_folds
from ..splits.nested_walk_forward import NestedFold
from ..evaluation.metrics import calculate_mase_scale, calculate_mae as eval_calculate_mae
from ..models.base import ModelRegistry
from ..models.traditional import get_model_param_space
from ..models.lstm import get_lstm_param_space
from .pso import PSO, GWO, get_optimizer


def create_objective_function(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    model_name: str,
    param_names: List[str],
    param_types: List[str],
    bounds: List[Tuple[float, float]],
    config: Config,
    param_space: Dict[str, Any] = None
) -> callable:
    """
    LEGACY (pre-nested-CV): builds a PSO objective evaluated over whatever
    `cv_plan` is passed in.

    BUG 3.5 CONTEXT: if `cv_plan` is the same flat plan later used for final
    model evaluation (as in the original scripts/03 + scripts/04 pipeline),
    hyperparameters get selected by validating on the same folds used to
    report "final" out-of-sample performance - the outer test data is not
    actually held out from tuning. Kept for backward compatibility with the
    original single-level walk-forward scripts; new code should use
    `optimize_model_nested`, which is constructed so `cv_plan` here is
    always an INNER plan built strictly from one outer fold's training data
    (see splits.nested_walk_forward.NestedFold) and can therefore never see
    the outer test fold.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        model_name: Name of the model
        param_names: List of parameter names
        param_types: List of parameter types
        bounds: Parameter bounds
        config: Configuration

    Returns:
        Objective function that takes position array and returns fitness
    """
    logger = get_logger()
    horizon_weights = config.splits.horizon_weights

    def objective(position: np.ndarray) -> float:
        # Convert position to parameters
        params = {}
        for i, (name, ptype, value) in enumerate(zip(param_names, param_types, position)):
            if ptype == 'integer':
                # Check if this is actually a categorical parameter
                if param_space and name in param_space and param_space[name].get('type') == 'categorical':
                    # Convert index to actual choice value
                    choices = param_space[name]['choices']
                    idx = int(round(value))
                    idx = max(0, min(idx, len(choices) - 1))  # Clamp to valid range
                    params[name] = choices[idx]
                else:
                    params[name] = int(round(value))
            elif ptype == 'log':
                params[name] = float(value)
            elif ptype == 'categorical':
                # Handle categorical by index - convert to actual choice value
                if param_space and name in param_space:
                    choices = param_space[name]['choices']
                    idx = int(round(value))
                    idx = max(0, min(idx, len(choices) - 1))  # Clamp to valid range
                    params[name] = choices[idx]
                else:
                    params[name] = int(round(value))
            else:
                params[name] = float(value)

        # Evaluate with walk-forward CV. Primary metric is configurable
        # (spec section 13); MASE requires the training-fold history to
        # compute its seasonal-naive scale, so it is always available here
        # since X_train/y_train are exactly the in-sample data for this fold.
        primary_metric = getattr(config.optimization, 'metric', 'mae')
        horizon_errors = {h: [] for h in config.splits.horizons}

        try:
            for X_train, y_train, X_test, y_test, fold in generate_cv_folds(X, y, cv_plan):
                # Create and train model
                model = ModelRegistry.create(model_name, params)
                model.fit(X_train, y_train)

                # Predict. LSTM requires lookback history immediately before
                # the test rows (bug 3.4) - it cannot predict from X_test
                # alone when X_test is shorter than lookback (see the
                # identical handling in scripts/04_evaluate_and_safeguards.py).
                if model_name == 'lstm' and hasattr(model, 'build_predict_input'):
                    predict_input = model.build_predict_input(X_train, X_test)
                    y_pred = model.predict(predict_input, n_targets=len(X_test))
                else:
                    y_pred = model.predict(X_test)

                if primary_metric == 'mase':
                    try:
                        scale = calculate_mase_scale(
                            y_train.values, seasonal_period=config.evaluation.seasonal_period
                        )
                        error = eval_calculate_mae(y_test.values, y_pred) / scale
                    except (ValueError, ZeroDivisionError):
                        error = np.mean(np.abs(y_test.values - y_pred))
                else:
                    error = np.mean(np.abs(y_test.values - y_pred))

                horizon_errors[fold.horizon].append(error)

        except Exception as e:
            logger.warning(f"Model training failed with params {params}: {e}")
            return float('inf')

        # Calculate weighted error across horizons
        mean_errors = {}
        for h in config.splits.horizons:
            if horizon_errors[h]:
                mean_errors[h] = np.mean(horizon_errors[h])
            else:
                mean_errors[h] = float('inf')

        weighted_error = calculate_weighted_mae(mean_errors, horizon_weights)

        # Optional: add annual consistency penalty
        # (simplified - full implementation in evaluation module)
        if hasattr(config.optimization, 'annual_penalty_weight'):
            penalty_weight = config.optimization.annual_penalty_weight
            if penalty_weight > 0:
                # Add small penalty for unstable models
                stability_penalty = np.std(list(mean_errors.values()))
                weighted_error += penalty_weight * stability_penalty

        return weighted_error

    return objective


def optimize_model(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    model_name: str,
    config: Config,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    LEGACY (pre-nested-CV) - see create_objective_function docstring for why
    this must not be used with a cv_plan that overlaps the folds used for
    final evaluation. Prefer `optimize_model_nested`.

    Optimize hyperparameters for a single model.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        model_name: Name of the model
        config: Configuration
        output_dir: Directory to save results

    Returns:
        Dictionary with optimization results
    """
    logger = get_logger()
    logger.info(f"Optimizing {model_name}...")

    # Get parameter space
    if model_name == 'lstm':
        param_space = get_lstm_param_space()
    else:
        param_space = get_model_param_space(model_name)

    if not param_space:
        logger.warning(f"No parameter space defined for {model_name}, using defaults")
        return {
            'model': model_name,
            'best_params': {},
            'best_fitness': None,
            'history': None
        }

    # Build bounds, names, and types
    param_names = []
    param_types = []
    bounds = []

    for name, spec in param_space.items():
        param_names.append(name)

        if spec['type'] == 'int':
            param_types.append('integer')
            bounds.append((spec['low'], spec['high']))
        elif spec['type'] == 'log_uniform':
            param_types.append('continuous')
            bounds.append((spec['low'], spec['high']))
        elif spec['type'] == 'uniform':
            param_types.append('continuous')
            bounds.append((spec['low'], spec['high']))
        elif spec['type'] == 'categorical':
            param_types.append('integer')
            bounds.append((0, len(spec['choices']) - 1))
        else:
            param_types.append('continuous')
            bounds.append((spec.get('low', 0), spec.get('high', 1)))

    # Create objective function
    objective = create_objective_function(
        X, y, cv_plan, model_name,
        param_names, param_types, bounds, config,
        param_space=param_space
    )

    # Get optimizer
    optimizer = get_optimizer(
        config.optimization.optimizer,
        n_particles=config.optimization.n_particles,
        n_iterations=config.optimization.n_iterations,
        seed=config.seed
    )

    # Run optimization
    best_position, best_fitness, history = optimizer.optimize(
        objective, bounds, param_types
    )

    # Convert position to parameters
    best_params = {}
    for i, (name, ptype, value) in enumerate(zip(param_names, param_types, best_position)):
        spec = param_space[name]
        if ptype == 'integer':
            if spec['type'] == 'categorical':
                best_params[name] = spec['choices'][int(round(value))]
            else:
                best_params[name] = int(round(value))
        else:
            best_params[name] = float(value)

    results = {
        'model': model_name,
        'best_params': best_params,
        'best_fitness': float(best_fitness),
        'history': {
            'iterations': history['iterations'],
            'best_fitness': [float(f) for f in history['best_fitness']],
            'mean_fitness': [float(f) for f in history['mean_fitness']]
        },
        'param_space': param_space,
        'optimizer': config.optimization.optimizer
    }

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json_numpy(results, output_dir / f"{model_name}_optimization.json")

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best fitness: {best_fitness:.6f}")

    return results


def optimize_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    config: Config,
    output_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Optimize hyperparameters for all models.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        config: Configuration
        output_dir: Directory to save results

    Returns:
        Dictionary with optimization results for all models
    """
    logger = get_logger()
    logger.info("Optimizing all models...")

    results = {}

    for model_name in config.model.models:
        try:
            model_results = optimize_model(
                X, y, cv_plan, model_name, config, output_dir
            )
            results[model_name] = model_results
        except Exception as e:
            logger.error(f"Optimization failed for {model_name}: {e}")
            results[model_name] = {
                'model': model_name,
                'error': str(e)
            }

    # Save summary
    if output_dir:
        summary = []
        for name, res in results.items():
            summary.append({
                'model': name,
                'best_fitness': res.get('best_fitness'),
                'best_params': res.get('best_params', {})
            })

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_dir / 'optimization_summary.csv', index=False)

    return results


def train_optimized_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    best_params: Dict[str, Any]
) -> 'BaseForecaster':
    """
    Train a model with optimized parameters.

    Args:
        X_train: Training features
        y_train: Training target
        model_name: Model name
        best_params: Optimized parameters

    Returns:
        Trained model
    """
    model = ModelRegistry.create(model_name, best_params)
    model.fit(X_train, y_train)
    return model


def optimize_model_nested(
    X: pd.DataFrame,
    y: pd.Series,
    nested_fold: NestedFold,
    model_name: str,
    config: Config,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a single model, for a single OUTER fold,
    using ONLY that outer fold's inner CV plan (bug 3.5 / spec section 11).

    This is the nested-CV-correct replacement for `optimize_model`: PSO's
    objective is evaluated exclusively on `nested_fold.inner_cv_plan`, which
    is built strictly from `X.iloc[nested_fold.outer.train_indices]` and can
    structurally never include a row from the outer test fold. The caller is
    responsible for refitting the winning hyperparameters on the *complete*
    outer training data and forecasting the untouched outer test target
    (this function does not do that refit itself, since the outer
    train/test split belongs to the caller's cross-validation loop, not to
    the optimizer).

    Args:
        X: Full feature DataFrame (only the outer-training slice is ever
            touched via nested_fold.outer.train_indices).
        y: Full target Series.
        nested_fold: A NestedFold from create_nested_walk_forward_splits.
        model_name: Model to tune.
        config: Configuration.
        output_dir: Optional directory to save this outer fold's
            optimization result.

    Returns:
        Same shape as `optimize_model`'s return dict, plus 'outer_fold_id'.
    """
    logger = get_logger()
    X_outer_train = X.iloc[nested_fold.outer.train_indices]
    y_outer_train = y.iloc[nested_fold.outer.train_indices]

    logger.info(
        f"Optimizing {model_name} for outer fold {nested_fold.outer_fold_id} "
        f"(h={nested_fold.outer.horizon}) using "
        f"{nested_fold.inner_cv_plan.n_folds} inner folds only"
    )

    result = optimize_model(
        X_outer_train, y_outer_train, nested_fold.inner_cv_plan,
        model_name, config, output_dir=None
    )
    result['outer_fold_id'] = nested_fold.outer_fold_id
    result['outer_horizon'] = nested_fold.outer.horizon
    result['n_inner_folds'] = nested_fold.inner_cv_plan.n_folds

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json_numpy(
            result,
            output_dir / f"{model_name}_outer{nested_fold.outer_fold_id}_optimization.json"
        )

    return result
