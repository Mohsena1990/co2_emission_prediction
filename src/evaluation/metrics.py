"""
Evaluation metrics for CO2 forecasting.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

from ..core.logging_utils import get_logger
from ..core.utils import calculate_weighted_mae, inverse_log_transform, aggregate_to_annual


@dataclass
class ForecastMetrics:
    """Container for forecast evaluation metrics."""
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    smape: float = float('nan')
    mase: float = float('nan')
    mse: float = 0.0
    r2: float = 0.0
    bias: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'smape': self.smape,
            'mase': self.mase,
            'mse': self.mse,
            'r2': self.r2,
            'bias': self.bias,
            'n_samples': self.n_samples
        }


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def calculate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Bias (systematic error)."""
    return np.mean(y_pred - y_true)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Symmetric Mean Absolute Percentage Error (%), bounded in [0, 200].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < epsilon, epsilon, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100)


def calculate_mase_scale(
    y_train_history: np.ndarray,
    seasonal_period: int = 4
) -> float:
    """
    Compute the MASE scaling denominator: the in-sample mean absolute error
    of a seasonal-naive forecast (y_hat[t] = y[t - seasonal_period]).

    This MUST be computed on the training/in-sample history only (never on
    the test fold) - it is the scale against which out-of-sample MAE is
    compared in `calculate_mase`.

    Args:
        y_train_history: In-sample target series in original (untransformed)
            units, in temporal order.
        seasonal_period: Seasonal lag (4 for quarterly data).

    Returns:
        Scale (mean absolute seasonal-naive error). Raises ValueError if the
        history is too short to compute a seasonal-naive baseline.
    """
    y_train_history = np.asarray(y_train_history, dtype=float)
    if len(y_train_history) <= seasonal_period:
        raise ValueError(
            f"MASE scale requires at least {seasonal_period + 1} in-sample "
            f"observations (seasonal_period={seasonal_period}), got "
            f"{len(y_train_history)}."
        )
    naive_errors = np.abs(y_train_history[seasonal_period:] - y_train_history[:-seasonal_period])
    scale = float(np.mean(naive_errors))
    return scale


def calculate_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: np.ndarray,
    seasonal_period: int = 4
) -> float:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler 2006), the primary
    comparative metric for this framework.

    MASE = mean(|y_true - y_pred|) / mean(|y_train[t] - y_train[t-m]|)

    where the denominator (seasonal-naive in-sample MAE) is computed strictly
    from `y_train_history` (the outer-fold training data available at the
    forecast origin) - never from the test fold being scored.

    Args:
        y_true: Test-fold actual values (original scale).
        y_pred: Test-fold predicted values (original scale).
        y_train_history: In-sample (training) series used only to compute the
            naive-forecast scale.
        seasonal_period: Seasonal lag for the naive baseline (4 = quarterly).

    Returns:
        MASE value. A value < 1 means the model beats the seasonal-naive
        in-sample benchmark on average; >= 1 means it does not.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    scale = calculate_mase_scale(y_train_history, seasonal_period=seasonal_period)
    if scale == 0 or not np.isfinite(scale):
        raise ValueError(
            "MASE scale is zero or non-finite (degenerate/constant training "
            "history) - cannot compute a meaningful MASE."
        )

    mae = calculate_mae(y_true, y_pred)
    return float(mae / scale)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: Optional[np.ndarray] = None,
    seasonal_period: int = 4
) -> ForecastMetrics:
    """
    Compute all forecast metrics on original-scale values.

    Args:
        y_true: True values
        y_pred: Predicted values
        y_train_history: In-sample training series (original scale, temporal
            order) used to compute the MASE seasonal-naive scale. If omitted,
            `mase` is returned as NaN (explicitly, not silently dropped) - the
            primary pipeline must always supply this.
        seasonal_period: Seasonal lag for MASE (4 = quarterly).

    Returns:
        ForecastMetrics object
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have same length")

    if len(y_true) == 0:
        return ForecastMetrics()

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mase = float('nan')
    if y_train_history is not None:
        logger = get_logger()
        try:
            mase = calculate_mase(y_true, y_pred, y_train_history, seasonal_period=seasonal_period)
        except ValueError as e:
            logger.warning(f"MASE could not be computed: {e}")
            mase = float('nan')

    return ForecastMetrics(
        mae=calculate_mae(y_true, y_pred),
        rmse=calculate_rmse(y_true, y_pred),
        mape=calculate_mape(y_true, y_pred),
        smape=calculate_smape(y_true, y_pred),
        mase=mase,
        mse=np.mean((y_true - y_pred) ** 2),
        r2=calculate_r2(y_true, y_pred),
        bias=calculate_bias(y_true, y_pred),
        n_samples=len(y_true)
    )


def calculate_weighted_mase(
    mase_by_horizon: Dict[int, float],
    horizon_weights: Dict[int, float]
) -> float:
    """
    Weighted MASE across horizons: WMASE = sum(w_h * MASE_h).
    Mirrors `core.utils.calculate_weighted_mae` but for MASE, which is the
    primary criterion used throughout evaluation, Pareto filtering and MCDA.

    Returns NaN (rather than a partial/renormalized sum) if any horizon with
    non-zero weight is missing a valid MASE - a partial weighted average is
    not the quantity defined in the spec and must not be reported as if it
    were.
    """
    total = 0.0
    for h, w in horizon_weights.items():
        if w == 0:
            continue
        mase = mase_by_horizon.get(h)
        if mase is None or (isinstance(mase, float) and np.isnan(mase)):
            return float('nan')
        total += w * mase
    return total


def evaluate_by_horizon(
    predictions_df: pd.DataFrame,
    horizons: List[int],
    train_history_by_fold: Optional[Dict[Any, np.ndarray]] = None,
    seasonal_period: int = 4
) -> Dict[int, ForecastMetrics]:
    """
    Evaluate metrics by forecast horizon.

    Args:
        predictions_df: DataFrame with columns ['horizon', 'actual', 'predicted'],
            and 'fold_id' if `train_history_by_fold` is supplied.
        horizons: List of horizons to evaluate
        train_history_by_fold: Optional dict mapping fold_id -> the outer-fold
            training series (original scale, temporal order) for that fold.
            When supplied, MASE is computed per horizon by pooling the
            seasonal-naive errors from every fold contributing to that
            horizon's predictions (each fold's naive scale is computed from
            its own training-only history, never from test data).
        seasonal_period: Seasonal lag for MASE (4 = quarterly).

    Returns:
        Dictionary mapping horizon to metrics
    """
    results = {}

    for h in horizons:
        h_data = predictions_df[predictions_df['horizon'] == h]

        if len(h_data) == 0:
            results[h] = ForecastMetrics()
            continue

        y_true = h_data['actual'].values
        y_pred = h_data['predicted'].values

        metrics = compute_all_metrics(y_true, y_pred)

        # MASE is pooled across every fold contributing to this horizon: each
        # fold's seasonal-naive scale is computed from that fold's own
        # training-only history, then all folds' naive errors are pooled
        # before averaging (never mixing in test-fold data).
        if train_history_by_fold is not None and 'fold_id' in h_data.columns:
            pooled_naive_errors = []
            for fold_id in h_data['fold_id'].unique():
                history = train_history_by_fold.get(fold_id)
                if history is None:
                    continue
                history = np.asarray(history, dtype=float)
                if len(history) <= seasonal_period:
                    continue
                pooled_naive_errors.append(
                    np.abs(history[seasonal_period:] - history[:-seasonal_period])
                )
            if pooled_naive_errors:
                scale = float(np.mean(np.concatenate(pooled_naive_errors)))
                if scale > 0 and np.isfinite(scale):
                    metrics.mase = float(calculate_mae(y_true, y_pred) / scale)

        results[h] = metrics

    return results


def evaluate_stability(
    predictions_df: pd.DataFrame,
    group_col: str = 'fold_id'
) -> Dict[str, float]:
    """
    Evaluate prediction stability across folds.

    Args:
        predictions_df: DataFrame with predictions
        group_col: Column to group by (e.g., 'fold_id')

    Returns:
        Stability metrics
    """
    if group_col not in predictions_df.columns:
        return {}

    # Calculate error per fold
    fold_errors = []
    for fold_id, group in predictions_df.groupby(group_col):
        y_true = group['actual'].values
        y_pred = group['predicted'].values
        mae = calculate_mae(y_true, y_pred)
        fold_errors.append(mae)

    if not fold_errors:
        return {}

    return {
        'error_mean': np.mean(fold_errors),
        'error_std': np.std(fold_errors),
        'error_min': np.min(fold_errors),
        'error_max': np.max(fold_errors),
        'error_range': np.max(fold_errors) - np.min(fold_errors),
        'stability_score': 1 / (1 + np.std(fold_errors)),  # Higher is better
        'worst_fold_error': np.max(fold_errors),
        'n_folds': len(fold_errors)
    }


def create_evaluation_summary(
    model_name: str,
    horizon_metrics: Dict[int, ForecastMetrics],
    stability_metrics: Dict[str, float],
    horizon_weights: Dict[int, float]
) -> Dict[str, Any]:
    """
    Create a comprehensive evaluation summary.

    Args:
        model_name: Name of the model
        horizon_metrics: Metrics by horizon
        stability_metrics: Stability metrics
        horizon_weights: Weights for each horizon

    Returns:
        Evaluation summary dictionary
    """
    # Calculate weighted MAE and weighted MASE (MASE is the primary criterion)
    mae_by_horizon = {h: m.mae for h, m in horizon_metrics.items()}
    weighted_mae = calculate_weighted_mae(mae_by_horizon, horizon_weights)

    mase_by_horizon = {h: m.mase for h, m in horizon_metrics.items()}
    weighted_mase = calculate_weighted_mase(mase_by_horizon, horizon_weights)

    finite_mases = [m.mase for m in horizon_metrics.values() if np.isfinite(m.mase)]
    worst_horizon_mase = max(finite_mases) if finite_mases else float('nan')

    summary = {
        'model': model_name,
        'weighted_mae': weighted_mae,
        'weighted_mase': weighted_mase,
        'worst_horizon_mase': worst_horizon_mase,
        'stability_score': stability_metrics.get('stability_score', 0),
        'error_std': stability_metrics.get('error_std', 0),
        'worst_fold_error': stability_metrics.get('worst_fold_error', 0)
    }

    # Add per-horizon metrics
    for h, metrics in horizon_metrics.items():
        summary[f'mae_h{h}'] = metrics.mae
        summary[f'rmse_h{h}'] = metrics.rmse
        summary[f'mape_h{h}'] = metrics.mape
        summary[f'smape_h{h}'] = metrics.smape
        summary[f'mase_h{h}'] = metrics.mase
        summary[f'r2_h{h}'] = metrics.r2

    return summary


def compare_models(
    model_results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create comparison table of all models.

    Args:
        model_results: Dictionary of model evaluation results

    Returns:
        Comparison DataFrame
    """
    rows = []

    for model_name, results in model_results.items():
        row = {'model': model_name}
        row.update({k: v for k, v in results.items() if k != 'model'})
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by weighted MASE (primary criterion); fall back to weighted MAE
    if 'weighted_mase' in df.columns:
        df = df.sort_values('weighted_mase')
    elif 'weighted_mae' in df.columns:
        df = df.sort_values('weighted_mae')

    return df
