"""
Evaluation module for CO2 forecasting framework.
"""
from .metrics import (
    ForecastMetrics,
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_r2,
    calculate_bias,
    compute_all_metrics,
    evaluate_by_horizon,
    evaluate_stability,
    create_evaluation_summary,
    compare_models,
    calculate_weighted_mase
)
from .statistical_tests import (
    paired_bootstrap_ci,
    diebold_mariano_test,
    wilcoxon_signed_rank_test,
    bonferroni_correction,
    benjamini_hochberg_correction
)
from .incremental_value import (
    paired_fold_errors,
    compute_incremental_value_table,
    INCREMENTAL_COMPARISONS
)

__all__ = [
    'ForecastMetrics',
    'calculate_mae', 'calculate_rmse', 'calculate_mape',
    'calculate_r2', 'calculate_bias',
    'compute_all_metrics', 'evaluate_by_horizon', 'evaluate_stability',
    'create_evaluation_summary', 'compare_models', 'calculate_weighted_mase',
    'paired_bootstrap_ci', 'diebold_mariano_test', 'wilcoxon_signed_rank_test',
    'bonferroni_correction', 'benjamini_hochberg_correction',
    'paired_fold_errors', 'compute_incremental_value_table', 'INCREMENTAL_COMPARISONS',
]
