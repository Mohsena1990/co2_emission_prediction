"""
Interpretability module for CO2 forecasting framework.
SHAP analysis and feature importance.
"""
from .shap_analysis import (
    compute_shap_values,
    get_feature_importance_from_shap,
    analyze_regime_shap,
    analyze_seasonal_shap,
    compute_permutation_importance,
    manual_permutation_importance,
    analyze_regime_permutation_importance,
    get_ridge_coefficients,
    generate_interpretation_report
)
from .cross_model_importance import (
    rank_importance,
    build_cross_model_importance_table,
    compute_regime_stability,
    compute_family_contribution,
)

__all__ = [
    'compute_shap_values',
    'get_feature_importance_from_shap',
    'analyze_regime_shap',
    'analyze_seasonal_shap',
    'compute_permutation_importance',
    'manual_permutation_importance',
    'analyze_regime_permutation_importance',
    'get_ridge_coefficients',
    'generate_interpretation_report',
    'rank_importance',
    'build_cross_model_importance_table',
    'compute_regime_stability',
    'compute_family_contribution',
]
