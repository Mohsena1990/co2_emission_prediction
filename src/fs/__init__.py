"""
Feature selection module for CO2 forecasting framework.

Provides comprehensive feature selection methods:
- Filter methods: Mutual Information, F-test, Correlation, Variance
- Wrapper methods: RFE, Sequential Forward/Backward Selection
- Embedded methods: LASSO, ElasticNet, Gradient Boosting, Random Forest
"""
from .vif import calculate_vif, filter_by_vif
from .linear import (
    ridge_stability_selection,
    elasticnet_stability_selection,
    fs_linear
)
from .nonlinear import (
    random_forest_importance,
    lightgbm_importance,
    catboost_importance,
    fs_nonlinear,
    xgboost_shap_stability_selection,
    fs_xgboost_shap
)
from .consensus import (
    vote_based_selection,
    stability_based_selection,
    fs_consensus,
    fs_hybrid,
    run_all_fs_options
)
from .evaluation import (
    evaluate_fs_option_with_shap,
    create_fs_evaluation_matrix
)

# New comprehensive feature selection methods
from .filter_methods import (
    mutual_information_selection,
    f_test_selection,
    correlation_based_selection,
    variance_threshold_selection,
    run_all_filter_methods
)
from .wrapper_methods import (
    rfe_selection,
    sequential_forward_selection,
    sequential_backward_selection,
    run_all_wrapper_methods,
    fs_wrapper
)
from .embedded_methods import (
    lasso_selection,
    elasticnet_selection,
    gradient_boosting_selection,
    random_forest_selection,
    catboost_selection,
    run_all_embedded_methods,
    permutation_stability_selection,
    fs_permutation_stability
)

__all__ = [
    # VIF
    'calculate_vif', 'filter_by_vif',
    # Linear methods
    'ridge_stability_selection', 'elasticnet_stability_selection', 'fs_linear',
    # Nonlinear methods
    'random_forest_importance', 'lightgbm_importance', 'catboost_importance',
    'fs_nonlinear',
    # FS3: XGBoost-SHAP stability selection
    'xgboost_shap_stability_selection', 'fs_xgboost_shap',
    # Consensus methods
    'vote_based_selection', 'stability_based_selection', 'fs_consensus',
    'fs_hybrid', 'run_all_fs_options',
    # Evaluation
    'evaluate_fs_option_with_shap', 'create_fs_evaluation_matrix',
    # Filter methods (NEW)
    'mutual_information_selection', 'f_test_selection',
    'correlation_based_selection', 'variance_threshold_selection',
    'run_all_filter_methods',
    # Wrapper methods (NEW)
    'rfe_selection', 'sequential_forward_selection', 'sequential_backward_selection',
    'run_all_wrapper_methods', 'fs_wrapper',
    # Embedded methods (NEW)
    'lasso_selection', 'elasticnet_selection', 'gradient_boosting_selection',
    'random_forest_selection', 'catboost_selection', 'run_all_embedded_methods',
    # FS4: model-agnostic permutation stability selection
    'permutation_stability_selection', 'fs_permutation_stability'
]
