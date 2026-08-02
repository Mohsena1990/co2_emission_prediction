"""
Cross-validation splits module for CO2 forecasting framework.
"""
from .walk_forward import (
    CVFold,
    CVPlan,
    create_walk_forward_splits,
    create_expanding_window_splits,
    generate_cv_folds,
    validate_no_leakage,
    save_cv_plan,
    load_cv_plan
)
from .nested_walk_forward import (
    NestedFold,
    create_nested_walk_forward_splits,
    validate_nested_isolation,
    build_tuning_cv_plan
)
from .panels import (
    build_common_period_cv_plan,
    panel2_split_config,
    slice_configurations_to_common_period,
    assert_cv_plan_fits_matrix,
)

__all__ = [
    'CVFold',
    'CVPlan',
    'create_walk_forward_splits',
    'create_expanding_window_splits',
    'generate_cv_folds',
    'validate_no_leakage',
    'save_cv_plan',
    'load_cv_plan',
    'NestedFold',
    'create_nested_walk_forward_splits',
    'validate_nested_isolation',
    'build_tuning_cv_plan',
    'build_common_period_cv_plan',
    'panel2_split_config',
    'slice_configurations_to_common_period',
    'assert_cv_plan_fits_matrix',
]
