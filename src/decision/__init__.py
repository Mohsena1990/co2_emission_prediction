"""
Decision making module for CO2 forecasting framework.
MCDA methods: VIKOR and TOPSIS with sensitivity analysis.
"""
from .mcda import (
    normalize_matrix,
    pareto_filter,
    topsis,
    vikor,
    select_best_fs_option,
    select_best_model,
    assign_deterministic_rank
)
from .sensitivity import (
    vikor_v_sensitivity,
    weight_sensitivity,
    criterion_removal_sensitivity,
    compute_rank_stability_score
)
from .experiment_ranking import (
    build_pareto_mcda_table,
    rank_correlation_across_weight_sets,
    select_best_overall,
    CRITERIA as EXPERIMENT_RANKING_CRITERIA,
    WEIGHT_SETS as EXPERIMENT_RANKING_WEIGHT_SETS,
)

__all__ = [
    'normalize_matrix',
    'pareto_filter',
    'topsis',
    'vikor',
    'select_best_fs_option',
    'select_best_model',
    'assign_deterministic_rank',
    # Sensitivity analysis
    'vikor_v_sensitivity',
    'weight_sensitivity',
    'criterion_removal_sensitivity',
    'compute_rank_stability_score',
    # Experimental-grid Pareto/MCDA ranking (Table 10)
    'build_pareto_mcda_table',
    'rank_correlation_across_weight_sets',
    'select_best_overall',
    'EXPERIMENT_RANKING_CRITERIA',
    'EXPERIMENT_RANKING_WEIGHT_SETS',
]
