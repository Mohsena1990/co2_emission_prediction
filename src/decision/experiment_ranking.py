"""
Pareto filtering + MCDA ranking over the experimental grid (spec section 18,
Table 10). Builds entirely on the existing generic pareto_filter/vikor/
topsis/weight_sensitivity machinery in mcda.py/sensitivity.py - this module
only supplies the specific criteria/weight-set choices for ranking
configuration x FS x model cells from Stage 1/2 metrics.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..core.logging_utils import get_logger
from .mcda import pareto_filter, vikor, topsis, assign_deterministic_rank

# Spec section 18 Pareto/MCDA criteria for ranking experimental-grid cells -
# all "minimise" (cost): weighted MASE, worst-horizon MASE, temporal error
# variability (error_std, spec's "fold-level error standard deviation"),
# feature count (parsimony), runtime.
CRITERIA = ['weighted_mase', 'worst_horizon_mase', 'error_std', 'n_features', 'total_runtime_seconds']
CRITERIA_TYPES = {c: 'cost' for c in CRITERIA}

# Spec section 18: "sensitivity analysis with equal weights; accuracy-emphasis
# weights; stability-emphasis weights; parsimony-emphasis weights;
# runtime-emphasis weights".
WEIGHT_SETS: Dict[str, Dict[str, float]] = {
    'equal': {c: 1 / len(CRITERIA) for c in CRITERIA},
    'accuracy_emphasis': {'weighted_mase': 0.50, 'worst_horizon_mase': 0.30, 'error_std': 0.10,
                           'n_features': 0.05, 'total_runtime_seconds': 0.05},
    'horizon_robustness_emphasis': {'weighted_mase': 0.20, 'worst_horizon_mase': 0.60, 'error_std': 0.10,
                                     'n_features': 0.05, 'total_runtime_seconds': 0.05},
    'stability_emphasis': {'weighted_mase': 0.20, 'worst_horizon_mase': 0.20, 'error_std': 0.50,
                            'n_features': 0.05, 'total_runtime_seconds': 0.05},
    'parsimony_emphasis': {'weighted_mase': 0.20, 'worst_horizon_mase': 0.20, 'error_std': 0.10,
                            'n_features': 0.45, 'total_runtime_seconds': 0.05},
    'runtime_emphasis': {'weighted_mase': 0.20, 'worst_horizon_mase': 0.20, 'error_std': 0.10,
                          'n_features': 0.05, 'total_runtime_seconds': 0.45},
}
PRIMARY_WEIGHT_SET = 'equal'


def _cell_label(row) -> str:
    return f"{row['configuration']}/{row.get('fs_option', 'all_features')}/{row['model']}"


def _prepare(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.dropna(subset=CRITERIA).copy()
    if df.empty:
        return df
    df['cell_label'] = df.apply(_cell_label, axis=1)
    return df.set_index('cell_label', drop=False)


def build_pareto_mcda_table(
    metrics_df: pd.DataFrame,
    weight_set_name: str = PRIMARY_WEIGHT_SET,
    vikor_v: float = 0.5,
) -> pd.DataFrame:
    """
    Table 10 (spec section 20): Pareto-filter the config x FS x model grid,
    then rank the non-dominated set with VIKOR. Per spec section 18: "If
    only one non-dominated configuration remains, select it directly" -
    handled without invoking VIKOR on a degenerate single-alternative set.
    """
    logger = get_logger()
    df = _prepare(metrics_df)
    if df.empty:
        logger.warning("build_pareto_mcda_table: no rows with complete criteria - returning empty table")
        return pd.DataFrame()

    pareto_df = pareto_filter(df, CRITERIA, CRITERIA_TYPES)
    df['pareto_status'] = np.where(df.index.isin(pareto_df.index), 'non_dominated', 'dominated')

    weights = WEIGHT_SETS[weight_set_name]

    if len(pareto_df) == 1:
        ranked = pareto_df.copy()
        ranked['vikor_S'] = np.nan
        ranked['vikor_R'] = np.nan
        ranked['vikor_Q'] = 0.0
        ranked['vikor_rank'] = 1
        ranked['final_decision'] = 'selected (only non-dominated alternative)'
        logger.info(f"Only one non-dominated cell: {ranked.index[0]} - selected directly")
    else:
        ranked = vikor(pareto_df, CRITERIA, weights, CRITERIA_TYPES, v=vikor_v, tie_break_col='n_features')
        ranked['final_decision'] = np.where(ranked['vikor_rank'] == 1, 'selected', '')

    result = df.merge(
        ranked[['vikor_S', 'vikor_R', 'vikor_Q', 'vikor_rank', 'final_decision']],
        left_index=True, right_index=True, how='left'
    )
    result['final_decision'] = result['final_decision'].fillna('')
    result = result.sort_values(['pareto_status', 'vikor_rank'], na_position='last')

    # spec Table 4/5/7 all require the per-horizon MASE breakdown alongside
    # the aggregate criteria - carried through here (not just CRITERIA)
    # even though the ranking/Pareto/VIKOR math itself never uses them.
    horizon_cols = [c for c in ('mase_h1', 'mase_h2', 'mase_h4') if c in result.columns]
    cols = ['configuration', 'panel', 'fs_option', 'model', 'pareto_status'] + CRITERIA + horizon_cols + [
        'vikor_S', 'vikor_R', 'vikor_Q', 'vikor_rank', 'final_decision'
    ]
    return result[[c for c in cols if c in result.columns]]


def rank_correlation_across_weight_sets(
    metrics_df: pd.DataFrame,
    weight_sets: Optional[Dict[str, Dict[str, float]]] = None,
    vikor_v: float = 0.5,
) -> pd.DataFrame:
    """
    Spec section 18: "Calculate rank correlations and identify whether the
    selected result changes" - pairwise Spearman's rho between every
    weight-sensitivity VIKOR scheme, plus TOPSIS and a simple weighted sum,
    over the Pareto-non-dominated set.
    """
    logger = get_logger()
    weight_sets = weight_sets or WEIGHT_SETS
    df = _prepare(metrics_df)
    if df.empty:
        return pd.DataFrame()

    pareto_df = pareto_filter(df, CRITERIA, CRITERIA_TYPES)
    if len(pareto_df) < 2:
        logger.warning(
            "rank_correlation_across_weight_sets: fewer than 2 non-dominated "
            "cells - no correlation to compute"
        )
        return pd.DataFrame()

    schemes: Dict[str, pd.Series] = {}
    for name, weights in weight_sets.items():
        ranked = vikor(pareto_df, CRITERIA, weights, CRITERIA_TYPES, v=vikor_v, tie_break_col='n_features')
        schemes[f'vikor_{name}'] = ranked['vikor_rank']

    equal_weights = weight_sets.get('equal', {c: 1 / len(CRITERIA) for c in CRITERIA})
    schemes['topsis_equal'] = topsis(
        pareto_df, CRITERIA, equal_weights, CRITERIA_TYPES, tie_break_col='n_features'
    )['topsis_rank']

    # Simple weighted sum over min-max-normalized cost criteria (lower is better).
    normalized = (pareto_df[CRITERIA] - pareto_df[CRITERIA].min()) / (
        pareto_df[CRITERIA].max() - pareto_df[CRITERIA].min() + 1e-12
    )
    weighted_sum = sum(normalized[c] * equal_weights[c] for c in CRITERIA)
    ws_df = pd.DataFrame({'score': weighted_sum, 'n_features': pareto_df['n_features']})
    schemes['weighted_sum_equal'] = assign_deterministic_rank(
        ws_df, 'score', ascending=True, tie_break_col='n_features'
    )

    scheme_names = list(schemes.keys())
    rows = []
    for i, name_a in enumerate(scheme_names):
        for name_b in scheme_names[i + 1:]:
            common_idx = schemes[name_a].index.intersection(schemes[name_b].index)
            rho, p_value = spearmanr(schemes[name_a].loc[common_idx], schemes[name_b].loc[common_idx])
            top_a = schemes[name_a].idxmin()
            top_b = schemes[name_b].idxmin()
            rows.append({
                'scheme_a': name_a, 'scheme_b': name_b,
                'spearman_rho': rho, 'p_value': p_value,
                'top_choice_a': top_a, 'top_choice_b': top_b,
                'top_choice_agrees': top_a == top_b,
            })

    return pd.DataFrame(rows)


def select_best_overall(
    best_a: pd.Series,
    best_b: pd.Series,
    weight_set_name: str = PRIMARY_WEIGHT_SET,
    vikor_v: float = 0.5,
) -> Tuple[str, pd.DataFrame, Dict[str, object]]:
    """
    Final cross-stream decision (spec section 21.3): compare Best_A/Best_B
    on the same normalized criteria/weights as each stream's own ranking,
    explicitly check Pareto dominance between the two finalists, then run
    VIKOR on just the two rows to select Best_Overall.

    Args:
        best_a: The winning row from Stream A's global ranking (a pandas
            Series with at least the CRITERIA columns plus
            configuration/fs_option/model).
        best_b: The winning row from Stream B's global ranking.
        weight_set_name: Which WEIGHT_SETS entry to use (default: 'equal',
            matching each stream's own primary ranking).
        vikor_v: VIKOR's compromise parameter (0.5 = balanced).

    Returns:
        (overall_label, table7, dominance_result):
          - overall_label: 'Best_A' or 'Best_B'.
          - table7: 2-row DataFrame (spec Table 7) with a `stream_winner`
            column ('Best_A'/'Best_B') and `is_best_overall` boolean.
          - dominance_result: dict with 'one_dominates_other' (bool),
            'dominant' (label or None), 'non_dominated' (list of labels) -
            spec section 21.3's transparency requirement ("state whether
            one finalist Pareto-dominates the other").
    """
    finalists = pd.DataFrame([best_a, best_b])
    finalists.index = ['Best_A', 'Best_B']

    dominance_pareto = pareto_filter(finalists, CRITERIA, CRITERIA_TYPES)
    one_dominates = len(dominance_pareto) == 1
    dominance_result = {
        'one_dominates_other': bool(one_dominates),
        'dominant': dominance_pareto.index[0] if one_dominates else None,
        'non_dominated': list(dominance_pareto.index),
    }

    weights = WEIGHT_SETS[weight_set_name]
    if len(finalists) == 1:
        ranked = finalists.copy()
        ranked['vikor_rank'] = 1
    else:
        ranked = vikor(finalists, CRITERIA, weights, CRITERIA_TYPES, v=vikor_v, tie_break_col='n_features')

    ranked = ranked.merge(
        finalists[[c for c in finalists.columns if c not in ranked.columns]],
        left_index=True, right_index=True, how='left'
    )
    overall_label = ranked['vikor_rank'].idxmin()

    table7 = ranked.reset_index().rename(columns={'index': 'stream_winner'})
    table7['is_best_overall'] = table7['stream_winner'] == overall_label

    return overall_label, table7, dominance_result
