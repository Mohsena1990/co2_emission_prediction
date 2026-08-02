"""
Cross-model feature-importance comparison (spec section 19.3 / Table 11)
and regime-stability summarization (spec section 19.4).

Different model classes produce importance on different, non-comparable
scales (Ridge |coefficient|, tree-based mean-|SHAP|, permutation-importance
MAE increase) - this module RANKS each model's own importance values
(1 = most important) before comparing across models, since rank is the one
thing that IS comparable across model classes without a shared unit (spec
section 19.3: "Do not compare raw importance magnitudes across model
classes without normalisation").
"""
from typing import Dict

import numpy as np
import pandas as pd

from ..core.logging_utils import get_logger


def rank_importance(importance_df: pd.DataFrame, value_col: str = 'importance') -> pd.Series:
    """Rank features by |value_col| descending (1 = most important), indexed by feature name."""
    df = importance_df.set_index('feature') if 'feature' in importance_df.columns else importance_df
    return df[value_col].abs().rank(ascending=False, method='min').astype(int)


def build_cross_model_importance_table(
    importance_by_model: Dict[str, pd.DataFrame],
    value_col: str = 'importance',
) -> pd.DataFrame:
    """
    Table 11 (spec section 20): one row per feature, one rank column per
    model, plus average rank and rank dispersion (std across models that
    scored that feature - NaN, not 0, for models that didn't include it, so
    a feature absent from one model's selected set doesn't spuriously pull
    down its average rank).

    Args:
        importance_by_model: Dict mapping model name (e.g. 'ridge',
            'random_forest', 'lightgbm', 'catboost', 'lstm') to that
            model's importance DataFrame (columns: 'feature', value_col).
        value_col: Column in each DataFrame holding the importance value to
            rank by (magnitude, descending).

    Returns:
        DataFrame with columns: feature, <model>_rank (one per model),
        average_rank, rank_dispersion (std), n_models_scored.
    """
    logger = get_logger()
    rank_series = {}
    for model_name, df in importance_by_model.items():
        if df is None or df.empty:
            logger.warning(f"build_cross_model_importance_table: empty importance for '{model_name}', skipping")
            continue
        rank_series[model_name] = rank_importance(df, value_col=value_col)

    if not rank_series:
        return pd.DataFrame()

    rank_df = pd.DataFrame(rank_series)
    rank_df.columns = [f'{m}_rank' for m in rank_df.columns]
    rank_df['average_rank'] = rank_df.mean(axis=1, skipna=True)
    rank_df['rank_dispersion'] = rank_df.drop(columns=['average_rank']).std(axis=1, skipna=True)
    rank_df['n_models_scored'] = rank_df.drop(columns=['average_rank', 'rank_dispersion']).notna().sum(axis=1)

    rank_df = rank_df.reset_index().rename(columns={'index': 'feature'})
    return rank_df.sort_values('average_rank').reset_index(drop=True)


def compute_regime_stability(
    regime_importance: Dict[str, pd.DataFrame],
    value_col: str = 'importance',
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Regime stability (spec section 19.4 / Table 11's regime_stability
    column): for each feature, the fraction of regimes (e.g. pre/during/
    post-COVID) in which it ranks in the top-`top_k` by `value_col` - 1.0
    means consistently important across every regime, 0.0 means never
    important in any regime. This is predictive/descriptive interpretation
    of rank consistency, not a causal regime-shift claim (spec 19.4).

    Args:
        regime_importance: Dict mapping regime name to that regime's
            importance DataFrame (output of analyze_regime_shap /
            analyze_regime_permutation_importance for ONE model).
        value_col: Importance column to rank by.
        top_k: Top-K threshold per regime.

    Returns:
        DataFrame with columns: feature, n_regimes, n_regimes_in_top_k,
        regime_stability.
    """
    logger = get_logger()
    if not regime_importance:
        return pd.DataFrame()

    top_k_membership: Dict[str, list] = {}
    for regime_name, df in regime_importance.items():
        if df is None or df.empty:
            continue
        ranked = rank_importance(df, value_col=value_col)
        top_features = set(ranked[ranked <= top_k].index)
        for feature in ranked.index:
            top_k_membership.setdefault(feature, []).append(feature in top_features)

    n_regimes = len(regime_importance)
    rows = []
    for feature, memberships in top_k_membership.items():
        rows.append({
            'feature': feature,
            'n_regimes': len(memberships),
            'n_regimes_in_top_k': sum(memberships),
            'regime_stability': sum(memberships) / len(memberships) if memberships else np.nan,
        })

    logger.info(f"Regime stability computed for {len(rows)} features across {n_regimes} regimes")
    return pd.DataFrame(rows).sort_values('regime_stability', ascending=False).reset_index(drop=True)


def compute_family_contribution(
    importance_df: pd.DataFrame,
    family_map: pd.Series,
    value_col: str = 'importance',
) -> pd.DataFrame:
    """
    Spec section 23: aggregate |importance| by registry family
    (mobility_exogenous, grid_exogenous, owid_exogenous vs. everything
    else) - answers "do mobility/grid variables explain this regime's
    deviations" without cherry-picking individual features.

    Args:
        importance_df: One model/regime's importance DataFrame (columns:
            'feature', value_col).
        family_map: Series mapping feature name -> family string (e.g.
            FeatureRegistry.to_dataframe().set_index('name')['family']).
        value_col: Importance column to aggregate.

    Returns:
        DataFrame with columns: family, sum, mean, count, share_of_total
        (share_of_total sums to 1.0 across families), sorted descending by
        share_of_total. Features absent from `family_map` are grouped
        under 'other' rather than dropped or raising.
    """
    if importance_df is None or importance_df.empty:
        return pd.DataFrame(columns=['family', 'sum', 'mean', 'count', 'share_of_total'])

    df = importance_df.copy()
    df['family'] = df['feature'].map(family_map).fillna('other')
    agg = df.groupby('family')[value_col].agg(['sum', 'mean', 'count']).reset_index()
    total = agg['sum'].sum()
    agg['share_of_total'] = agg['sum'] / total if total else float('nan')
    return agg.sort_values('share_of_total', ascending=False).reset_index(drop=True)
