"""
Incremental-value comparisons (spec section 20's supplementary factor-level
diagnostic, formerly Table 6): A2-A1 (feature engineering), A3-A1 (grid
fusion), A4-A2 (grid after engineering), A4-A3 (engineering after grid
fusion). Built entirely from saved fold-level predictions (spec section 25)
- paired on (fold_id, target_date) so every comparison is apples-to-apples
on identical outer folds.

NOT the primary decision structure (spec section 20 explicitly bans
organizing the study around pairwise comparisons like A1 vs A2) - this is a
supplementary diagnostic, called only on Stream A's all_features cells by
scripts/11_pareto_mcda_and_incremental.py, never fed into the Stream A/B
Pareto/MCDM ranking itself.
"""
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.logging_utils import get_logger
from ..splits.panels import PRIMARY_PERIOD_PANEL
from .statistical_tests import paired_bootstrap_ci, diebold_mariano_test, benjamini_hochberg_correction

# (alternative, baseline, panel) - the four primary factor-level
# comparisons. All four configurations now share the single primary common
# period (spec section 14, src/splits/panels.py::PRIMARY_PERIOD_PANEL) -
# Stream A/B predictions carry no other panel label, so every comparison
# must use it (the earlier 'panel1_full_period' variant for A2-A1 predates
# the Stream A/B rebuild and no longer has any matching predictions).
INCREMENTAL_COMPARISONS: List[Tuple[str, str, str]] = [
    ('A2', 'A1', PRIMARY_PERIOD_PANEL),
    ('A3', 'A1', PRIMARY_PERIOD_PANEL),
    ('A4', 'A2', PRIMARY_PERIOD_PANEL),
    ('A4', 'A3', PRIMARY_PERIOD_PANEL),
]


def paired_fold_errors(
    predictions_df: pd.DataFrame,
    config_a: str,
    config_b: str,
    panel: str,
    model: str,
    fs_option: str = 'all_features',
    horizon: Optional[int] = None,
) -> pd.DataFrame:
    """
    Inner-join two configurations' predictions on (fold_id, target_date) -
    and horizon, if not already filtered - so every row is a genuinely
    paired outer-fold comparison. Returns the merged DataFrame with
    `residual_a`/`residual_b` (predicted - actual) columns; empty if no
    fold pairs exist for these tags.
    """
    def _subset(df, configuration):
        mask = (
            (df['configuration'] == configuration) & (df['panel'] == panel)
            & (df['model'] == model) & (df['fs_option'] == fs_option)
        )
        if horizon is not None:
            mask &= df['horizon'] == horizon
        return df[mask]

    a = _subset(predictions_df, config_a)
    b = _subset(predictions_df, config_b)
    if a.empty or b.empty:
        return pd.DataFrame()

    merged = a.merge(
        b, on=['horizon', 'fold_id', 'target_date'], suffixes=('_a', '_b')
    )
    return merged


def compute_incremental_value_table(
    predictions_df: pd.DataFrame,
    models: Iterable[str],
    fs_option: str = 'all_features',
    horizons: Iterable[int] = (1, 2, 4),
    comparisons: Optional[List[Tuple[str, str, str]]] = None,
) -> pd.DataFrame:
    """
    Table 6 (spec section 20): for each of the four primary comparisons x
    model x horizon, paired absolute-error change, percentage improvement
    (spec section 10's formula), bootstrap CI, and Diebold-Mariano test -
    with Benjamini-Hochberg correction across the whole table (spec section
    17: "Correct for multiple comparisons when necessary").
    """
    logger = get_logger()
    comparisons = comparisons or INCREMENTAL_COMPARISONS
    rows = []

    for alt, baseline, panel in comparisons:
        for model in models:
            for h in horizons:
                merged = paired_fold_errors(
                    predictions_df, alt, baseline, panel, model,
                    fs_option=fs_option, horizon=h
                )
                if merged.empty:
                    continue

                residual_a = merged['residual_a'].values
                residual_b = merged['residual_b'].values
                errors_a = np.abs(residual_a)
                errors_b = np.abs(residual_b)

                boot = paired_bootstrap_ci(errors_a, errors_b)
                try:
                    dm = diebold_mariano_test(residual_a, residual_b, h=h)
                except ValueError as e:
                    logger.warning(f"DM test skipped for {alt}-{baseline}/{model}/h{h}: {e}")
                    dm = {'dm_statistic': float('nan'), 'p_value': float('nan'), 'n_obs': len(merged)}

                mae_alt, mae_base = float(errors_a.mean()), float(errors_b.mean())
                pct_improvement = (
                    (mae_base - mae_alt) / mae_base * 100 if mae_base != 0 else float('nan')
                )

                rows.append({
                    'comparison': f'{alt}-{baseline}',
                    'alternative': alt,
                    'baseline': baseline,
                    'panel': panel,
                    'model': model,
                    'fs_option': fs_option,
                    'horizon': h,
                    'mae_alternative': mae_alt,
                    'mae_baseline': mae_base,
                    'absolute_error_change': mae_alt - mae_base,
                    'percentage_improvement': pct_improvement,
                    'ci_low': boot['ci_low'],
                    'ci_high': boot['ci_high'],
                    'dm_statistic': dm['dm_statistic'],
                    'dm_p_value': dm['p_value'],
                    'n_pairs': boot['n_pairs'],
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df['dm_p_value_bh'] = benjamini_hochberg_correction(df['dm_p_value'].tolist())
    return df
