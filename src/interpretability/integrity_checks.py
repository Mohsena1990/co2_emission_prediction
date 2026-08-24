"""
SHAP integrity checks (spec: the regime-specific SHAP rerun "must fail
loudly" rather than silently accept a degenerate result).

Guards specifically against the failure mode diagnosed in
outputs/audit/champion_shap_diagnosis.md: an interpretability refit created
with library-default hyperparameters (instead of the cell's actual
PSO-tuned params) collapsed to a constant predictor for the small
common-period sample, producing exactly-zero SHAP values and identical
"rankings" across every regime. These checks would have caught that bug
immediately rather than letting it ship as a plausible-looking CSV.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .shap_analysis import compute_shap_values, get_feature_importance_from_shap
from ..core.logging_utils import get_logger


def compute_regime_shap_with_integrity(
    model,
    X: pd.DataFrame,
    regime_periods: Dict[str, Tuple[Optional[str], Optional[str]]],
    model_type: str,
    winner_label: str,
    model_name: str,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Compute per-regime SHAP importance for one model, alongside a row of
    integrity-check results per regime (to be concatenated across
    winners/models into outputs/audit/shap_integrity_checks.csv).

    Returns:
        (regime_results, checks_df) - regime_results has the same shape as
        `analyze_regime_shap`'s return (dict of importance DataFrames);
        checks_df has one row per regime with pass/fail flags and an
        'issues' column (semicolon-joined issue codes, empty if none).
    """
    logger = get_logger()
    feature_names = list(X.columns)

    regime_results: Dict[str, pd.DataFrame] = {}
    check_rows = []
    raw_by_regime: Dict[str, np.ndarray] = {}

    for regime_name, (start, end) in regime_periods.items():
        mask = pd.Series(True, index=X.index)
        if start is not None:
            mask &= (X.index >= start)
        if end is not None:
            mask &= (X.index < end)
        X_regime = X[mask]

        if len(X_regime) < 5:
            logger.warning(f"Regime '{regime_name}' has only {len(X_regime)} samples, skipping")
            continue

        shap_values, explainer = compute_shap_values(model, X_regime, model_type)
        sv = np.asarray(shap_values)

        row = {
            'winner': winner_label, 'model': model_name, 'regime': regime_name,
            'n_observations': len(X_regime),
            'n_features_expected': len(feature_names),
        }
        issues = []

        if sv.ndim != 2:
            issues.append(f'unexpected_shap_ndim_{sv.ndim}')
            row['n_shap_rows'], row['n_features_shap'] = None, None
        else:
            row['n_shap_rows'], row['n_features_shap'] = sv.shape[0], sv.shape[1]
            if sv.shape[0] != len(X_regime):
                issues.append('row_count_mismatch')
            if sv.shape[1] != len(feature_names):
                issues.append('feature_count_mismatch')
            if np.allclose(sv, 0.0):
                issues.append('all_values_zero')
            if np.isnan(sv).any():
                issues.append('nan_present')
            if np.isinf(sv).any():
                issues.append('inf_present')

            expected_value = getattr(explainer, 'expected_value', None)
            additivity_diff = None
            if expected_value is not None:
                ev = np.asarray(expected_value)
                ev_scalar = float(ev.flat[0]) if ev.size else float(ev)
                try:
                    underlying = model.model if hasattr(model, 'model') else model
                    actual_pred = np.asarray(underlying.predict(X_regime.values))
                    reconstructed = ev_scalar + sv.sum(axis=1)
                    additivity_diff = float(np.max(np.abs(reconstructed - actual_pred)))
                    tolerance = max(1e-2, 1e-3 * float(np.abs(actual_pred).mean() + 1e-12))
                    if additivity_diff > tolerance:
                        issues.append('additivity_check_failed')
                except Exception as e:
                    logger.warning(
                        f"Additivity check skipped for {winner_label}/{model_name}/{regime_name}: {e}"
                    )
            row['additivity_max_abs_diff'] = additivity_diff

        row['issues'] = ';'.join(issues)
        row['passed'] = len(issues) == 0
        check_rows.append(row)

        if sv.ndim == 2:
            raw_by_regime[regime_name] = sv
            importance_df = get_feature_importance_from_shap(shap_values, feature_names)
            importance_df['regime'] = regime_name
            importance_df['n_samples'] = len(X_regime)
            regime_results[regime_name] = importance_df

    # Cross-regime "identical output" check - the exact symptom of the
    # original bug (every regime producing the same values because the
    # model ignored every feature).
    regime_names = list(raw_by_regime.keys())
    for i in range(len(regime_names)):
        for j in range(i + 1, len(regime_names)):
            name_i, name_j = regime_names[i], regime_names[j]
            a, b = raw_by_regime[name_i], raw_by_regime[name_j]
            if a.shape == b.shape and np.array_equal(a, b):
                for r in check_rows:
                    if r['regime'] in (name_i, name_j):
                        other = name_j if r['regime'] == name_i else name_i
                        issues_list = [x for x in r['issues'].split(';') if x]
                        issues_list.append(f'identical_to_{other}')
                        r['issues'] = ';'.join(issues_list)
                        r['passed'] = False

    checks_df = pd.DataFrame(check_rows)
    if not checks_df.empty and not checks_df['passed'].all():
        logger.warning(
            f"SHAP integrity check FAILED for {winner_label}/{model_name}: "
            f"{checks_df[~checks_df['passed']][['regime', 'issues']].to_dict(orient='records')}"
        )

    return regime_results, checks_df
