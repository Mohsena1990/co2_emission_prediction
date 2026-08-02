"""
Table generation (spec section 20): every table is built from saved
CSV/JSON/Parquet outputs (or, for Table 12, a hand-maintained changelog of
verified fixes) - never hand-entered numeric results.
"""
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..core.logging_utils import get_logger
from ..core.utils import calculate_weighted_mae
from ..features.registry import FeatureRegistry, CONFIGURATION_TAGS, STREAM_B_TAGS, STREAM_B_TO_A

FS_NAMES = ['fs_linear', 'fs_wrapper', 'fs_xgboost_shap', 'fs_permutation_stability', 'fs_consensus']
FS_PRIMARY_NAMES = ['fs_linear', 'fs_wrapper', 'fs_xgboost_shap', 'fs_permutation_stability']


def build_table1_raw_variable_inventory(
    registry: FeatureRegistry,
    data_quality_report: dict,
    target_column: str = 'CO2e',
) -> pd.DataFrame:
    """
    Table 1: raw variable inventory (target + 7 raw predictors).

    'source'/'unit' are extracted from the registry's documented formula
    text where available; where the upstream provenance (e.g. which
    government agency) isn't independently recorded anywhere in this
    repository, that is stated explicitly rather than invented.
    """
    missing_by_col = {
        row['column']: row['missing_count']
        for row in data_quality_report.get('missing_values', [])
    }
    date_range = data_quality_report.get('summary', {}).get('date_range', {})

    rows = [{
        'variable': target_column,
        'target_or_predictor': 'target',
        'source': 'raw input data file (upstream agency not recorded in repo)',
        'unit': 'CO2e (as supplied)',
        'frequency': 'quarterly',
        'start_date': date_range.get('start'),
        'end_date': date_range.get('end'),
        'missing_count': missing_by_col.get(target_column, None),
        'publication_timing': 'n/a (target)',
        'target_derived': False,
        'safe_at_h1': False,  # the target itself is never a "safe predictor"
        'safe_at_h2': False,
        'safe_at_h4': False,
    }]

    for name, entry in registry.entries.items():
        if entry['kind'] != 'raw' or entry['family'] == 'grid_exogenous':
            continue
        formula = entry.get('formula', '')
        unit = formula.split(',', 1)[1].strip(') ') if ',' in formula else 'not recorded'
        rows.append({
            'variable': name,
            'target_or_predictor': 'predictor',
            'source': 'raw input data file (upstream agency not recorded in repo)',
            'unit': unit,
            'frequency': 'quarterly',
            'start_date': date_range.get('start'),
            'end_date': date_range.get('end'),
            'missing_count': missing_by_col.get(name, None),
            'publication_timing': f"release_delay_quarters={entry['release_delay_quarters']}",
            'target_derived': entry['target_derived'],
            'safe_at_h1': entry['safe_at_h1'],
            'safe_at_h2': entry['safe_at_h2'],
            'safe_at_h4': entry['safe_at_h4'],
        })

    return pd.DataFrame(rows)


def build_table2_feature_registry(registry: FeatureRegistry) -> pd.DataFrame:
    """Table 2: complete candidate-feature registry - registry already builds this directly."""
    return registry.to_dataframe()


def build_table3_configuration_definitions(manifest: dict) -> pd.DataFrame:
    """
    Table 3: configuration definitions (A1-A4), from
    configurations_manifest.json (spec section 8/24).
    """
    purposes = {
        'A1': 'Raw quarterly predictors only - baseline.',
        'A2': 'Raw + engineered predictors - isolates feature-engineering value.',
        'A3': 'Raw + grid features - isolates grid-fusion value.',
        'A4': 'Raw + engineered + grid - combined/interaction effect.',
    }
    rows = []
    for tag in ('A1', 'A2', 'A3', 'A4'):
        info = manifest.get(tag, {})
        rows.append({
            'configuration': tag,
            'raw_feature_count': info.get('n_raw'),
            'engineered_feature_count': info.get('n_engineered'),
            'grid_feature_count': info.get('n_grid'),
            'nominal_total': info.get('nominal_max'),
            'audited_total': info.get('n_features_total'),
            'sample_period': 'panel1_full_period' if tag in ('A1', 'A2') else 'panel2_common_period (also A1/A2)',
            'purpose': purposes[tag],
        })
    return pd.DataFrame(rows)


def build_table4_grid_aggregation_summary(
    registry: FeatureRegistry,
    grid_quality_report: pd.DataFrame,
) -> pd.DataFrame:
    """
    Table 4: grid aggregation summary - one row per grid feature, combining
    the registry's static documentation with the dynamic completeness
    numbers already saved in grid_quality_report.csv (spec section 6.3/24).
    """
    agg_methods = {
        'Grid_CI_mean': 'mean', 'Grid_CI_p90': 'P90',
        'Grid_CI_std': 'std', 'Grid_CI_high_share': 'share of high/very-high intervals',
        'Grid_CI_low_share': 'share of low/very-low intervals',
        'Grid_renewable_share': 'mean of summed renewable generation-mix %',
        'Grid_low_carbon_share': 'mean of summed renewable+nuclear generation-mix %',
        'Grid_fossil_share': 'mean of summed fossil generation-mix %',
        'Grid_gas_share': 'mean generation-mix %', 'Grid_wind_share': 'mean generation-mix %',
    }
    included = grid_quality_report[grid_quality_report['inclusion_status'] == 'included']
    n_missing_quarters = (grid_quality_report['inclusion_status'] != 'included').sum()
    first_q = included['quarter'].min() if len(included) else None
    last_q = included['quarter'].max() if len(included) else None
    completeness_threshold = (
        grid_quality_report['min_completeness_threshold'].iloc[0]
        if len(grid_quality_report) else None
    )

    rows = []
    for name, entry in registry.entries.items():
        if entry['family'] != 'grid_exogenous':
            continue
        rows.append({
            'grid_feature': name,
            'original_frequency': 'half-hourly',
            'aggregation_method': agg_methods.get(name, 'mean'),
            'completeness_threshold': completeness_threshold,
            'missing_count_quarters': n_missing_quarters,
            'unit': 'gCO2/kWh' if name.startswith('Grid_CI') and 'share' not in name else '%',
            'first_available_quarter': first_q,
            'final_available_quarter': last_q,
        })
    return pd.DataFrame(rows)


def build_table5_configuration_level_performance(
    stage1_metrics: pd.DataFrame,
    horizon_weights: Dict[int, float],
) -> pd.DataFrame:
    """
    Table 5: configuration-level performance (spec section 20) - A1-A4 x 5
    models, straight from Stage 1's saved metrics.csv (spec section 12
    "Fit all five forecasting models using all audited features"). RMSE/
    sMAPE/R2 are horizon-weighted the same way weighted_mase/weighted_mae
    already are, for a single comparable summary column per metric.
    """
    if stage1_metrics.empty:
        return pd.DataFrame()

    rows = []
    for _, row in stage1_metrics.iterrows():
        weighted_rmse = calculate_weighted_mae(
            {h: row.get(f'rmse_h{h}', np.nan) for h in horizon_weights}, horizon_weights
        )
        weighted_smape = calculate_weighted_mae(
            {h: row.get(f'smape_h{h}', np.nan) for h in horizon_weights}, horizon_weights
        )
        weighted_r2 = calculate_weighted_mae(
            {h: row.get(f'r2_h{h}', np.nan) for h in horizon_weights}, horizon_weights
        )
        rows.append({
            'configuration': row['configuration'],
            'panel': row.get('panel'),
            'model': row['model'],
            'h1_mase': row.get('mase_h1'),
            'h2_mase': row.get('mase_h2'),
            'h4_mase': row.get('mase_h4'),
            'weighted_mase': row.get('weighted_mase'),
            'weighted_mae': row.get('weighted_mae'),
            'weighted_rmse': weighted_rmse,
            'weighted_smape': weighted_smape,
            'weighted_r2': weighted_r2,
            'stability_score': row.get('stability_score'),
            'total_runtime_seconds': row.get('total_runtime_seconds'),
            'n_features': row.get('n_features'),
        })
    return pd.DataFrame(rows).sort_values(['configuration', 'model'])


def build_table_fs_performance(
    stage2_metrics: pd.DataFrame,
    configuration: str,
) -> pd.DataFrame:
    """
    Tables 8 (A2) and 9 (A4): FS1-FS5 x 5 models performance for a single
    configuration, straight from Stage 2's saved metrics.csv.
    """
    if stage2_metrics.empty or 'configuration' not in stage2_metrics.columns:
        return pd.DataFrame()
    sub = stage2_metrics[
        (stage2_metrics['configuration'] == configuration) & (stage2_metrics['fs_option'] != 'all_features')
    ].copy()
    if sub.empty:
        return sub
    cols = [
        'configuration', 'panel', 'fs_option', 'model', 'n_features',
        'weighted_mase', 'weighted_mae', 'worst_horizon_mase', 'stability_score',
        'total_runtime_seconds',
    ]
    return sub[[c for c in cols if c in sub.columns]].sort_values(['fs_option', 'model'])


def build_table7_fs_membership(
    fs_results: dict,
    all_features: list,
    vote_threshold: int = 3,
) -> pd.DataFrame:
    """
    Table 7: feature-selection membership (spec section 20) - one row per
    feature, binary membership in FS1-FS4 (FS5 consensus is the vote
    outcome, not an independent vote), vote_count, selection_frequency.

    `average_rank`/`stability` (spec's other two Table 7 columns) require
    per-fold rank/stability data that scripts/10_run_experiment_grid.py
    does not currently persist beyond each FS option's final selected-set
    JSON (src/fs/*.py compute it internally but it isn't saved) - this is a
    known, documented gap rather than a fabricated column; a future run
    that also saves each FS method's stability DataFrame could extend this.

    Args:
        fs_results: Loaded stage2_fs_results_<config>_<panel>.json, i.e.
            {fs_name: {'selected_features': [...], 'n_selected': int}}.
        all_features: Full candidate feature list for this configuration.
        vote_threshold: Matches config.fs.vote_threshold (for reference only
            - the consensus column itself already reflects whatever
            threshold FS5 was run with).
    """
    fs_primary = ['fs_linear', 'fs_wrapper', 'fs_xgboost_shap', 'fs_permutation_stability']
    fs_present = [fs for fs in fs_primary if fs in fs_results]

    rows = []
    for feature in all_features:
        row = {'feature': feature}
        vote_count = 0
        for fs in fs_present:
            member = int(feature in fs_results[fs]['selected_features'])
            row[fs] = member
            vote_count += member
        row['vote_count'] = vote_count
        row['selection_frequency'] = vote_count / len(fs_present) if fs_present else np.nan
        if 'fs_consensus' in fs_results:
            row['fs_consensus'] = int(feature in fs_results['fs_consensus']['selected_features'])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values('vote_count', ascending=False)
    return df


def build_table12_bug_fix_verification() -> pd.DataFrame:
    """
    Table 12: bug-fix verification log. Hand-maintained (not derivable
    purely from saved run outputs - it records what changed and why, which
    only exists in the development history), but every row names the exact
    regression test that verifies it; run `pytest tests/ -q` to confirm all
    pass before trusting this table's 'status' column.
    """
    rows = [
        {
            'issue': 'Intensity ratios (CO2e_per_Population, CO2e_per_TEC) used contemporaneous CO2e',
            'previous_behaviour': 'Numerator/denominator both used CO2e_t, leaking the target into its own predictor',
            'corrected_behaviour': 'Both numerator and denominator lagged by >=1 quarter',
            'validation_test': 'tests/test_feature_leakage_fixes.py',
            'status': 'fixed',
        },
        {
            'issue': 'delta_log target transform had no valid inverse',
            'previous_behaviour': 'Predictions left in delta-log space or inverted incorrectly',
            'corrected_behaviour': 'invert_delta_log(): reference level -> cumulative sum -> exp, round-trip tested',
            'validation_test': 'tests/test_core_utils.py::TestDeltaLogInversion',
            'status': 'fixed',
        },
        {
            'issue': 'LSTM prediction length/date alignment (bug 3.3)',
            'previous_behaviour': 'len(y_pred) could silently mismatch len(y_test); risk of misaligned target dates',
            'corrected_behaviour': 'build_predict_input() + strict length assertions (assert_prediction_alignment)',
            'validation_test': 'tests/test_lstm_alignment.py',
            'status': 'fixed',
        },
        {
            'issue': 'VIKOR/TOPSIS tied ranks used Series.rank().astype(int)',
            'previous_behaviour': 'Fractional ranks truncated, collapsing distinct alternatives onto the same rank',
            'corrected_behaviour': 'assign_deterministic_rank(): stable sort on (score, tie_break_col, label)',
            'validation_test': 'tests/test_mcda_ties.py',
            'status': 'fixed',
        },
        {
            'issue': 'Duplicated/inconsistent quarter-string parsing',
            'previous_behaviour': 'Multiple ad-hoc parsers scattered across the codebase',
            'corrected_behaviour': 'quarter_to_date(): single canonical parser, all formats',
            'validation_test': 'tests/test_core_utils.py::TestQuarterParsing',
            'status': 'fixed',
        },
        {
            'issue': 'Feature selection ran on the flat outer cv_plan (audit finding A-7)',
            'previous_behaviour': 'scripts/01_run_fs.py passed the same plan later used for final evaluation into FS - no isolation from outer-test data',
            'corrected_behaviour': 'Builds an isolated build_tuning_cv_plan first; FS never sees outer-test rows',
            'validation_test': 'tests/test_leakage_sentinel.py (invariance tests)',
            'status': 'fixed',
        },
        {
            'issue': 'FS2 wrapper methods (RFE/SFS/SBS) leaked via full-sample fit / non-time-aware KFold',
            'previous_behaviour': 'RFE fit on the entire sample (train+test together); SFS/SBS used plain sklearn KFold=5',
            'corrected_behaviour': 'RFE fits per-fold on training indices only (stability selection); SFS/SBS use the project\'s own expanding-window folds as sklearn cv=',
            'validation_test': 'tests/test_leakage_sentinel.py::test_wrapper_fs_result_invariant_to_contamination',
            'status': 'fixed',
        },
        {
            'issue': 'aggregate_to_quarterly crashed when every quarter was excluded for incompleteness',
            'previous_behaviour': "pd.DataFrame([]).set_index('quarter') raised KeyError (empty list has no columns)",
            'corrected_behaviour': 'Returns a well-formed empty frame with the expected 12 grid-feature columns',
            'validation_test': 'tests/test_grid.py::TestAggregateToQuarterly',
            'status': 'fixed',
        },
        {
            'issue': "configuration_manifest()'s grid-feature count used registry absence as the detector",
            'previous_behaviour': "grid_count = sum(c not in registry.entries) - became always-0 once grid rows were registered, silently disabling the Panel 2 trigger in scripts/00",
            'corrected_behaviour': "Keys off family == 'grid_exogenous' instead of absence-from-registry",
            'validation_test': 'tests/test_configurations.py::test_manifest_counts_grid_features_separately_from_raw',
            'status': 'fixed',
        },
        {
            'issue': 'A3/A4 matrices contained NaN from a partially-missing grid quarter (2018Q1)',
            'previous_behaviour': 'Quarter passed overall completeness check but had a fully-missing generation-mix sub-field -> NaN crashed Ridge.fit()',
            'corrected_behaviour': 'build_configuration_matrices drops any row with residual NaN after the grid merge, with a warning',
            'validation_test': 'tests/test_configurations.py::test_partially_missing_grid_quarter_is_dropped_not_left_as_nan',
            'status': 'fixed',
        },
        {
            'issue': "vikor()'s S/R denominator used the signed f_star-f_minus for both benefit and cost criteria",
            'previous_behaviour': 'Negative for every cost criterion (accuracy/MAE is always cost) - S could go negative; R stuck at exactly 0.0 for every alternative, degrading VIKOR to ranking by S alone in every existing caller (scripts/02, scripts/05)',
            'corrected_behaviour': 'abs() on the denominator (no-op for benefit criteria, fixes cost criteria)',
            'validation_test': 'tests/test_mcda_ties.py::TestVikorCostCriteriaDenominatorBug',
            'status': 'fixed',
        },
        {
            'issue': "Cross-model LSTM importance used compute_permutation_importance (sklearn), which assumes predict(X) returns len(X) rows",
            'previous_behaviour': "LSTM's predict() returns len(X)-lookback rows by contract (bug 3.3) - sklearn's permutation_importance crashed with 'inconsistent numbers of samples'",
            'corrected_behaviour': 'Uses manual_permutation_importance with y sliced by lookback, matching the pattern scripts/06 already used for the same LSTM contract',
            'validation_test': 'verified via scripts/12 smoke test on real data (no dedicated unit test written)',
            'status': 'fixed',
        },
    ]
    return pd.DataFrame(rows)


# =============================================================================
# Stream A/B mega-spec tables (spec section 24, Tables 3/6/8/9/10). Table
# numbering here follows the REVISED spec, distinct from the legacy
# Table1-12 numbering above (kept for backward compatibility/existing
# tests) - build_table3_configuration_definitions above still answers
# "what are A1-A4" for anyone using it standalone; the function below
# additionally covers B1-B4 (spec section 24's Table 3: "Rows: A1-A4;
# B1-B4").
# =============================================================================

def build_table3_ab_configuration_definitions(
    manifest: dict,
    registry: FeatureRegistry,
) -> pd.DataFrame:
    """
    Table 3 (spec section 24): A1-A4 and B1-B4 configuration definitions.
    B{n}'s candidate pool always mirrors A{n} exactly (spec section 12) -
    audited/nominal counts are identical between e.g. A2 and B2; what
    differs is that B applies FS1-FS5 on top (spec section 12) and both
    streams share the single primary common period (spec section 14).
    """
    purposes = {
        'A1': 'Raw quarterly + mobility-shock predictors only - baseline, no feature selection.',
        'A2': 'Raw + engineered predictors, no feature selection - isolates feature-engineering value.',
        'A3': 'Raw + grid features, no feature selection - isolates grid-fusion value.',
        'A4': 'Raw + engineered + grid, no feature selection - combined/interaction effect.',
        'B1': 'Same candidate pool as A1, with FS1-FS5 applied.',
        'B2': 'Same candidate pool as A2, with FS1-FS5 applied.',
        'B3': 'Same candidate pool as A3, with FS1-FS5 applied.',
        'B4': 'Same candidate pool as A4, with FS1-FS5 applied.',
    }
    rows = []
    for tag in CONFIGURATION_TAGS:
        info = manifest.get(tag, {})
        rows.append({
            'row': tag, 'stream': 'A', 'candidate_data_blocks': tag,
            'nominal_count': info.get('nominal_max'), 'audited_count': info.get('n_features_total'),
            'fs_applied': 'none', 'sample_period': 'panel2_common_period (primary)',
            'purpose': purposes[tag],
        })
    for b_tag in STREAM_B_TAGS:
        a_tag = STREAM_B_TO_A[b_tag]
        info = manifest.get(a_tag, {})
        rows.append({
            'row': b_tag, 'stream': 'B', 'candidate_data_blocks': f'{a_tag} candidate pool',
            'nominal_count': info.get('nominal_max'), 'audited_count': info.get('n_features_total'),
            'fs_applied': 'FS1-FS5', 'sample_period': 'panel2_common_period (primary)',
            'purpose': purposes[b_tag],
        })
    return pd.DataFrame(rows)


def build_table6_global_factor_summary(
    stream_a_ranked: pd.DataFrame,
    stream_b_ranked: pd.DataFrame,
    models: Optional[list] = None,
) -> pd.DataFrame:
    """
    Table 6 (spec section 24): global factor-level summary - one row per
    A1-A4, B1-B4, FS1-FS5, and each model. Never the primary decision
    structure (spec section 20 bans organizing the study around isolated
    factor comparisons) - a supplementary "how much does each factor
    matter on average" view, built entirely from the two streams' own
    global rankings (spec sections 20/21 - `stream_a_ranked`/
    `stream_b_ranked` are `build_pareto_mcda_table`'s output for each
    stream, so `vikor_rank`/`pareto_status` already reflect that stream's
    OWN joint ranking, never re-derived here).

    'failure_rate' compares the number of candidates with a result against
    the NOMINAL count that factor level should contribute across the two
    streams (Stream A: 4 configs x 5 models = 20; Stream B: 4 configs x 5
    FS x 5 models = 100) - e.g. a config's nominal count is 5 (its Stream A
    models) + 25 (its Stream B FS x models) = 30.
    """
    models = models or ['ridge', 'random_forest', 'lightgbm', 'catboost', 'lstm']
    n_models, n_fs, n_configs = len(models), len(FS_NAMES), len(CONFIGURATION_TAGS)

    def _summarize(sub: pd.DataFrame, label: str, nominal: int) -> Optional[dict]:
        if sub.empty:
            return {'row': label, 'n_candidates': 0, 'nominal_count': nominal, 'failure_rate': 1.0,
                    'mean_weighted_mase': np.nan, 'median_weighted_mase': np.nan,
                    'median_global_rank': np.nan, 'rank_dispersion': np.nan, 'mean_runtime_seconds': np.nan}
        return {
            'row': label, 'n_candidates': len(sub), 'nominal_count': nominal,
            'failure_rate': 1.0 - (len(sub) / nominal) if nominal else np.nan,
            'mean_weighted_mase': sub['weighted_mase'].mean(),
            'median_weighted_mase': sub['weighted_mase'].median(),
            'median_global_rank': sub['vikor_rank'].median() if 'vikor_rank' in sub else np.nan,
            'rank_dispersion': sub['vikor_rank'].std() if 'vikor_rank' in sub else np.nan,
            'mean_runtime_seconds': sub['total_runtime_seconds'].mean() if 'total_runtime_seconds' in sub else np.nan,
        }

    rows = []
    for tag in CONFIGURATION_TAGS:
        a_sub = stream_a_ranked[stream_a_ranked['configuration'] == tag] if not stream_a_ranked.empty else pd.DataFrame()
        b_sub = stream_b_ranked[stream_b_ranked['configuration'] == tag] if not stream_b_ranked.empty else pd.DataFrame()
        combined = pd.concat([a_sub, b_sub], ignore_index=True) if not (a_sub.empty and b_sub.empty) else pd.DataFrame()
        rows.append(_summarize(combined, tag, nominal=n_models + n_fs * n_models))

    for b_tag in STREAM_B_TAGS:
        a_tag = STREAM_B_TO_A[b_tag]
        sub = stream_b_ranked[stream_b_ranked['configuration'] == a_tag] if not stream_b_ranked.empty else pd.DataFrame()
        rows.append(_summarize(sub, b_tag, nominal=n_fs * n_models))

    for fs in FS_NAMES:
        sub = stream_b_ranked[stream_b_ranked['fs_option'] == fs] if not stream_b_ranked.empty else pd.DataFrame()
        rows.append(_summarize(sub, fs, nominal=n_configs * n_models))

    for model in models:
        a_sub = stream_a_ranked[stream_a_ranked['model'] == model] if not stream_a_ranked.empty else pd.DataFrame()
        b_sub = stream_b_ranked[stream_b_ranked['model'] == model] if not stream_b_ranked.empty else pd.DataFrame()
        combined = pd.concat([a_sub, b_sub], ignore_index=True) if not (a_sub.empty and b_sub.empty) else pd.DataFrame()
        rows.append(_summarize(combined, model, nominal=n_configs + n_configs * n_fs))

    return pd.DataFrame(rows)


def build_table8_fs_outputs(
    fs_results_by_config: Dict[str, dict],
    feature_pool_by_config: Dict[str, list],
    vote_threshold: int = 3,
) -> pd.DataFrame:
    """
    Table 8 (spec section 24): one row per (B-configuration, candidate
    feature) - binary FS1-FS4 membership, FS5 consensus outcome, vote
    count, selection frequency. `average_rank`/`rank_stability` (spec's
    other two columns) require per-fold rank data FS1-FS4 compute
    internally but scripts/10_run_experiment_grid.py does not currently
    persist beyond each method's final selected-set JSON - documented gap,
    not a fabricated column (same limitation the legacy
    build_table7_fs_membership already documented).

    Args:
        fs_results_by_config: {b_tag: fs_results dict} where fs_results is
            streamB_fs_results_<A{n}>.json's loaded content
            ({fs_name: {'selected_features': [...], ...}}).
        feature_pool_by_config: {b_tag: candidate feature list} (B{n}'s own
            audited candidate pool, i.e. A{n}'s columns).
    """
    rows = []
    for b_tag, fs_results in fs_results_by_config.items():
        all_features = feature_pool_by_config.get(b_tag, [])
        fs_present = [fs for fs in FS_PRIMARY_NAMES if fs in fs_results]
        for feature in all_features:
            row = {'b_configuration': b_tag, 'feature': feature}
            vote_count = 0
            for fs in FS_PRIMARY_NAMES:
                member = int(fs in fs_results and feature in fs_results[fs]['selected_features'])
                row[fs] = member
                vote_count += member if fs in fs_present else 0
            row['fs5_consensus'] = int('fs_consensus' in fs_results and feature in fs_results['fs_consensus']['selected_features'])
            row['vote_count'] = vote_count
            row['selection_frequency'] = vote_count / len(fs_present) if fs_present else np.nan
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['b_configuration', 'vote_count'], ascending=[True, False])


def build_table9_covid_regime_interpretation(
    winner_regime_data: Dict[str, pd.DataFrame],
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Table 9 (spec section 24): pre/COVID/post-COVID metrics + top-K SHAP
    predictors for Best_A/Best_B/Best_Overall (spec section 23).

    Args:
        winner_regime_data: {label ('Best_A'/'Best_B'/'Best_Overall'):
            that winner's regime_importance.csv (columns: feature,
            importance, regime, n_samples), i.e.
            interpretability/{best_A,best_B,best_overall}/regime_importance.csv
            loaded as a DataFrame}.

    Returns:
        One row per (winner, regime): n_samples, top_k_predictors
        (comma-joined, ranked by |importance| descending).
    """
    rows = []
    for label, df in winner_regime_data.items():
        if df is None or df.empty:
            rows.append({'winner': label, 'regime': None, 'n_samples': 0, 'top_predictors': None})
            continue
        for regime, regime_df in df.groupby('regime'):
            top = regime_df.reindex(regime_df['importance'].abs().sort_values(ascending=False).index)
            top_predictors = ', '.join(top['feature'].head(top_k).tolist())
            rows.append({
                'winner': label, 'regime': regime,
                'n_samples': regime_df['n_samples'].iloc[0] if 'n_samples' in regime_df else len(regime_df),
                'top_predictors': top_predictors,
            })
    return pd.DataFrame(rows)


def build_table10_source_data_quality(
    mobility_metadata: Optional[dict] = None,
    grid_quality_report: Optional[pd.DataFrame] = None,
    owid_renewable_metadata: Optional[dict] = None,
    owid_low_carbon_metadata: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Table 10 (spec section 24): source and data-quality validation - one
    row per external data source (Google mobility, NESO grid, OWID
    renewable, OWID low-carbon), each from its own retrieval-metadata
    sidecar file rather than re-derived/estimated here.
    """
    rows = []

    if mobility_metadata:
        rows.append({
            'source': 'Google COVID-19 mobility', 'retrieval_date': mobility_metadata.get('retrieval_date'),
            'checksum': mobility_metadata.get('file_checksum_sha256') or mobility_metadata.get('checksum'),
            'first_observation': mobility_metadata.get('first_observation'),
            'last_observation': mobility_metadata.get('last_observation'),
            'n_records': mobility_metadata.get('n_daily_records'),
            'coverage_note': 'daily national UK, 2020-02-15 through Google archive end (~2022-10-15)',
        })

    if grid_quality_report is not None and not grid_quality_report.empty:
        included = grid_quality_report[grid_quality_report['inclusion_status'] == 'included']
        rows.append({
            'source': 'NESO GB Carbon Intensity API', 'retrieval_date': None, 'checksum': None,
            'first_observation': included['quarter'].min() if len(included) else None,
            'last_observation': included['quarter'].max() if len(included) else None,
            'n_records': len(included),
            'coverage_note': f"{len(grid_quality_report) - len(included)} quarters excluded for incompleteness",
        })

    for label, meta in (('OWID renewable electricity share', owid_renewable_metadata),
                        ('OWID low-carbon electricity share', owid_low_carbon_metadata)):
        if meta:
            rows.append({
                'source': label,
                'retrieval_date': meta.get('dateDownloaded') or meta.get('retrieved_date') or meta.get('retrieval_date'),
                'checksum': None, 'first_observation': None, 'last_observation': None,
                'n_records': None, 'coverage_note': 'annual UK series, one-year-lagged before use (spec 8.2)',
            })

    return pd.DataFrame(rows)
