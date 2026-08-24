#!/usr/bin/env python
"""
Script 16: corrected global + regime-specific SHAP deliverables for the
A3/LightGBM champion (audit task, section 4/5/6), built on the fixed
hyperparameter-lookup path (see outputs/audit/champion_shap_diagnosis.md
and scripts/12_interpretability_and_sensitivity.py's
`_load_cell_hyperparameters`).

Runs independently of script 12 (which also recomputes this winner's
regime SHAP as part of its broader Best_A/Best_B/Best_Overall loop) so the
three literal deliverable files the audit brief names can be produced
without waiting on script 12's much slower target-derived-sensitivity
PSO reruns for the other winners.

Outputs:
    outputs/interpretability/champion_A3_LightGBM_global_shap.csv
    outputs/interpretability/champion_A3_LightGBM_regime_shap.csv
    outputs/interpretability/champion_A3_LightGBM_regime_rank_changes.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.core import Config, set_seed
from src.data_io import load_processed_data
from src.features import FeatureRegistry
from src.splits import slice_configurations_to_common_period
from src.models import ModelRegistry
from src.interpretability import compute_shap_values, compute_regime_shap_with_integrity

RUN_ID = 'final_rerun_2026'
OUT_DIR = Path('outputs/interpretability')

# Finer-grained predictor families than the registry's 4 coarse tags
# (grid_exogenous/mobility_exogenous/owid_exogenous/raw_exogenous), matching
# the 6 families the audit brief asks for. COVID_Deaths does not fit any of
# the 6 requested families (macroeconomic/weather/mobility/grid carbon
# intensity/grid generation mix/annual transition) - it is a shock/health
# indicator, not macro or weather - so it gets its own explicit 'shock_health'
# bucket rather than being silently forced into an ill-fitting one.
FAMILY_MAP = {
    'GDP': 'macroeconomic', 'Population': 'macroeconomic',
    'Air_Temp': 'weather', 'Rainfall': 'weather',
    'COVID_Deaths': 'shock_health',
    'Mobility_Retail_Recreation': 'mobility', 'Mobility_Grocery_Pharmacy': 'mobility',
    'Mobility_Parks': 'mobility', 'Mobility_Transit_Stations': 'mobility',
    'Mobility_Workplaces': 'mobility', 'Mobility_Residential': 'mobility',
    'Grid_CI_mean': 'grid_carbon_intensity', 'Grid_CI_p90': 'grid_carbon_intensity',
    'Grid_CI_std': 'grid_carbon_intensity', 'Grid_CI_high_share': 'grid_carbon_intensity',
    'Grid_CI_low_share': 'grid_carbon_intensity',
    'Grid_renewable_share': 'grid_generation_mix', 'Grid_low_carbon_share': 'grid_generation_mix',
    'Grid_fossil_share': 'grid_generation_mix', 'Grid_gas_share': 'grid_generation_mix',
    'Grid_wind_share': 'grid_generation_mix',
    'OWID_RenewableShare_L1Y': 'annual_transition', 'OWID_LowCarbonShare_L1Y': 'annual_transition',
}


def load_cell_hyperparameters(run_dir, configuration, fs_option, model_name):
    rows = []
    for stream in ('A', 'B'):
        prov_path = run_dir / 'fold_predictions' / f'stream_{stream}' / 'provenance.json'
        if not prov_path.exists():
            continue
        with open(prov_path) as f:
            prov = json.load(f)
        rows.extend([
            r for r in prov
            if r['configuration'] == configuration and r['fs_option'] == fs_option and r['model'] == model_name
        ])
    if not rows:
        raise ValueError(f"No provenance found for {configuration}/{fs_option}/{model_name}")
    rows.sort(key=lambda r: r['train_end'], reverse=True)
    return rows[0]['hyperparameters']


def get_regime_periods(config):
    covid_start = pd.Period(config.features.covid_start).start_time
    covid_end = pd.Period(config.features.covid_end).end_time
    return {
        'pre_covid': (None, str(covid_start.date())),
        'covid': (str(covid_start.date()), str(covid_end.date())),
        'post_covid': (str(covid_end.date()), None),
    }


def add_stats_columns(df, sv, feature_names):
    """Attach mean/median/signed-mean |SHAP|, rank, share_of_total, family."""
    mean_abs = np.mean(np.abs(sv), axis=0)
    median_abs = np.median(np.abs(sv), axis=0)
    signed_mean = np.mean(sv, axis=0)
    total = mean_abs.sum()
    out = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs,
        'median_abs_shap': median_abs,
        'signed_mean_shap': signed_mean,
        'share_of_total_attribution': mean_abs / total if total > 0 else np.nan,
        'family': [FAMILY_MAP.get(f, 'other') for f in feature_names],
    })
    out['rank'] = out['mean_abs_shap'].rank(ascending=False, method='min').astype(int)
    return out.sort_values('rank').reset_index(drop=True)


def main():
    config = Config()
    config.run_id = RUN_ID
    set_seed(config.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table7 = pd.read_csv(config.run_dir / 'pareto_mcda' / 'final' / 'table7_stream_winners_and_final.csv')
    champ = table7[table7['stream_winner'] == 'Best_A'].iloc[0]
    configuration, fs_option, model_name = champ['configuration'], champ.get('fs_option', 'all_features'), champ['model']
    print(f"Champion: {configuration}/{fs_option}/{model_name} (WMASE={champ['weighted_mase']:.4f})")
    assert configuration == 'A3' and model_name == 'lightgbm', \
        f"Expected A3/lightgbm champion, found {configuration}/{model_name} - check table7 before trusting file names below."

    processed_dir = Path('data/processed')
    X_by_config = {tag: load_processed_data(processed_dir / f'X_{tag}') for tag in ('A1', 'A2', 'A3', 'A4')}
    y = load_processed_data(processed_dir / 'y')['target']
    X_by_config = slice_configurations_to_common_period(X_by_config, y, reference_tag='A3')
    X_config = X_by_config[configuration]
    y_slice = y.loc[X_config.index]
    feature_names = list(X_config.columns)

    params = load_cell_hyperparameters(config.run_dir, configuration, fs_option, model_name)
    print(f"Using tuned hyperparameters: {params}")
    model = ModelRegistry.create(model_name, params)
    model.fit(X_config, y_slice)
    preds = model.predict(X_config)
    pred_std = float(np.std(preds))
    print(f"Full-sample prediction std: {pred_std:.6f} "
          f"({'OK - not constant' if pred_std > 1e-6 else 'DEGENERATE - constant prediction, aborting'})")
    if pred_std <= 1e-6:
        raise RuntimeError(
            "Champion refit collapsed to a constant prediction even with tuned hyperparameters - "
            "this must not be silently reported as valid SHAP. Investigate before proceeding."
        )

    # ---- Global SHAP (full common-period sample) ----
    shap_values, explainer = compute_shap_values(model, X_config, 'tree')
    sv = np.asarray(shap_values)
    assert sv.shape == (len(X_config), len(feature_names)), f"Unexpected SHAP shape {sv.shape}"
    assert not np.allclose(sv, 0.0), "Global SHAP is all-zero - integrity failure, aborting."

    global_df = add_stats_columns(sv, sv, feature_names)
    global_df.insert(0, 'n_observations', len(X_config))
    global_df.to_csv(OUT_DIR / 'champion_A3_LightGBM_global_shap.csv', index=False)
    print(f"Wrote {OUT_DIR / 'champion_A3_LightGBM_global_shap.csv'} ({len(global_df)} features)")
    print("Top 5 global predictors:")
    print(global_df.head(5)[['feature', 'mean_abs_shap', 'share_of_total_attribution', 'family']].to_string(index=False))

    # ---- Regime SHAP + integrity ----
    regime_periods = get_regime_periods(config)
    regime_results, checks_df = compute_regime_shap_with_integrity(
        model, X_config, regime_periods, 'tree', 'Best_A_champion_A3_LightGBM', model_name
    )
    if not checks_df['passed'].all():
        print(checks_df[~checks_df['passed']])
        raise RuntimeError("Regime SHAP integrity check failed - see printed rows above.")

    # Recompute raw shap per regime (compute_regime_shap_with_integrity only
    # returns importance summaries) so we can attach the full stats columns.
    regime_rows = []
    raw_by_regime = {}
    for regime_name, (start, end) in regime_periods.items():
        mask = pd.Series(True, index=X_config.index)
        if start is not None:
            mask &= (X_config.index >= start)
        if end is not None:
            mask &= (X_config.index < end)
        X_regime = X_config[mask]
        if len(X_regime) < 5:
            continue
        sv_r, _ = compute_shap_values(model, X_regime, 'tree')
        sv_r = np.asarray(sv_r)
        raw_by_regime[regime_name] = sv_r
        stats = add_stats_columns(sv_r, sv_r, feature_names)
        stats.insert(0, 'regime', regime_name)
        stats.insert(1, 'n_observations', len(X_regime))
        regime_rows.append(stats)

    regime_df = pd.concat(regime_rows, ignore_index=True)
    regime_df.to_csv(OUT_DIR / 'champion_A3_LightGBM_regime_shap.csv', index=False)
    print(f"Wrote {OUT_DIR / 'champion_A3_LightGBM_regime_shap.csv'} ({len(regime_df)} rows)")

    # ---- Rank-change / correlation / overlap across regimes ----
    regime_names = list(raw_by_regime.keys())
    rank_by_regime = {
        r: regime_df[regime_df['regime'] == r].set_index('feature')['rank'] for r in regime_names
    }

    corr_rows = []
    rank_change_rows = []
    top5_overlap_rows = []
    for i in range(len(regime_names)):
        for j in range(i + 1, len(regime_names)):
            a, b = regime_names[i], regime_names[j]
            ranks_a, ranks_b = rank_by_regime[a].align(rank_by_regime[b])
            rho, pval = spearmanr(ranks_a, ranks_b)
            corr_rows.append({'regime_a': a, 'regime_b': b, 'spearman_rho': rho, 'p_value': pval,
                               'n_features': len(ranks_a)})

            top5_a = set(rank_by_regime[a][rank_by_regime[a] <= 5].index)
            top5_b = set(rank_by_regime[b][rank_by_regime[b] <= 5].index)
            overlap = top5_a & top5_b
            top5_overlap_rows.append({
                'regime_a': a, 'regime_b': b, 'top5_a': ','.join(sorted(top5_a)),
                'top5_b': ','.join(sorted(top5_b)), 'overlap_count': len(overlap),
                'overlap_features': ','.join(sorted(overlap)),
            })

    for feature in feature_names:
        row = {'feature': feature, 'family': FAMILY_MAP.get(feature, 'other')}
        for r in regime_names:
            row[f'rank_{r}'] = int(rank_by_regime[r].get(feature, np.nan))
        ranks_present = [row[f'rank_{r}'] for r in regime_names]
        row['max_rank_change'] = max(ranks_present) - min(ranks_present)
        rank_change_rows.append(row)
    rank_change_df = pd.DataFrame(rank_change_rows).sort_values('max_rank_change', ascending=False)

    # Grid vs non-grid importance share by regime (grid = carbon intensity + generation mix)
    grid_share_rows = []
    for r in regime_names:
        sub = regime_df[regime_df['regime'] == r]
        total = sub['mean_abs_shap'].sum()
        grid = sub[sub['family'].isin(['grid_carbon_intensity', 'grid_generation_mix'])]['mean_abs_shap'].sum()
        grid_share_rows.append({
            'regime': r, 'grid_feature_share': grid / total if total else np.nan,
            'non_grid_feature_share': (total - grid) / total if total else np.nan,
        })
    grid_share_df = pd.DataFrame(grid_share_rows)

    with open(OUT_DIR / 'champion_A3_LightGBM_regime_rank_changes.csv', 'w') as f:
        f.write("# spearman_rank_correlation_between_regimes\n")
        pd.DataFrame(corr_rows).to_csv(f, index=False)
        f.write("\n# top5_overlap_between_regimes\n")
        pd.DataFrame(top5_overlap_rows).to_csv(f, index=False)
        f.write("\n# per_feature_rank_by_regime_and_max_change\n")
        rank_change_df.to_csv(f, index=False)
        f.write("\n# grid_vs_non_grid_importance_share_by_regime\n")
        grid_share_df.to_csv(f, index=False)

    print(f"Wrote {OUT_DIR / 'champion_A3_LightGBM_regime_rank_changes.csv'}")
    print("\nSpearman rank correlation between regimes:")
    print(pd.DataFrame(corr_rows).to_string(index=False))
    print("\nGrid vs non-grid feature importance share by regime:")
    print(grid_share_df.to_string(index=False))


if __name__ == '__main__':
    main()
