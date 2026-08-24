#!/usr/bin/env python
"""
Script 21: forecast-origin-level diagnostic for the A3/LightGBM champion
(audit task section 10) - answers whether the aggregate WMASE advantage is
driven by one or two exceptional quarters, or is broadly distributed.

For each of the champion's 20 walk-forward (horizon, target_date) forecast
origins (outputs/runs/my_run/fold_predictions/stream_A/predictions.csv):
prediction, actual, error, absolute error, the seasonal-naive (naive_lag4)
error at the same date for comparison, the main grid-feature values
actually seen by the model that quarter, and each feature's SHAP
contribution AT THAT OBSERVATION.

SHAP contributions here come from the single full-sample champion refit
(same model as the global/regime SHAP deliverables - tuned hyperparameters
from scripts/16_champion_shap_deliverables.py's provenance lookup), not
from each walk-forward fold's own out-of-sample model (those use different,
fold-specific PSO-tuned parameters re-optimised per outer fold). This
diagnostic therefore describes cross-sectional feature association at each
quarter, not a re-derivation of the exact mechanics behind each historical
walk-forward forecast - stated explicitly rather than implied.

Outputs:
    outputs/robustness/champion_origin_diagnostics.csv
    outputs/figures/pdf/sensitivity/fig_origin_level_grid_gain.pdf
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.core import Config, set_seed
from src.data_io import load_processed_data
from src.splits import slice_configurations_to_common_period
from src.models import ModelRegistry
from src.interpretability import compute_shap_values
from src.reporting.pdf_figures import configure_matplotlib_for_pdf

RUN_ID = 'final_rerun_2026'
GRID_FEATURES_HEADLINE = ['Grid_CI_std', 'Grid_CI_mean', 'Grid_CI_p90']


def load_cell_hyperparameters(run_dir, configuration, fs_option, model_name):
    rows = []
    for stream in ('A', 'B'):
        prov_path = run_dir / 'fold_predictions' / f'stream_{stream}' / 'provenance.json'
        if not prov_path.exists():
            continue
        with open(prov_path) as f:
            prov = json.load(f)
        rows.extend([r for r in prov if r['configuration'] == configuration
                     and r['fs_option'] == fs_option and r['model'] == model_name])
    rows.sort(key=lambda r: r['train_end'], reverse=True)
    return rows[0]['hyperparameters']


def main():
    config = Config()
    config.run_id = RUN_ID
    set_seed(config.seed)

    processed_dir = Path('data/processed')
    X_by_config = {tag: load_processed_data(processed_dir / f'X_{tag}') for tag in ('A1', 'A2', 'A3', 'A4')}
    y = load_processed_data(processed_dir / 'y')['target']
    X_by_config = slice_configurations_to_common_period(X_by_config, y, reference_tag='A3')
    X_A3 = X_by_config['A3']
    y_slice = y.loc[X_A3.index]

    params = load_cell_hyperparameters(config.run_dir, 'A3', 'all_features', 'lightgbm')
    model = ModelRegistry.create('lightgbm', params)
    model.fit(X_A3, y_slice)
    shap_values, _ = compute_shap_values(model, X_A3, 'tree')
    shap_df = pd.DataFrame(np.asarray(shap_values), index=X_A3.index, columns=X_A3.columns)

    preds = pd.read_csv('outputs/runs/my_run/fold_predictions/stream_A/predictions.csv')
    preds['target_date'] = pd.to_datetime(preds['target_date'], format='mixed')
    champ = preds[(preds.configuration == 'A3') & (preds.model == 'lightgbm')].copy()
    naive = preds[(preds.configuration == 'naive_reference') & (preds.model == 'naive_lag4')].copy()
    naive_by_key = naive.set_index(['horizon', 'target_date'])[['actual', 'predicted']]

    rows = []
    for _, r in champ.iterrows():
        key = (r['horizon'], r['target_date'])
        naive_row = naive_by_key.loc[key] if key in naive_by_key.index else None
        naive_error = (naive_row['actual'] - naive_row['predicted']) if naive_row is not None else np.nan

        row = {
            'horizon': r['horizon'], 'fold_id': r['fold_id'], 'target_date': r['target_date'].date(),
            'actual': r['actual'], 'predicted': r['predicted'],
            'error': r['predicted'] - r['actual'], 'abs_error': abs(r['predicted'] - r['actual']),
            'seasonal_naive_error': naive_error,
            'abs_seasonal_naive_error': abs(naive_error) if pd.notna(naive_error) else np.nan,
            'grid_advantage_abs_error_reduction': (
                abs(naive_error) - abs(r['predicted'] - r['actual']) if pd.notna(naive_error) else np.nan
            ),
        }
        if r['target_date'] in X_A3.index:
            for feat in GRID_FEATURES_HEADLINE:
                row[feat] = X_A3.loc[r['target_date'], feat]
            sv_row = shap_df.loc[r['target_date']].abs().sort_values(ascending=False)
            for rank, (feat, val) in enumerate(sv_row.head(3).items(), start=1):
                row[f'top{rank}_shap_feature'] = feat
                row[f'top{rank}_shap_abs_value'] = val
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values('target_date')
    Path('outputs/robustness').mkdir(parents=True, exist_ok=True)
    result_df.to_csv('outputs/robustness/champion_origin_diagnostics.csv', index=False)
    print(f"Wrote outputs/robustness/champion_origin_diagnostics.csv ({len(result_df)} rows)")
    print(result_df[['target_date', 'horizon', 'abs_error', 'abs_seasonal_naive_error',
                      'grid_advantage_abs_error_reduction', 'top1_shap_feature']].to_string(index=False))

    # ---- Figure: chronological grid-advantage diagnostic ----
    configure_matplotlib_for_pdf()
    fig_dir = Path('outputs/figures/pdf/sensitivity')
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, height_ratios=[1.3, 1])

    for h, marker in zip([1, 2, 4], ['o', 's', '^']):
        sub = result_df[result_df['horizon'] == h]
        ax1.plot(sub['target_date'], sub['abs_error'], marker=marker, label=f'A3/LightGBM (H{h})',
                  color='#1b5e20', alpha=0.5 + 0.15 * (h == 1))
        ax1.plot(sub['target_date'], sub['abs_seasonal_naive_error'], marker=marker, linestyle='--',
                  label=f'Seasonal-naive (H{h})', color='#c62828', alpha=0.4)
    ax1.set_ylabel('Absolute error (CO2e, thousand tonnes)')
    ax1.legend(fontsize=6.5, ncol=3, loc='upper left')
    ax1.spines[['top', 'right']].set_visible(False)

    offsets = {1: -18, 2: 0, 4: 18}
    for h in (1, 2, 4):
        sub = result_df[result_df['horizon'] == h]
        dates_shifted = pd.to_datetime(sub['target_date']) + pd.to_timedelta(offsets[h], unit='D')
        colors = ['#2e7d32' if v > 0 else '#c62828' for v in sub['grid_advantage_abs_error_reduction']]
        ax2.bar(dates_shifted, sub['grid_advantage_abs_error_reduction'], width=16, color=colors)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel('Error reduction vs.\nseasonal-naive')
    ax2.set_xlabel('Forecast target quarter')
    ax2.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig_origin_level_grid_gain.pdf', format='pdf', bbox_inches='tight',
                metadata={'Title': 'Origin-level grid gain', 'Creator': 'Q-DECEM reproducible pipeline'})
    png_dir = Path('outputs/figures/png/sensitivity')
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / 'fig_origin_level_grid_gain.png', format='png', dpi=600, bbox_inches='tight')
    print(f"Saved {fig_dir / 'fig_origin_level_grid_gain.pdf'}")


if __name__ == '__main__':
    main()
