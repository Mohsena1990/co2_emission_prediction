#!/usr/bin/env python
"""
Script 25: what is Grid_CI_std actually a proxy for?

The champion's leading predictor is the within-quarter STANDARD DEVIATION
of half-hourly carbon intensity. "The grid was more variable this quarter"
is a description, not a mechanism. Two candidate mechanisms, both
constructible from data already cached locally (no new API calls):

  1. System *predictability* stress. The NESO Carbon Intensity API ships
     its own half-hourly `forecast` alongside `actual` in every cached
     intensity_*.json file (src/grid/aggregate.py currently only uses
     `forecast` as a fallback for missing `actual` - the forecast ERROR
     itself, actual-forecast, is not engineered as a feature anywhere in
     this pipeline). If Grid_CI_std is highly correlated with how wrong
     the operator's own real-time forecast was, that reframes "dispersion"
     from an abstract statistic into "how much the system deviated from
     its own operating expectations" - a concept NESO already tracks
     operationally.
  2. Wind volatility. Wind is the most intermittent major generation
     source; if Grid_CI_std tracks the within-quarter standard deviation
     of the wind generation share, that points at renewable intermittency
     specifically, rather than a generic "grid volatility" story.

This script computes both candidate series from the raw half-hourly cache,
correlates them with the existing Grid_CI_std (and, for context,
Grid_CI_mean/p90) over the 28-quarter common period, and runs ONE
additional single-statistic ablation cell (forecast-error std alone vs.
the baseline) so the mechanism finding is tied to a concrete predictive
result, not just a correlation coefficient.

Outputs:
    outputs/robustness/ci_std_mechanism_correlations.csv
    outputs/robustness/ci_std_mechanism_ablation.csv
    outputs/figures/pdf/sensitivity/fig_ci_std_mechanism.pdf
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.core import Config, create_run_directories, setup_logging, set_seed
from src.data_io import load_processed_data
from src.grid.aggregate import load_raw_halfhourly, _impute_intensity, HIGH_INDEX_BANDS, LOW_INDEX_BANDS
from src.splits import load_cv_plan, slice_configurations_to_common_period
from src.splits.panels import panel2_split_config
from src.reporting.pdf_figures import configure_matplotlib_for_pdf, OKABE_ITO

_spec = importlib.util.spec_from_file_location(
    "run_experiment_grid_10", Path(__file__).parent / "10_run_experiment_grid.py"
)
run_grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_grid)

RUN_ID = 'final_rerun_2026_ci_mechanism'
GRID_CACHE_DIR = Path('data/raw/grid_cache')
CI_COLS = ['Grid_CI_mean', 'Grid_CI_p90', 'Grid_CI_std', 'Grid_CI_high_share', 'Grid_CI_low_share']
GENMIX_COLS = ['Grid_renewable_share', 'Grid_low_carbon_share', 'Grid_fossil_share',
               'Grid_gas_share', 'Grid_wind_share']
OWID_COLS = ['OWID_RenewableShare_L1Y', 'OWID_LowCarbonShare_L1Y']


def build_mechanism_candidates(min_completeness: float = 0.95) -> pd.DataFrame:
    """Quarterly forecast-error and wind-volatility candidates, same quarter
    boundaries/completeness rule as src.grid.aggregate.aggregate_to_quarterly,
    so they align exactly with the already-built Grid_CI_* columns."""
    df = load_raw_halfhourly(GRID_CACHE_DIR)
    df['intensity_imputed'], _ = _impute_intensity(df)
    # Forecast error is only meaningful where `actual` was genuinely
    # reported (not the fallback-imputed value) - using the imputed series
    # here would compare forecast to itself for those rows.
    has_actual = df['intensity_actual'].notna() & df['intensity_forecast'].notna()
    df['forecast_error'] = np.nan
    df.loc[has_actual, 'forecast_error'] = (
        df.loc[has_actual, 'intensity_actual'] - df.loc[has_actual, 'intensity_forecast']
    )
    df['gen_wind'] = pd.to_numeric(df.get('gen_wind'), errors='coerce')
    df['_quarter'] = df.index.to_period('Q')

    rows = []
    for period, g in df.groupby('_quarter'):
        n_days = (period.end_time.normalize() - period.start_time.normalize()).days + 1
        expected_obs = n_days * 48
        observed_obs = int(g['intensity_imputed'].notna().sum())
        completeness = observed_obs / expected_obs if expected_obs > 0 else 0.0
        if completeness < min_completeness:
            continue
        fe = g['forecast_error'].dropna()
        rows.append({
            'quarter': period.start_time,
            'Grid_forecast_error_mean_abs': fe.abs().mean() if len(fe) else np.nan,
            'Grid_forecast_error_std': fe.std() if len(fe) else np.nan,
            'n_forecast_error_obs': len(fe),
            'Grid_wind_share_std': g['gen_wind'].std(),
        })
    return pd.DataFrame(rows).set_index('quarter').sort_index()


def main():
    config = Config()
    config.run_id = RUN_ID
    set_seed(config.seed)
    dirs = create_run_directories(config)
    logger = setup_logging(log_dir=dirs['logs'], run_id=config.run_id)
    logger.info("=" * 60)
    logger.info("Script 25: what does Grid_CI_std proxy for?")
    logger.info("=" * 60)

    mechanism = build_mechanism_candidates()
    logger.info(f"Built {len(mechanism)} quarters of mechanism candidates:\n{mechanism.describe()}")

    processed_dir = Path('data/processed')
    X_A3 = load_processed_data(processed_dir / 'X_A3')
    y = load_processed_data(processed_dir / 'y')['target']
    X_by_config = slice_configurations_to_common_period({'A3': X_A3}, y, reference_tag='A3')
    X_A3 = X_by_config['A3']
    X_A3.index = pd.PeriodIndex(X_A3.index, freq='Q').start_time

    merged = X_A3[CI_COLS].join(mechanism, how='inner')
    logger.info(f"Merged {len(merged)} common-period quarters for correlation analysis")

    corr_pearson = merged.corr(method='pearson')
    corr_spearman = merged.corr(method='spearman')
    corr_out = pd.DataFrame({
        'pearson_vs_Grid_CI_std': corr_pearson['Grid_CI_std'],
        'spearman_vs_Grid_CI_std': corr_spearman['Grid_CI_std'],
    }).drop(index='Grid_CI_std')
    Path('outputs/robustness').mkdir(parents=True, exist_ok=True)
    corr_out.to_csv('outputs/robustness/ci_std_mechanism_correlations.csv')
    logger.info(f"Correlations with Grid_CI_std:\n{corr_out.to_string()}")

    # ---- One ablation cell: forecast-error std alone vs. baseline ----
    y_slice = y.loc[y.index.isin(X_A3.index)]
    baseline_cols = [c for c in X_A3.columns if c not in CI_COLS + GENMIX_COLS + OWID_COLS]
    X_baseline = X_A3[baseline_cols].join(mechanism[['Grid_forecast_error_std']], how='inner')
    y_aligned = y_slice.loc[X_baseline.index]

    eval_plan = load_cv_plan(processed_dir / 'panel2_cv_plan.pkl')
    split_config = panel2_split_config(config.splits)
    config.optimization.n_particles = 8
    config.optimization.n_iterations = 10

    ablation_rows = []
    for model_name in config.model.models:
        cell = run_grid.ExperimentCell(
            configuration='A3_forecast_error_std_only', panel=run_grid.PRIMARY_PERIOD_PANEL,
            fs_option='all_features', model=model_name, seed=config.seed
        )
        try:
            result = run_grid.run_configuration_model(
                X_baseline, y_aligned, eval_plan, cell, config, config.run_id, outer_split_config=split_config
            )
        except Exception as e:
            logger.error(f"forecast_error_std_only/{model_name} failed: {e}")
            continue
        summary = result['summary']
        summary['variant'] = 'A3_forecast_error_std_only'
        summary['n_features'] = len(X_baseline.columns)
        ablation_rows.append(summary)
        logger.info(f"  {model_name} -> weighted_mase={summary.get('weighted_mase')}")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv('outputs/robustness/ci_std_mechanism_ablation.csv', index=False)
    logger.info(f"Wrote outputs/robustness/ci_std_mechanism_ablation.csv ({len(ablation_df)} rows)")

    # ---- Figure: correlation bars + ablation comparison ----
    configure_matplotlib_for_pdf()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))

    ax = axes[0]
    plot_corr = corr_out.loc[['Grid_forecast_error_mean_abs', 'Grid_forecast_error_std', 'Grid_wind_share_std']]
    ax.barh(plot_corr.index.str.replace('Grid_', '').str.replace('_', ' '),
            plot_corr['pearson_vs_Grid_CI_std'], color=OKABE_ITO[4])
    ax.axvline(0, color='#616161', linewidth=0.8)
    ax.set_xlabel('Pearson correlation with Grid_CI_std', fontsize=7.5)
    ax.set_title('(a) What correlates with dispersion?', fontsize=8.5)
    ax.set_xlim(-1, 1)
    for i, v in enumerate(plot_corr['pearson_vs_Grid_CI_std']):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:.2f}", va='center',
                 ha='left' if v >= 0 else 'right', fontsize=7)
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    if len(ablation_df):
        best_fe = ablation_df.loc[ablation_df['weighted_mase'].idxmin()]
        ci_decomp_path = Path('outputs/robustness/ci_decomposition_results.csv')
        compare_rows = {'Forecast-error\nstd only': best_fe['weighted_mase']}
        if ci_decomp_path.exists():
            decomp = pd.read_csv(ci_decomp_path)
            for v, lbl in [('A3_no_grid', 'No grid data'), ('A3_CI_mean_only', 'Mean only'),
                           ('A3_CI_std_only', 'Dispersion\n(std) only')]:
                sub = decomp[decomp['variant'] == v]
                if len(sub):
                    compare_rows[lbl] = sub['weighted_mase'].min()
        order = ['No grid data', 'Forecast-error\nstd only', 'Dispersion\n(std) only', 'Mean only']
        vals = [compare_rows[k] for k in order if k in compare_rows]
        labels = [k for k in order if k in compare_rows]
        colors = ['#9e9e9e' if l == 'No grid data' else OKABE_ITO[4] for l in labels]
        ax.barh(labels, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=7)
        ax.set_xlabel('Weighted MASE (lower is better)', fontsize=7.5)
    ax.set_title('(b) Does forecast-error std forecast as well?', fontsize=8.5)
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Investigating the mechanism behind Grid_CI_std', fontsize=9.5, y=1.03)
    fig.tight_layout()
    fig_dir = Path('outputs/figures/pdf/sensitivity')
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / 'fig_ci_std_mechanism.pdf', format='pdf', bbox_inches='tight',
                metadata={'Title': 'CI-std mechanism investigation', 'Creator': 'Q-DECEM reproducible pipeline'})
    png_dir = Path('outputs/figures/png/sensitivity')
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / 'fig_ci_std_mechanism.png', format='png', dpi=600, bbox_inches='tight')
    logger.info(f"Saved {fig_dir / 'fig_ci_std_mechanism.pdf'}")

    logger.info("=" * 60)
    logger.info("Script 25 complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
