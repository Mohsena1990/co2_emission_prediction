#!/usr/bin/env python
"""
Script 22: carbon-intensity statistic decomposition (follow-up flagged in
FINAL_RERUN_REPORT.md section G and manuscript Discussion 5.4 as "not yet
run" - whether Grid_CI_std's dispersion signal adds information beyond the
mean, independently of the other carbon-intensity summary statistics,
rather than the CI-family-as-a-whole vs generation-mix-as-a-whole
comparison already reported in Table 12 / scripts/19_grid_ablation.py).

Starting from the same A3 baseline (11 raw+mobility columns) used by
script 19, adds each of the five Grid_CI_* statistics ALONE:

    A3_no_grid          - baseline only (11); floor, no grid information
    A3_CI_mean_only      - baseline + Grid_CI_mean (12)
    A3_CI_p90_only       - baseline + Grid_CI_p90 (12)
    A3_CI_std_only       - baseline + Grid_CI_std (12)
    A3_CI_highshare_only - baseline + Grid_CI_high_share (12)
    A3_CI_lowshare_only  - baseline + Grid_CI_low_share (12)
    A3_CI_all            - baseline + all five CI statistics (16)

Generation-mix and OWID columns are excluded throughout, so every
variant here isolates carbon-intensity information only - this is a
decomposition WITHIN the carbon-intensity family, not a repeat of
script 19's CI-vs-generation-mix comparison. Same evaluation dates,
same reduced PSO budget (n_particles=8, n_iterations=10) as scripts 18
and 19, for the same reason: this is a sensitivity/mechanism-decomposition
sweep, not the primary champion selection.

Outputs:
    outputs/robustness/ci_decomposition_results.csv
    outputs/figures/pdf/sensitivity/fig_ci_decomposition.pdf
"""
import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.core import Config, create_run_directories, setup_logging, get_logger, set_seed
from src.data_io import load_processed_data
from src.splits import load_cv_plan, slice_configurations_to_common_period
from src.splits.panels import panel2_split_config
from src.reporting.pdf_figures import configure_matplotlib_for_pdf

_spec = importlib.util.spec_from_file_location(
    "run_experiment_grid_10", Path(__file__).parent / "10_run_experiment_grid.py"
)
run_grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_grid)

RUN_ID = 'final_rerun_2026_ci_decomposition'

CI_COLS = ['Grid_CI_mean', 'Grid_CI_p90', 'Grid_CI_std', 'Grid_CI_high_share', 'Grid_CI_low_share']
GENMIX_COLS = ['Grid_renewable_share', 'Grid_low_carbon_share', 'Grid_fossil_share',
               'Grid_gas_share', 'Grid_wind_share']
OWID_COLS = ['OWID_RenewableShare_L1Y', 'OWID_LowCarbonShare_L1Y']

VARIANTS = {
    'A3_no_grid':          [],
    'A3_CI_mean_only':     ['Grid_CI_mean'],
    'A3_CI_p90_only':      ['Grid_CI_p90'],
    'A3_CI_std_only':      ['Grid_CI_std'],
    'A3_CI_highshare_only': ['Grid_CI_high_share'],
    'A3_CI_lowshare_only': ['Grid_CI_low_share'],
    'A3_CI_all':           CI_COLS,
}


def main():
    config = Config()
    config.run_id = RUN_ID
    # Reduced PSO budget for this decomposition sweep only - matches the
    # rationale already documented in scripts/18 and 19 (champion selection
    # itself is untouched; this only shrinks the search budget for the
    # 7-variant x 5-model decomposition grid).
    config.optimization.n_particles = 8
    config.optimization.n_iterations = 10
    set_seed(config.seed)
    dirs = create_run_directories(config)
    logger = setup_logging(log_dir=dirs['logs'], run_id=config.run_id)
    logger.info("=" * 60)
    logger.info("Script 22: carbon-intensity statistic decomposition")
    logger.info("=" * 60)

    processed_dir = Path('data/processed')
    X_by_config = {tag: load_processed_data(processed_dir / f'X_{tag}') for tag in ('A1', 'A2', 'A3', 'A4')}
    y = load_processed_data(processed_dir / 'y')['target']
    X_by_config = slice_configurations_to_common_period(X_by_config, y, reference_tag='A3')
    X_A3 = X_by_config['A3']
    y_slice = y.loc[X_A3.index]

    baseline_cols = [c for c in X_A3.columns if c not in CI_COLS + GENMIX_COLS + OWID_COLS]
    logger.info(f"Baseline (raw+mobility) columns ({len(baseline_cols)}): {baseline_cols}")

    eval_plan = load_cv_plan(processed_dir / 'panel2_cv_plan.pkl')
    split_config = panel2_split_config(config.splits)
    models = config.model.models
    logger.info(f"Models: {models}")

    all_summaries = []
    for variant_name, extra_cols in VARIANTS.items():
        t0 = time.time()
        cols = baseline_cols + [c for c in extra_cols if c in X_A3.columns]
        X_variant = X_A3[cols]
        logger.info(f"--- Variant {variant_name}: {len(cols)} features ---")

        for model_name in models:
            cell = run_grid.ExperimentCell(
                configuration=variant_name, panel=run_grid.PRIMARY_PERIOD_PANEL,
                fs_option='all_features', model=model_name, seed=config.seed
            )
            try:
                result = run_grid.run_configuration_model(
                    X_variant, y_slice, eval_plan, cell, config, config.run_id, outer_split_config=split_config
                )
            except Exception as e:
                logger.error(f"{variant_name}/{model_name} failed: {e}")
                continue
            summary = result['summary']
            summary['variant'] = variant_name
            summary['n_features'] = len(cols)
            all_summaries.append(summary)
            logger.info(f"  {model_name} -> weighted_mase={summary.get('weighted_mase')}")

        pd.DataFrame(all_summaries).to_csv('outputs/robustness/ci_decomposition_results.csv', index=False)
        logger.info(f"Variant {variant_name} complete in {time.time() - t0:.1f}s")

    result_df = pd.DataFrame(all_summaries)
    Path('outputs/robustness').mkdir(parents=True, exist_ok=True)
    result_df.to_csv('outputs/robustness/ci_decomposition_results.csv', index=False)
    logger.info(f"Wrote outputs/robustness/ci_decomposition_results.csv ({len(result_df)} rows)")

    # ---- Figure: best-model-per-variant WMASE, ordered as defined above ----
    configure_matplotlib_for_pdf()
    fig_dir = Path('outputs/figures/pdf/sensitivity')
    fig_dir.mkdir(parents=True, exist_ok=True)
    best_per_variant = result_df.loc[result_df.groupby('variant')['weighted_mase'].idxmin()]
    best_per_variant = best_per_variant.set_index('variant').reindex(VARIANTS.keys()).reset_index()

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    # Colour communicates the finding, not just which bar to look at: the
    # BEST standalone statistic (mean) is highlighted as the positive
    # result; the statistic that carries the largest SHAP share in the
    # full joint model (std) is flagged with a callout instead of a
    # "winner" colour, since alone it is actually *worse* than the
    # no-grid baseline - a dark highlight there would misleadingly read
    # as "this is the good one".
    def _bar_color(v):
        if v == 'A3_CI_mean_only':
            return '#1b5e20'
        if v == 'A3_no_grid':
            return '#9e9e9e'
        return '#78909c'

    colors = [_bar_color(v) for v in best_per_variant['variant']]
    ax.barh(best_per_variant['variant'][::-1], best_per_variant['weighted_mase'][::-1], color=colors[::-1])
    for i, (_, row) in enumerate(best_per_variant[::-1].iterrows()):
        ax.text(row['weighted_mase'] + 0.005, i, f"{row['weighted_mase']:.3f} ({row['model']})",
                 va='center', fontsize=7)
    no_grid_val = best_per_variant.loc[best_per_variant['variant'] == 'A3_no_grid', 'weighted_mase'].values
    if len(no_grid_val):
        ax.axvline(no_grid_val[0], color='#c62828', linestyle='--', linewidth=1)
        ax.text(no_grid_val[0], -0.85, 'no grid data', color='#c62828', fontsize=7, ha='center')
    std_row = best_per_variant[best_per_variant['variant'] == 'A3_CI_std_only']
    if len(std_row):
        std_idx = best_per_variant[::-1].reset_index(drop=True)
        pos = std_idx[std_idx['variant'] == 'A3_CI_std_only'].index[0]
        ax.annotate(
            'dispersion alone is\nWORSE than no grid data\n(despite the largest SHAP\nshare in the full model)',
            xy=(std_row['weighted_mase'].values[0], pos), xytext=(0.15, pos - 1.15),
            fontsize=6.3, color='#c62828', ha='left',
            arrowprops=dict(arrowstyle='-', color='#c62828', lw=0.7),
        )
    ax.set_xlabel('Weighted MASE (best model per variant, lower is better)')
    ax.set_xlim(0, best_per_variant['weighted_mase'].max() + 0.16)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig_ci_decomposition.pdf', format='pdf', bbox_inches='tight',
                metadata={'Title': 'Carbon-intensity statistic decomposition', 'Creator': 'Q-DECEM reproducible pipeline'})
    png_dir = Path('outputs/figures/png/sensitivity')
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / 'fig_ci_decomposition.png', format='png', dpi=600, bbox_inches='tight')
    logger.info(f"Saved {fig_dir / 'fig_ci_decomposition.pdf'}")

    logger.info("=" * 60)
    logger.info("Script 22 complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
