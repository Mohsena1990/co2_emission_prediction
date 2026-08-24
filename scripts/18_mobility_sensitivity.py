#!/usr/bin/env python
"""
Script 18: mobility-treatment robustness test (audit task section 7).

Tests whether the A3 electricity-grid advantage survives changing how the
6 Google mobility predictors are encoded outside their real reporting
window (2020-02-15 to 2022-10-15 -> quarterly "included" 2020Q2-2022Q3 per
data/processed/mobility_quality_report.csv's inclusion_status column).

Scenarios:
    M0 - current primary convention (neutral-zero outside the reporting
         window). Not rerun here - reused directly from the existing
         validated outputs/runs/my_run/metrics/stream_A/metrics.csv, since
         that IS this exact scenario already correctly computed.
    M1 - remove all 6 Mobility_* columns entirely from A1-A4.
    M2 - keep mobility values (same neutral-zero fill as M0) but add an
         explicit Mobility_Available indicator (1 inside the real Google
         reporting window, 0 outside), so the model can distinguish "zero
         because genuinely observed" from "zero because unavailable".
    M3 - same feature encoding as M2, but the OUTER evaluation is
         restricted to forecast origins whose target falls inside/near the
         real mobility-covered window (2020Q1-2023Q1) rather than
         evaluating forecast origins where mobility information does not
         exist. This is the leakage-safe operationalisation of "use
         mobility only where legitimately available" that this pipeline's
         single-fixed-X-per-run architecture supports without introducing
         per-fold imputation (which would require deeper surgery to
         run_configuration_model). Labelled exploratory/sensitivity-only
         given the resulting sample is a subset of an already-small
         28-quarter common period.

Runs Stream A (4 configs x 5 models, no FS) - the "principal models
necessary to establish robustness" per spec, and here all 5 since
computationally retained per user instruction - for M1/M2/M3, reusing
`run_stream_a` from scripts/10_run_experiment_grid.py directly (same
common-period matrices, same eval_plan/split_config) so every scenario is
evaluated on identical forecast origins except where M3 deliberately
restricts them (documented above).

Outputs:
    outputs/robustness/mobility_sensitivity_results.csv
    outputs/figures/pdf/sensitivity/fig_mobility_robustness.pdf
"""
import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.core import Config, create_run_directories, setup_logging, get_logger, set_seed
from src.splits import load_cv_plan, slice_configurations_to_common_period, assert_cv_plan_fits_matrix
from src.splits.panels import panel2_split_config
from src.splits.walk_forward import CVPlan
from src.reporting.pdf_figures import configure_matplotlib_for_pdf

_spec = importlib.util.spec_from_file_location(
    "run_experiment_grid_10", Path(__file__).parent / "10_run_experiment_grid.py"
)
run_grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_grid)

CONFIGURATIONS = ['A1', 'A2', 'A3', 'A4']
MOBILITY_COLS = [
    'Mobility_Retail_Recreation', 'Mobility_Grocery_Pharmacy', 'Mobility_Parks',
    'Mobility_Transit_Stations', 'Mobility_Workplaces', 'Mobility_Residential',
]
RUN_ID = 'final_rerun_2026_mobility'
M3_WINDOW = (pd.Timestamp('2020-01-01'), pd.Timestamp('2023-01-01'))


def load_availability_series():
    q = pd.read_csv('data/processed/mobility_quality_report.csv', parse_dates=['quarter']).set_index('quarter')
    return (q['inclusion_status'] == 'included').astype(float)


def build_scenario_matrices(X_by_config, availability, scenario):
    out = {}
    for tag, X in X_by_config.items():
        X = X.copy()
        if scenario == 'M1':
            X = X.drop(columns=[c for c in MOBILITY_COLS if c in X.columns])
        elif scenario in ('M2', 'M3'):
            avail_aligned = availability.reindex(X.index).fillna(0.0)
            X['Mobility_Available'] = avail_aligned.values
        out[tag] = X
    return out


def summarize_scenario(summaries_df, scenario_label):
    """Per-configuration best-model view (matches the repo's own Table 4
    convention) plus overall-rank-of-A3 view."""
    rows = []
    df = summaries_df[summaries_df['configuration'] != 'naive_reference'].copy()
    df['overall_rank'] = df['weighted_mase'].rank(method='min')
    best_per_config = df.loc[df.groupby('configuration')['weighted_mase'].idxmin()].set_index('configuration')
    config_rank = best_per_config['weighted_mase'].rank(method='min')

    a3_row = best_per_config.loc['A3'] if 'A3' in best_per_config.index else None
    for tag in CONFIGURATIONS:
        if tag not in best_per_config.index:
            continue
        row = best_per_config.loc[tag]
        rows.append({
            'mobility_scenario': scenario_label,
            'configuration': tag,
            'best_model': row['model'],
            'weighted_mase': row['weighted_mase'],
            'mase_h1': row.get('mase_h1'), 'mase_h2': row.get('mase_h2'), 'mase_h4': row.get('mase_h4'),
            'worst_horizon_mase': row.get('worst_horizon_mase'),
            'error_std': row.get('error_std'),
            'config_rank_among_A1_A4': int(config_rank[tag]),
            'a3_minus_this_config_weighted_mase': (
                None if a3_row is None or tag == 'A3' else float(a3_row['weighted_mase'] - row['weighted_mase'])
            ),
            'n_cells_in_scenario': len(df),
        })
    return pd.DataFrame(rows)


def main():
    config = Config()
    config.run_id = RUN_ID
    # Reduced PSO budget for this sensitivity sweep only (champion selection
    # itself already ran, and stands, under the full 20-particle/30-iteration
    # budget) - full budget across 60 Stream-A cells (3 scenarios x 4 configs
    # x 5 models) measured at well over a day of wall-clock time on this
    # machine (CatBoost/RandomForest PSO cells alone take 10-45 minutes each
    # at full budget); this cuts runtime roughly 3-6x while still tuning
    # every model for every scenario, per the user's explicit choice to keep
    # all 5 models but shrink the search budget rather than drop model classes.
    config.optimization.n_particles = 8
    config.optimization.n_iterations = 10
    set_seed(config.seed)
    dirs = create_run_directories(config)
    logger = setup_logging(log_dir=dirs['logs'], run_id=config.run_id)
    logger.info("=" * 60)
    logger.info("Script 18: mobility-treatment robustness (M0-M3)")
    logger.info("=" * 60)

    processed_dir = Path('data/processed')
    X_by_config, y = run_grid._load_matrices(processed_dir)
    X_by_config = slice_configurations_to_common_period(X_by_config, y, reference_tag='A3')
    eval_plan = load_cv_plan(processed_dir / 'panel2_cv_plan.pkl')
    split_config = panel2_split_config(config.splits)
    models = config.model.models
    logger.info(f"Models: {models}, base eval_plan folds: {len(eval_plan.folds)}")

    availability = load_availability_series()

    m3_folds = [f for f in eval_plan.folds if M3_WINDOW[0] <= f.test_start <= M3_WINDOW[1]]
    m3_plan = CVPlan(folds=m3_folds, config=eval_plan.config, y_by_horizon=eval_plan.y_by_horizon)
    logger.info(f"M3 restricted eval plan: {len(m3_folds)}/{len(eval_plan.folds)} outer folds retained")

    scenario_plans = {'M1': eval_plan, 'M2': eval_plan, 'M3': m3_plan}

    all_scenario_rows = []

    # ---- M0: reuse the existing validated Stream A run ----
    m0_path = Path('outputs/runs/my_run/metrics/stream_A/metrics.csv')
    m0_df = pd.read_csv(m0_path)
    all_scenario_rows.append(summarize_scenario(m0_df, 'M0'))
    logger.info(f"M0: reused {len(m0_df)} rows from {m0_path}")

    only_scenarios = set(sys.argv[1:]) or {'M1', 'M2', 'M3'}
    for scenario in ('M1', 'M2', 'M3'):
        if scenario not in only_scenarios:
            cached_path = dirs['metrics'] / f'mobility_{scenario}_metrics.csv'
            if cached_path.exists():
                all_scenario_rows.append(summarize_scenario(pd.read_csv(cached_path), scenario))
                logger.info(f"{scenario}: reused cached {cached_path} (not in {only_scenarios})")
            continue
        t_scenario = time.time()
        plan = scenario_plans[scenario]
        if len(plan.folds) == 0:
            logger.warning(f"{scenario}: 0 folds in eval plan after restriction - skipping entirely")
            continue
        X_scenario = build_scenario_matrices(X_by_config, availability, scenario)
        for tag in CONFIGURATIONS:
            assert_cv_plan_fits_matrix(plan, X_scenario[tag], tag=tag)

        logger.info(f"--- Scenario {scenario}: {len(plan.folds)} outer folds, "
                    f"{len(X_scenario['A3'].columns)} features in A3 ---")
        preds, prov, summaries = run_grid.run_stream_a(
            X_scenario, y, plan, split_config, config, config.run_id, logger, models
        )
        summ_df = pd.DataFrame(summaries)
        pred_dir = dirs['root'] / 'fold_predictions' / f'mobility_{scenario}'
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(preds).to_csv(pred_dir / 'predictions.csv', index=False)
        summ_df.to_csv(dirs['metrics'] / f'mobility_{scenario}_metrics.csv', index=False)

        all_scenario_rows.append(summarize_scenario(summ_df, scenario))
        logger.info(f"Scenario {scenario} complete in {time.time() - t_scenario:.1f}s")

        # Persist incrementally so a partial run still leaves usable results.
        pd.concat(all_scenario_rows, ignore_index=True).to_csv(
            'outputs/robustness/mobility_sensitivity_results.csv', index=False
        )

    result_df = pd.concat(all_scenario_rows, ignore_index=True)
    Path('outputs/robustness').mkdir(parents=True, exist_ok=True)
    result_df.to_csv('outputs/robustness/mobility_sensitivity_results.csv', index=False)
    logger.info(f"Wrote outputs/robustness/mobility_sensitivity_results.csv ({len(result_df)} rows)")

    # ---- Figure ----
    configure_matplotlib_for_pdf()
    fig_dir = Path('outputs/figures/pdf/sensitivity')
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    scenarios_present = [s for s in ('M0', 'M1', 'M2', 'M3') if s in result_df['mobility_scenario'].unique()]
    colors = {'A1': '#8e24aa', 'A2': '#1565c0', 'A3': '#1b5e20', 'A4': '#ef6c00'}
    for tag in CONFIGURATIONS:
        sub = result_df[result_df['configuration'] == tag].set_index('mobility_scenario').reindex(scenarios_present)
        ax.plot(scenarios_present, sub['weighted_mase'], marker='o', label=tag,
                color=colors.get(tag), linewidth=2 if tag == 'A3' else 1.3,
                markersize=8 if tag == 'A3' else 5)
    ax.set_ylabel('Weighted MASE (best model per configuration)')
    ax.set_xlabel('Mobility-treatment scenario')
    ax.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig_mobility_robustness.pdf', format='pdf', bbox_inches='tight',
                metadata={'Title': 'Mobility treatment robustness', 'Creator': 'Q-DECEM reproducible pipeline'})
    png_dir = Path('outputs/figures/png/sensitivity')
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / 'fig_mobility_robustness.png', format='png', dpi=600, bbox_inches='tight')
    logger.info(f"Saved {fig_dir / 'fig_mobility_robustness.pdf'}")

    logger.info("=" * 60)
    logger.info("Script 18 complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
