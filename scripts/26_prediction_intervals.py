#!/usr/bin/env python
"""
Script 26: empirical prediction intervals for the A3/LightGBM champion.

The manuscript and every downstream policy figure report point forecasts
only (a single MASE number per horizon). For the policy/monitoring use
cases in RESEARCH_OVERVIEW.md ("an early-warning signal", "a transition-
risk indicator") a point forecast is a weak decision input on its own - a
government or market user needs a RANGE ("an 80% chance next quarter's
emissions fall between X and Y"), not just a number.

With only 8-11 forecasts per horizon, anything beyond an empirical
residual-quantile interval (leave-one-out to keep each observation's own
interval honestly out-of-sample) would overstate what this sample size can
support - no quantile regression, no distributional assumption beyond
"the future residual looks like a residual we've already seen at this
horizon". This is the standard small-sample-appropriate approach, and is
reported with its own honest calibration check (empirical coverage is
itself imprecise at n=8-11 - a coverage estimate is not a guarantee).

Outputs:
    outputs/robustness/champion_prediction_intervals.csv
    outputs/robustness/champion_interval_calibration.csv
    outputs/figures/pdf/main/fig_champion_prediction_intervals.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.reporting.pdf_figures import configure_matplotlib_for_pdf, save_figure_pdf, OKABE_ITO

SRC = Path('outputs/robustness/champion_origin_diagnostics.csv')
LEVELS = [0.50, 0.80]  # nominal central-interval coverage levels


def loo_quantile_interval(errors: pd.Series, idx, level: float):
    """Leave-one-out empirical quantile interval for `errors.loc[idx]`'s own
    forecast: quantiles are computed from every OTHER observation at this
    horizon, so an observation never informs its own interval."""
    others = errors.drop(index=idx)
    alpha = (1 - level) / 2
    q_low, q_high = others.quantile(alpha), others.quantile(1 - alpha)
    return q_low, q_high


def main():
    df = pd.read_csv(SRC, parse_dates=['target_date'])
    # error = predicted - actual (verified against the source columns), so
    # actual = predicted - error; an interval on future `error` translates
    # to [predicted - q_high, predicted - q_low] for the actual value.
    interval_rows = []
    calibration_rows = []

    for h, g in df.groupby('horizon'):
        g = g.set_index(g.index)
        for level in LEVELS:
            covered = 0
            for idx in g.index:
                q_low, q_high = loo_quantile_interval(g['error'], idx, level)
                predicted = g.loc[idx, 'predicted']
                actual = g.loc[idx, 'actual']
                lower = predicted - q_high
                upper = predicted - q_low
                is_covered = lower <= actual <= upper
                covered += int(is_covered)
                interval_rows.append({
                    'horizon': h, 'target_date': g.loc[idx, 'target_date'],
                    'predicted': predicted, 'actual': actual,
                    'interval_level': level, 'lower': lower, 'upper': upper,
                    'covered': is_covered, 'half_width': (upper - lower) / 2,
                })
            calibration_rows.append({
                'horizon': h, 'nominal_level': level, 'n': len(g),
                'n_covered': covered, 'empirical_coverage': covered / len(g),
            })

    interval_df = pd.DataFrame(interval_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    Path('outputs/robustness').mkdir(parents=True, exist_ok=True)
    interval_df.to_csv('outputs/robustness/champion_prediction_intervals.csv', index=False)
    calibration_df.to_csv('outputs/robustness/champion_interval_calibration.csv', index=False)
    print("Calibration (leave-one-out empirical coverage vs. nominal level):")
    print(calibration_df.to_string(index=False))
    print()
    width_summary = interval_df.groupby(['horizon', 'interval_level'])['half_width'].mean().reset_index()
    print("Mean interval half-width (thousand tonnes CO2e):")
    print(width_summary.to_string(index=False))

    # ---- Figure: fan chart of the 80% interval per horizon over time ----
    configure_matplotlib_for_pdf()
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.2), sharex=False)
    for ax, h in zip(axes, sorted(df['horizon'].unique())):
        sub = interval_df[(interval_df['horizon'] == h) & (interval_df['interval_level'] == 0.80)].sort_values('target_date')
        ax.plot(sub['target_date'], sub['actual'], color='#263238', marker='o', markersize=3,
                linewidth=1.0, label='Actual', zorder=3)
        ax.plot(sub['target_date'], sub['predicted'], color=OKABE_ITO[4], marker='s', markersize=3,
                linewidth=1.0, label='Champion point forecast', zorder=3)
        ax.fill_between(sub['target_date'], sub['lower'], sub['upper'], color=OKABE_ITO[4],
                         alpha=0.18, label='80% empirical interval (leave-one-out)', zorder=1)
        cov = calibration_df[(calibration_df['horizon'] == h) & (calibration_df['nominal_level'] == 0.80)]['empirical_coverage'].values[0]
        ax.set_title(f"H{h}: 80% interval, empirical LOO coverage = {cov:.0%} (n={len(sub)})", fontsize=8.5)
        ax.set_ylabel('CO2e (thousand tonnes)', fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=6.5, frameon=False, loc='upper right')
    fig.suptitle('Champion (A3/LightGBM) forecasts with empirical 80% prediction intervals', fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plot_data = interval_df[interval_df['interval_level'] == 0.80].copy()
    save_figure_pdf(
        fig, Path('outputs/figures/pdf/main/fig_champion_prediction_intervals.pdf'),
        title='Champion forecasts with empirical prediction intervals',
        plot_data=plot_data,
    )
    png_dir = Path('outputs/figures/png/main')
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / 'fig_champion_prediction_intervals.png', format='png', dpi=600, bbox_inches='tight')
    print("\nSaved outputs/figures/pdf/main/fig_champion_prediction_intervals.pdf")


if __name__ == '__main__':
    main()
