#!/usr/bin/env python
"""
Script 23: surface per-horizon statistical significance alongside the
pooled result already reported in manuscript Table 12 / Section 4.6.

scripts/20_statistical_robustness.py already computes per-horizon (H1,
H2, H4) AND pooled Wilcoxon signed-rank tests for A3 vs A1/A2/A4/seasonal-
naive, but the manuscript text and tables only surface the pooled n=27
number. Pooling three horizons whose forecast origins are not independent
(overlapping training windows, shared underlying series) is a real
methodological soft spot - showing the per-horizon breakdown costs
nothing and lets a reader see whether the pooled significance is driven
by one horizon or is broadly consistent.

Outputs:
    outputs/tables/table13_per_horizon_significance.csv
    outputs/figures/pdf/sensitivity/fig_per_horizon_significance.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.reporting.pdf_figures import configure_matplotlib_for_pdf

SRC = Path('outputs/robustness/grid_signal_paired_tests.csv')
TABLE_OUT = Path('outputs/tables/table13_per_horizon_significance.csv')
FIG_OUT = Path('outputs/figures/pdf/sensitivity/fig_per_horizon_significance.pdf')

COMPARISON_LABELS = {
    'A3_vs_A1': 'A3 vs A1 (baseline)',
    'A3_vs_A2': 'A3 vs A2 (engineered-only)',
    'A3_vs_A4': 'A3 vs A4 (full fusion)',
    'A3_vs_naive_lag4': 'A3 vs seasonal-naive',
}
HORIZON_LABELS = {1: 'H1', 2: 'H2', 4: 'H4', 'pooled': 'Pooled'}


def main():
    df = pd.read_csv(SRC)
    df['horizon_label'] = df['horizon'].apply(
        lambda h: HORIZON_LABELS.get(h, HORIZON_LABELS.get(str(h), str(h)))
    )
    df['comparison_label'] = df['comparison'].map(COMPARISON_LABELS)
    df['significant'] = df['significant_after_correction_alpha_0.05']

    out = df[['comparison_label', 'horizon_label', 'n_pairs', 'pct_pairs_a_better',
              'mean_abs_error_diff', 'wilcoxon_p', 'wilcoxon_p_holm_corrected', 'significant',
              'dm_statistic', 'dm_p_value', 'dm_p_holm_corrected']].copy()
    out = out.rename(columns={
        'comparison_label': 'comparison', 'horizon_label': 'horizon',
        'pct_pairs_a_better': 'pct_A3_better', 'mean_abs_error_diff': 'mean_abs_error_diff_tonnes',
        'wilcoxon_p_holm_corrected': 'holm_corrected_p',
    })
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLE_OUT, index=False)
    print(f"Wrote {TABLE_OUT} ({len(out)} rows)")
    print(out.to_string(index=False))

    # ---- Figure: per-horizon % pairs A3 better, with pooled + significance marker ----
    configure_matplotlib_for_pdf()
    comparisons = list(COMPARISON_LABELS.values())
    horizons = ['H1', 'H2', 'H4', 'Pooled']
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y_positions = {}
    y = 0
    yticks, yticklabels = [], []
    for comp in comparisons:
        sub = out[out['comparison'] == comp].set_index('horizon').reindex(horizons)
        for h in horizons:
            row = sub.loc[h]
            marker = 'o' if h != 'Pooled' else 's'
            face = '#1b5e20' if bool(row['significant']) else '#b0bec5'
            ax.scatter(row['pct_A3_better'], y, marker=marker, s=70 if h == 'Pooled' else 45,
                       color=face, edgecolor='#263238', linewidth=0.6, zorder=3)
            yticks.append(y)
            yticklabels.append(f"{comp}  ({h})" if h == 'Pooled' else h)
            y -= 1
        y -= 0.6
    ax.axvline(50, color='#9e9e9e', linestyle=':', linewidth=1)
    ax.set_xlim(30, 100)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=7)
    ax.set_xlabel('% of paired forecast origins where A3 has lower absolute error')
    ax.spines[['top', 'right']].set_visible(False)
    handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#1b5e20', markeredgecolor='#263238', label='Significant (Holm-corrected, pooled or per-horizon)', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#b0bec5', markeredgecolor='#263238', label='Not significant at this sample size', markersize=8),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=6.5, frameon=False)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, format='pdf', bbox_inches='tight',
                metadata={'Title': 'Per-horizon statistical robustness', 'Creator': 'Q-DECEM reproducible pipeline'})
    png_out = Path('outputs/figures/png/sensitivity/fig_per_horizon_significance.png')
    png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_out, format='png', dpi=600, bbox_inches='tight')
    print(f"Saved {FIG_OUT}")


if __name__ == '__main__':
    main()
