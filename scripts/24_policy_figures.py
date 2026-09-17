#!/usr/bin/env python
"""
Script 24: manuscript-ready figures for the policy/managerial-implications
section (spec-compliant: vector PDF, embedded fonts, companion plot-data
CSV, via src.reporting.pdf_figures.save_figure_pdf - same convention as
every other numbered figure in outputs/figures/pdf).

Two figures, both built only from already-audited outputs already on disk
(outputs/robustness/grid_ablation_results.csv,
outputs/robustness/ci_decomposition_results.csv):

    fig_policy_capacity_vs_flexibility.pdf
        Two panels. (a) family-level ablation (baseline / carbon-intensity
        family / generation-mix family / full A3) - the "capacity vs
        flexibility" framing. (b) within-family single-statistic
        decomposition - reports HONESTLY that Grid_CI_mean alone
        outperforms Grid_CI_std alone (0.598 vs 0.662), which qualifies
        (does not overturn) the manuscript's SHAP-attribution claim that
        dispersion carries the largest attribution SHARE inside the full
        joint model: TreeSHAP reports marginal contribution given the
        other features present, not standalone predictive power. Both
        panels are captioned to make that distinction explicit rather
        than letting the figure imply "dispersion always wins".

    fig_policy_stakeholder_map.pdf
        Static vector redraw of the technical finding -> policy/economic
        lever -> institution map (originally prototyped as an HTML
        artifact) as a proper journal figure: three columns, boxes and
        arrows, Okabe-Ito colour-coded tracks, no interactivity.

Outputs:
    outputs/figures/pdf/main/fig_policy_capacity_vs_flexibility.pdf
    outputs/figures/pdf/main/fig_policy_stakeholder_map.pdf
    outputs/figures/plot_data/fig_policy_capacity_vs_flexibility.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.reporting.pdf_figures import configure_matplotlib_for_pdf, save_figure_pdf, OKABE_ITO, DOUBLE_COLUMN_WIDTH_IN

FIG_DIR = Path('outputs/figures/pdf/main')
PNG_DIR = Path('outputs/figures/png/main')

FAMILY_ORDER = ['A3_no_grid', 'A3_CI_only', 'A3_genmix_only', 'A3_full']
FAMILY_LABELS = {
    'A3_no_grid': 'No grid data\n(baseline)',
    'A3_CI_only': 'Carbon-intensity\nfamily only',
    'A3_genmix_only': 'Generation-mix\nfamily only',
    'A3_full': 'Full A3\n(both families)',
}
STAT_ORDER = ['A3_no_grid', 'A3_CI_mean_only', 'A3_CI_lowshare_only', 'A3_CI_p90_only',
              'A3_CI_all', 'A3_CI_std_only', 'A3_CI_highshare_only']
STAT_LABELS = {
    'A3_no_grid': 'No grid data',
    'A3_CI_mean_only': 'Mean only',
    'A3_CI_lowshare_only': 'Low-intensity\nshare only',
    'A3_CI_p90_only': '90th pctile\nonly',
    'A3_CI_all': 'All 5 CI\nstatistics',
    'A3_CI_std_only': 'Dispersion (std)\nonly',
    'A3_CI_highshare_only': 'High-intensity\nshare only',
}


def fig_capacity_vs_flexibility():
    ablation = pd.read_csv('outputs/robustness/grid_ablation_results.csv')
    decomp = pd.read_csv('outputs/robustness/ci_decomposition_results.csv')

    fam_best = ablation.loc[ablation.groupby('variant')['weighted_mase'].idxmin()]
    fam_best = fam_best.set_index('variant').reindex(FAMILY_ORDER)

    stat_best = decomp.loc[decomp.groupby('variant')['weighted_mase'].idxmin()]
    stat_best = stat_best.set_index('variant').reindex(STAT_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH_IN, 4.4))
    no_grid_val = fam_best.loc['A3_no_grid', 'weighted_mase']

    ax = axes[0]
    fam_plot = fam_best.iloc[::-1]
    colors = ['#9e9e9e' if v == 'A3_no_grid' else (OKABE_ITO[4] if v == 'A3_full' else '#78909c')
              for v in fam_plot.index]
    ax.barh([FAMILY_LABELS[v].replace('\n', ' ') for v in fam_plot.index], fam_plot['weighted_mase'], color=colors)
    for i, v in enumerate(fam_plot.index):
        ax.text(fam_plot.loc[v, 'weighted_mase'] + 0.012, i, f"{fam_plot.loc[v, 'weighted_mase']:.3f}",
                 va='center', fontsize=7)
    ax.axvline(no_grid_val, color='#c62828', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Weighted MASE (lower is better)', fontsize=7.5)
    ax.set_title('(a) Which information family helps?', fontsize=8.5)
    ax.set_xlim(0, fam_best['weighted_mase'].max() + 0.14)
    ax.tick_params(axis='y', labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    stat_plot = stat_best.iloc[::-1]
    colors2 = []
    for v in stat_plot.index:
        if v == 'A3_no_grid':
            colors2.append('#9e9e9e')
        elif v == 'A3_CI_mean_only':
            colors2.append('#1b5e20')
        else:
            colors2.append('#78909c')
    ax.barh([STAT_LABELS[v].replace('\n', ' ') for v in stat_plot.index], stat_plot['weighted_mase'], color=colors2)
    for i, v in enumerate(stat_plot.index):
        ax.text(stat_plot.loc[v, 'weighted_mase'] + 0.012, i, f"{stat_plot.loc[v, 'weighted_mase']:.3f}",
                 va='center', fontsize=7)
    ax.axvline(no_grid_val, color='#c62828', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Weighted MASE (lower is better)', fontsize=7.5)
    ax.set_title('(b) Which single statistic helps alone?', fontsize=8.5)
    ax.set_xlim(0, stat_best['weighted_mase'].max() + 0.14)
    ax.tick_params(axis='y', labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(
        'Grid carbon-intensity forecasting advantage: family-level vs. single-statistic',
        fontsize=9.5, y=1.01,
    )
    fig.text(
        0.5, -0.05,
        "Dashed line = no-grid-data baseline. Panel (b): the mean alone (green) beats every other single statistic, including dispersion,\n"
        "as a STANDALONE predictor - this qualifies, but does not overturn, the SHAP finding that dispersion carries the largest attribution\n"
        "SHARE inside the full joint model (Fig. 7a): SHAP reports marginal contribution given the other grid features present, not\n"
        "standalone predictive power. The deployable conclusion is the FULL grid feature set, not any one statistic, drives the gain.",
        ha='center', fontsize=6.3, style='italic', wrap=True,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    plot_data = pd.concat([
        fam_best.reset_index().assign(panel='a_family'),
        stat_best.reset_index().assign(panel='b_statistic'),
    ], ignore_index=True, sort=False)[['panel', 'variant', 'model', 'weighted_mase', 'mase_h1', 'mase_h2', 'mase_h4']]

    save_figure_pdf(
        fig, FIG_DIR / 'fig_policy_capacity_vs_flexibility.pdf',
        title='Grid carbon-intensity forecasting advantage: family vs. single-statistic',
        plot_data=plot_data,
    )
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_DIR / 'fig_policy_capacity_vs_flexibility.png', format='png', dpi=600, bbox_inches='tight')


def _box(ax, x, y, w, h, text, color, text_fontsize=6.3):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.04',
        linewidth=0.8, edgecolor='#37474f', facecolor='#f5f5f5', zorder=2,
    )
    ax.add_patch(box)
    ax.add_patch(FancyBboxPatch((x, y), 0.025, h, boxstyle='square,pad=0', linewidth=0, facecolor=color, zorder=3))
    ax.text(x + 0.05, y + h / 2, text, va='center', ha='left', fontsize=text_fontsize, zorder=4, linespacing=1.35)


def _arrow(ax, xy_from, xy_to):
    ax.annotate(
        '', xy=xy_to, xytext=xy_from,
        arrowprops=dict(arrowstyle='-|>', color='#78909c', lw=1.0, shrinkA=2, shrinkB=2,
                         connectionstyle='arc3,rad=0.15'),
        zorder=1,
    )


def fig_stakeholder_map():
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 4.6))
    ax.set_xlim(-0.05, 3.4)
    ax.set_ylim(0, 4.4)
    ax.axis('off')

    blue, vermillion, green, pink, grey = OKABE_ITO[4], OKABE_ITO[5], OKABE_ITO[2], OKABE_ITO[6], '#90a4ae'
    col_x = [0.05, 1.2, 2.35]
    col_w = 0.95
    box_h = 0.55

    ax.text(col_x[0] + col_w / 2, 4.25, 'TECHNICAL SIGNAL', ha='center', fontsize=6.5, color='#607d8b', fontweight='bold')
    ax.text(col_x[1] + col_w / 2, 4.25, 'POLICY / ECONOMIC LEVER', ha='center', fontsize=6.5, color='#607d8b', fontweight='bold')
    ax.text(col_x[2] + col_w / 2, 4.25, 'INSTITUTION / SECTOR', ha='center', fontsize=6.5, color='#607d8b', fontweight='bold')

    rows_c1 = {'A': (3.55, 'Within-quarter CI\ndispersion signal', blue),
               'H1': (2.15, 'Quarterly H1 nowcast\nMASE 0.376', grey),
               'D': (0.35, 'Regime-stable\nranking (ρ 0.92-0.97)', pink)}
    rows_c2 = {'A': (3.55, 'Flexibility & storage\ninvestment case', blue),
               'B': (2.60, 'In-year carbon-budget\nearly warning', vermillion),
               'C': (1.65, 'Transition-risk\nmarket signal', green),
               'D': (0.35, 'Monitoring layer, not\ninventory replacement', pink)}
    rows_c3 = {'A': (3.55, 'NESO & Ofgem', blue),
               'B': (2.60, 'DESNZ & Climate\nChange Committee', vermillion),
               'C': (1.65, 'Energy investors &\ncarbon-market participants', green),
               'D': (0.35, 'HM Treasury/ONS ·\ndevolved & local bodies', pink)}

    for key, (yc, label, color) in rows_c1.items():
        _box(ax, col_x[0], yc - box_h / 2, col_w, box_h, label, color)
    for key, (yc, label, color) in rows_c2.items():
        _box(ax, col_x[1], yc - box_h / 2, col_w, box_h, label, color)
    for key, (yc, label, color) in rows_c3.items():
        _box(ax, col_x[2], yc - box_h / 2, col_w, box_h, label, color)

    x1r, x2l, x2r, x3l = col_x[0] + col_w, col_x[1], col_x[1] + col_w, col_x[2]
    _arrow(ax, (x1r, rows_c1['A'][0]), (x2l, rows_c2['A'][0]))
    _arrow(ax, (x1r, rows_c1['H1'][0]), (x2l, rows_c2['B'][0]))
    _arrow(ax, (x1r, rows_c1['H1'][0]), (x2l, rows_c2['C'][0]))
    _arrow(ax, (x1r, rows_c1['D'][0]), (x2l, rows_c2['D'][0]))
    for k in ('A', 'B', 'C', 'D'):
        _arrow(ax, (x2r, rows_c2[k][0]), (x3l, rows_c3[k][0]))

    ax.set_title(
        'From grid dispersion to decision-makers: a technical-finding-to-stakeholder map',
        fontsize=8.5, pad=10,
    )
    fig.text(
        0.5, 0.01,
        'The H1 nowcast forks: the same forecast underwrites both an early-warning use (DESNZ/CCC) and a market-risk use (investors).\n'
        'Predictive attribution, not a causal policy lever - see Section 5.4/5.5 for the accompanying caveats.',
        ha='center', fontsize=6.0, style='italic',
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    save_figure_pdf(
        fig, FIG_DIR / 'fig_policy_stakeholder_map.pdf',
        title='From grid dispersion to decision-makers',
        plot_data=None,
    )
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_DIR / 'fig_policy_stakeholder_map.png', format='png', dpi=600, bbox_inches='tight')


def main():
    configure_matplotlib_for_pdf()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_capacity_vs_flexibility()
    fig_stakeholder_map()
    print("Wrote fig_policy_capacity_vs_flexibility.pdf and fig_policy_stakeholder_map.pdf")


if __name__ == '__main__':
    main()
