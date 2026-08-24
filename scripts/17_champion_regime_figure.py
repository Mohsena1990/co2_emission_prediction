#!/usr/bin/env python
"""
Script 17: publication figure - "Evolution of predictive importance across
decarbonisation regimes" for the A3/LightGBM champion (audit task section 6).

Three panels, built directly from the corrected SHAP deliverables produced
by scripts/16_champion_shap_deliverables.py:
    (a) top-10 global predictors by mean |SHAP|
    (b) heatmap of normalized mean |SHAP| across pre/COVID/post-COVID regimes
    (c) family-level (grid carbon intensity / grid generation mix / weather /
        macroeconomic / mobility / shock-health / annual transition)
        importance-share evolution across regimes

Outputs:
    outputs/figures/pdf/main/fig_champion_regime_importance.pdf
    outputs/figures/png/main/fig_champion_regime_importance.png
    outputs/reporting/figure_champion_regime_caption.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.reporting.pdf_figures import configure_matplotlib_for_pdf

IN_DIR = Path('outputs/interpretability')
PDF_DIR = Path('outputs/figures/pdf/main')
PNG_DIR = Path('outputs/figures/png/main')
CAPTION_PATH = Path('outputs/reporting/figure_champion_regime_caption.txt')

REGIME_ORDER = ['pre_covid', 'covid', 'post_covid']
REGIME_LABELS = {'pre_covid': 'Pre-COVID', 'covid': 'COVID', 'post_covid': 'Post-COVID'}
FAMILY_LABELS = {
    'grid_carbon_intensity': 'Grid carbon intensity', 'grid_generation_mix': 'Grid generation mix',
    'weather': 'Weather', 'macroeconomic': 'Macroeconomic', 'mobility': 'Mobility',
    'shock_health': 'Shock/health (COVID deaths)', 'annual_transition': 'Annual transition (OWID)',
}
FAMILY_COLORS = {
    'grid_carbon_intensity': '#1b5e20', 'grid_generation_mix': '#66bb6a',
    'weather': '#1565c0', 'macroeconomic': '#8e24aa', 'mobility': '#ef6c00',
    'shock_health': '#c62828', 'annual_transition': '#00838f',
}


def main():
    configure_matplotlib_for_pdf()
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    CAPTION_PATH.parent.mkdir(parents=True, exist_ok=True)

    global_df = pd.read_csv(IN_DIR / 'champion_A3_LightGBM_global_shap.csv')
    regime_df = pd.read_csv(IN_DIR / 'champion_A3_LightGBM_regime_shap.csv')

    top10 = global_df.sort_values('mean_abs_shap', ascending=False).head(10)

    fig = plt.figure(figsize=(11, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.42, wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # --- Panel (a): top-10 global predictors ---
    order = top10.iloc[::-1]
    colors_a = [FAMILY_COLORS.get(f, '#757575') for f in order['family']]
    ax_a.barh(order['feature'], order['mean_abs_shap'], color=colors_a)
    ax_a.set_xlabel('Mean |SHAP| (full common-period sample, n=28)')
    ax_a.set_title('(a)', loc='left', fontweight='bold')
    ax_a.tick_params(axis='y', labelsize=7.5)

    # --- Panel (b): heatmap of normalized mean|SHAP| across regimes ---
    pivot = regime_df.pivot(index='feature', columns='regime', values='mean_abs_shap')
    pivot = pivot[REGIME_ORDER]
    pivot_norm = pivot.div(pivot.max(axis=1), axis=0).fillna(0.0)
    top10_features = top10['feature'].tolist()
    pivot_norm = pivot_norm.loc[top10_features]

    im = ax_b.imshow(pivot_norm.values, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
    ax_b.set_xticks(range(len(REGIME_ORDER)))
    ax_b.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER])
    ax_b.set_yticks(range(len(top10_features)))
    ax_b.set_yticklabels(top10_features, fontsize=7.5)
    ax_b.set_title('(b)', loc='left', fontweight='bold')
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized mean |SHAP|\n(within-feature, 0-1)', fontsize=7.5)
    for i in range(pivot_norm.shape[0]):
        for j in range(pivot_norm.shape[1]):
            val = pivot_norm.values[i, j]
            ax_b.text(j, i, f'{val:.2f}', ha='center', va='center',
                      fontsize=6.5, color='white' if val > 0.55 else 'black')

    # --- Panel (c): family-level importance share by regime ---
    family_sum = regime_df.groupby(['regime', 'family'])['mean_abs_shap'].sum()
    family_share_series = family_sum / family_sum.groupby(level='regime').transform('sum')
    family_share = (
        family_share_series.unstack(fill_value=0.0)[
            [f for f in FAMILY_LABELS if f in regime_df['family'].unique()]
        ]
        .loc[REGIME_ORDER]
    )
    bottom = np.zeros(len(REGIME_ORDER))
    x = np.arange(len(REGIME_ORDER))
    for family in family_share.columns:
        vals = family_share[family].values
        ax_c.bar(x, vals, bottom=bottom, color=FAMILY_COLORS.get(family, '#757575'),
                  label=FAMILY_LABELS.get(family, family), width=0.6)
        bottom += vals
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER])
    ax_c.set_ylabel('Share of total |SHAP| attribution')
    ax_c.set_ylim(0, 1.02)
    ax_c.set_title('(c)', loc='left', fontweight='bold')
    ax_c.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=7.5)

    fig.suptitle('')  # no in-figure title, per spec

    for path in (PDF_DIR / 'fig_champion_regime_importance.pdf', ):
        fig.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0.05,
                    metadata={'Title': 'Champion regime importance', 'Creator': 'Q-DECEM reproducible pipeline'})
    fig.savefig(PNG_DIR / 'fig_champion_regime_importance.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)

    plot_data_dir = Path('outputs/figures/plot_data')
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    top10.to_csv(plot_data_dir / 'fig_champion_regime_importance_panel_a.csv', index=False)
    pivot_norm.reset_index().to_csv(plot_data_dir / 'fig_champion_regime_importance_panel_b.csv', index=False)
    family_share.reset_index().to_csv(plot_data_dir / 'fig_champion_regime_importance_panel_c.csv', index=False)

    print(f"Saved {PDF_DIR / 'fig_champion_regime_importance.pdf'}")
    print(f"Saved {PNG_DIR / 'fig_champion_regime_importance.png'}")

    top_family_per_regime = family_share.idxmax(axis=1)
    grid_share = family_share.get('grid_carbon_intensity', pd.Series(0, index=REGIME_ORDER)) + \
                 family_share.get('grid_generation_mix', pd.Series(0, index=REGIME_ORDER))

    caption = (
        "Figure. Evolution of predictive importance across decarbonisation regimes for the "
        "validated A3/LightGBM champion (WMASE = 0.454; common evaluation period 2018Q2-2025Q1, "
        "n=28 quarterly observations). (a) Top-10 predictors ranked by mean absolute SHAP value "
        "over the full common-period sample. (b) Heatmap of mean |SHAP| for the same top-10 "
        "predictors, normalized within each feature (0-1) across the Pre-COVID (n=7), COVID "
        "(n=8) and Post-COVID (n=13) regimes; cell values are the normalized magnitude, not "
        "causal effects. (c) Share of total |SHAP| attribution held by each predictor family "
        f"in each regime. Electricity-grid variables (carbon-intensity distributional statistics "
        f"plus generation-mix shares) account for {grid_share.min()*100:.0f}-{grid_share.max()*100:.0f}% "
        "of total attribution across all three regimes, with grid carbon-intensity dispersion "
        "(Grid_CI_std) the single largest contributor in every regime (Spearman rank correlation "
        "of feature importance between regimes: 0.91-0.97; see "
        "outputs/interpretability/champion_A3_LightGBM_regime_rank_changes.csv). SHAP values are "
        "predictive attribution only and do not imply a causal relationship."
    )
    CAPTION_PATH.write_text(caption + "\n")
    print(f"Saved {CAPTION_PATH}")


if __name__ == '__main__':
    main()
