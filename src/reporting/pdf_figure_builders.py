"""
Individual PDF figure builders (spec section 22). Each function takes
already-loaded DataFrames (read by the caller from saved CSV/JSON outputs -
spec 21.1: figures are generated programmatically from saved data, never
computed inline from raw numbers) and calls save_figure_pdf as its final
step. Covers the figures with the most direct, unambiguous mapping from
already-existing saved tables; the remaining figures in spec section 22
follow the same save_figure_pdf pattern once their upstream data exists.
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.logging_utils import get_logger
from .pdf_figures import save_figure_pdf, configure_matplotlib_for_pdf, OKABE_ITO, DOUBLE_COLUMN_WIDTH_IN

MODEL_ORDER = ['ridge', 'random_forest', 'lightgbm', 'catboost', 'lstm']
CONFIG_ORDER = ['A1', 'A2', 'A3', 'A4']


def fig04_configuration_model_wmase_heatmap(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Figure 4: rows=models, columns=A1-A4, cell=weighted MASE, annotated."""
    configure_matplotlib_for_pdf()
    logger = get_logger()

    pivot = metrics_df.pivot_table(index='model', columns='configuration', values='weighted_mase', aggfunc='mean')
    pivot = pivot.reindex(index=[m for m in MODEL_ORDER if m in pivot.index],
                           columns=[c for c in CONFIG_ORDER if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.5))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7)
    ax.set_title('Weighted MASE by Configuration and Model', fontweight='bold')
    fig.colorbar(im, ax=ax, label='Weighted MASE (lower is better)')
    fig.tight_layout()

    save_figure_pdf(fig, output_path, 'Configuration x Model Weighted-MASE Heatmap',
                     plot_data=pivot.reset_index())
    plt.close(fig)
    logger.info(f"fig04: {pivot.shape[0]}x{pivot.shape[1]} cells")


def fig05_incremental_configuration_value(table6: pd.DataFrame, output_path: Path) -> None:
    """Figure 5: percentage improvement + CI per comparison x model, zero-reference line."""
    configure_matplotlib_for_pdf()
    logger = get_logger()

    if table6.empty:
        logger.warning("fig05: Table 6 is empty, skipping")
        return

    comparisons = sorted(table6['comparison'].unique())
    fig, axes = plt.subplots(1, len(comparisons), figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.0), sharey=True)
    if len(comparisons) == 1:
        axes = [axes]

    for ax, comp in zip(axes, comparisons):
        sub = table6[table6['comparison'] == comp].sort_values(['model', 'horizon'])
        x = range(len(sub))
        labels = [f"{r['model']}\nH{r['horizon']}" for _, r in sub.iterrows()]
        errors_low = sub['percentage_improvement'] - (sub['ci_low'] / sub['mae_baseline'] * -100)
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.bar(x, sub['percentage_improvement'], color=OKABE_ITO[0])
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax.set_title(comp, fontsize=8)

    axes[0].set_ylabel('% improvement vs. baseline')
    fig.suptitle('Incremental Configuration Value', fontweight='bold', y=1.02)
    fig.tight_layout()

    save_figure_pdf(fig, output_path, 'Incremental Configuration Value', plot_data=table6)
    plt.close(fig)
    logger.info(f"fig05: {len(comparisons)} comparisons")


def fig06_multi_horizon_configuration_performance(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Figure 6: H1/H2/H4 MASE for A1-A4 x 5 models, small multiples per model."""
    configure_matplotlib_for_pdf()
    logger = get_logger()

    horizon_cols = [c for c in ('mase_h1', 'mase_h2', 'mase_h4') if c in metrics_df.columns]
    if not horizon_cols:
        logger.warning("fig06: no per-horizon MASE columns found, skipping")
        return

    models = [m for m in MODEL_ORDER if m in metrics_df['model'].unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(DOUBLE_COLUMN_WIDTH_IN, 2.8), sharey=True)
    if len(models) == 1:
        axes = [axes]

    plot_rows = []
    for ax, model in zip(axes, models):
        sub = metrics_df[metrics_df['model'] == model]
        for i, config in enumerate([c for c in CONFIG_ORDER if c in sub['configuration'].unique()]):
            row = sub[sub['configuration'] == config]
            if row.empty:
                continue
            values = [row.iloc[0].get(c, np.nan) for c in horizon_cols]
            horizons = [int(c.split('_h')[1]) for c in horizon_cols]
            ax.plot(horizons, values, marker='o', label=config, color=OKABE_ITO[i % len(OKABE_ITO)], markersize=3)
            for h, v in zip(horizons, values):
                plot_rows.append({'model': model, 'configuration': config, 'horizon': h, 'mase': v})
        ax.set_title(model, fontsize=8)
        ax.set_xlabel('Horizon')
        ax.set_xticks([1, 2, 4])

    axes[0].set_ylabel('MASE')
    axes[-1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.suptitle('Multi-Horizon Performance by Configuration', fontweight='bold', y=1.05)
    fig.tight_layout()

    save_figure_pdf(fig, output_path, 'Multi-Horizon Configuration Performance',
                     plot_data=pd.DataFrame(plot_rows))
    plt.close(fig)
    logger.info(f"fig06: {len(models)} models")


def fig10_model_fs_heatmap(metrics_df: pd.DataFrame, configuration: str, output_path: Path) -> None:
    """Figures 10/11: rows=models, columns=FS1-FS5, cell=weighted MASE, for one configuration."""
    configure_matplotlib_for_pdf()
    logger = get_logger()

    sub = metrics_df[(metrics_df['configuration'] == configuration) & (metrics_df['fs_option'] != 'all_features')]
    if sub.empty:
        logger.warning(f"fig10/11 ({configuration}): no FS-comparison rows found, skipping")
        return

    pivot = sub.pivot_table(index='model', columns='fs_option', values='weighted_mase', aggfunc='mean')
    fs_order = ['fs_linear', 'fs_wrapper', 'fs_xgboost_shap', 'fs_permutation_stability', 'fs_consensus']
    pivot = pivot.reindex(index=[m for m in MODEL_ORDER if m in pivot.index],
                           columns=[c for c in fs_order if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.5))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace('fs_', '') for c in pivot.columns], rotation=30, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7)
    ax.set_title(f'{configuration}: Model x Feature-Selection Weighted MASE', fontweight='bold')
    fig.colorbar(im, ax=ax, label='Weighted MASE (lower is better)')
    fig.tight_layout()

    save_figure_pdf(fig, output_path, f'{configuration} Model x FS Heatmap', plot_data=pivot.reset_index())
    plt.close(fig)
    logger.info(f"fig10/11 ({configuration}): {pivot.shape[0]}x{pivot.shape[1]} cells")


def fig01_revised_framework(output_path: Path) -> None:
    """Figure 1: revised complete framework schematic (spec section 22) - a
    static diagram, no data dependency."""
    configure_matplotlib_for_pdf()
    logger = get_logger()

    stages = [
        ("Quarterly raw data", 0),
        ("A1: raw predictors  |  Feature engineering -> A2", 1),
        ("High-frequency grid data -> quarterly aggregation -> A3  |  Engineering + grid fusion -> A4", 2),
        ("Feature selection (FS1-FS5)", 3),
        ("Five forecasting models (Ridge/RF/LightGBM/CatBoost/LSTM)", 4),
        ("Nested expanding-window evaluation (outer + inner CV)", 5),
        ("Pareto filtering", 6),
        ("VIKOR (when >1 non-dominated)", 7),
        ("Forecasting and interpretation", 8),
    ]

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 7.5))
    box_h = 0.7
    for text, i in stages:
        y = -i
        ax.add_patch(plt.Rectangle((0, y - box_h / 2), 10, box_h, fill=True,
                                    facecolor=OKABE_ITO[1], edgecolor='black', linewidth=0.8, alpha=0.85))
        ax.text(5, y, text, ha='center', va='center', fontsize=6.5, wrap=True)
        if i > 0:
            ax.annotate('', xy=(5, y + box_h / 2 + 0.05), xytext=(5, y + 1 - box_h / 2 - 0.05),
                        arrowprops=dict(arrowstyle='->', lw=0.8))

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-len(stages) + 0.3, 0.8)
    ax.axis('off')
    ax.set_title('Q-DECEM Revised Framework', fontweight='bold', fontsize=10)
    fig.tight_layout()

    save_figure_pdf(fig, output_path, 'Revised Q-DECEM Framework')
    plt.close(fig)
    logger.info(f"fig01: {len(stages)} stages")


def fig02_predictor_governance_map(table2: pd.DataFrame, output_path: Path) -> None:
    """Figure 2: predictor-governance and leakage map (spec section 22) -
    one row per feature, columns for target-derived/safe-at-horizon/
    configuration membership, from the saved feature registry (Table 2)."""
    configure_matplotlib_for_pdf()
    logger = get_logger()

    if table2.empty:
        logger.warning("fig02: Table 2 is empty, skipping")
        return

    df = table2.copy()
    bool_cols = ['target_derived', 'safe_at_h1', 'safe_at_h2', 'safe_at_h4', 'retained_after_audit']
    for tag in ('A1', 'A2', 'A3', 'A4'):
        df[tag] = df['configuration_membership'].apply(
            lambda s: tag in str(s).replace("'", "").split(', ')
        )
    matrix_cols = bool_cols + ['A1', 'A2', 'A3', 'A4']
    matrix = df[matrix_cols].astype(int)
    matrix.index = df['name']

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, max(4.0, 0.15 * len(matrix))))
    ax.imshow(matrix.values, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix_cols)))
    ax.set_xticklabels(matrix_cols, rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=5)
    ax.set_title('Predictor Governance and Leakage Map', fontweight='bold')
    fig.tight_layout()

    save_figure_pdf(fig, output_path, 'Predictor Governance Map',
                     plot_data=matrix.reset_index().rename(columns={'index': 'feature'}))
    plt.close(fig)
    logger.info(f"fig02: {len(matrix)} features x {len(matrix_cols)} governance columns")


def fig14_mcda_rank_sensitivity(rank_correlation: pd.DataFrame, output_path: Path) -> None:
    """Figure 14: MCDA rank sensitivity - pairwise Spearman's rho between
    weighting schemes (equal/accuracy/stability/parsimony/runtime-emphasis
    VIKOR, TOPSIS, weighted sum), from rank_correlation_across_weight_sets'
    saved output (spec section 18)."""
    configure_matplotlib_for_pdf()
    logger = get_logger()

    if rank_correlation.empty:
        logger.warning("fig14: rank correlation table is empty, skipping")
        return

    schemes = sorted(set(rank_correlation['scheme_a']) | set(rank_correlation['scheme_b']))
    matrix = pd.DataFrame(1.0, index=schemes, columns=schemes)
    for _, row in rank_correlation.iterrows():
        matrix.loc[row['scheme_a'], row['scheme_b']] = row['spearman_rho']
        matrix.loc[row['scheme_b'], row['scheme_a']] = row['spearman_rho']

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, DOUBLE_COLUMN_WIDTH_IN * 0.85))
    im = ax.imshow(matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels([s.replace('_', ' ') for s in schemes], rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(len(schemes)))
    ax.set_yticklabels([s.replace('_', ' ') for s in schemes], fontsize=6)
    for i in range(len(schemes)):
        for j in range(len(schemes)):
            ax.text(j, i, f'{matrix.values[i, j]:.2f}', ha='center', va='center', fontsize=5)
    ax.set_title('MCDA Rank Sensitivity (Spearman\'s rho)', fontweight='bold')
    fig.colorbar(im, ax=ax, label="Spearman's rho")
    fig.tight_layout()

    n_disagree = (~rank_correlation['top_choice_agrees']).sum()
    save_figure_pdf(fig, output_path, 'MCDA Rank Sensitivity', plot_data=rank_correlation)
    plt.close(fig)
    logger.info(f"fig14: {len(schemes)} schemes, top choice disagrees in {n_disagree}/{len(rank_correlation)} pairs")


def fig08_fs_membership_heatmap(
    fs_results: dict, all_features: list, configuration_label: str, output_path: Path
) -> None:
    """Figures 8/9: binary FS1-FS5 membership per feature, for A2 or A4."""
    configure_matplotlib_for_pdf()
    logger = get_logger()

    fs_order = ['fs_linear', 'fs_wrapper', 'fs_xgboost_shap', 'fs_permutation_stability', 'fs_consensus']
    fs_present = [fs for fs in fs_order if fs in fs_results]
    if not fs_present:
        logger.warning(f"fig08/09 ({configuration_label}): no FS results found, skipping")
        return

    membership = pd.DataFrame(
        {fs: [1 if f in fs_results[fs]['selected_features'] else 0 for f in all_features] for fs in fs_present},
        index=all_features,
    )
    membership = membership.loc[membership.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN * 0.7, max(3.0, 0.18 * len(all_features))))
    ax.imshow(membership.values, cmap='Greys', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(fs_present)))
    ax.set_xticklabels([c.replace('fs_', '') for c in fs_present], rotation=30, ha='right')
    ax.set_yticks(range(len(membership.index)))
    ax.set_yticklabels(membership.index, fontsize=6)
    ax.set_title(f'{configuration_label}: Feature-Selection Membership', fontweight='bold')
    fig.tight_layout()

    save_figure_pdf(fig, output_path, f'{configuration_label} FS Membership',
                     plot_data=membership.reset_index().rename(columns={'index': 'feature'}))
    plt.close(fig)
    logger.info(f"fig08/09 ({configuration_label}): {len(all_features)} features x {len(fs_present)} FS methods")


def fig15_cross_model_feature_importance(table11: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    """Figure 15: normalized (ranked) importance across Ridge/RF/LightGBM/CatBoost/LSTM."""
    configure_matplotlib_for_pdf()
    logger = get_logger()

    if table11.empty:
        logger.warning("fig15: Table 11 is empty, skipping")
        return

    rank_cols = [c for c in table11.columns if c.endswith('_rank') and c != 'average_rank']
    top = table11.sort_values('average_rank').head(top_n)

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, max(3.0, 0.22 * len(top))))
    y = np.arange(len(top))
    width = 0.8 / max(len(rank_cols), 1)
    for i, col in enumerate(rank_cols):
        ax.barh(y + i * width, top[col], height=width, label=col.replace('_rank', ''),
                color=OKABE_ITO[i % len(OKABE_ITO)])
    ax.set_yticks(y + width * (len(rank_cols) - 1) / 2)
    ax.set_yticklabels(top['feature'], fontsize=6)
    ax.invert_yaxis()
    ax.invert_xaxis()  # rank 1 (best) at the edge
    ax.set_xlabel('Rank (1 = most important)')
    ax.set_title('Cross-Model Feature-Importance Ranks', fontweight='bold')
    ax.legend(fontsize=6, loc='lower right')
    fig.tight_layout()

    save_figure_pdf(fig, output_path, 'Cross-Model Feature Importance', plot_data=table11)
    plt.close(fig)
    logger.info(f"fig15: top {len(top)} of {len(table11)} features")


def fig17_regime_specific_importance(regime_importance: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    """Figure 17: importance by regime (pre/COVID/post-COVID) for the champion model."""
    configure_matplotlib_for_pdf()
    logger = get_logger()

    if regime_importance.empty:
        logger.warning("fig17: no regime importance data, skipping")
        return

    top_features = (
        regime_importance.groupby('feature')['importance'].mean().nlargest(top_n).index
    )
    sub = regime_importance[regime_importance['feature'].isin(top_features)]
    pivot = sub.pivot_table(index='feature', columns='regime', values='importance', aggfunc='mean')
    regime_order = [r for r in ('pre_covid', 'covid', 'post_covid') if r in pivot.columns]
    pivot = pivot.reindex(index=top_features, columns=regime_order)

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.5))
    pivot.plot(kind='bar', ax=ax, width=0.8, color=OKABE_ITO[:len(regime_order)])
    ax.set_ylabel('Importance')
    ax.set_title('Regime-Specific Feature Importance (pre/COVID/post-COVID)', fontweight='bold')
    ax.legend(title='Regime', fontsize=6)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)
    fig.tight_layout()

    save_figure_pdf(fig, output_path, 'Regime-Specific Importance', plot_data=pivot.reset_index())
    plt.close(fig)
    logger.info(f"fig17: {len(top_features)} features x {len(regime_order)} regimes")


def fig18_target_derived_feature_sensitivity(sensitivity_df: pd.DataFrame, output_path: Path) -> None:
    """Figure 18: weighted MASE for full pool vs. CEI/intensity-excluded vs. all-target-derived-excluded."""
    configure_matplotlib_for_pdf()
    logger = get_logger()

    if sensitivity_df.empty:
        logger.warning("fig18: no target-derived sensitivity data, skipping")
        return

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN * 0.6, 3.0))
    ax.bar(sensitivity_df['variant'], sensitivity_df['weighted_mase'], color=OKABE_ITO[0])
    ax.set_ylabel('Weighted MASE')
    ax.set_title('Target-Derived-Feature Sensitivity', fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=6)
    fig.tight_layout()

    save_figure_pdf(fig, output_path, 'Target-Derived Feature Sensitivity', plot_data=sensitivity_df)
    plt.close(fig)
    logger.info(f"fig18: {len(sensitivity_df)} variants")


def fig03_grid_aggregation_quality(grid_quality_report: pd.DataFrame, output_path: Path) -> None:
    """Figure 3: half-hourly-to-quarterly completeness over time."""
    configure_matplotlib_for_pdf()
    logger = get_logger()

    if grid_quality_report.empty:
        logger.warning("fig03: grid quality report is empty, skipping")
        return

    df = grid_quality_report.copy()
    df['quarter'] = pd.to_datetime(df['quarter'])
    df = df.sort_values('quarter')

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.0))
    colors = np.where(df['inclusion_status'] == 'included', OKABE_ITO[2], OKABE_ITO[5])
    ax.bar(df['quarter'], df['completeness_ratio'], width=60, color=colors)
    threshold = df['min_completeness_threshold'].iloc[0] if len(df) else 0.95
    ax.axhline(threshold, color='grey', linestyle='--', linewidth=0.8, label=f'threshold={threshold}')
    ax.set_ylabel('Completeness ratio')
    ax.set_title('Grid Data: Quarterly Completeness', fontweight='bold')
    ax.legend(fontsize=6)
    fig.tight_layout()

    save_figure_pdf(fig, output_path, 'Grid Aggregation Quality', plot_data=df)
    plt.close(fig)
    logger.info(f"fig03: {len(df)} quarters, {(df['inclusion_status'] != 'included').sum()} excluded")


def fig13_pareto_frontier(table10: pd.DataFrame, output_path: Path) -> None:
    """Figure 13: weighted MASE vs. worst-horizon MASE, non-dominated points labelled."""
    configure_matplotlib_for_pdf()
    logger = get_logger()

    if table10.empty:
        logger.warning("fig13: Table 10 is empty, skipping")
        return

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN * 0.6, 3.5))
    model_markers = {m: marker for m, marker in zip(MODEL_ORDER, ['o', 's', '^', 'D', 'v'])}

    for status, group in table10.groupby('pareto_status'):
        for model, sub in group.groupby('model'):
            ax.scatter(
                sub['weighted_mase'], sub['worst_horizon_mase'],
                s=30 + sub['n_features'] * 2,
                marker=model_markers.get(model, 'o'),
                facecolors=OKABE_ITO[0] if status == 'non_dominated' else 'none',
                edgecolors=OKABE_ITO[0] if status == 'non_dominated' else 'grey',
                alpha=0.9 if status == 'non_dominated' else 0.4,
                label=f'{model} ({status})' if status == 'non_dominated' else None,
            )

    non_dominated = table10[table10['pareto_status'] == 'non_dominated']
    for _, row in non_dominated.iterrows():
        ax.annotate(f"{row['configuration']}/{row['model']}",
                     (row['weighted_mase'], row['worst_horizon_mase']), fontsize=5)

    ax.set_xlabel('Weighted MASE')
    ax.set_ylabel('Worst-horizon MASE')
    ax.set_title('Pareto Frontier: Weighted vs. Worst-Horizon MASE', fontweight='bold')
    ax.legend(fontsize=5, loc='best')
    fig.tight_layout()

    save_figure_pdf(fig, output_path, 'Pareto Frontier', plot_data=table10)
    plt.close(fig)
    logger.info(f"fig13: {len(non_dominated)} non-dominated of {len(table10)} total")


# =============================================================================
# Spec section 26: replacements for the old single-panel "Incremental
# Improvement" figure - four figures covering the Stream A/B global
# comparison, never a pairwise config-vs-config framing.
# =============================================================================

FS_ORDER = ['fs_linear', 'fs_wrapper', 'fs_xgboost_shap', 'fs_permutation_stability', 'fs_consensus']


def _assign_global_rank(ranked_df: pd.DataFrame) -> pd.Series:
    """Global rank across a whole stream (spec 26's 'global Stream A/B
    rank' cell annotation): non-dominated cells first (already MCDM-ranked
    by vikor_rank within that group), then dominated cells ordered by
    weighted_mase - a single total ordering over every candidate in the
    stream, documented here since neither Pareto filtering nor VIKOR alone
    produces one.

    `pareto_status` is sorted via an explicit category order, NOT
    alphabetically - 'dominated' < 'non_dominated' lexicographically would
    silently rank every dominated (worse) candidate ahead of every
    non-dominated one.
    """
    status_rank = ranked_df['pareto_status'].map({'non_dominated': 0, 'dominated': 1}).fillna(1)
    order = (
        pd.DataFrame({
            'status_rank': status_rank,
            'vikor_rank': ranked_df['vikor_rank'],
            'weighted_mase': ranked_df['weighted_mase'],
        }, index=ranked_df.index)
        .sort_values(['status_rank', 'vikor_rank', 'weighted_mase'], na_position='last')
        .index
    )
    return pd.Series(range(1, len(order) + 1), index=order, name='global_rank')


def fig03a_stream_A_global_comparison(stream_a_ranked: pd.DataFrame, output_path: Path) -> None:
    """
    Figure 3A (spec section 26): all 20 Stream A candidates. Rows=models,
    columns=A1-A4, cell=weighted MASE, annotated with weighted MASE AND
    the candidate's global Stream A rank.
    """
    configure_matplotlib_for_pdf()
    logger = get_logger()

    if stream_a_ranked.empty:
        logger.warning("fig03a: Stream A ranking is empty, skipping")
        return

    df = stream_a_ranked.copy()
    df['global_rank'] = _assign_global_rank(df)

    pivot_mase = df.pivot_table(index='model', columns='configuration', values='weighted_mase', aggfunc='first')
    pivot_rank = df.pivot_table(index='model', columns='configuration', values='global_rank', aggfunc='first')
    pivot_mase = pivot_mase.reindex(index=[m for m in MODEL_ORDER if m in pivot_mase.index],
                                     columns=[c for c in CONFIG_ORDER if c in pivot_mase.columns])
    pivot_rank = pivot_rank.reindex(index=pivot_mase.index, columns=pivot_mase.columns)

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.5))
    im = ax.imshow(pivot_mase.values, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(pivot_mase.columns)))
    ax.set_xticklabels(pivot_mase.columns)
    ax.set_yticks(range(len(pivot_mase.index)))
    ax.set_yticklabels(pivot_mase.index)
    for i in range(pivot_mase.shape[0]):
        for j in range(pivot_mase.shape[1]):
            mase_val = pivot_mase.values[i, j]
            rank_val = pivot_rank.values[i, j]
            if np.isfinite(mase_val):
                label = f'{mase_val:.3f}\n(#{int(rank_val)})' if np.isfinite(rank_val) else f'{mase_val:.3f}'
                ax.text(j, i, label, ha='center', va='center', fontsize=6)
    ax.set_title('Stream A: All 20 Candidates (no feature selection)', fontweight='bold')
    fig.colorbar(im, ax=ax, label='Weighted MASE (lower is better)')
    fig.tight_layout()

    save_figure_pdf(fig, output_path, 'Stream A Global Comparison', plot_data=df)
    plt.close(fig)
    logger.info(f"fig03a: {len(df)} Stream A candidates")


def fig03b_stream_B_global_overview(stream_b_ranked: pd.DataFrame, output_path: Path) -> None:
    """
    Figure 3B (spec section 26): all 100 Stream B candidates as 4 aligned
    heatmap panels (B1-B4). Within each panel: rows=models, columns=FS1-5,
    cells=weighted MASE annotated; non-dominated (Pareto) cells get a
    black border; the Stream B winner cell gets a gold border.
    """
    configure_matplotlib_for_pdf()
    logger = get_logger()

    if stream_b_ranked.empty:
        logger.warning("fig03b: Stream B ranking is empty, skipping")
        return

    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.2), sharey=True)
    configs = CONFIG_ORDER
    vmin, vmax = stream_b_ranked['weighted_mase'].min(), stream_b_ranked['weighted_mase'].max()

    for ax, config in zip(axes, configs):
        sub = stream_b_ranked[stream_b_ranked['configuration'] == config]
        pivot = sub.pivot_table(index='model', columns='fs_option', values='weighted_mase', aggfunc='first')
        pivot = pivot.reindex(index=MODEL_ORDER, columns=[f for f in FS_ORDER if f in pivot.columns])
        im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f.replace('fs_', '') for f in pivot.columns], fontsize=6, rotation=45, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=6)
        ax.set_title(f'B{configs.index(config) + 1} ({config} pool)', fontsize=7)

        for i, model in enumerate(pivot.index):
            for j, fs in enumerate(pivot.columns):
                cell_rows = sub[(sub['model'] == model) & (sub['fs_option'] == fs)]
                if cell_rows.empty:
                    continue
                cell = cell_rows.iloc[0]
                val = cell['weighted_mase']
                if not np.isfinite(val):
                    continue
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5)
                if cell.get('pareto_status') == 'non_dominated':
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=1.2))
                if str(cell.get('final_decision', '')).startswith('selected'):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='gold', linewidth=1.8))

    fig.suptitle('Stream B: All 100 Candidates (black = Pareto-optimal, gold = Stream B winner)',
                 fontweight='bold', fontsize=8, y=1.04)
    fig.colorbar(im, ax=axes, label='Weighted MASE', shrink=0.7)

    save_figure_pdf(fig, output_path, 'Stream B Global Overview', plot_data=stream_b_ranked)
    plt.close(fig)
    logger.info(f"fig03b: {len(stream_b_ranked)} Stream B candidates across 4 panels")


def fig03c_stream_B_shortlist_magnified(stream_b_ranked: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    """
    Figure 3C (spec section 26): magnified ranked horizontal dot plot of
    the top Pareto-filtered Stream B candidates - weighted MASE (+/-
    error_std as an uncertainty band, the fold-level variability measure
    already computed per cell - a true bootstrap CI is a section 19
    sensitivity extension, not persisted per-cell today), worst-horizon
    MASE, feature count, FS strategy, model, configuration, MCDM rank.
    """
    configure_matplotlib_for_pdf()
    logger = get_logger()

    non_dominated = stream_b_ranked[stream_b_ranked['pareto_status'] == 'non_dominated']
    if non_dominated.empty:
        logger.warning("fig03c: no non-dominated Stream B candidates, skipping")
        return

    shortlist = non_dominated.sort_values('vikor_rank', na_position='last').head(top_n).iloc[::-1]
    labels = [f"{r.configuration}/{r.fs_option.replace('fs_', '')}/{r.model} (#{int(r.vikor_rank) if np.isfinite(r.vikor_rank) else '-'})"
              for r in shortlist.itertuples()]

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, max(3.0, 0.22 * len(shortlist))))
    y = range(len(shortlist))
    ax.errorbar(shortlist['weighted_mase'], y, xerr=shortlist['error_std'], fmt='o',
                color=OKABE_ITO[4], ecolor=OKABE_ITO[4], elinewidth=1, capsize=2, markersize=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel('Weighted MASE (+/- fold error std.)')
    ax.set_title(f'Stream B Shortlist: Top {len(shortlist)} Pareto-Optimal Candidates', fontweight='bold')
    fig.tight_layout()

    save_figure_pdf(fig, output_path, 'Stream B Shortlist (Magnified)', plot_data=shortlist)
    plt.close(fig)
    logger.info(f"fig03c: {len(shortlist)} shortlisted candidates")


def fig03d_best_A_vs_best_B(table7: pd.DataFrame, output_path: Path) -> None:
    """
    Figure 3D (spec section 26): Best_A vs Best_B on H1/H2/H4 MASE,
    weighted MASE, stability (error_std), feature count, runtime, and
    normalized MCDM criterion (vikor_Q) - grouped dot plot (not a radar
    chart, per spec section 26's explicit preference for readability).
    """
    configure_matplotlib_for_pdf()
    logger = get_logger()

    if table7.empty or 'stream_winner' not in table7.columns:
        logger.warning("fig03d: Table 7 is empty or malformed, skipping")
        return

    criteria = [c for c in ('mase_h1', 'mase_h2', 'mase_h4', 'weighted_mase', 'worst_horizon_mase',
                             'error_std', 'n_features', 'total_runtime_seconds') if c in table7.columns]
    winners = table7[table7['stream_winner'].isin(['Best_A', 'Best_B'])].set_index('stream_winner')
    if len(winners) < 2:
        logger.warning("fig03d: need both Best_A and Best_B rows, skipping")
        return

    # Normalize each criterion to [0, 1] across just these two rows so both
    # are visually comparable on one axis regardless of native units.
    normalized = pd.DataFrame(index=winners.index)
    for c in criteria:
        lo, hi = winners[c].min(), winners[c].max()
        normalized[c] = 0.5 if hi == lo else (winners[c] - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.2))
    y = range(len(criteria))
    for i, label in enumerate(winners.index):
        color = OKABE_ITO[0] if label == 'Best_A' else OKABE_ITO[4]
        ax.scatter(normalized.loc[label, criteria], y, color=color, s=40, label=label, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(criteria, fontsize=7)
    ax.set_xlabel('Normalized value (0 = better of the two, 1 = worse of the two, per criterion)')
    ax.set_title('Best_A vs. Best_B: Criterion Profile', fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(axis='x', linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    save_figure_pdf(fig, output_path, 'Best_A vs. Best_B', plot_data=winners[criteria].reset_index())
    plt.close(fig)
    logger.info("fig03d: Best_A vs Best_B criterion profile")
