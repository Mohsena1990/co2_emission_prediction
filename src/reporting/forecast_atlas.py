"""
Forecast atlas (spec section 25): genuine out-of-sample forecast figures.
36 PDFs total - 24 family winners (A1-A4/B1-B4 x H1/H2/H4), 6 stream
winners (Best_A/Best_B x H1/H2/H4), 3 final winner (Best_Overall x
H1/H2/H4), 3 Best_A-vs-Best_B comparisons (x H1/H2/H4).

Every plotted point is read directly from a saved fold_predictions/
stream_{A,B}/predictions.csv row (spec 25's "genuine out-of-sample
forecast" requirement - never a refit/in-sample fitted value). The actual
CO2e series is the FULL history (data/processed/y.parquet), not just the
test-window rows, so the forecast panel reads naturally as "one line of
real history, dots wherever this cell forecasted forward" rather than a
disconnected fragment.
"""
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..core.logging_utils import get_logger
from ..core.utils import quarter_to_date
from .pdf_figures import save_figure_pdf, configure_matplotlib_for_pdf, OKABE_ITO, DOUBLE_COLUMN_WIDTH_IN

COVID_PERIOD = (quarter_to_date('2020Q1'), quarter_to_date('2021Q4') + pd.DateOffset(months=3) - pd.Timedelta(days=1))
ENERGY_CRISIS_PERIOD = (quarter_to_date('2022Q1'), quarter_to_date('2023Q4') + pd.DateOffset(months=3) - pd.Timedelta(days=1))


def select_family_winner(metrics_df: pd.DataFrame, configuration: str) -> Optional[pd.Series]:
    """
    Spec section 25.1: "the highest-ranked candidate within each family" -
    the single (fs_option, model) cell with the lowest weighted_mase among
    every candidate carrying this `configuration` label, WITHOUT
    re-invoking Pareto/VIKOR (that's the global stream-level MCDM, spec
    section 25.1 explicitly says family winners must not replace it - this
    is a simple local sort for visual reporting only).
    """
    if metrics_df.empty or 'configuration' not in metrics_df.columns:
        return None
    sub = metrics_df[metrics_df['configuration'] == configuration]
    if sub.empty or sub['weighted_mase'].isna().all():
        return None
    return sub.loc[sub['weighted_mase'].idxmin()]


def _cell_predictions(predictions_df: pd.DataFrame, configuration: str, fs_option: str, model: str,
                       horizon: int) -> pd.DataFrame:
    mask = (
        (predictions_df['configuration'] == configuration) & (predictions_df['fs_option'] == fs_option) &
        (predictions_df['model'] == model) & (predictions_df['horizon'] == horizon)
    )
    return predictions_df[mask].sort_values('target_date')


def fig_forecast_single(
    actual_full: pd.Series,
    cell_predictions: pd.DataFrame,
    meta: Dict,
    output_path: Path,
) -> None:
    """
    One family/stream/overall-winner forecast figure (spec section 25):
    top panel = full actual history + out-of-sample forecast points;
    bottom panel = residuals with a zero reference line. COVID and
    energy-crisis periods shaded on both panels. Title carries
    configuration/model/FS/horizon/MASE/feature-count (spec 25.1's
    required annotations).
    """
    configure_matplotlib_for_pdf()
    logger = get_logger()

    if cell_predictions.empty:
        logger.warning(f"fig_forecast_single ({meta.get('label', '?')}): no predictions, skipping")
        return

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(DOUBLE_COLUMN_WIDTH_IN, 4.2), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    for ax in (ax_top, ax_bottom):
        ax.axvspan(*COVID_PERIOD, color=OKABE_ITO[5], alpha=0.12, label='COVID (2020Q1-2021Q4)', zorder=0)
        ax.axvspan(*ENERGY_CRISIS_PERIOD, color=OKABE_ITO[1], alpha=0.12, label='Energy crisis (2022Q1-2023Q4)', zorder=0)

    ax_top.plot(actual_full.index, actual_full.values, color='grey', linewidth=1.0, label='Actual CO2e', zorder=2)
    ax_top.scatter(cell_predictions['target_date'], cell_predictions['predicted'],
                    color=OKABE_ITO[4], s=18, zorder=3, label='Out-of-sample forecast')
    ax_top.scatter(cell_predictions['target_date'], cell_predictions['actual'],
                    facecolors='none', edgecolors='grey', s=18, zorder=3, label='Actual (forecast window)')
    ax_top.set_ylabel('CO2e')
    ax_top.legend(fontsize=6, loc='upper left', ncol=2)

    ax_bottom.axhline(0, color='black', linewidth=0.6, zorder=2)
    ax_bottom.scatter(cell_predictions['target_date'], cell_predictions['residual'],
                       color=OKABE_ITO[5], s=14, zorder=3)
    ax_bottom.set_ylabel('Residual\n(predicted - actual)')
    ax_bottom.set_xlabel('Target date')

    title = (
        f"{meta.get('label', '')}: {meta['configuration']}/{meta.get('fs_option', 'all_features')}/"
        f"{meta['model']}, H{meta['horizon']} "
        f"(MASE={meta.get('mase', float('nan')):.3f}, n_features={meta.get('n_features', '?')})"
    )
    fig.suptitle(title, fontweight='bold', fontsize=8, y=0.99)
    fig.tight_layout()

    save_figure_pdf(fig, output_path, title, plot_data=cell_predictions)
    plt.close(fig)
    logger.info(f"forecast atlas: {output_path.name} ({len(cell_predictions)} forecast points)")


def fig_forecast_comparison(
    actual_full: pd.Series,
    cell_predictions_a: pd.DataFrame,
    cell_predictions_b: pd.DataFrame,
    meta_a: Dict,
    meta_b: Dict,
    output_path: Path,
) -> None:
    """Spec section 25.4: Best_A vs Best_B, aligned small panels (not a
    crowded single overlay) for one horizon."""
    configure_matplotlib_for_pdf()
    logger = get_logger()

    if cell_predictions_a.empty and cell_predictions_b.empty:
        logger.warning("fig_forecast_comparison: no predictions for either winner, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.0), sharey=True)

    for ax, preds, meta, color in (
        (axes[0], cell_predictions_a, meta_a, OKABE_ITO[0]),
        (axes[1], cell_predictions_b, meta_b, OKABE_ITO[4]),
    ):
        ax.axvspan(*COVID_PERIOD, color=OKABE_ITO[5], alpha=0.12, zorder=0)
        ax.axvspan(*ENERGY_CRISIS_PERIOD, color=OKABE_ITO[1], alpha=0.12, zorder=0)
        ax.plot(actual_full.index, actual_full.values, color='grey', linewidth=1.0, zorder=2)
        if not preds.empty:
            ax.scatter(preds['target_date'], preds['predicted'], color=color, s=18, zorder=3)
        mase = meta.get('mase', float('nan'))
        ax.set_title(f"{meta.get('label', '')}: {meta['configuration']}/{meta.get('fs_option', 'all_features')}/"
                     f"{meta['model']}\nMASE={mase:.3f}", fontsize=7)
        ax.set_xlabel('Target date')

    axes[0].set_ylabel('CO2e')
    fig.suptitle(f"Best_A vs. Best_B: H{meta_a.get('horizon', '?')} Out-of-Sample Forecasts",
                 fontweight='bold', fontsize=8, y=1.04)
    fig.tight_layout()

    combined_plot_data = pd.concat([
        cell_predictions_a.assign(winner='Best_A'), cell_predictions_b.assign(winner='Best_B')
    ], ignore_index=True) if not (cell_predictions_a.empty and cell_predictions_b.empty) else pd.DataFrame()
    save_figure_pdf(fig, output_path, 'Best_A vs. Best_B Forecasts', plot_data=combined_plot_data)
    plt.close(fig)
    logger.info(f"forecast atlas: {output_path.name}")
