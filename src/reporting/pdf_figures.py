"""
PDF figure infrastructure (spec section 21): vector-first output, embedded
TrueType fonts, journal-ready dimensions, and a companion plot-data CSV for
every figure. All 19 required figures (spec section 22) should call
`save_figure_pdf` as their final step rather than a bare `plt.savefig`.
"""
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from ..core.logging_utils import get_logger

# Journal-ready dimensions (spec section 21.2), in inches.
SINGLE_COLUMN_WIDTH_IN = 3.5   # ~89 mm
DOUBLE_COLUMN_WIDTH_IN = 7.0   # ~178 mm

# Colourblind-accessible qualitative palette (Okabe-Ito), used consistently
# across all figures needing categorical colour (spec 21.1: "colourblind-
# accessible palettes").
OKABE_ITO = [
    '#E69F00', '#56B4E9', '#009E73', '#F0E442',
    '#0072B2', '#D55E00', '#CC79A7', '#000000',
]


def configure_matplotlib_for_pdf() -> None:
    """
    Global rcParams for spec section 21.1's PDF quality requirements:
    embedded TrueType fonts (not converted to paths), 600 dpi for any
    raster content, consistent typography.
    """
    matplotlib.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 600,
        'figure.dpi': 150,
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'font.family': 'sans-serif',
        'axes.unicode_minus': False,
        'svg.fonttype': 'none',
    })


def save_figure_pdf(
    fig: 'plt.Figure',
    output_path: Path,
    title: str,
    plot_data: Optional[pd.DataFrame] = None,
    plot_data_path: Optional[Path] = None,
) -> None:
    """
    Save a figure as a spec-compliant PDF (vector, embedded fonts, metadata)
    plus its underlying plot-data CSV (spec 21.1: "Every figure must also
    have an underlying plot-data CSV").

    Args:
        fig: The matplotlib Figure to save.
        output_path: Destination .pdf path (parent dirs created if needed).
        title: Embedded PDF title metadata and (if not already set) figure title.
        plot_data: DataFrame backing this figure's content - saved alongside
            the PDF as plot_data_path (or output_path with .csv suffix if
            plot_data_path is not given).
        plot_data_path: Explicit path for the plot-data CSV (default:
            same directory/stem as output_path, under a `plot_data/` sibling
            directory per spec section 24's `figures/plot_data/` layout).
    """
    logger = get_logger()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.02,
        metadata={'Title': title, 'Creator': 'Q-DECEM reproducible pipeline'},
    )
    logger.info(f"Saved figure: {output_path}")

    if plot_data is not None:
        if plot_data_path is None:
            plot_data_dir = output_path.parent.parent / 'plot_data'
            plot_data_dir.mkdir(parents=True, exist_ok=True)
            plot_data_path = plot_data_dir / f'{output_path.stem}.csv'
        else:
            plot_data_path = Path(plot_data_path)
            plot_data_path.parent.mkdir(parents=True, exist_ok=True)
        plot_data.to_csv(plot_data_path, index=False)
        logger.info(f"Saved plot data: {plot_data_path}")
