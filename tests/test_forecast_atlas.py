"""
Tests for src/reporting/forecast_atlas.py (spec section 25): family/stream/
overall forecast-figure selection and PDF generation.
"""
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.reporting.forecast_atlas import (
    select_family_winner, fig_forecast_single, fig_forecast_comparison,
    COVID_PERIOD, ENERGY_CRISIS_PERIOD,
)

HAS_PDFFONTS = shutil.which('pdffonts') is not None


def _assert_valid_pdf(path: Path):
    assert path.exists()
    assert path.stat().st_size > 0
    with open(path, 'rb') as f:
        assert f.read(5) == b'%PDF-'


def _fake_metrics(configs=('A1', 'A2'), models=('ridge', 'catboost')):
    rng = np.random.RandomState(0)
    rows = []
    for c in configs:
        for m in models:
            rows.append({
                'configuration': c, 'fs_option': 'all_features', 'model': m,
                'weighted_mase': rng.uniform(0.5, 1.5), 'n_features': 11,
                'mase_h1': rng.uniform(0.4, 1.2), 'mase_h2': rng.uniform(0.5, 1.3), 'mase_h4': rng.uniform(0.6, 1.4),
            })
    return pd.DataFrame(rows)


def _fake_predictions(configuration='A1', fs_option='all_features', model='ridge', horizon=1, n=8):
    dates = pd.date_range('2022-04-01', periods=n, freq='QS')
    rng = np.random.RandomState(1)
    actual = 100000 + rng.randn(n) * 5000
    predicted = actual + rng.randn(n) * 3000
    return pd.DataFrame({
        'configuration': configuration, 'fs_option': fs_option, 'model': model, 'horizon': horizon,
        'target_date': dates, 'actual': actual, 'predicted': predicted, 'residual': predicted - actual,
    })


def _fake_actual_history(n=40):
    dates = pd.date_range('2015-01-01', periods=n, freq='QS')
    rng = np.random.RandomState(2)
    return pd.Series(100000 + np.cumsum(rng.randn(n) * 2000), index=dates)


class TestSelectFamilyWinner:
    def test_picks_lowest_weighted_mase_within_configuration(self):
        metrics = pd.DataFrame([
            {'configuration': 'A1', 'model': 'ridge', 'weighted_mase': 1.2, 'n_features': 11},
            {'configuration': 'A1', 'model': 'catboost', 'weighted_mase': 0.7, 'n_features': 11},
            {'configuration': 'A2', 'model': 'ridge', 'weighted_mase': 0.5, 'n_features': 25},
        ])
        winner = select_family_winner(metrics, 'A1')
        assert winner['model'] == 'catboost'

    def test_empty_metrics_returns_none(self):
        """Regression: an empty (columnless) DataFrame - e.g. Stream B not
        yet computed - previously raised KeyError('configuration') instead
        of returning None, crashing the whole forecast-atlas script."""
        assert select_family_winner(pd.DataFrame(), 'A1') is None

    def test_missing_configuration_returns_none(self):
        metrics = _fake_metrics(configs=('A2',))
        assert select_family_winner(metrics, 'A1') is None

    def test_all_nan_weighted_mase_returns_none(self):
        metrics = pd.DataFrame([
            {'configuration': 'A1', 'model': 'ridge', 'weighted_mase': np.nan, 'n_features': 11},
        ])
        assert select_family_winner(metrics, 'A1') is None


class TestFigForecastSingle:
    def test_produces_valid_pdf(self, tmp_path):
        actual = _fake_actual_history()
        preds = _fake_predictions()
        meta = {'label': 'A1 family winner', 'configuration': 'A1', 'fs_option': 'all_features',
                'model': 'ridge', 'horizon': 1, 'mase': 0.8, 'n_features': 11}
        output_path = tmp_path / 'fig_family_A1_H1.pdf'
        fig_forecast_single(actual, preds, meta, output_path)
        _assert_valid_pdf(output_path)

    def test_handles_empty_predictions(self, tmp_path):
        actual = _fake_actual_history()
        output_path = tmp_path / 'fig_empty.pdf'
        fig_forecast_single(actual, pd.DataFrame(), {'label': 'x', 'configuration': 'A1', 'model': 'ridge',
                                                       'horizon': 1}, output_path)
        assert not output_path.exists()

    @pytest.mark.skipif(not HAS_PDFFONTS, reason="pdffonts not available")
    def test_fonts_embedded(self, tmp_path):
        actual = _fake_actual_history()
        preds = _fake_predictions()
        meta = {'label': 'Best_A', 'configuration': 'A1', 'fs_option': 'all_features',
                'model': 'ridge', 'horizon': 1, 'mase': 0.8, 'n_features': 11}
        output_path = tmp_path / 'fig.pdf'
        fig_forecast_single(actual, preds, meta, output_path)
        result = subprocess.run(['pdffonts', str(output_path)], capture_output=True, text=True)
        lines = [l for l in result.stdout.splitlines() if l and not l.startswith('name') and not l.startswith('---')]
        for line in lines:
            assert 'no' not in line.split()[-2:], f"a font is not embedded: {line}"

    def test_plot_data_matches_predictions(self, tmp_path):
        actual = _fake_actual_history()
        preds = _fake_predictions(n=6)
        meta = {'label': 'x', 'configuration': 'A1', 'model': 'ridge', 'horizon': 1, 'mase': 0.8, 'n_features': 11}
        output_path = tmp_path / 'pdf' / 'forecasting' / 'fig.pdf'
        fig_forecast_single(actual, preds, meta, output_path)
        plot_data = pd.read_csv(tmp_path / 'pdf' / 'plot_data' / 'fig.csv')
        assert len(plot_data) == 6


class TestFigForecastComparison:
    def test_produces_valid_pdf(self, tmp_path):
        actual = _fake_actual_history()
        preds_a = _fake_predictions(configuration='A2', model='ridge')
        preds_b = _fake_predictions(configuration='A1', fs_option='fs_xgboost_shap', model='catboost')
        meta_a = {'label': 'Best_A', 'configuration': 'A2', 'fs_option': 'all_features', 'model': 'ridge',
                  'horizon': 1, 'mase': 0.9}
        meta_b = {'label': 'Best_B', 'configuration': 'A1', 'fs_option': 'fs_xgboost_shap', 'model': 'catboost',
                  'horizon': 1, 'mase': 0.7}
        output_path = tmp_path / 'fig_best_A_vs_B_forecast_H1.pdf'
        fig_forecast_comparison(actual, preds_a, preds_b, meta_a, meta_b, output_path)
        _assert_valid_pdf(output_path)

    def test_handles_both_empty(self, tmp_path):
        actual = _fake_actual_history()
        output_path = tmp_path / 'fig_empty.pdf'
        fig_forecast_comparison(actual, pd.DataFrame(), pd.DataFrame(),
                                 {'configuration': 'A1', 'model': 'ridge'},
                                 {'configuration': 'A2', 'model': 'ridge'}, output_path)
        assert not output_path.exists()


class TestRegimePeriods:
    def test_covid_period_is_2020q1_through_2021q4(self):
        start, end = COVID_PERIOD
        assert start.year == 2020 and start.month == 1
        assert end.year == 2021 and end.month == 12

    def test_energy_crisis_period_is_2022q1_through_2023q4(self):
        start, end = ENERGY_CRISIS_PERIOD
        assert start.year == 2022 and start.month == 1
        assert end.year == 2023 and end.month == 12

    def test_periods_do_not_overlap(self):
        assert COVID_PERIOD[1] < ENERGY_CRISIS_PERIOD[0]


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
