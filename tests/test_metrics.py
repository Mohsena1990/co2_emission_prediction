"""
Tests for evaluation metrics, in particular MASE/sMAPE/weighted-MASE, which
were previously entirely absent despite MASE being specified as the primary
comparative metric.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    calculate_mase, calculate_mase_scale, calculate_smape,
    calculate_weighted_mase, compute_all_metrics, evaluate_by_horizon,
)


class TestMASE:
    def test_perfect_forecast_gives_zero_mase(self):
        y_train = np.array([100, 102, 98, 105, 101, 103, 99, 106], dtype=float)
        y_true = np.array([104.0, 108.0])
        y_pred = y_true.copy()
        mase = calculate_mase(y_true, y_pred, y_train, seasonal_period=4)
        assert mase == pytest.approx(0.0, abs=1e-10)

    def test_seasonal_naive_test_forecast_close_to_one(self):
        # A seasonally-patterned but trending (non-degenerate) series: using
        # the seasonal-naive prediction on the test period should give a
        # MASE near 1, since the test-period naive error is drawn from the
        # same generating process as the in-sample scale.
        rng = np.random.RandomState(0)
        base = np.tile([100.0, 110.0, 90.0, 120.0], 12)
        trend = np.arange(len(base)) * 0.5
        noise = rng.normal(0, 1.0, size=len(base))
        series = base + trend + noise

        y_train = series[:40]
        y_test_true = series[40:44]
        y_test_naive_pred = series[36:40]  # y_{t-4} for the test period
        mase = calculate_mase(y_test_true, y_test_naive_pred, y_train, seasonal_period=4)
        assert 0.3 < mase < 3.0

    def test_mase_scale_requires_enough_history(self):
        with pytest.raises(ValueError):
            calculate_mase_scale(np.array([1.0, 2.0, 3.0]), seasonal_period=4)

    def test_mase_raises_on_length_mismatch(self):
        with pytest.raises(ValueError):
            calculate_mase(np.array([1.0, 2.0]), np.array([1.0]), np.arange(10, dtype=float))

    def test_mase_never_uses_test_fold_for_scale(self):
        # Regression guard: the scale must depend only on y_train_history,
        # not on y_true/y_pred. Changing y_true/y_pred must not change the
        # scale-derived denominator's dependency on train history.
        y_train = np.array([100, 105, 95, 110, 102, 108, 96, 112], dtype=float)
        y_true_a = np.array([120.0, 130.0])
        y_true_b = np.array([50.0, 10.0])  # wildly different test period
        y_pred = np.array([100.0, 100.0])
        scale = calculate_mase_scale(y_train, seasonal_period=4)
        mase_a = calculate_mase(y_true_a, y_pred, y_train, seasonal_period=4)
        mase_b = calculate_mase(y_true_b, y_pred, y_train, seasonal_period=4)
        # scale is identical regardless of the test fold contents
        assert mase_a * scale == pytest.approx(np.mean(np.abs(y_true_a - y_pred)))
        assert mase_b * scale == pytest.approx(np.mean(np.abs(y_true_b - y_pred)))


class TestSMAPE:
    def test_perfect_forecast_zero(self):
        y = np.array([10.0, 20.0, 30.0])
        assert calculate_smape(y, y) == pytest.approx(0.0)

    def test_bounded_0_200(self):
        y_true = np.array([1.0, -5.0, 100.0])
        y_pred = np.array([100.0, 5.0, -1.0])
        s = calculate_smape(y_true, y_pred)
        assert 0.0 <= s <= 200.0


class TestWeightedMASE:
    def test_matches_manual_weighted_sum(self):
        mase_by_h = {1: 0.8, 2: 1.0, 4: 1.2}
        weights = {1: 0.5, 2: 0.3, 4: 0.2}
        expected = 0.5 * 0.8 + 0.3 * 1.0 + 0.2 * 1.2
        assert calculate_weighted_mase(mase_by_h, weights) == pytest.approx(expected)

    def test_missing_horizon_returns_nan_not_partial_average(self):
        mase_by_h = {1: 0.8, 4: 1.2}  # horizon 2 missing
        weights = {1: 0.5, 2: 0.3, 4: 0.2}
        result = calculate_weighted_mase(mase_by_h, weights)
        assert np.isnan(result)


class TestComputeAllMetrics:
    def test_mase_nan_when_no_history_supplied(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        metrics = compute_all_metrics(y_true, y_pred)
        assert np.isnan(metrics.mase)
        assert np.isfinite(metrics.smape)

    def test_mase_populated_when_history_supplied(self):
        y_true = np.array([104.0, 108.0])
        y_pred = np.array([103.0, 109.0])
        y_train = np.array([100, 102, 98, 105, 101, 103, 99, 106], dtype=float)
        metrics = compute_all_metrics(y_true, y_pred, y_train_history=y_train)
        assert np.isfinite(metrics.mase)


class TestEvaluateByHorizon:
    def test_pooled_mase_across_folds(self):
        predictions_df = pd.DataFrame({
            'fold_id': [0, 0, 1, 1],
            'horizon': [1, 1, 1, 1],
            'actual': [100.0, 105.0, 110.0, 108.0],
            'predicted': [98.0, 107.0, 111.0, 106.0],
        })
        train_history_by_fold = {
            0: np.array([90, 95, 85, 100, 92, 96, 88, 102], dtype=float),
            1: np.array([90, 95, 85, 100, 92, 96, 88, 102, 94, 99], dtype=float),
        }
        results = evaluate_by_horizon(
            predictions_df, horizons=[1],
            train_history_by_fold=train_history_by_fold
        )
        assert 1 in results
        assert np.isfinite(results[1].mase)
        assert results[1].n_samples == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
