"""
Tests for the naive reference baselines (spec section 18):
src/models/naive.py (NaiveLag1Forecaster, NaiveLag4Forecaster) and
src/pipeline/experiment.py::run_naive_baseline.

Regression coverage for a real bug caught during Phase 6 smoke-testing:
run_naive_baseline originally called `model.predict(X_test)`, but X_test in
this direct-horizon framework is indexed by the forecast ORIGIN, not the
TARGET date - NaiveLag4Forecaster's seasonal (t-4) lookup silently
degenerated to persistence (identical to naive_lag1) at every horizon
except h=4, where origin+h-4 happens to equal the origin itself. The fix
passes `y_test.to_frame()` (indexed by target date) instead.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.core.config import Config
from src.models.naive import NaiveLag1Forecaster, NaiveLag4Forecaster
from src.splits.walk_forward import create_walk_forward_splits
from src.pipeline.experiment import run_naive_baseline


def _quarterly_series(n=24, start='2000-01-01'):
    idx = pd.date_range(start, periods=n, freq='QS')
    return pd.Series(np.arange(n, dtype=float), index=idx)


class TestNaiveLag1Forecaster:
    def test_predicts_last_train_value_regardless_of_target_date(self):
        y = _quarterly_series(12)
        model = NaiveLag1Forecaster()
        model.fit(None, y)

        for h in (1, 2, 4):
            target = pd.DataFrame(index=[y.index[-1] + pd.DateOffset(months=3 * h)])
            preds = model.predict(target)
            assert np.allclose(preds, y.iloc[-1])


class TestNaiveLag4Forecaster:
    def test_seasonal_lookup_uses_target_date_minus_four_quarters(self):
        y = _quarterly_series(20)
        model = NaiveLag4Forecaster()
        model.fit(None, y)

        last_train_date = y.index[-1]
        for h in (1, 2, 4):
            target_date = last_train_date + pd.DateOffset(months=3 * h)
            lookup_date = target_date - pd.DateOffset(months=12)
            expected = y.loc[lookup_date] if lookup_date in y.index else y.iloc[-1]
            pred = model.predict(pd.DataFrame(index=[target_date]))
            assert np.allclose(pred, expected), f"h={h}: expected {expected}, got {pred}"

    def test_h4_seasonal_naive_equals_persistence_at_the_origin(self):
        """At h=4, origin+h-4 == origin exactly, so seasonal-naive and
        persistence must coincide - the one case the pre-fix bug happened
        to get right by accident."""
        y = _quarterly_series(20)
        naive1, naive4 = NaiveLag1Forecaster(), NaiveLag4Forecaster()
        naive1.fit(None, y)
        naive4.fit(None, y)

        target = pd.DataFrame(index=[y.index[-1] + pd.DateOffset(months=12)])
        assert np.allclose(naive1.predict(target), naive4.predict(target))

    def test_h1_seasonal_naive_differs_from_persistence(self):
        """At h=1 (and h=2), seasonal-naive must NOT collapse to
        persistence - this is exactly the regression the bug introduced."""
        y = _quarterly_series(20)
        naive1, naive4 = NaiveLag1Forecaster(), NaiveLag4Forecaster()
        naive1.fit(None, y)
        naive4.fit(None, y)

        for h in (1, 2):
            target = pd.DataFrame(index=[y.index[-1] + pd.DateOffset(months=3 * h)])
            pred1 = naive1.predict(target)
            pred4 = naive4.predict(target)
            assert not np.allclose(pred1, pred4), f"h={h}: naive_lag4 collapsed to naive_lag1"

    def test_predict_requires_datetime_index(self):
        y = _quarterly_series(20)
        model = NaiveLag4Forecaster()
        model.fit(None, y)
        with pytest.raises(ValueError):
            model.predict(np.zeros((3, 2)))


class TestRunNaiveBaselineIntegration:
    """End-to-end through run_naive_baseline with a real multi-horizon
    CVPlan - the level at which the X_test-vs-y_test bug actually manifested."""

    def _config(self):
        config = Config()
        config.splits.min_train_size = 20
        config.splits.test_size = 4
        config.splits.horizons = [1, 2, 4]
        config.splits.horizon_weights = {1: 0.5, 2: 0.3, 4: 0.2}
        config.data.target_transform = 'none'
        return config

    def _Xy(self, n=40):
        dates = pd.date_range('2015-01-01', periods=n, freq='QS')
        rng = np.random.RandomState(0)
        X = pd.DataFrame({'feat1': rng.randn(n)}, index=dates)
        y = pd.Series(np.arange(n, dtype=float) + rng.randn(n) * 0.01, index=dates, name='CO2e')
        return X, y

    def test_naive_lag4_predictions_vary_by_horizon_not_just_origin(self):
        X, y = self._Xy()
        config = self._config()
        eval_plan = create_walk_forward_splits(X, y, config.splits)
        assert eval_plan.n_folds > 0

        result = run_naive_baseline(X, y, eval_plan, 'naive_lag4', config, run_id='test_run')
        preds_df = pd.DataFrame(result['predictions'])
        prov_df = pd.DataFrame(result['fold_provenance'])
        assert not preds_df.empty

        # Each fold_id in this direct-horizon CV plan is single-horizon (one
        # fold per (origin, horizon) pair), so compare across folds that
        # share the same forecast_origin instead: h=1 and h=4 predictions
        # from the SAME origin must differ (h=4 looks back to the origin
        # itself; h=1 looks back 3 quarters before it) - the pre-fix bug
        # made every horizon at a given origin identical.
        merged = preds_df.merge(prov_df[['fold_id', 'forecast_origin']], on='fold_id')
        checked_any_origin = False
        for origin, group in merged.groupby('forecast_origin'):
            h1 = group[group['horizon'] == 1]['predicted']
            h4 = group[group['horizon'] == 4]['predicted']
            if len(h1) and len(h4):
                checked_any_origin = True
                assert h1.values[0] != h4.values[0], f"origin {origin}: h1 collapsed to h4's value"
        assert checked_any_origin, "no forecast origin had both h1 and h4 predictions to compare"

    def test_naive_lag1_and_naive_lag4_not_identical_across_all_horizons(self):
        X, y = self._Xy()
        config = self._config()
        eval_plan = create_walk_forward_splits(X, y, config.splits)

        result1 = run_naive_baseline(X, y, eval_plan, 'naive_lag1', config, run_id='test_run')
        result4 = run_naive_baseline(X, y, eval_plan, 'naive_lag4', config, run_id='test_run')

        df1 = pd.DataFrame(result1['predictions']).sort_values(['fold_id', 'horizon', 'target_date'])
        df4 = pd.DataFrame(result4['predictions']).sort_values(['fold_id', 'horizon', 'target_date'])
        assert not df1.empty and not df4.empty

        # h=4 rows may legitimately coincide; h=1/h=2 rows must not, unless
        # the underlying series happens to be perfectly seasonally flat
        # (not the case for this synthetic linearly-increasing series).
        merged = df1.merge(
            df4, on=['fold_id', 'horizon', 'target_date'], suffixes=('_lag1', '_lag4')
        )
        non_h4 = merged[merged['horizon'] != 4]
        assert not non_h4.empty
        assert not np.allclose(non_h4['predicted_lag1'], non_h4['predicted_lag4'])
