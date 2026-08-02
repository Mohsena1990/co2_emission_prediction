"""
Tests for feature-engineering leakage fixes: intensity features (3.1),
and direct horizon-target alignment (section 12). CEI-lagging is no longer
part of this module - CEI is fully removed per the revised spec section 2
(see test_tec_cei_removed.py), superseding the earlier bug-3.2 governance
approach that used to be tested here.

Includes a leakage-sentinel test (section 25): an artificial signal placed
only in what would be the outer-test period must not leak into any
lag/intensity/horizon-target feature derived from data before that period.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    create_intensity_features,
    create_direct_horizon_targets, create_lag_features,
)


def _quarterly_df(n=40, seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.date_range('2000-01-01', periods=n, freq='QS')
    return pd.DataFrame({
        'CO2e': 1000 + rng.normal(0, 10, n).cumsum(),
        'Population': 50000 + np.arange(n) * 10.0,
    }, index=dates)


class TestIntensityFeatureLeakageFix:
    def test_numerator_and_denominator_are_lagged(self):
        df = _quarterly_df()
        out = create_intensity_features(df, target_col='CO2e', denominator_cols=['Population'], min_lag=1)
        col = 'CO2e_per_Population'
        assert col in out.columns

        for i in range(1, len(df)):
            expected = df['CO2e'].iloc[i - 1] / df['Population'].iloc[i - 1]
            actual = out[col].iloc[i]
            if pd.isna(actual):
                continue
            assert actual == pytest.approx(expected)

        # Row 0 must be NaN - no lag-1 data available at the first observation
        assert pd.isna(out[col].iloc[0])

    def test_contemporaneous_co2e_never_appears_in_ratio(self):
        # Regression guard for bug 3.1: row t's intensity feature must not
        # equal CO2e_t / Population_t (the old, leaky formula).
        df = _quarterly_df()
        out = create_intensity_features(df, target_col='CO2e', denominator_cols=['Population'], min_lag=1)
        leaky_formula = df['CO2e'] / df['Population']
        col = 'CO2e_per_Population'
        # They must differ (except by coincidence) for the interior of the series
        diffs = (out[col].iloc[5:] - leaky_formula.iloc[5:]).abs()
        assert (diffs > 1e-9).all()

    def test_min_lag_must_be_at_least_one(self):
        df = _quarterly_df()
        with pytest.raises(ValueError):
            create_intensity_features(df, target_col='CO2e', denominator_cols=['Population'], min_lag=0)


class TestDirectHorizonTargets:
    def test_shift_alignment(self):
        dates = pd.date_range('2000-01-01', periods=20, freq='QS')
        y = pd.Series(np.arange(20, dtype=float), index=dates, name='CO2e')

        targets = create_direct_horizon_targets(y, horizons=[1, 2, 4])

        for h in [1, 2, 4]:
            yh = targets[h]
            # y_h.loc[t] == y.loc[t + h]
            for i in range(len(y) - h):
                assert yh.iloc[i] == y.iloc[i + h]
            # Last h rows must be NaN (target date falls outside the series)
            assert yh.iloc[-h:].isna().all()

    def test_requires_datetime_index(self):
        y = pd.Series(np.arange(10, dtype=float))
        with pytest.raises(ValueError):
            create_direct_horizon_targets(y, horizons=[1])

    def test_rejects_horizon_below_one(self):
        dates = pd.date_range('2000-01-01', periods=10, freq='QS')
        y = pd.Series(np.arange(10, dtype=float), index=dates)
        with pytest.raises(ValueError):
            create_direct_horizon_targets(y, horizons=[0])


class TestLeakageSentinel:
    """
    Plant an artificial signal that only exists in the held-out "future"
    tail of the series and confirm it cannot leak backwards into features
    or horizon targets computed for earlier origins.
    """

    def test_sentinel_signal_does_not_leak_into_past_lag_features(self):
        n = 40
        dates = pd.date_range('2000-01-01', periods=n, freq='QS')
        values = np.ones(n) * 100.0
        # Sentinel: an enormous, obviously-anomalous spike placed ONLY in
        # the last 4 rows (the simulated "outer test" period).
        sentinel_start = n - 4
        values[sentinel_start:] = 1e9
        df = pd.DataFrame({'CO2e': values}, index=dates)

        lag_df = create_lag_features(df, ['CO2e'], lags=[1, 2, 3, 4])

        # No lag feature for any row BEFORE the sentinel period may see the
        # sentinel value (lags only look backwards). NaNs from the initial
        # lag warm-up are expected and excluded from the check.
        pre_sentinel = lag_df.iloc[:sentinel_start]
        assert (pre_sentinel.fillna(0).abs() < 1e8).all().all(), (
            "Sentinel value from the future leaked into a lag feature for "
            "an earlier origin"
        )

    def test_sentinel_signal_does_not_leak_into_intensity_features(self):
        n = 40
        dates = pd.date_range('2000-01-01', periods=n, freq='QS')
        co2e = np.ones(n) * 100.0
        pop = np.ones(n) * 1000.0
        sentinel_start = n - 4
        co2e[sentinel_start:] = 1e9
        pop[sentinel_start:] = 1e-3  # would blow up the ratio if leaked
        df = pd.DataFrame({'CO2e': co2e, 'Population': pop}, index=dates)

        out = create_intensity_features(df, target_col='CO2e', denominator_cols=['Population'], min_lag=1)

        # Rows strictly before the sentinel period (accounting for the lag)
        # must be unaffected by the sentinel values. The leading NaN from
        # the lag warm-up is expected and excluded from the check.
        safe_rows = out.iloc[:sentinel_start]
        assert (safe_rows.fillna(0).abs() < 1e6).all().all()

    def test_sentinel_does_not_leak_into_direct_horizon_targets_for_origin(self):
        # The whole point of direct horizon targets is that origin t's
        # target IS y_{t+h} - that's by design, not leakage. The leakage
        # sentinel here checks the FEATURE side (X_t) never contains
        # information from t+h; this is exercised via the lag/intensity
        # tests above. Here we just confirm the horizon-target shift itself
        # doesn't reach further than h steps ahead.
        n = 20
        dates = pd.date_range('2000-01-01', periods=n, freq='QS')
        y = pd.Series(np.arange(n, dtype=float), index=dates)
        targets = create_direct_horizon_targets(y, horizons=[1])
        # Origin at index 0 must see exactly y[1], not any later value.
        assert targets[1].iloc[0] == y.iloc[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
