"""
Tests for electricity-grid data fusion (spec section 6): quarterly
aggregation correctness, completeness-threshold exclusion, imputation, and
raw-response cache idempotency (spec: "the package must cache raw responses
and avoid repeatedly requesting the same data").
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.grid.aggregate import (
    aggregate_to_quarterly, load_raw_halfhourly, _longest_true_streak, _impute_intensity
)
from src.grid.fetch import fetch_and_cache_range, _date_chunks


def _halfhourly_index(start: str, n_periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n_periods, freq='30min')


def _complete_quarter_df(quarter_start: str = '2020-01-01', n_days: int = 91) -> pd.DataFrame:
    """A fully-complete quarter (every expected half-hourly slot present)."""
    n = n_days * 48
    idx = _halfhourly_index(quarter_start, n)
    rng = np.random.RandomState(0)
    intensity = rng.uniform(50, 350, n)
    index_band = np.where(intensity > 300, 'high', np.where(intensity < 100, 'low', 'moderate'))
    df = pd.DataFrame({
        'intensity_actual': intensity,
        'intensity_forecast': intensity + rng.normal(0, 2, n),
        'intensity_index': index_band,
        'gen_gas': rng.uniform(20, 40, n),
        'gen_coal': rng.uniform(0, 2, n),
        'gen_wind': rng.uniform(10, 30, n),
        'gen_solar': rng.uniform(0, 15, n),
        'gen_biomass': rng.uniform(2, 8, n),
        'gen_hydro': rng.uniform(1, 3, n),
        'gen_imports': rng.uniform(3, 10, n),
        'gen_nuclear': rng.uniform(10, 20, n),
    }, index=idx)
    return df


class TestAggregateToQuarterly:
    def test_complete_quarter_is_included_with_correct_stats(self):
        df = _complete_quarter_df()
        quarterly, quality = aggregate_to_quarterly(df, min_completeness=0.95)

        assert len(quarterly) == 1
        assert quality.iloc[0]['inclusion_status'] == 'included'
        assert quality.iloc[0]['completeness_ratio'] == pytest.approx(1.0)

        expected_mean = df['intensity_actual'].mean()
        assert quarterly.iloc[0]['Grid_CI_mean'] == pytest.approx(expected_mean)
        assert quarterly.iloc[0]['Grid_CI_std'] == pytest.approx(df['intensity_actual'].std())

    def test_renewable_and_fossil_shares_sum_correctly(self):
        df = _complete_quarter_df()
        quarterly, _ = aggregate_to_quarterly(df, min_completeness=0.95)

        expected_renewable = df[['gen_biomass', 'gen_hydro', 'gen_solar', 'gen_wind']].sum(axis=1).mean()
        expected_fossil = df[['gen_gas', 'gen_coal']].sum(axis=1).mean()
        assert quarterly.iloc[0]['Grid_renewable_share'] == pytest.approx(expected_renewable)
        assert quarterly.iloc[0]['Grid_fossil_share'] == pytest.approx(expected_fossil)
        # nuclear is neither renewable nor fossil in this classification -
        # sanity check the two shares don't silently include it.
        assert quarterly.iloc[0]['Grid_renewable_share'] + quarterly.iloc[0]['Grid_fossil_share'] < 100

    def test_low_carbon_share_is_renewable_plus_nuclear_and_distinct(self):
        # spec section 7.2: nuclear is low-carbon but NOT renewable -
        # Grid_low_carbon_share must be strictly greater than
        # Grid_renewable_share whenever nuclear generation is present.
        df = _complete_quarter_df()
        quarterly, _ = aggregate_to_quarterly(df, min_completeness=0.95)

        expected_low_carbon = df[
            ['gen_biomass', 'gen_hydro', 'gen_solar', 'gen_wind', 'gen_nuclear']
        ].sum(axis=1).mean()
        assert quarterly.iloc[0]['Grid_low_carbon_share'] == pytest.approx(expected_low_carbon)
        assert quarterly.iloc[0]['Grid_low_carbon_share'] > quarterly.iloc[0]['Grid_renewable_share']

    def test_severely_incomplete_quarter_is_excluded_not_imputed(self):
        df = _complete_quarter_df()
        # Keep only 50% of rows -> well below the 0.95 default threshold.
        df_incomplete = df.iloc[::2]

        quarterly, quality = aggregate_to_quarterly(df_incomplete, min_completeness=0.95)

        assert len(quarterly) == 0  # excluded quarter contributes no feature row
        assert quality.iloc[0]['inclusion_status'] == 'excluded_incomplete'
        assert quality.iloc[0]['completeness_ratio'] < 0.95

    def test_completeness_threshold_is_configurable(self):
        df = _complete_quarter_df()
        df_half = df.iloc[::2]  # ~50% completeness

        _, quality_strict = aggregate_to_quarterly(df_half, min_completeness=0.95)
        _, quality_lenient = aggregate_to_quarterly(df_half, min_completeness=0.40)

        assert quality_strict.iloc[0]['inclusion_status'] == 'excluded_incomplete'
        assert quality_lenient.iloc[0]['inclusion_status'] == 'included'

    def test_quality_report_has_one_row_per_quarter_regardless_of_inclusion(self):
        df1 = _complete_quarter_df('2020-01-01', n_days=91)
        df2 = _complete_quarter_df('2020-04-01', n_days=91).iloc[::3]  # ~33% complete
        df = pd.concat([df1, df2])

        quarterly, quality = aggregate_to_quarterly(df, min_completeness=0.95)

        assert len(quality) == 2
        assert len(quarterly) == 1  # only the complete quarter passes


class TestImputation:
    def test_forecast_fallback_used_when_actual_missing(self):
        idx = _halfhourly_index('2020-01-01', 10)
        df = pd.DataFrame({
            'intensity_actual': [100.0, np.nan, 120.0, np.nan, np.nan, 90, 95, 100, 105, 110],
            'intensity_forecast': [101.0, 98.0, 119.0, 85.0, np.nan, 91, 96, 101, 106, 111],
        }, index=idx)

        imputed, n_fallback = _impute_intensity(df)

        # actual is NaN at indices 1, 3, 4; forecast is also NaN at index 4,
        # so only indices 1 and 3 count as a fallback (index 4 stays NaN).
        assert n_fallback == 2
        assert imputed.iloc[1] == 98.0  # fell back to forecast
        assert imputed.iloc[3] == 85.0
        assert pd.isna(imputed.iloc[4])  # both missing -> stays NaN


class TestStreak:
    def test_longest_true_streak(self):
        mask = pd.Series([False, True, True, True, False, True, False, True, True])
        assert _longest_true_streak(mask) == 3

    def test_longest_true_streak_all_false(self):
        mask = pd.Series([False, False, False])
        assert _longest_true_streak(mask) == 0

    def test_longest_true_streak_all_true(self):
        mask = pd.Series([True, True, True, True])
        assert _longest_true_streak(mask) == 4


class TestLoadRawHalfhourly:
    def test_parses_cached_intensity_and_generation_files(self, tmp_path):
        intensity_payload = {
            "data": [
                {"from": "2020-01-01T00:00Z", "to": "2020-01-01T00:30Z",
                 "intensity": {"actual": 150, "forecast": 148, "index": "moderate"}},
                {"from": "2020-01-01T00:30Z", "to": "2020-01-01T01:00Z",
                 "intensity": {"actual": 160, "forecast": 158, "index": "moderate"}},
            ]
        }
        generation_payload = {
            "data": [
                {"from": "2020-01-01T00:00Z", "to": "2020-01-01T00:30Z",
                 "generationmix": [{"fuel": "gas", "perc": 30.0}, {"fuel": "wind", "perc": 20.0}]},
                {"from": "2020-01-01T00:30Z", "to": "2020-01-01T01:00Z",
                 "generationmix": [{"fuel": "gas", "perc": 28.0}, {"fuel": "wind", "perc": 22.0}]},
            ]
        }
        (tmp_path / "intensity_20200101_20200102.json").write_text(json.dumps(intensity_payload))
        (tmp_path / "generation_20200101_20200102.json").write_text(json.dumps(generation_payload))

        df = load_raw_halfhourly(tmp_path)

        assert len(df) == 2
        assert df.iloc[0]['intensity_actual'] == 150
        assert df.iloc[0]['gen_gas'] == 30.0
        assert df.iloc[1]['gen_wind'] == 22.0

    def test_raises_if_no_cached_intensity_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw_halfhourly(tmp_path)


class TestFetchAndCacheIdempotency:
    def test_second_fetch_over_same_range_makes_no_new_requests(self, tmp_path):
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 15)  # single chunk (< CHUNK_DAYS)

        call_count = {'n': 0}

        def fake_fetch_json(url, *args, **kwargs):
            call_count['n'] += 1
            return {"data": []}

        with patch('src.grid.fetch._fetch_json', side_effect=fake_fetch_json):
            fetch_and_cache_range(start, end, tmp_path)
            first_call_count = call_count['n']
            assert first_call_count > 0

            fetch_and_cache_range(start, end, tmp_path)  # re-run, same range

        assert call_count['n'] == first_call_count, (
            "second fetch_and_cache_range call over an already-cached range "
            "must not issue any new HTTP requests"
        )

    def test_date_chunks_cover_full_range_without_gaps(self):
        start = datetime(2020, 1, 1)
        end = datetime(2020, 3, 15)
        chunks = list(_date_chunks(start, end, chunk_days=28))

        assert chunks[0][0] == start
        assert chunks[-1][1] == end
        for (a_start, a_end), (b_start, b_end) in zip(chunks, chunks[1:]):
            assert a_end == b_start  # contiguous, no gaps or overlaps


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
