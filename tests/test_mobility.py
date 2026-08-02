"""
Tests for Google COVID-19 mobility-shock data fusion (spec section 4):
UK-national filtering, chunked-read correctness, quarterly aggregation +
completeness threshold, the neutral-zero convention outside the official
reporting window, and cache fetch idempotency.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.mobility.aggregate import aggregate_to_quarterly, apply_neutral_zero_convention
from src.mobility.fetch import (
    CATEGORY_COLUMN_MAP, MOBILITY_COLUMNS, _uk_national_mask,
    load_uk_google_mobility, fetch_and_cache_uk_mobility, load_cached_uk_mobility,
)

RAW_CATEGORY_COLS = list(CATEGORY_COLUMN_MAP.keys())


def _raw_mobility_frame(n_days: int = 120, start: str = '2020-04-01', seed: int = 0) -> pd.DataFrame:
    """Synthetic multi-country daily mobility data matching Google's raw schema."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start, periods=n_days, freq='D')

    rows = []
    for d in dates:
        # UK national row (what we want to keep).
        rows.append({
            'country_region_code': 'GB', 'country_region': 'United Kingdom',
            'sub_region_1': np.nan, 'sub_region_2': np.nan, 'date': d.strftime('%Y-%m-%d'),
            **{c: rng.uniform(-50, 20) for c in RAW_CATEGORY_COLS},
        })
        # UK sub-region row (must be excluded - not the national aggregate).
        rows.append({
            'country_region_code': 'GB', 'country_region': 'United Kingdom',
            'sub_region_1': 'England', 'sub_region_2': np.nan, 'date': d.strftime('%Y-%m-%d'),
            **{c: rng.uniform(-50, 20) for c in RAW_CATEGORY_COLS},
        })
        # US national row (must be excluded - wrong country).
        rows.append({
            'country_region_code': 'US', 'country_region': 'United States',
            'sub_region_1': np.nan, 'sub_region_2': np.nan, 'date': d.strftime('%Y-%m-%d'),
            **{c: rng.uniform(-50, 20) for c in RAW_CATEGORY_COLS},
        })
    return pd.DataFrame(rows)[MOBILITY_COLUMNS]


class TestUkNationalMask:
    def test_keeps_only_uk_national_rows(self):
        df = _raw_mobility_frame(n_days=5)
        mask = _uk_national_mask(df)
        selected = df.loc[mask]

        assert len(selected) == 5  # one national row per day
        assert (selected['country_region_code'] == 'GB').all()
        assert selected['sub_region_1'].isna().all()
        assert selected['sub_region_2'].isna().all()

    def test_excludes_uk_subregions_and_other_countries(self):
        df = _raw_mobility_frame(n_days=3)
        mask = _uk_national_mask(df)
        excluded = df.loc[~mask]
        assert len(excluded) == 6  # 3 sub-region + 3 US rows
        assert not ((excluded['country_region_code'] == 'GB') & excluded['sub_region_1'].isna()).any()


class TestChunkedLoading:
    def test_chunked_read_matches_single_pass_filtering(self):
        full_df = _raw_mobility_frame(n_days=20)
        expected = full_df.loc[_uk_national_mask(full_df)].copy()
        expected['date'] = pd.to_datetime(expected['date'])
        expected = expected.sort_values('date').reset_index(drop=True)

        def fake_read_csv(url, usecols, chunksize, low_memory):
            # Split into several chunks straddling day/country boundaries,
            # like the real chunked download would.
            chunk_size = 17
            for start in range(0, len(full_df), chunk_size):
                yield full_df.iloc[start:start + chunk_size][usecols]

        with patch('src.mobility.fetch.pd.read_csv', side_effect=fake_read_csv):
            result = load_uk_google_mobility(url='fake://mobility.csv', chunksize=17)

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True), check_like=True
        )

    def test_raises_if_no_uk_rows_found(self):
        df = pd.DataFrame({
            'country_region_code': ['US'], 'country_region': ['United States'],
            'sub_region_1': [np.nan], 'sub_region_2': [np.nan], 'date': ['2020-04-01'],
            **{c: [0.0] for c in RAW_CATEGORY_COLS},
        })[MOBILITY_COLUMNS]

        def fake_read_csv(url, usecols, chunksize, low_memory):
            yield df[usecols]

        with patch('src.mobility.fetch.pd.read_csv', side_effect=fake_read_csv):
            with pytest.raises(RuntimeError):
                load_uk_google_mobility(url='fake://mobility.csv')


class TestAggregateToQuarterly:
    def _daily_df(self, n_days=91, start='2020-04-01', seed=0):
        rng = np.random.RandomState(seed)
        dates = pd.date_range(start, periods=n_days, freq='D')
        data = {'date': dates}
        for col in RAW_CATEGORY_COLS:
            data[col] = rng.uniform(-60, 10, n_days)
        return pd.DataFrame(data)

    def test_complete_quarter_is_included_with_correct_mean(self):
        df = self._daily_df()
        quarterly, quality = aggregate_to_quarterly(df, minimum_quarter_completeness=0.80)

        assert len(quarterly) == 1
        assert quality.iloc[0]['inclusion_status'] == 'included'
        expected = df['retail_and_recreation_percent_change_from_baseline'].mean()
        assert quarterly.iloc[0]['Mobility_Retail_Recreation'] == pytest.approx(expected)

    def test_severely_incomplete_quarter_is_excluded_not_imputed(self):
        df = self._daily_df().iloc[::3]  # ~33% of days present
        quarterly, quality = aggregate_to_quarterly(df, minimum_quarter_completeness=0.80)

        assert len(quarterly) == 0
        assert quality.iloc[0]['inclusion_status'] == 'excluded_incomplete'

    def test_completeness_threshold_is_configurable(self):
        df = self._daily_df().iloc[::2]  # ~50% completeness
        _, quality_strict = aggregate_to_quarterly(df, minimum_quarter_completeness=0.80)
        _, quality_lenient = aggregate_to_quarterly(df, minimum_quarter_completeness=0.30)

        assert quality_strict.iloc[0]['inclusion_status'] == 'excluded_incomplete'
        assert quality_lenient.iloc[0]['inclusion_status'] == 'included'

    def test_partial_2022q4_excluded_from_primary_by_cutoff(self):
        # 2022Q4 data is complete-enough by day-count, but must still be
        # excluded from the PRIMARY series because it's after the official
        # reporting window's cutoff quarter (spec 4.3).
        df = self._daily_df(n_days=60, start='2022-10-01')  # within 2022Q4
        quarterly, quality = aggregate_to_quarterly(
            df, minimum_quarter_completeness=0.10, primary_cutoff_quarter='2022Q3'
        )
        assert len(quarterly) == 0
        assert quality.iloc[0]['inclusion_status'] == 'excluded_post_primary_window'

    def test_partial_2022q4_included_when_cutoff_disabled(self):
        # The sensitivity variant: pass primary_cutoff_quarter=None to
        # include partial 2022Q4 (spec 4.3/29).
        df = self._daily_df(n_days=60, start='2022-10-01')
        quarterly, quality = aggregate_to_quarterly(
            df, minimum_quarter_completeness=0.10, primary_cutoff_quarter=None
        )
        assert len(quarterly) == 1
        assert quality.iloc[0]['inclusion_status'] == 'included'

    def test_quality_report_has_per_category_stats(self):
        df = self._daily_df()
        _, quality = aggregate_to_quarterly(df, minimum_quarter_completeness=0.80)
        for feat_name in CATEGORY_COLUMN_MAP.values():
            assert f'{feat_name}_observed_days' in quality.columns
            assert f'{feat_name}_coverage_ratio' in quality.columns
            assert f'{feat_name}_missing_days' in quality.columns


class TestNeutralZeroConvention:
    def test_quarters_outside_window_are_zero_not_nan(self):
        # Only one real observed quarter; the full sample spans many more.
        idx = pd.date_range('2019-01-01', periods=8, freq='QS')
        quarterly = pd.DataFrame(
            {'Mobility_Retail_Recreation': [-25.0]},
            index=[pd.Timestamp('2020-04-01')],
        )
        filled = apply_neutral_zero_convention(quarterly, idx)

        assert len(filled) == len(idx)
        assert filled.loc['2020-04-01', 'Mobility_Retail_Recreation'] == -25.0
        other_quarters = filled.drop(index=pd.Timestamp('2020-04-01'))
        assert (other_quarters['Mobility_Retail_Recreation'] == 0.0).all()
        assert not filled.isna().any().any()

    def test_empty_quarterly_features_still_produces_full_zero_frame(self):
        idx = pd.date_range('2019-01-01', periods=4, freq='QS')
        empty = pd.DataFrame(columns=['Mobility_Parks'])
        filled = apply_neutral_zero_convention(empty, idx)
        assert len(filled) == len(idx)
        assert (filled['Mobility_Parks'] == 0.0).all()


class TestFetchAndCacheIdempotency:
    def test_second_fetch_reuses_cache_without_new_download(self, tmp_path):
        df = self._synthetic_uk_daily()
        call_count = {'n': 0}

        def fake_load(url=None, chunksize=None):
            call_count['n'] += 1
            return df

        with patch('src.mobility.fetch.load_uk_google_mobility', side_effect=fake_load):
            fetch_and_cache_uk_mobility(tmp_path)
            assert call_count['n'] == 1

            fetch_and_cache_uk_mobility(tmp_path)  # re-run, cache already present
            assert call_count['n'] == 1, "second fetch must not re-download"

        cached = load_cached_uk_mobility(tmp_path)
        assert len(cached) == len(df)

    def test_force_re_fetches(self, tmp_path):
        df = self._synthetic_uk_daily()
        call_count = {'n': 0}

        def fake_load(url=None, chunksize=None):
            call_count['n'] += 1
            return df

        with patch('src.mobility.fetch.load_uk_google_mobility', side_effect=fake_load):
            fetch_and_cache_uk_mobility(tmp_path)
            fetch_and_cache_uk_mobility(tmp_path, force=True)

        assert call_count['n'] == 2

    @staticmethod
    def _synthetic_uk_daily(n=30):
        rng = np.random.RandomState(0)
        dates = pd.date_range('2020-04-01', periods=n, freq='D')
        data = {'date': dates}
        for col in RAW_CATEGORY_COLS:
            data[col] = rng.uniform(-40, 10, n)
        return pd.DataFrame(data)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
