"""
Tests for OWID annual renewable/low-carbon electricity share ingestion
(spec section 8): UK-entity filtering, value-column resolution against
metadata (the two OWID series use DIFFERENT column names - never
hardcode), one-year-lag alignment, and cache fetch idempotency.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.owid.fetch import (
    load_owid_uk_series, resolve_value_column,
    fetch_and_cache_owid, load_cached_owid_uk_series,
)
from src.owid.lag import build_owid_lagged_feature


def _owid_csv_frame(value_col_name: str, years=range(2015, 2023), other_entity='France'):
    rows = []
    for y in years:
        rows.append({'Entity': 'United Kingdom', 'Code': 'GBR', 'Year': y, value_col_name: 30.0 + y % 5})
        rows.append({'Entity': other_entity, 'Code': 'FRA', 'Year': y, value_col_name: 70.0 + y % 3})
    return pd.DataFrame(rows)


def _uk_owid_frame(value_col_name: str, years=range(2015, 2023)):
    """UK-only rows, as build_owid_lagged_feature expects (post-filtering)."""
    full = _owid_csv_frame(value_col_name, years=years)
    return full[full['Entity'] == 'United Kingdom'].reset_index(drop=True)


def _fake_metadata(n_columns=1):
    columns = {f'col_{i}': {'shortUnit': '%'} for i in range(n_columns)}
    return {'columns': columns, 'dateDownloaded': '2026-01-01'}


class TestLoadOwidUkSeries:
    def test_filters_to_united_kingdom_only(self):
        df = _owid_csv_frame('Renewables')

        with patch('src.owid.fetch.pd.read_csv', return_value=df), \
             patch('src.owid.fetch.requests.get') as mock_get:
            mock_get.return_value.json.return_value = _fake_metadata()
            mock_get.return_value.raise_for_status = lambda: None

            uk_df, metadata = load_owid_uk_series('fake://csv', 'fake://meta')

        assert (uk_df['Entity'] == 'United Kingdom').all()
        assert len(uk_df) == 8  # one row per year, 2015-2022
        assert metadata == _fake_metadata()

    def test_raises_if_uk_not_present(self):
        df = _owid_csv_frame('Renewables', other_entity='Germany')
        df = df[df['Entity'] != 'United Kingdom']

        with patch('src.owid.fetch.pd.read_csv', return_value=df), \
             patch('src.owid.fetch.requests.get') as mock_get:
            mock_get.return_value.json.return_value = _fake_metadata()
            mock_get.return_value.raise_for_status = lambda: None

            with pytest.raises(RuntimeError):
                load_owid_uk_series('fake://csv', 'fake://meta')


class TestResolveValueColumn:
    def test_resolves_renewable_style_column_name(self):
        uk_df = _owid_csv_frame('Renewables')
        col = resolve_value_column(uk_df, _fake_metadata(n_columns=1))
        assert col == 'Renewables'

    def test_resolves_low_carbon_style_column_name(self):
        # The two real OWID series use DIFFERENT column names - this is the
        # exact scenario spec 8.1 warns against hardcoding.
        uk_df = _owid_csv_frame('Share of electricity from low-carbon sources')
        col = resolve_value_column(uk_df, _fake_metadata(n_columns=1))
        assert col == 'Share of electricity from low-carbon sources'

    def test_raises_on_unexpected_extra_columns(self):
        uk_df = _owid_csv_frame('Renewables')
        uk_df['UnexpectedExtraColumn'] = 1.0
        with pytest.raises(ValueError):
            resolve_value_column(uk_df, _fake_metadata(n_columns=1))

    def test_raises_on_metadata_column_count_mismatch(self):
        uk_df = _owid_csv_frame('Renewables')
        with pytest.raises(ValueError):
            resolve_value_column(uk_df, _fake_metadata(n_columns=2))


class TestBuildOwidLaggedFeature:
    def test_quarter_uses_prior_year_value_only(self):
        uk_df = _uk_owid_frame('Renewables', years=range(2018, 2022))
        metadata = _fake_metadata(n_columns=1)
        idx = pd.date_range('2019-01-01', periods=8, freq='QS')  # 2019Q1..2020Q4

        result = build_owid_lagged_feature(uk_df, metadata, 'OWID_RenewableShare_L1Y', idx)

        expected_2019 = uk_df.set_index('Year').loc[2018, 'Renewables']
        expected_2020 = uk_df.set_index('Year').loc[2019, 'Renewables']
        for q in idx[idx.year == 2019]:
            assert result.loc[q] == pytest.approx(expected_2019)
        for q in idx[idx.year == 2020]:
            assert result.loc[q] == pytest.approx(expected_2020)

    def test_never_uses_same_year_value(self):
        # Regression guard: a quarter in year y must never see year y's own
        # (possibly still-incomplete) annual value.
        uk_df = _uk_owid_frame('Renewables', years=range(2018, 2023))
        metadata = _fake_metadata(n_columns=1)
        idx = pd.date_range('2021-01-01', periods=4, freq='QS')  # all of 2021

        result = build_owid_lagged_feature(uk_df, metadata, 'x', idx)
        same_year_value = uk_df.set_index('Year').loc[2021, 'Renewables']
        assert not (result == same_year_value).any()

    def test_nan_before_series_start(self):
        uk_df = _uk_owid_frame('Renewables', years=range(2018, 2020))
        metadata = _fake_metadata(n_columns=1)
        idx = pd.date_range('2016-01-01', periods=4, freq='QS')  # before any data - 1

        result = build_owid_lagged_feature(uk_df, metadata, 'x', idx)
        assert result.isna().all()


class TestFetchAndCacheIdempotency:
    def test_second_fetch_reuses_cache(self, tmp_path):
        call_count = {'n': 0}

        def fake_load(csv_url, metadata_url):
            call_count['n'] += 1
            value_col = 'Renewables' if 'renewables' in csv_url else 'Share of electricity from low-carbon sources'
            return _uk_owid_frame(value_col), _fake_metadata()

        with patch('src.owid.fetch.load_owid_uk_series', side_effect=fake_load):
            fetch_and_cache_owid(tmp_path)
            first_calls = call_count['n']
            assert first_calls == 2  # renewable + low_carbon

            fetch_and_cache_owid(tmp_path)  # re-run, cache present

        assert call_count['n'] == first_calls, "second fetch must not re-download"

    def test_force_re_fetches(self, tmp_path):
        call_count = {'n': 0}

        def fake_load(csv_url, metadata_url):
            call_count['n'] += 1
            value_col = 'Renewables' if 'renewables' in csv_url else 'Share of electricity from low-carbon sources'
            return _uk_owid_frame(value_col), _fake_metadata()

        with patch('src.owid.fetch.load_owid_uk_series', side_effect=fake_load):
            fetch_and_cache_owid(tmp_path)
            fetch_and_cache_owid(tmp_path, force=True)

        assert call_count['n'] == 4

    def test_load_cached_round_trips(self, tmp_path):
        def fake_load(csv_url, metadata_url):
            value_col = 'Renewables' if 'renewables' in csv_url else 'Share of electricity from low-carbon sources'
            return _uk_owid_frame(value_col), _fake_metadata()

        with patch('src.owid.fetch.load_owid_uk_series', side_effect=fake_load):
            fetch_and_cache_owid(tmp_path)

        uk_df, metadata = load_cached_owid_uk_series(tmp_path, 'renewable')
        assert 'Renewables' in uk_df.columns
        assert (uk_df['Entity'] == 'United Kingdom').all()

    def test_unknown_kind_raises(self, tmp_path):
        with pytest.raises(ValueError):
            load_cached_owid_uk_series(tmp_path, 'bogus')

    def test_missing_cache_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_cached_owid_uk_series(tmp_path, 'renewable')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
