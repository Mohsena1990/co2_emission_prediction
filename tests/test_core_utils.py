"""
Tests for core utilities: quarter parsing and delta_log inversion.

Covers spec items 3.3 (delta_log inversion round-trip) and 3.7 (quarter
parsing formats), including the leakage-adjacent requirement that decimal
quarter encodings outside {.1, .2, .3, .4} raise rather than being guessed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.core.utils import quarter_to_date, year_quarter_to_date, invert_delta_log, to_original_scale


class TestQuarterParsing:
    """Tests for the single canonical quarter_to_date parser (bug 3.7)."""

    @pytest.mark.parametrize("value,expected", [
        ("1999Q1", pd.Timestamp("1999-01-01")),
        ("1999Q4", pd.Timestamp("1999-10-01")),
        ("1999-Q1", pd.Timestamp("1999-01-01")),
        ("1999-Q4", pd.Timestamp("1999-10-01")),
        ("1999 Q1", pd.Timestamp("1999-01-01")),
        ("1999_Q2", pd.Timestamp("1999-04-01")),
        ("1999/Q3", pd.Timestamp("1999-07-01")),
        ("1999q1", pd.Timestamp("1999-01-01")),  # lowercase
    ])
    def test_string_formats(self, value, expected):
        assert quarter_to_date(value) == expected

    @pytest.mark.parametrize("value,expected", [
        (1999.1, pd.Timestamp("1999-01-01")),
        (1999.2, pd.Timestamp("1999-04-01")),
        (1999.3, pd.Timestamp("1999-07-01")),
        (1999.4, pd.Timestamp("1999-10-01")),
        (2025.1, pd.Timestamp("2025-01-01")),
    ])
    def test_decimal_encoding(self, value, expected):
        assert quarter_to_date(value) == expected

    def test_decimal_encoding_invalid_fraction_raises(self):
        # .5, .0, .15 etc. do not map to a quarter 1-4 and must raise,
        # not be silently coerced (this is the exact failure mode called
        # out in spec 3.7).
        for bad in (1999.0, 1999.5, 1999.15, 1999.99):
            with pytest.raises(ValueError):
                quarter_to_date(bad)

    def test_timestamp_passthrough(self):
        ts = pd.Timestamp("2020-05-15")  # mid Q2
        assert quarter_to_date(ts) == pd.Timestamp("2020-04-01")

    def test_datetime_passthrough(self):
        import datetime
        d = datetime.date(2020, 11, 3)  # Q4
        assert quarter_to_date(d) == pd.Timestamp("2020-10-01")

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            quarter_to_date("not-a-quarter")

    def test_year_quarter_columns(self):
        assert year_quarter_to_date(1999, 1) == pd.Timestamp("1999-01-01")
        assert year_quarter_to_date("2020", "3") == pd.Timestamp("2020-07-01")


class TestDeltaLogInversion:
    """Round-trip tests for y -> delta_log(y) -> y_hat (bug 3.3)."""

    def test_round_trip_reconstructs_original_levels(self):
        rng = np.random.RandomState(0)
        y = pd.Series(1000 + np.cumsum(rng.uniform(-5, 20, size=30)))
        assert (y > 0).all()

        log_y = np.log(y)
        deltas = log_y.diff().dropna().values

        # Forecast origin is the last observed value before the deltas begin
        reference_log_level = log_y.iloc[0]

        reconstructed = invert_delta_log(deltas, reference_log_level)

        np.testing.assert_allclose(reconstructed, y.values[1:], rtol=1e-10)

    def test_single_step_reconstruction(self):
        reference = np.log(500.0)
        delta = np.array([np.log(520.0 / 500.0)])
        reconstructed = invert_delta_log(delta, reference)
        np.testing.assert_allclose(reconstructed, [520.0], rtol=1e-10)

    def test_raises_on_missing_reference_level(self):
        with pytest.raises(ValueError):
            invert_delta_log(np.array([0.01, 0.02]), None)
        with pytest.raises(ValueError):
            invert_delta_log(np.array([0.01, 0.02]), np.nan)

    def test_raises_on_non_finite_diffs(self):
        with pytest.raises(ValueError):
            invert_delta_log(np.array([0.01, np.nan]), np.log(100.0))
        with pytest.raises(ValueError):
            invert_delta_log(np.array([0.01, np.inf]), np.log(100.0))

    def test_empty_input_returns_empty(self):
        result = invert_delta_log(np.array([]), np.log(100.0))
        assert len(result) == 0


class TestToOriginalScale:
    """Regression coverage for spec section 2/17's original-scale
    requirement, and the specific bug found 2026-08-01: this inversion was
    missing entirely from src/pipeline/experiment.py (the new A1-A4
    orchestrator) and from scripts/04_evaluate_and_safeguards.py's
    quarterly-level metrics (only its separate annual-consistency check
    ever called the equivalent inversion)."""

    def test_log_transform_inverts_correctly(self):
        original = np.array([1000.0, 2000.0, 500.0])
        log_values = np.log(original)
        result = to_original_scale(log_values, 'log')
        np.testing.assert_allclose(result, original)

    def test_none_transform_passes_through_unchanged(self):
        values = np.array([1000.0, 2000.0, 500.0])
        result = to_original_scale(values, 'none')
        np.testing.assert_array_equal(result, values)

    def test_none_type_transform_passes_through_unchanged(self):
        values = np.array([1000.0, 2000.0, 500.0])
        result = to_original_scale(values, None)
        np.testing.assert_array_equal(result, values)

    def test_unsupported_transform_raises_value_error(self):
        with pytest.raises(ValueError, match='delta_log'):
            to_original_scale(np.array([0.01, 0.02]), 'delta_log')

    def test_result_is_physically_plausible_scale(self):
        """The exact sanity check that caught the original bug: a log-CO2e
        value (~11-12 for UK quarterly emissions in the hundreds of
        thousands of tonnes) must invert to a value in the right order of
        magnitude, not stay in log space."""
        log_co2e = np.array([11.8, 11.9, 12.0])  # realistic UK quarterly log(CO2e)
        result = to_original_scale(log_co2e, 'log')
        assert (result > 10000).all()  # original-scale CO2e, not log-scale (~12)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
