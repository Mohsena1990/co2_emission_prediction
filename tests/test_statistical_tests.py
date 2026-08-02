"""
Tests for src/evaluation/statistical_tests.py (spec section 17) and
src/evaluation/incremental_value.py (spec section 10 / Table 6).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistical_tests import (
    paired_bootstrap_ci, diebold_mariano_test, wilcoxon_signed_rank_test,
    bonferroni_correction, benjamini_hochberg_correction
)
from src.evaluation.incremental_value import (
    paired_fold_errors, compute_incremental_value_table, INCREMENTAL_COMPARISONS
)


class TestPairedBootstrapCI:
    def test_identical_errors_give_zero_mean_diff(self):
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paired_bootstrap_ci(errors, errors)
        assert result['mean_diff'] == pytest.approx(0.0)
        assert result['ci_low'] <= 0.0 <= result['ci_high']

    def test_a_systematically_smaller_gives_negative_mean_diff(self):
        rng = np.random.RandomState(0)
        errors_b = rng.uniform(5, 10, 50)
        errors_a = errors_b - 2.0  # `a` is uniformly 2.0 better
        result = paired_bootstrap_ci(errors_a, errors_b)
        assert result['mean_diff'] == pytest.approx(-2.0)
        assert result['ci_high'] < 0  # entire CI below zero -> confidently better

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap_ci(np.array([1, 2]), np.array([1, 2, 3]))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap_ci(np.array([]), np.array([]))


class TestDieboldMariano:
    def test_identical_forecasts_give_nan_or_near_zero_statistic(self):
        errors = np.array([1.0, -2.0, 3.0, -1.5, 2.5, -0.5])
        result = diebold_mariano_test(errors, errors)
        # d = |e|-|e| = 0 everywhere -> zero variance -> NaN by design (no
        # spurious "significant" result from a degenerate variance).
        assert np.isnan(result['dm_statistic'])

    def test_systematically_better_forecast_is_significant(self):
        rng = np.random.RandomState(1)
        errors_b = rng.normal(0, 5, 60)
        errors_a = rng.normal(0, 1, 60)  # much smaller errors
        result = diebold_mariano_test(errors_a, errors_b)
        assert result['dm_statistic'] < 0  # a has lower loss
        assert result['p_value'] < 0.05

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            diebold_mariano_test(np.array([1, 2]), np.array([1, 2, 3]))

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError):
            diebold_mariano_test(np.array([1.0, 2.0]), np.array([1.0, 2.0]), loss='bogus')


class TestWilcoxon:
    def test_identical_errors_give_p_value_one(self):
        errors = np.array([1.0, 2.0, 3.0, 4.0])
        result = wilcoxon_signed_rank_test(errors, errors)
        assert result['p_value'] == 1.0

    def test_systematic_difference_is_significant(self):
        rng = np.random.RandomState(2)
        errors_b = rng.uniform(5, 10, 30)
        errors_a = errors_b - 3.0
        result = wilcoxon_signed_rank_test(errors_a, errors_b)
        assert result['p_value'] < 0.05


class TestMultipleComparisonCorrection:
    def test_bonferroni_scales_by_count(self):
        p_values = [0.01, 0.02, 0.5]
        adjusted = bonferroni_correction(p_values)
        assert adjusted == [pytest.approx(0.03), pytest.approx(0.06), pytest.approx(1.0)]

    def test_bh_less_conservative_than_bonferroni(self):
        p_values = [0.001, 0.01, 0.02, 0.5, 0.8]
        bonf = bonferroni_correction(p_values)
        bh = benjamini_hochberg_correction(p_values)
        assert sum(bh) <= sum(bonf)

    def test_bh_is_monotonic_in_sorted_order(self):
        p_values = [0.5, 0.01, 0.3, 0.001]
        bh = benjamini_hochberg_correction(p_values)
        order = np.argsort(p_values)
        sorted_bh = np.array(bh)[order]
        assert np.all(np.diff(sorted_bh) >= -1e-12)


def _fake_predictions(configs, panel, model, horizons, n_folds, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for config in configs:
        bias = {'A1': 3.0, 'A2': 2.0, 'A3': 2.5, 'A4': 1.5}[config]
        for h in horizons:
            for fold_id in range(n_folds):
                actual = 100.0 + rng.randn()
                predicted = actual + bias + rng.normal(0, 0.3)
                rows.append({
                    'configuration': config, 'panel': panel, 'model': model,
                    'fs_option': 'all_features', 'horizon': h, 'fold_id': fold_id,
                    'target_date': pd.Timestamp('2020-01-01') + pd.DateOffset(months=3 * fold_id),
                    'actual': actual, 'predicted': predicted, 'residual': predicted - actual,
                })
    return pd.DataFrame(rows)


class TestIncrementalValueTable:
    def test_paired_fold_errors_merges_on_fold_and_date(self):
        df = _fake_predictions(['A1', 'A2'], 'panel1_full_period', 'ridge', [1], n_folds=5)
        merged = paired_fold_errors(df, 'A2', 'A1', 'panel1_full_period', 'ridge')
        assert len(merged) == 5
        assert 'residual_a' in merged.columns and 'residual_b' in merged.columns

    def test_paired_fold_errors_empty_when_no_match(self):
        df = _fake_predictions(['A1', 'A2'], 'panel1_full_period', 'ridge', [1], n_folds=5)
        merged = paired_fold_errors(df, 'A4', 'A1', 'panel1_full_period', 'ridge')
        assert merged.empty

    def test_incremental_table_shows_a2_beats_a1(self):
        df = _fake_predictions(['A1', 'A2'], 'panel1_full_period', 'ridge', [1, 2], n_folds=10)
        table = compute_incremental_value_table(
            df, models=['ridge'], horizons=(1, 2),
            comparisons=[('A2', 'A1', 'panel1_full_period')]
        )
        assert len(table) == 2  # one row per horizon
        assert (table['percentage_improvement'] > 0).all()  # A2 has smaller bias -> improvement
        assert 'dm_p_value_bh' in table.columns

    def test_incremental_table_empty_for_missing_configuration(self):
        df = _fake_predictions(['A1', 'A2'], 'panel1_full_period', 'ridge', [1], n_folds=5)
        table = compute_incremental_value_table(
            df, models=['ridge'], horizons=(1,),
            comparisons=[('A4', 'A3', 'panel2_common_period')]
        )
        assert table.empty

    def test_default_comparisons_cover_all_four(self):
        assert len(INCREMENTAL_COMPARISONS) == 4
        pairs = {(a, b) for a, b, _ in INCREMENTAL_COMPARISONS}
        assert pairs == {('A2', 'A1'), ('A3', 'A1'), ('A4', 'A2'), ('A4', 'A3')}


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
