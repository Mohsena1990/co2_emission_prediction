"""
Tests for the primary/sensitivity period designation (spec section 14) and
the common-period CV plan builder (src/splits/panels.py).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.core.config import Config
from src.grid.fetch import EARLIEST_AVAILABLE_DATE
from src.splits.walk_forward import create_walk_forward_splits
from src.splits.panels import (
    PRIMARY_PERIOD_PANEL, SENSITIVITY_LONG_PERIOD_PANEL,
    panel2_split_config, build_common_period_cv_plan,
    slice_configurations_to_common_period, assert_cv_plan_fits_matrix,
)


class TestPrimarySensitivityDesignation:
    def test_primary_period_panel_is_common_period(self):
        assert PRIMARY_PERIOD_PANEL == 'panel2_common_period'

    def test_sensitivity_panel_is_full_period(self):
        assert SENSITIVITY_LONG_PERIOD_PANEL == 'panel1_full_period'

    def test_both_panels_are_distinct_and_in_config_defaults(self):
        panels = Config().configurations.panels
        assert PRIMARY_PERIOD_PANEL in panels
        assert SENSITIVITY_LONG_PERIOD_PANEL in panels
        assert PRIMARY_PERIOD_PANEL != SENSITIVITY_LONG_PERIOD_PANEL


class TestGridStartDateReconciliation:
    def test_earliest_available_date_is_2018_not_spec_assumed_2009(self):
        # spec section 6/14 assumes NESO grid data begin 2009Q1; a live
        # probe of the GB Carbon Intensity API (documented in
        # src/grid/fetch.py) confirms the real start is 2018-01-01. The
        # PRIMARY common period must be bounded by this actual date.
        assert EARLIEST_AVAILABLE_DATE.year == 2018
        assert EARLIEST_AVAILABLE_DATE.month == 1


class TestPanel2SplitConfig:
    def test_uses_panel2_specific_sizing_not_base_sizing(self):
        base = Config().splits
        panel2_cfg = panel2_split_config(base)

        assert panel2_cfg.min_train_size == base.panel2_min_train_size
        assert panel2_cfg.test_size == base.panel2_test_size
        assert panel2_cfg.inner_min_train_size == base.panel2_inner_min_train_size
        assert panel2_cfg.inner_test_size == base.panel2_inner_test_size
        # Base sizing (sized for the ~101-quarter full period) must differ
        # from Panel 2's (sized for the ~29-quarter common period) - if they
        # were ever accidentally set equal, Panel 2 would yield zero folds.
        assert panel2_cfg.min_train_size != base.min_train_size

    def test_horizons_and_weights_carry_over_unchanged(self):
        base = Config().splits
        panel2_cfg = panel2_split_config(base)
        assert panel2_cfg.horizons == base.horizons
        assert panel2_cfg.horizon_weights == base.horizon_weights


class TestBuildCommonPeriodCvPlan:
    def _X_y(self, n=40, seed=0):
        dates = pd.date_range('2015-01-01', periods=n, freq='QS')
        rng = np.random.RandomState(seed)
        X = pd.DataFrame({'f1': rng.randn(n), 'f2': rng.randn(n)}, index=dates)
        y = pd.Series(1000 + rng.normal(0, 10, n).cumsum(), index=dates)
        return X, y

    def test_restricts_folds_to_common_period_only(self):
        X, y = self._X_y(n=40)
        common_start = X.index[20]  # last 20 quarters only
        split_config = Config().splits
        split_config.min_train_size = 10
        split_config.test_size = 2
        split_config.horizons = [1]

        plan = build_common_period_cv_plan(X, y, split_config, common_period_start=common_start)

        for fold in plan.folds:
            assert fold.train_start >= common_start

    def test_raises_on_empty_window(self):
        X, y = self._X_y(n=10)
        split_config = Config().splits
        with pytest.raises(ValueError):
            build_common_period_cv_plan(
                X, y, split_config,
                common_period_start=X.index.max() + pd.Timedelta(days=400),
            )

    def test_default_end_is_x_max(self):
        X, y = self._X_y(n=40)
        split_config = Config().splits
        split_config.min_train_size = 10
        split_config.test_size = 2
        split_config.horizons = [1]

        plan = build_common_period_cv_plan(X, y, split_config, common_period_start=X.index[15])
        assert plan.n_folds > 0


class TestSliceConfigurationsToCommonPeriod:
    """Regression coverage for a real Phase 6 bug: generate_cv_folds indexes
    X/y POSITIONALLY, so every configuration passed to it must be sliced to
    the exact same [common_start, common_end] window the CV plan was built
    against - passing an unsliced, longer-history matrix (e.g. A1/A2, which
    are NOT naturally bounded like A3/A4) silently walks the same integer
    fold positions onto the wrong calendar dates."""

    def _configs(self):
        # A1/A2: full history back to 2000. A3/A4: grid-bounded, 2018 on -
        # mirrors the real repo's A1/A2 vs A3/A4 asymmetry.
        long_idx = pd.date_range('2000-01-01', periods=100, freq='QS')
        short_idx = pd.date_range('2018-01-01', periods=28, freq='QS')
        y = pd.Series(np.arange(len(long_idx), dtype=float), index=long_idx)
        X_by_config = {
            'A1': pd.DataFrame({'f1': np.arange(len(long_idx))}, index=long_idx),
            'A2': pd.DataFrame({'f1': np.arange(len(long_idx))}, index=long_idx),
            'A3': pd.DataFrame({'f1': np.arange(len(short_idx))}, index=short_idx),
            'A4': pd.DataFrame({'f1': np.arange(len(short_idx))}, index=short_idx),
        }
        return X_by_config, y

    def test_every_configuration_resliced_to_reference_window(self):
        X_by_config, y = self._configs()
        sliced = slice_configurations_to_common_period(X_by_config, y, reference_tag='A3')

        for tag in ('A1', 'A2', 'A3', 'A4'):
            assert sliced[tag].index.min() == X_by_config['A3'].index.min()
            assert sliced[tag].index.max() == X_by_config['A3'].index.max()
            assert len(sliced[tag]) == len(X_by_config['A3'])

    def test_a1_a2_row_count_shrinks_from_full_history(self):
        X_by_config, y = self._configs()
        assert len(X_by_config['A1']) == 100
        sliced = slice_configurations_to_common_period(X_by_config, y, reference_tag='A3')
        assert len(sliced['A1']) == 28

    def test_positional_alignment_matches_reference_after_slicing(self):
        """The actual bug scenario: build a CV plan against the (already
        short) A3 matrix, then confirm A1's SLICED matrix - not the raw
        one - resolves fold position 0 to the same calendar date as A3's."""
        X_by_config, y = self._configs()
        split_config = Config().splits
        split_config.min_train_size = 10
        split_config.test_size = 2
        split_config.horizons = [1]

        plan = create_walk_forward_splits(X_by_config['A3'], y.loc[X_by_config['A3'].index], split_config)
        assert plan.n_folds > 0

        sliced = slice_configurations_to_common_period(X_by_config, y, reference_tag='A3')
        first_fold = plan.folds[0]
        date_from_a3 = X_by_config['A3'].index[first_fold.train_indices[0]]
        date_from_sliced_a1 = sliced['A1'].index[first_fold.train_indices[0]]
        assert date_from_a3 == date_from_sliced_a1

        # The bug this guards against: using the UNSLICED A1 matrix would
        # resolve the same position to a wildly earlier (wrong) date.
        date_from_unsliced_a1 = X_by_config['A1'].index[first_fold.train_indices[0]]
        assert date_from_unsliced_a1 != date_from_a3


class TestAssertCvPlanFitsMatrix:
    def _plan_and_matrix(self, n=30):
        idx = pd.date_range('2018-01-01', periods=n, freq='QS')
        y = pd.Series(np.arange(n, dtype=float), index=idx)
        X = pd.DataFrame({'f1': np.arange(n)}, index=idx)
        split_config = Config().splits
        split_config.min_train_size = 10
        split_config.test_size = 2
        split_config.horizons = [1]
        plan = create_walk_forward_splits(X, y, split_config)
        return plan, X

    def test_passes_for_matching_matrix(self):
        plan, X = self._plan_and_matrix()
        assert_cv_plan_fits_matrix(plan, X, tag='A3')  # must not raise

    def test_raises_for_truncated_matrix(self):
        plan, X = self._plan_and_matrix()
        truncated = X.iloc[:5]
        with pytest.raises(ValueError):
            assert_cv_plan_fits_matrix(plan, truncated, tag='A1')

    def test_noop_for_empty_plan(self):
        idx = pd.date_range('2018-01-01', periods=5, freq='QS')
        X = pd.DataFrame({'f1': range(5)}, index=idx)
        from src.splits.walk_forward import CVPlan
        empty_plan = CVPlan(folds=[])
        assert_cv_plan_fits_matrix(empty_plan, X, tag='A1')  # must not raise


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
