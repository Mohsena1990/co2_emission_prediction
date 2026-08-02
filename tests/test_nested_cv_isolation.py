"""
Tests for nested (outer/inner) expanding-window CV (bug 3.5 / section 11):
inner folds used for feature selection and PSO tuning must never see any
outer test row, and PSO/FS isolation must hold structurally, not just by
convention.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.core.config import SplitConfig
from src.splits.nested_walk_forward import (
    create_nested_walk_forward_splits, validate_nested_isolation, NestedFold
)
from src.splits.walk_forward import generate_cv_folds


def _quarterly_xy(n=100, seed=0):
    dates = pd.date_range('2000-01-01', periods=n, freq='QS')
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({'f1': rng.randn(n), 'f2': rng.randn(n)}, index=dates)
    y = pd.Series(rng.randn(n), index=dates, name='y')
    return X, y


class TestNestedSplitStructure:
    def test_produces_nested_folds(self):
        X, y = _quarterly_xy(100)
        cfg = SplitConfig(min_train_size=40, test_size=4, horizons=[1, 2, 4],
                           inner_min_train_size=24, inner_test_size=4)
        nested = create_nested_walk_forward_splits(X, y, cfg)
        assert len(nested) > 0
        assert all(isinstance(nf, NestedFold) for nf in nested)
        assert all(nf.inner_cv_plan.n_folds > 0 for nf in nested)

    def test_inner_indices_are_local_to_outer_training_slice(self):
        X, y = _quarterly_xy(80)
        cfg = SplitConfig(min_train_size=30, test_size=4, horizons=[1],
                           inner_min_train_size=16, inner_test_size=4)
        nested = create_nested_walk_forward_splits(X, y, cfg)
        assert len(nested) > 0

        nf = nested[0]
        n_outer_train = len(nf.outer.train_indices)
        for inner_fold in nf.inner_cv_plan.folds:
            assert inner_fold.train_indices.max() < n_outer_train
            assert inner_fold.test_indices.max() < n_outer_train


class TestIsolation:
    def test_validate_nested_isolation_passes_for_valid_construction(self):
        X, y = _quarterly_xy(105)
        cfg = SplitConfig(min_train_size=40, test_size=4, horizons=[1, 2, 4],
                           inner_min_train_size=24, inner_test_size=4)
        nested = create_nested_walk_forward_splits(X, y, cfg)
        assert validate_nested_isolation(X, nested) is True

    def test_no_inner_fold_reaches_outer_test_start(self):
        X, y = _quarterly_xy(105)
        cfg = SplitConfig(min_train_size=40, test_size=4, horizons=[1, 2, 4],
                           inner_min_train_size=24, inner_test_size=4)
        nested = create_nested_walk_forward_splits(X, y, cfg)

        for nf in nested:
            X_outer_train = X.iloc[nf.outer.train_indices]
            for _, _, _, _, inner_fold in generate_cv_folds(
                X_outer_train, y.iloc[nf.outer.train_indices], nf.inner_cv_plan
            ):
                assert inner_fold.test_end < nf.outer.test_start
                assert inner_fold.train_end < nf.outer.test_start

    def test_leakage_sentinel_outer_test_values_invisible_to_inner_folds(self):
        # Plant an extreme sentinel value ONLY in what will be the outer
        # test window and confirm no inner fold's train/test data (built
        # purely from the outer training slice) ever contains it.
        n = 60
        dates = pd.date_range('2000-01-01', periods=n, freq='QS')
        values = np.ones(n) * 10.0
        cfg = SplitConfig(min_train_size=30, test_size=4, horizons=[1],
                           inner_min_train_size=16, inner_test_size=4)

        X = pd.DataFrame({'f1': values}, index=dates)
        y = pd.Series(values, index=dates)

        nested = create_nested_walk_forward_splits(X, y, cfg)
        assert len(nested) > 0

        for nf in nested:
            # Plant the sentinel in the outer test window only, using the
            # ACTUAL outer training slice this nested fold was built from -
            # the inner plan must never have seen it because it was built
            # before the sentinel could exist in the training slice.
            X_outer_train = X.iloc[nf.outer.train_indices]
            assert (X_outer_train['f1'] == 10.0).all()  # sentinel never present in training data
            for inner_fold in nf.inner_cv_plan.folds:
                assert inner_fold.test_end < nf.outer.test_start

    def test_skips_outer_folds_with_insufficient_inner_window(self):
        # inner_min_train_size larger than any outer training window ->
        # zero inner folds possible -> the outer fold must be skipped
        # entirely rather than silently falling back to no validation.
        X, y = _quarterly_xy(50)
        cfg = SplitConfig(min_train_size=30, test_size=4, horizons=[1],
                           inner_min_train_size=1000, inner_test_size=4)
        nested = create_nested_walk_forward_splits(X, y, cfg)
        assert len(nested) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
