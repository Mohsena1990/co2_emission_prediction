"""
Tests that PSO hyperparameter tuning is isolated to inner folds and never
influenced by the outer test fold (bug 3.5 / spec section 11, 13, 25).

Uses a cheap Ridge model with a tiny PSO budget so this runs fast.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.core.config import Config, SplitConfig, OptimizationConfig
from src.splits.nested_walk_forward import create_nested_walk_forward_splits
from src.optimization.model_optimizer import optimize_model_nested


def _make_config():
    config = Config()
    config.splits = SplitConfig(
        min_train_size=30, test_size=4, horizons=[1],
        inner_min_train_size=16, inner_test_size=4,
        horizon_weights={1: 1.0, 2: 0.0, 4: 0.0}
    )
    config.optimization = OptimizationConfig(
        optimizer='pso', n_particles=4, n_iterations=2, seed=42, metric='mae'
    )
    config.model.models = ['ridge']
    return config


class TestPSOIsolation:
    def test_optimizer_never_receives_outer_test_rows(self, monkeypatch):
        n = 60
        dates = pd.date_range('2000-01-01', periods=n, freq='QS')
        rng = np.random.RandomState(0)
        X = pd.DataFrame({'f1': rng.randn(n), 'f2': rng.randn(n)}, index=dates)
        y = pd.Series(rng.randn(n), index=dates)

        config = _make_config()
        nested = create_nested_walk_forward_splits(X, y, config.splits)
        assert len(nested) > 0
        nf = nested[0]

        seen_dates = set()

        # Patch generate_cv_folds as used inside model_optimizer to record
        # every date the PSO objective actually trains/tests on.
        import src.optimization.model_optimizer as mod
        original_generate = mod.generate_cv_folds

        def spying_generate_cv_folds(X_arg, y_arg, cv_plan):
            for X_train, y_train, X_test, y_test, fold in original_generate(X_arg, y_arg, cv_plan):
                seen_dates.update(X_train.index.tolist())
                seen_dates.update(X_test.index.tolist())
                yield X_train, y_train, X_test, y_test, fold

        monkeypatch.setattr(mod, 'generate_cv_folds', spying_generate_cv_folds)

        optimize_model_nested(X, y, nf, 'ridge', config, output_dir=None)

        outer_test_dates = set(X.index[nf.outer.test_indices].tolist())
        overlap = seen_dates & outer_test_dates
        assert not overlap, f"PSO objective touched outer test dates: {overlap}"

    def test_returns_outer_fold_metadata(self):
        n = 60
        dates = pd.date_range('2000-01-01', periods=n, freq='QS')
        rng = np.random.RandomState(1)
        X = pd.DataFrame({'f1': rng.randn(n)}, index=dates)
        y = pd.Series(rng.randn(n), index=dates)

        config = _make_config()
        nested = create_nested_walk_forward_splits(X, y, config.splits)
        nf = nested[0]

        result = optimize_model_nested(X, y, nf, 'ridge', config, output_dir=None)
        assert result['outer_fold_id'] == nf.outer_fold_id
        assert result['outer_horizon'] == nf.outer.horizon
        assert result['n_inner_folds'] == nf.inner_cv_plan.n_folds
        assert 'best_params' in result

    def test_mase_metric_option_runs_without_error(self):
        n = 60
        dates = pd.date_range('2000-01-01', periods=n, freq='QS')
        rng = np.random.RandomState(2)
        X = pd.DataFrame({'f1': rng.randn(n)}, index=dates)
        y = pd.Series(100 + rng.randn(n).cumsum(), index=dates)

        config = _make_config()
        config.optimization.metric = 'mase'
        nested = create_nested_walk_forward_splits(X, y, config.splits)
        nf = nested[0]

        result = optimize_model_nested(X, y, nf, 'ridge', config, output_dir=None)
        assert np.isfinite(result['best_fitness'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
