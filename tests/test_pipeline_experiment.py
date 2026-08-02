"""
Tests for the single-cell experiment orchestrator (spec sections 12/15):
src/pipeline/experiment.py::run_configuration_model.

Covers both tuning modes (config.optimization.nested_retuning False/True),
provenance shape, and a leakage-sentinel invariance check specific to this
new orchestration layer (PSO tuning + refit + predict all happen inside
this function, so it needs its own isolation proof, not just reliance on
the primitives it calls).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.core.config import Config
from src.splits.walk_forward import create_walk_forward_splits
from src.pipeline.experiment import run_configuration_model, ExperimentCell

SENTINEL_COL = "LEAKAGE_SENTINEL"
SENTINEL_SAFE_VALUE = 0.0
SENTINEL_CONTAMINATED_VALUE = 1e9


def _fast_config():
    config = Config()
    config.splits.min_train_size = 20
    config.splits.test_size = 4
    config.splits.horizons = [1, 2]
    config.splits.horizon_weights = {1: 0.6, 2: 0.4}
    config.splits.inner_min_train_size = 10
    config.splits.inner_test_size = 4
    config.optimization.n_particles = 3
    config.optimization.n_iterations = 2
    config.model.models = ['ridge']
    # These orchestration-mechanics tests build `y` as plain raw values, not
    # log-transformed - target_transform='none' keeps that consistent with
    # run_configuration_model's original-scale inversion step (see
    # TestOriginalScaleInversion below for dedicated log-transform coverage).
    config.data.target_transform = 'none'
    return config


def _synthetic_Xy(n=40, n_features=3, seed=0):
    dates = pd.date_range('2015-01-01', periods=n, freq='QS')
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        {f'feat{i}': rng.randn(n) for i in range(1, n_features + 1)}, index=dates
    )
    y = pd.Series(
        1.5 * X['feat1'] - 0.8 * X['feat2'] + 0.05 * rng.randn(n) + 50.0,
        index=dates, name='CO2e'
    )
    return X, y


class TestRunConfigurationModelSweepMode:
    def test_produces_predictions_and_provenance(self):
        X, y = _synthetic_Xy()
        config = _fast_config()
        eval_plan = create_walk_forward_splits(X, y, config.splits)
        assert eval_plan.n_folds > 0

        cell = ExperimentCell(configuration='A1', panel='panel1_full_period',
                               fs_option='all_features', model='ridge', seed=42)
        result = run_configuration_model(X, y, eval_plan, cell, config, run_id='test_run')

        assert len(result['predictions']) > 0
        assert len(result['fold_provenance']) == eval_plan.n_folds

        pred = result['predictions'][0]
        for key in ('run_id', 'configuration', 'panel', 'fs_option', 'model',
                    'horizon', 'fold_id', 'target_date', 'actual', 'predicted', 'residual'):
            assert key in pred
        assert pred['configuration'] == 'A1'
        assert pred['model'] == 'ridge'

        prov = result['fold_provenance'][0]
        for key in ('train_start', 'train_end', 'forecast_origin', 'test_start',
                    'test_end', 'selected_features', 'hyperparameters', 'runtime_seconds'):
            assert key in prov
        assert prov['selected_features'] == list(X.columns)

        summary = result['summary']
        assert summary['configuration'] == 'A1'
        assert 'weighted_mase' in summary
        assert summary['n_folds'] == eval_plan.n_folds

    def test_same_hyperparameters_reused_across_folds_in_sweep_mode(self):
        """Sweep mode tunes once (shared tuning plan) and reuses those
        hyperparameters for every outer fold's refit - unlike nested mode,
        which retunes per fold."""
        X, y = _synthetic_Xy()
        config = _fast_config()
        eval_plan = create_walk_forward_splits(X, y, config.splits)

        cell = ExperimentCell(configuration='A1', panel='panel1_full_period',
                               fs_option='all_features', model='ridge')
        result = run_configuration_model(X, y, eval_plan, cell, config, run_id='test_run')

        hyperparams = [fp['hyperparameters'] for fp in result['fold_provenance']]
        assert all(h == hyperparams[0] for h in hyperparams)


class TestRunConfigurationModelNestedMode:
    def test_nested_retuning_produces_predictions(self):
        X, y = _synthetic_Xy(n=50)
        config = _fast_config()
        config.optimization.nested_retuning = True
        eval_plan = create_walk_forward_splits(X, y, config.splits)
        assert eval_plan.n_folds > 0

        cell = ExperimentCell(configuration='A2', panel='panel2_common_period',
                               fs_option='all_features', model='ridge')
        result = run_configuration_model(X, y, eval_plan, cell, config, run_id='test_run')

        assert len(result['predictions']) > 0
        assert result['summary']['n_folds'] > 0
        for pred in result['predictions']:
            assert pred['panel'] == 'panel2_common_period'


class TestLeakageInvariance:
    """The orchestrator (tuning + refit + predict) must produce byte-identical
    results whether data beyond the outer-training window is contaminated or
    clean - see tests/test_leakage_sentinel.py for why this invariance
    framing (rather than 'sentinel never selected/used') is the correct,
    robust property to check."""

    def _run_with_sentinel(self, contaminated: bool, nested: bool):
        X, y = _synthetic_Xy(n=40, n_features=3, seed=1)
        config = _fast_config()
        config.optimization.nested_retuning = nested

        eval_plan = create_walk_forward_splits(X, y, config.splits)
        # Contaminate from the LAST outer fold's test_start onward - not the
        # earliest. Under expanding-window CV, a date that is fold N's test
        # period legitimately becomes ordinary TRAINING data for fold N+1
        # (that's the whole point of expanding folds) - contaminating from
        # the earliest fold's test_start would make later folds correctly
        # "see" it as past history, which is not leakage. Only contamination
        # strictly at/after every fold's own test_start is guaranteed to
        # never enter any fold's training window.
        contamination_start = max(f.test_start for f in eval_plan.folds)

        sentinel = pd.Series(SENTINEL_SAFE_VALUE, index=X.index)
        if contaminated:
            sentinel.loc[sentinel.index >= contamination_start] = SENTINEL_CONTAMINATED_VALUE
        X_s = X.copy()
        X_s[SENTINEL_COL] = sentinel

        cell = ExperimentCell(configuration='A1', panel='panel1_full_period',
                               fs_option='all_features', model='ridge')
        result = run_configuration_model(X_s, y, eval_plan, cell, config, run_id='test_run')
        return result, contamination_start

    @staticmethod
    def _pre_contamination_predictions(result, contamination_start):
        # Predictions AT/AFTER contamination_start legitimately differ - the
        # sentinel is a real input feature at that fold's own forecast time,
        # so a different feature value correctly produces a different
        # prediction (that's not leakage, that's just a different input).
        # Only dates strictly before contamination_start can prove isolation.
        return [
            (p['fold_id'], p['target_date'], round(p['predicted'], 8))
            for p in result['predictions']
            if p['target_date'] < contamination_start
        ]

    def test_sweep_mode_predictions_invariant_to_contamination(self):
        result_contaminated, cutoff = self._run_with_sentinel(contaminated=True, nested=False)
        result_clean, _ = self._run_with_sentinel(contaminated=False, nested=False)

        preds_c = self._pre_contamination_predictions(result_contaminated, cutoff)
        preds_clean = self._pre_contamination_predictions(result_clean, cutoff)
        assert len(preds_c) > 0  # sanity: comparison isn't vacuous
        assert preds_c == preds_clean, (
            "sweep-mode predictions changed depending on data outside the "
            "outer-training window - tuning or refit must have seen a "
            "contaminated row"
        )

    def test_nested_mode_predictions_invariant_to_contamination(self):
        result_contaminated, cutoff = self._run_with_sentinel(contaminated=True, nested=True)
        result_clean, _ = self._run_with_sentinel(contaminated=False, nested=True)

        preds_c = self._pre_contamination_predictions(result_contaminated, cutoff)
        preds_clean = self._pre_contamination_predictions(result_clean, cutoff)
        assert len(preds_c) > 0
        assert preds_c == preds_clean, (
            "nested-mode predictions changed depending on data outside the "
            "outer-training window - per-fold retuning or refit must have "
            "seen a contaminated row"
        )


class TestOriginalScaleInversion:
    """Regression: run_configuration_model previously reported every metric
    on whatever scale `y` happened to be in (log-transformed by default,
    config.data.target_transform='log'), never inverting back to original-
    scale CO2e - a direct violation of spec section 2/17 ("All evaluation
    metrics... must use original-scale values" / "Calculate all metrics on
    original-scale CO2e"). This was live for the entire A1-A4 experimental-
    grid pipeline (src/pipeline/experiment.py, scripts/10) from when it was
    first built until this fix - caught only by manually re-deriving what
    scale the reported weighted_mase numbers were actually in, not by any
    test, since no test had checked which scale predictions.actual/predicted
    were reported in."""

    def test_log_transform_is_inverted_in_reported_predictions(self):
        n = 40
        dates = pd.date_range('2015-01-01', periods=n, freq='QS')
        rng = np.random.RandomState(5)
        X = pd.DataFrame({'feat1': rng.randn(n), 'feat2': rng.randn(n)}, index=dates)

        # Original-scale CO2e-like values (always positive, as a real target
        # would be) - y fed into the model is log(true_value), exactly what
        # engineer_features()/create_target_variable() produce by default.
        true_original = pd.Series(1000.0 + rng.uniform(-50, 50, n), index=dates)
        y_log = np.log(true_original)

        config = _fast_config()
        config.data.target_transform = 'log'
        eval_plan = create_walk_forward_splits(X, y_log, config.splits)
        assert eval_plan.n_folds > 0

        cell = ExperimentCell(configuration='A1', panel='panel1_full_period',
                               fs_option='all_features', model='ridge')
        result = run_configuration_model(X, y_log, eval_plan, cell, config, run_id='test_run')

        assert len(result['predictions']) > 0
        for pred in result['predictions']:
            # A log-CO2e-scale value would be roughly ln(1000) ~= 6.9;
            # original-scale values are ~1000. This distinguishes "still in
            # log space" (bug) from "correctly inverted" (fix) unambiguously.
            assert pred['actual'] > 100, (
                f"predicted 'actual' value {pred['actual']} looks like it is "
                f"still in log-transformed space, not inverted to original-"
                f"scale CO2e (expected roughly 950-1050)"
            )
            assert 100 < pred['actual'] < 5000

    def test_none_transform_leaves_values_unchanged(self):
        X, y = _synthetic_Xy(seed=6)
        config = _fast_config()  # already target_transform='none'
        eval_plan = create_walk_forward_splits(X, y, config.splits)

        cell = ExperimentCell(configuration='A1', panel='panel1_full_period',
                               fs_option='all_features', model='ridge')
        result = run_configuration_model(X, y, eval_plan, cell, config, run_id='test_run')

        for pred in result['predictions']:
            original_value = y.loc[pred['target_date']]
            assert pred['actual'] == pytest.approx(original_value)

    def test_unsupported_transform_raises_clear_error(self):
        X, y = _synthetic_Xy(seed=7)
        config = _fast_config()
        config.data.target_transform = 'delta_log'  # not yet supported by this pipeline
        eval_plan = create_walk_forward_splits(X, y, config.splits)

        cell = ExperimentCell(configuration='A1', panel='panel1_full_period',
                               fs_option='all_features', model='ridge')
        with pytest.raises(ValueError, match='delta_log'):
            run_configuration_model(X, y, eval_plan, cell, config, run_id='test_run')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
