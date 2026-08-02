"""
Artificial leakage sentinel test (spec section 27 / audit finding A-9).

Injects a synthetic column that is a harmless constant (0.0) everywhere
EXCEPT the outer test period, where it takes on a huge, wildly
out-of-distribution value. Nothing upstream of the outer test fold should
ever be able to see, learn from, select, or respond to that contaminated
value - if it does, the CV isolation the pipeline is built on is broken.

This exercises the real production code paths (build_tuning_cv_plan,
NestedFold/validate_nested_isolation, and an actual feature-selection call),
not a mock of them.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.core.config import Config, SplitConfig
from src.splits.walk_forward import create_walk_forward_splits, generate_cv_folds
from src.splits.nested_walk_forward import (
    build_tuning_cv_plan, create_nested_walk_forward_splits, validate_nested_isolation
)
from src.fs.linear import ridge_stability_selection
from src.fs.wrapper_methods import fs_wrapper
from src.fs.consensus import run_all_fs_options

SENTINEL_COL = "LEAKAGE_SENTINEL"
SENTINEL_SAFE_VALUE = 0.0
SENTINEL_CONTAMINATED_VALUE = 1e9


def _quarterly_data_with_sentinel(n=80, seed=0):
    dates = pd.date_range('2000-01-01', periods=n, freq='QS')
    rng = np.random.RandomState(seed)
    # Four independent noise features: enough headroom above min_features so
    # a feature-selection test below can't force-select the sentinel just to
    # satisfy a "select at least K" floor when too few real candidates exist.
    X = pd.DataFrame({
        'feat1': rng.randn(n),
        'feat2': rng.randn(n),
        'feat3': rng.randn(n),
        'feat4': rng.randn(n),
    }, index=dates)
    y = pd.Series(rng.randn(n).cumsum() + 100.0, index=dates, name='CO2e')
    return X, y, dates


def _quarterly_data_with_many_features(n=80, n_features=8, seed=1):
    """Wider pool than `_quarterly_data_with_sentinel` - needed so that
    top-k-style selectors (wrapper methods, `run_all_fs_options`) have room
    to *not* select the sentinel, rather than trivially selecting every
    column because top_k >= total column count.

    Unlike `_quarterly_data_with_sentinel`, `y` is given real dependence on
    a few of the features (rather than being pure noise unrelated to
    everything). This matters for permutation-importance-based selection:
    with a pure-noise target, EVERY feature - including the sentinel -has
    permutation importance hovering near zero, so which near-zero column
    randomly ranks in the "top-k" is an artifact of estimation noise, not of
    leakage. With real signal features, their importance is reliably above
    the sentinel's (a genuinely constant column's permutation importance is
    ~0 by construction - shuffling a constant changes nothing), so a
    correct, leakage-free FS run should never need the sentinel to fill out
    top-k.
    """
    dates = pd.date_range('2000-01-01', periods=n, freq='QS')
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        {f'feat{i}': rng.randn(n) for i in range(1, n_features + 1)}, index=dates
    )
    y = pd.Series(
        2.0 * X['feat1'] - 1.5 * X['feat2'] + 1.0 * X['feat3']
        + 0.1 * rng.randn(n) + 100.0,
        index=dates, name='CO2e'
    )
    return X, y, dates


def _split_config():
    return SplitConfig(
        min_train_size=40, test_size=4, horizons=[1],
        inner_min_train_size=16, inner_test_size=4,
    )


def _inject_sentinel(X: pd.DataFrame, contamination_start) -> pd.DataFrame:
    """Add a column that is safe (0.0) before `contamination_start` and a
    huge contaminated constant from `contamination_start` onward - i.e. it
    exists (in its "leaked" form) only in what will be the outer test
    period."""
    X_s = X.copy()
    sentinel = pd.Series(SENTINEL_SAFE_VALUE, index=X.index)
    sentinel.loc[sentinel.index >= contamination_start] = SENTINEL_CONTAMINATED_VALUE
    X_s[SENTINEL_COL] = sentinel
    return X_s


class TestLeakageSentinel:
    def test_tuning_plan_never_touches_a_contaminated_row(self):
        """PSO isolation (bug 3.5): build_tuning_cv_plan must never build a
        fold whose train or test rows include a date at/after the earliest
        outer-eval test start - i.e. it must never see the sentinel's
        contaminated value."""
        X, y, dates = _quarterly_data_with_sentinel()
        config = _split_config()

        eval_plan = create_walk_forward_splits(X, y, config)
        assert eval_plan.n_folds > 0

        contamination_start = min(f.test_start for f in eval_plan.folds)
        X_s = _inject_sentinel(X, contamination_start)

        tuning_plan = build_tuning_cv_plan(X_s, y, eval_plan, config)
        assert tuning_plan.n_folds > 0

        touched_positions = set()
        for f in tuning_plan.folds:
            touched_positions.update(f.train_indices)
            touched_positions.update(f.test_indices)

        touched_dates = X_s.index[sorted(touched_positions)]
        touched_values = X_s.loc[touched_dates, SENTINEL_COL]

        assert (touched_values == SENTINEL_SAFE_VALUE).all(), (
            "build_tuning_cv_plan produced a fold that touches a row where "
            "the leakage sentinel holds its outer-test-only contaminated "
            f"value: contaminated dates seen = "
            f"{touched_dates[touched_values != SENTINEL_SAFE_VALUE].tolist()}"
        )
        # Sanity: the contaminated rows actually exist in X_s at all (test
        # would be vacuous otherwise).
        assert (X_s[SENTINEL_COL] == SENTINEL_CONTAMINATED_VALUE).any()

    def test_nested_inner_folds_never_touch_a_contaminated_row(self):
        """Same guarantee via the per-outer-fold NestedFold path (bug 3.5).

        Unlike build_tuning_cv_plan (one shared cutoff - the earliest test
        date used by ANY outer fold), NestedFold's isolation boundary is
        per-outer-fold: each outer fold's own test_start. Once a quarter has
        passed, it legitimately becomes ordinary training data for later
        outer folds under expanding-window CV, so a single fixed "sentinel
        turns on here and stays on" column (as used in the test above)
        cannot represent every outer fold's boundary at once. Instead this
        injects a FRESH sentinel per outer fold, contaminated only from that
        specific fold's own test_start onward, and checks that fold's inner
        folds never see it - directly exercising validate_nested_isolation
        with a real injected value rather than only comparing dates."""
        X, y, dates = _quarterly_data_with_sentinel()
        config = _split_config()

        eval_plan = create_walk_forward_splits(X, y, config)
        assert eval_plan.n_folds > 0

        nested = create_nested_walk_forward_splits(X, y, config)
        assert len(nested) > 0
        assert validate_nested_isolation(X, nested)

        for nf in nested:
            X_s = _inject_sentinel(X, nf.outer.test_start)
            X_outer_train = X_s.iloc[nf.outer.train_indices]

            # The outer fold's own training slice must itself predate its
            # test_start (pre-existing walk-forward invariant) - sanity
            # check before trusting the inner-fold assertion below.
            assert (X_outer_train[SENTINEL_COL] == SENTINEL_SAFE_VALUE).all()

            for inner_fold in nf.inner_cv_plan.folds:
                inner_train_vals = X_outer_train.iloc[inner_fold.train_indices][SENTINEL_COL]
                inner_test_vals = X_outer_train.iloc[inner_fold.test_indices][SENTINEL_COL]
                assert (inner_train_vals == SENTINEL_SAFE_VALUE).all(), (
                    f"outer fold {nf.outer_fold_id}: an inner training fold "
                    f"touched a row contaminated relative to this outer "
                    f"fold's own test_start ({nf.outer.test_start})"
                )
                assert (inner_test_vals == SENTINEL_SAFE_VALUE).all(), (
                    f"outer fold {nf.outer_fold_id}: an inner test fold "
                    f"touched a row contaminated relative to this outer "
                    f"fold's own test_start ({nf.outer.test_start})"
                )

    def test_feature_selection_never_selects_the_sentinel(self):
        """Feature selection (bug 3.4): running a real FS method (Ridge
        stability selection) restricted to the isolated tuning-plan rows
        must never select LEAKAGE_SENTINEL - by construction it is a
        constant (zero variance, zero correlation with y) everywhere those
        rows can see, so a correct implementation cannot select it. If FS
        were mistakenly run against rows including the contaminated period,
        the huge out-of-distribution values would make Ridge assign it a
        large, easily-detected coefficient - this test would then fail."""
        X, y, dates = _quarterly_data_with_sentinel()
        config = _split_config()

        eval_plan = create_walk_forward_splits(X, y, config)
        contamination_start = min(f.test_start for f in eval_plan.folds)
        X_s = _inject_sentinel(X, contamination_start)

        tuning_plan = build_tuning_cv_plan(X_s, y, eval_plan, config)

        touched_positions = sorted({
            i for f in tuning_plan.folds
            for i in list(f.train_indices) + list(f.test_indices)
        })
        X_tune = X_s.iloc[touched_positions]
        y_tune = y.iloc[touched_positions]

        # min_features is explicit and well below the 5 available candidate
        # columns (feat1-4 + sentinel): with only 3 candidates and the
        # default min_features=3, EVERY column would be force-selected
        # regardless of importance, making the assertion below vacuous.
        selected_features, _ = ridge_stability_selection(
            X_tune, y_tune, tuning_plan, min_features=2
        )

        assert SENTINEL_COL not in selected_features, (
            "feature selection selected the leakage sentinel - it must have "
            "seen the contaminated (outer-test-only) value"
        )

    def test_model_training_never_receives_a_contaminated_row(self):
        """Model training (spec section 15): fitting a model on every
        tuning-plan fold's train slice must never expose it to a
        contaminated row either."""
        X, y, dates = _quarterly_data_with_sentinel()
        config = _split_config()

        eval_plan = create_walk_forward_splits(X, y, config)
        contamination_start = min(f.test_start for f in eval_plan.folds)
        X_s = _inject_sentinel(X, contamination_start)

        tuning_plan = build_tuning_cv_plan(X_s, y, eval_plan, config)

        for X_train, y_train, X_test, y_test, fold in generate_cv_folds(X_s, y, tuning_plan):
            assert (X_train[SENTINEL_COL] == SENTINEL_SAFE_VALUE).all()
            assert (X_test[SENTINEL_COL] == SENTINEL_SAFE_VALUE).all()

    def test_wrapper_fs_result_invariant_to_contamination(self):
        """Bug 3.4 regression: FS2 (RFE/SFS/SBS union) previously fit RFE on
        the full sample and SFS/SBS on a plain, non-time-aware sklearn
        `KFold` - both leakage risks.

        Framed as an invariance property rather than "was the sentinel ever
        selected": SFS/SBS are forced-top-k selectors, so once true signal
        is exhausted they must still pick *something* to fill out top_k -
        and a literal zero-variance column can legitimately tie with
        near-zero-value noise columns for those remaining slots with no
        leakage involved at all. That makes "sentinel not in selected_features"
        an unreliable signal.

        The property that actually proves isolation: since `fs_wrapper` only
        ever sees tuning-plan-visible rows, its output must be byte-for-byte
        identical whether the sentinel is contaminated beyond the visible
        window or stays safe (0.0) everywhere - a correct implementation
        cannot tell the difference because it never looks past that window.
        """
        X, y, dates = _quarterly_data_with_many_features()
        config = _split_config()

        eval_plan = create_walk_forward_splits(X, y, config)
        contamination_start = min(f.test_start for f in eval_plan.folds)

        X_contaminated = _inject_sentinel(X, contamination_start)
        X_clean = X.copy()
        X_clean[SENTINEL_COL] = SENTINEL_SAFE_VALUE

        # Cutoff/fold structure only depends on dates, not sentinel values,
        # so this tuning plan applies identically to both X variants.
        tuning_plan = build_tuning_cv_plan(X_contaminated, y, eval_plan, config)

        full_config = Config()
        full_config.fs.top_k_features = 5  # strictly less than the 9 candidate columns
        full_config.fs.min_features = 3

        result_contaminated = fs_wrapper(X_contaminated, y, tuning_plan, full_config)
        result_clean = fs_wrapper(X_clean, y, tuning_plan, full_config)

        assert result_contaminated['selected_features'] == result_clean['selected_features'], (
            "fs_wrapper's output changed depending on data outside the "
            "tuning window - the contaminated (outer-test-only) sentinel "
            "value must be completely invisible to it"
        )

    def test_run_all_fs_options_result_invariant_to_contamination(self):
        """End-to-end regression for the exact fix in scripts/01_run_fs.py
        (audit finding A-7): the script now builds an isolated tuning CV
        plan via `build_tuning_cv_plan` and passes it - not the flat outer
        `cv_plan` - into `run_all_fs_options`. Reproduces that call sequence
        and applies the same contamination-invariance property as
        `test_wrapper_fs_result_invariant_to_contamination` above (see its
        docstring for why "was the sentinel selected" is the wrong check).

        Methods are narrowed to linear_stability + wrapper (the two most
        directly touched by this fix) rather than all five primary FS
        methods, purely to keep this test's runtime reasonable - xgboost_shap
        and permutation_stability already have their own isolation coverage
        via `ridge_stability_selection`-style per-fold fitting and are not
        specific to this bug.
        """
        X, y, dates = _quarterly_data_with_many_features()
        config = _split_config()

        eval_plan = create_walk_forward_splits(X, y, config)
        contamination_start = min(f.test_start for f in eval_plan.folds)

        X_contaminated = _inject_sentinel(X, contamination_start)
        X_clean = X.copy()
        X_clean[SENTINEL_COL] = SENTINEL_SAFE_VALUE

        tuning_plan = build_tuning_cv_plan(X_contaminated, y, eval_plan, config)

        full_config = Config()
        full_config.fs.top_k_features = 5
        full_config.fs.min_features = 3
        full_config.fs.methods = ['linear_stability', 'wrapper']

        results_contaminated = run_all_fs_options(X_contaminated, y, tuning_plan, full_config)
        results_clean = run_all_fs_options(X_clean, y, tuning_plan, full_config)

        assert len(results_contaminated) > 0
        assert results_contaminated.keys() == results_clean.keys()
        for fs_name in results_contaminated:
            assert (
                results_contaminated[fs_name]['selected_features']
                == results_clean[fs_name]['selected_features']
            ), (
                f"{fs_name}'s output changed depending on data outside the "
                f"tuning window - the contaminated (outer-test-only) "
                f"sentinel value must be completely invisible to it"
            )


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
