"""
Tests for src/interpretability/integrity_checks.py, guarding against the
exact failure mode diagnosed in outputs/audit/champion_shap_diagnosis.md:
an interpretability refit that silently collapses to a constant predictor
(e.g. via untuned/default hyperparameters on a tiny sample) must be flagged
loudly, not shipped as a plausible-looking all-zero SHAP table.
"""
import numpy as np
import pandas as pd
import pytest

from src.interpretability.integrity_checks import compute_regime_shap_with_integrity


class _ConstantModel:
    """Mimics a degenerate tree model that predicts the same value regardless
    of input - the exact shape of the original bug (LightGBM with
    min_child_samples exceeding the sample size)."""

    def __init__(self, value=1.0):
        self.value = value

    def predict(self, X):
        n = len(X) if hasattr(X, '__len__') else X.shape[0]
        return np.full(n, self.value)

    def fit(self, X, y):
        return self


class _LinearlyResponsiveModel:
    """A trivial model whose prediction genuinely depends on the first
    feature column, so TreeExplainer-style attribution should be non-zero
    and should vary across regimes with different feature distributions."""

    def predict(self, X):
        arr = X.values if hasattr(X, 'values') else np.asarray(X)
        return arr[:, 0] * 2.0 + arr[:, 1] * 0.1

    def fit(self, X, y):
        return self


def _make_X(n=30, seed=0):
    rng = np.random.RandomState(seed)
    idx = pd.date_range('2018-01-01', periods=n, freq='QS')
    return pd.DataFrame(
        {'feat_a': rng.normal(size=n), 'feat_b': rng.normal(size=n)},
        index=idx,
    )


REGIME_PERIODS = {
    'pre_covid': (None, '2020-01-01'),
    'covid': ('2020-01-01', '2022-01-01'),
    'post_covid': ('2022-01-01', None),
}


class TestComputeRegimeShapWithIntegrity:
    def test_constant_model_flagged_all_zero_and_identical(self, monkeypatch):
        """A model that ignores every feature must fail integrity checks -
        this is the exact bug the fix targets."""
        import src.interpretability.integrity_checks as mod

        def fake_compute_shap_values(model, X, model_type):
            n, p = len(X), X.shape[1]
            return np.zeros((n, p)), type('E', (), {'expected_value': 1.0})()

        monkeypatch.setattr(mod, 'compute_shap_values', fake_compute_shap_values)

        X = _make_X()
        model = _ConstantModel()
        regime_results, checks_df = compute_regime_shap_with_integrity(
            model, X, REGIME_PERIODS, 'tree', 'Best_A', 'lightgbm'
        )

        assert not checks_df.empty
        assert not checks_df['passed'].any(), "constant model must fail every regime's integrity check"
        assert (checks_df['issues'].str.contains('all_values_zero')).all()
        # at least one regime should also be flagged as identical to another
        assert checks_df['issues'].str.contains('identical_to_').any()

    def test_responsive_model_passes_and_regimes_differ(self, monkeypatch):
        """A model that genuinely uses features should produce non-zero,
        regime-varying SHAP and pass all integrity checks."""
        import src.interpretability.integrity_checks as mod

        rng = np.random.RandomState(1)

        def fake_compute_shap_values(model, X, model_type):
            n, p = len(X), X.shape[1]
            arr = X.values if hasattr(X, 'values') else np.asarray(X)
            # Deterministic, feature-dependent, regime-varying "SHAP" values
            sv = np.zeros((n, p))
            sv[:, 0] = arr[:, 0] * 2.0
            sv[:, 1] = arr[:, 1] * 0.1
            expected_value = float(model.predict(X).mean() - sv.sum(axis=1).mean())
            return sv, type('E', (), {'expected_value': expected_value})()

        monkeypatch.setattr(mod, 'compute_shap_values', fake_compute_shap_values)

        X = _make_X(n=40, seed=2)
        model = _LinearlyResponsiveModel()
        regime_results, checks_df = compute_regime_shap_with_integrity(
            model, X, REGIME_PERIODS, 'tree', 'Best_A', 'lightgbm'
        )

        assert checks_df['passed'].all(), checks_df[~checks_df['passed']]
        assert len(regime_results) == 3
        means = [df.set_index('feature')['importance'] for df in regime_results.values()]
        # Regimes should not all be pairwise identical (different underlying data)
        assert not all(means[0].equals(m) for m in means[1:])

    def test_row_and_feature_count_match_observations(self, monkeypatch):
        import src.interpretability.integrity_checks as mod

        def fake_compute_shap_values(model, X, model_type):
            n, p = len(X), X.shape[1]
            sv = np.ones((n, p)) * 0.5
            return sv, type('E', (), {'expected_value': 0.0})()

        monkeypatch.setattr(mod, 'compute_shap_values', fake_compute_shap_values)

        X = _make_X(n=25, seed=3)
        model = _LinearlyResponsiveModel()
        regime_results, checks_df = compute_regime_shap_with_integrity(
            model, X, REGIME_PERIODS, 'tree', 'Best_A', 'lightgbm'
        )

        for _, row in checks_df.iterrows():
            assert row['n_shap_rows'] == row['n_observations']
            assert row['n_features_shap'] == row['n_features_expected']

    def test_nan_values_flagged(self, monkeypatch):
        import src.interpretability.integrity_checks as mod

        def fake_compute_shap_values(model, X, model_type):
            n, p = len(X), X.shape[1]
            sv = np.ones((n, p))
            sv[0, 0] = np.nan
            return sv, type('E', (), {'expected_value': 0.0})()

        monkeypatch.setattr(mod, 'compute_shap_values', fake_compute_shap_values)

        X = _make_X(n=20, seed=4)
        model = _LinearlyResponsiveModel()
        _, checks_df = compute_regime_shap_with_integrity(
            model, X, REGIME_PERIODS, 'tree', 'Best_A', 'lightgbm'
        )
        assert checks_df['issues'].str.contains('nan_present').any()
        assert not checks_df['passed'].all()
