"""
Spec section 2: TEC, CEI, TEC_dlog, CO2e_per_TEC, and any TEC/CEI-derived
feature (including the earlier CEI_lag1 governance variant, spec 3.2's
now-superseded approach) must be removed completely - from the registry,
from feature engineering, from configuration matrices, and from the
governed families/candidate pools - not merely excluded-by-default behind
an opt-in flag. This file is the dedicated end-to-end check; the
individual modules also carry narrower regression tests
(test_feature_registry.py, test_configurations.py,
test_feature_leakage_fixes.py).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.core.config import Config
from src.features.registry import FeatureRegistry, CONFIGURATION_TAGS
from src.features.configurations import build_configuration_matrices
from src.features.engineering import engineer_features, create_intensity_features

REGISTRY_PATH = Path(__file__).parent.parent / "config" / "feature_registry.yaml"
REMOVED_NAMES = ('TEC', 'CEI', 'TEC_dlog', 'CO2e_per_TEC', 'CEI_lag1')


@pytest.fixture(scope="module")
def registry():
    return FeatureRegistry.load(str(REGISTRY_PATH))


def _raw_df_with_tec_cei(n=40, seed=0):
    dates = pd.date_range('2010-01-01', periods=n, freq='QS')
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        'CO2e': 100_000 + rng.normal(0, 1000, n).cumsum(),
        'GDP': 500_000 + rng.normal(0, 5000, n).cumsum(),
        'Population': 60_000_000 + np.arange(n) * 10_000.0,
        'Air_Temp': 10 + 8 * np.sin(np.arange(n) * np.pi / 2) + rng.normal(0, 1, n),
        'Rainfall': 100 + rng.normal(0, 20, n),
        'COVID_Deaths': np.zeros(n),
        'TEC': 30_000 + rng.normal(0, 100, n).cumsum(),
        'CEI': 0.3 - np.arange(n) * 0.001,
    }, index=dates)


class TestRegistryHardRemoval:
    def test_removed_names_absent_from_registry(self, registry):
        for name in REMOVED_NAMES:
            assert name not in registry.all_names()

    def test_removed_names_absent_from_every_configuration(self, registry):
        for tag in CONFIGURATION_TAGS:
            members = registry.configuration_members(tag)
            for name in REMOVED_NAMES:
                assert name not in members

    def test_removed_names_absent_from_every_governed_family(self, registry):
        for fam in registry.all_governed_families():
            members = registry.governed_family(fam)
            for name in REMOVED_NAMES:
                assert name not in members

    def test_enforce_raises_for_any_removed_name(self, registry):
        for name in REMOVED_NAMES:
            with pytest.raises(ValueError):
                registry.enforce(['GDP', name], horizon=1)


class TestEngineerFeaturesDropsTecCei:
    def test_raw_tec_cei_columns_never_reach_x(self):
        df = _raw_df_with_tec_cei()
        config = Config()
        X, y, metadata = engineer_features(df, config, target_col='CO2e')

        for name in REMOVED_NAMES:
            assert name not in X.columns
        assert metadata.get('removed_columns') == ['TEC', 'CEI']

    def test_registry_enforcement_still_passes_with_tec_cei_in_source(self):
        # Even though the raw source `df` contains TEC/CEI (as the real
        # source spreadsheet still does until the raw file itself is
        # regenerated), enforce_availability_registry=True must not raise -
        # TEC/CEI are dropped structurally BEFORE the registry check runs.
        df = _raw_df_with_tec_cei()
        config = Config()
        assert config.features.enforce_availability_registry is True
        X, y, metadata = engineer_features(df, config, target_col='CO2e')
        assert metadata['registry_governance']['enforced'] is True


class TestConfigDefaultsExcludeTec:
    def test_roc_columns_default_has_no_tec(self):
        assert 'TEC' not in Config().features.roc_columns

    def test_intensity_denominators_default_has_no_tec(self):
        assert 'TEC' not in Config().features.intensity_denominators

    def test_no_cei_config_fields_remain(self):
        fields = Config().features.__dataclass_fields__
        for leftover in ('cei_column', 'cei_min_lag', 'include_contemporaneous_cei'):
            assert leftover not in fields


class TestFeatureBuildersRejectExplicitTec:
    def test_intensity_features_rejects_tec_denominator(self):
        dates = pd.date_range('2010-01-01', periods=10, freq='QS')
        df = pd.DataFrame({
            'CO2e': np.arange(10, dtype=float),
            'TEC': np.arange(10, dtype=float),
        }, index=dates)
        # create_intensity_features itself doesn't guard (the guard lives in
        # engineer_features' caller, matching where roc_columns is guarded) -
        # confirm the *pipeline* entry point raises instead.
        config = Config()
        config.features.include_intensity_features = True
        config.features.intensity_denominators = ['TEC']
        with pytest.raises(ValueError):
            engineer_features(df.assign(GDP=1.0), config, target_col='CO2e')

    def test_roc_columns_with_tec_raises(self):
        dates = pd.date_range('2010-01-01', periods=10, freq='QS')
        df = pd.DataFrame({
            'CO2e': np.arange(10, dtype=float),
            'GDP': np.arange(10, dtype=float),
        }, index=dates)
        config = Config()
        config.features.include_roc_features = True
        config.features.roc_columns = ['TEC', 'GDP']
        with pytest.raises(ValueError):
            engineer_features(df, config, target_col='CO2e')


class TestBuildConfigurationMatricesExcludesTecCei:
    def test_a1_through_a4_never_contain_removed_names(self, registry):
        dates = pd.date_range('2015-01-01', periods=30, freq='QS')
        rng = np.random.RandomState(1)
        non_grid_names = [
            name for name in registry.retained_names()
            if set(registry.entries[name]['configuration_membership']) & {'A1', 'A2'}
        ]
        X = pd.DataFrame({name: rng.randn(30) for name in non_grid_names}, index=dates)
        grid_names = [
            name for name in registry.retained_names()
            if 'A3' in registry.entries[name]['configuration_membership']
        ]
        grid_df = pd.DataFrame({name: rng.randn(30) for name in grid_names}, index=dates)

        matrices = build_configuration_matrices(X, registry, grid_df=grid_df)

        for tag, mat in matrices.items():
            for name in REMOVED_NAMES:
                assert name not in mat.columns, f"{name} leaked into {tag}"


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
