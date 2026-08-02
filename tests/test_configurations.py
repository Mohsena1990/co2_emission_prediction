"""
Tests for the A1-A4 data-configuration architecture (spec section 8):
registry-driven membership and the matrix builder in
src/features/configurations.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.features.registry import FeatureRegistry, CONFIGURATION_TAGS
from src.features.configurations import build_configuration_matrices, configuration_manifest

REGISTRY_PATH = Path(__file__).parent.parent / "config" / "feature_registry.yaml"

A1_EXPECTED = {
    'GDP', 'Population', 'Air_Temp', 'Rainfall', 'COVID_Deaths',
    'Mobility_Retail_Recreation', 'Mobility_Grocery_Pharmacy', 'Mobility_Parks',
    'Mobility_Transit_Stations', 'Mobility_Workplaces', 'Mobility_Residential',
}
A2_ENGINEERED_ONLY_EXPECTED = {
    'CO2e_lag1', 'CO2e_lag2', 'CO2e_lag3', 'CO2e_lag4',
    'CO2e_dlog', 'GDP_dlog',
    'CO2e_per_Population',
    'HDD_proxy', 'CDD_proxy', 'Q2', 'Q3', 'Q4', 'COVID', 'EnergyCrisis',
}


@pytest.fixture(scope="module")
def registry():
    return FeatureRegistry.load(str(REGISTRY_PATH))


@pytest.fixture(scope="module")
def grid_feature_names(registry):
    return {
        name for name, e in registry.entries.items()
        if 'A3' in e['configuration_membership'] and 'A1' not in e['configuration_membership']
    }


def _synthetic_X_for_registry(registry, n=40, seed=0):
    """Build a fake feature matrix mirroring engineer_features()'s real
    output: every retained registry feature that is a member of A1 or A2
    (raw/engineered candidates), EXCLUDING grid-only features (A3/A4-only) -
    in the real pipeline, grid columns never come from X/engineer_features,
    only from `grid_df` (see build_configuration_matrices). Also adds one
    column NOT in the registry at all, to prove the builder never leaks
    unknown columns into a configuration."""
    dates = pd.date_range('2010-01-01', periods=n, freq='QS')
    rng = np.random.RandomState(seed)
    non_grid_names = [
        name for name in registry.retained_names()
        if set(registry.entries[name]['configuration_membership']) & {'A1', 'A2'}
    ]
    data = {name: rng.randn(n) for name in non_grid_names}
    data['not_in_registry'] = rng.randn(n)
    return pd.DataFrame(data, index=dates)


class TestConfigurationMembership:
    def test_a1_is_exactly_eleven_raw_plus_mobility(self, registry):
        members = set(registry.configuration_members('A1'))
        assert members == A1_EXPECTED
        assert len(members) == 11

    def test_a2_is_a1_plus_fourteen_engineered_equals_25(self, registry):
        members = set(registry.configuration_members('A2'))
        assert members == A1_EXPECTED | A2_ENGINEERED_ONLY_EXPECTED
        assert len(members) == 25

    def test_a1_is_subset_of_a2(self, registry):
        assert set(registry.configuration_members('A1')) <= set(registry.configuration_members('A2'))

    def test_a3_is_a1_plus_grid_and_a4_is_a2_plus_grid(self, registry, grid_feature_names):
        # A3 = A1 (11 raw+mobility) + K grid features; A4 = A2 (25) + K grid features.
        a3 = set(registry.configuration_members('A3'))
        a4 = set(registry.configuration_members('A4'))
        assert a3 == A1_EXPECTED | grid_feature_names
        assert a4 == (A1_EXPECTED | A2_ENGINEERED_ONLY_EXPECTED) | grid_feature_names
        assert len(a3) == 11 + len(grid_feature_names)
        assert len(a4) == 25 + len(grid_feature_names)

    def test_a1_and_a2_have_no_grid_features(self, registry, grid_feature_names):
        assert set(registry.configuration_members('A1')) & grid_feature_names == set()
        assert set(registry.configuration_members('A2')) & grid_feature_names == set()

    def test_tec_and_cei_excluded_from_every_configuration(self, registry):
        for tag in CONFIGURATION_TAGS:
            members = registry.configuration_members(tag)
            for removed in ('TEC', 'CEI', 'TEC_dlog', 'CO2e_per_TEC', 'CEI_lag1'):
                assert removed not in members

    def test_all_configuration_memberships_matches_individual_calls(self, registry):
        all_members = registry.all_configuration_memberships()
        for tag in CONFIGURATION_TAGS:
            assert all_members[tag] == registry.configuration_members(tag)

    def test_unknown_tag_raises(self, registry):
        with pytest.raises(ValueError):
            registry.configuration_members('A5')


class TestBuildConfigurationMatrices:
    def test_matrix_shapes_match_registry_membership(self, registry):
        X = _synthetic_X_for_registry(registry)
        matrices = build_configuration_matrices(X, registry, grid_df=None)

        assert set(matrices.keys()) == set(CONFIGURATION_TAGS)
        assert set(matrices['A1'].columns) == A1_EXPECTED
        assert len(matrices['A2'].columns) == 25
        # A3/A4 fall back to A1/A2 in the absence of grid_df.
        assert set(matrices['A3'].columns) == set(matrices['A1'].columns)
        assert set(matrices['A4'].columns) == set(matrices['A2'].columns)

    def test_unknown_column_never_enters_any_configuration(self, registry):
        X = _synthetic_X_for_registry(registry)
        matrices = build_configuration_matrices(X, registry, grid_df=None)
        for tag, mat in matrices.items():
            assert 'not_in_registry' not in mat.columns

    def test_matrix_values_are_untouched_slices_of_x(self, registry):
        X = _synthetic_X_for_registry(registry)
        matrices = build_configuration_matrices(X, registry, grid_df=None)
        for col in matrices['A1'].columns:
            pd.testing.assert_series_equal(matrices['A1'][col], X[col])

    def test_grid_df_extends_a3_and_a4_only(self, registry):
        X = _synthetic_X_for_registry(registry)
        rng = np.random.RandomState(1)
        # Grid data only available for a later sub-window (mimics real
        # grid-data availability starting well after 1999).
        grid_index = X.index[10:]
        grid_df = pd.DataFrame(
            {'Grid_CI_mean': rng.randn(len(grid_index)),
             'Grid_renewable_share': rng.rand(len(grid_index))},
            index=grid_index
        )

        matrices = build_configuration_matrices(X, registry, grid_df=grid_df)

        # A1/A2 unaffected by grid_df.
        assert set(matrices['A1'].columns) == A1_EXPECTED
        assert len(matrices['A2'].columns) == 25

    def test_partially_missing_grid_quarter_is_dropped_not_left_as_nan(self, registry):
        """Regression: a real quarter (2018Q1) passed the overall
        completeness check (based on carbon-intensity readings) but had a
        fully-missing generation-mix sub-field, producing NaN in A3/A4 that
        crashed model fitting downstream (sklearn Ridge: 'Input X contains
        NaN'). The affected quarter must be dropped entirely from A3/A4, not
        silently left with NaN and not imputed."""
        X = _synthetic_X_for_registry(registry)
        rng = np.random.RandomState(3)
        grid_index = X.index[5:]
        grid_df = pd.DataFrame(
            {'Grid_CI_mean': rng.randn(len(grid_index)),
             'Grid_renewable_share': rng.rand(len(grid_index))},
            index=grid_index
        )
        # Simulate one quarter with a fully-missing generation-mix field.
        grid_df.loc[grid_df.index[0], 'Grid_renewable_share'] = np.nan

        matrices = build_configuration_matrices(X, registry, grid_df=grid_df)

        assert not matrices['A3'].isnull().any().any()
        assert not matrices['A4'].isnull().any().any()
        assert grid_df.index[0] not in matrices['A3'].index
        assert len(matrices['A3']) == len(grid_index) - 1
        assert len(matrices['A4']) == len(grid_index) - 1


class TestMobilityDfMerge:
    def _X_without_mobility(self, registry, n=20, seed=0):
        """A1/A2 feature matrix EXCLUDING the mobility columns, mirroring
        the real pipeline where mobility comes only from `mobility_df`."""
        dates = pd.date_range('2010-01-01', periods=n, freq='QS')
        rng = np.random.RandomState(seed)
        non_mobility_non_grid = [
            name for name in registry.retained_names()
            if (set(registry.entries[name]['configuration_membership']) & {'A1', 'A2'})
            and registry.entries[name]['family'] != 'mobility_exogenous'
        ]
        return pd.DataFrame({name: rng.randn(n) for name in non_mobility_non_grid}, index=dates)

    def test_mobility_df_merges_into_all_four_configurations(self, registry):
        X = self._X_without_mobility(registry)
        rng = np.random.RandomState(4)
        mobility_cols = [
            'Mobility_Retail_Recreation', 'Mobility_Grocery_Pharmacy', 'Mobility_Parks',
            'Mobility_Transit_Stations', 'Mobility_Workplaces', 'Mobility_Residential',
        ]
        mobility_df = pd.DataFrame({c: rng.randn(len(X)) for c in mobility_cols}, index=X.index)

        matrices = build_configuration_matrices(X, registry, grid_df=None, mobility_df=mobility_df)

        for tag in CONFIGURATION_TAGS:
            for c in mobility_cols:
                assert c in matrices[tag].columns, f"{c} missing from {tag}"

    def test_mobility_df_does_not_truncate_sample_period(self, registry):
        # Unlike grid_df, mobility_df is expected to already cover X's full
        # index (neutral-zero-filled) - merging it must not drop any rows.
        X = self._X_without_mobility(registry)
        rng = np.random.RandomState(5)
        mobility_df = pd.DataFrame(
            {'Mobility_Retail_Recreation': rng.randn(len(X))}, index=X.index
        )
        matrices = build_configuration_matrices(X, registry, grid_df=None, mobility_df=mobility_df)
        assert len(matrices['A1']) == len(X)
        assert len(matrices['A2']) == len(X)

    def test_no_mobility_df_omits_mobility_columns_with_warning(self, registry):
        X = self._X_without_mobility(registry)
        matrices = build_configuration_matrices(X, registry, grid_df=None, mobility_df=None)
        for tag in CONFIGURATION_TAGS:
            assert not any(c.startswith('Mobility_') for c in matrices[tag].columns)


class TestConfigurationManifest:
    def test_manifest_counts_match_matrices(self, registry):
        X = _synthetic_X_for_registry(registry)
        matrices = build_configuration_matrices(X, registry, grid_df=None)
        manifest = configuration_manifest(matrices, registry)

        for tag in CONFIGURATION_TAGS:
            assert manifest[tag]['n_features_total'] == len(matrices[tag].columns)
            assert manifest[tag]['columns'] == list(matrices[tag].columns)

        assert manifest['A1']['n_raw'] == 11  # 5 raw + 6 mobility-shock predictors
        assert manifest['A1']['n_engineered'] == 0
        assert manifest['A2']['n_engineered'] == 14
        assert manifest['A1']['nominal_max'] == 11

    def test_manifest_counts_grid_features_separately_from_raw(self, registry):
        """Regression: grid features are registered with kind='raw' (they
        ARE raw electricity-system measurements) - configuration_manifest
        must not fold them into n_raw, and must not require them to be
        absent from the registry to be counted as grid (a real bug found
        when the registry actually gained grid rows in Phase 3: grid_count
        was computed as `c not in registry.entries`, which became always-0
        the moment grid rows were added, and silently broke the Panel 2
        common-period trigger in scripts/00_make_dataset.py, which checks
        manifest['A3']['n_grid'] > 0)."""
        X = _synthetic_X_for_registry(registry)
        rng = np.random.RandomState(2)
        grid_index = X.index[5:]
        grid_df = pd.DataFrame(
            {'Grid_CI_mean': rng.randn(len(grid_index)),
             'Grid_renewable_share': rng.rand(len(grid_index))},
            index=grid_index
        )

        matrices = build_configuration_matrices(X, registry, grid_df=grid_df)
        manifest = configuration_manifest(matrices, registry)

        assert manifest['A3']['n_grid'] == 2
        assert manifest['A3']['n_raw'] == 11  # unchanged - grid not folded into raw
        assert manifest['A4']['n_grid'] == 2
        assert manifest['A1']['n_grid'] == 0
        assert manifest['A2']['n_grid'] == 0
        assert manifest['A2']['nominal_max'] == 25


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
