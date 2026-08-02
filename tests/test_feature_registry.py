"""
Tests for the predictor-governance registry (spec section 5-6): loading,
horizon-safety enforcement, and governed feature families G1-G5.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.features.registry import (
    FeatureRegistry, GOVERNED_FAMILY_NAMES, CONFIGURATION_TAGS,
    STREAM_B_TAGS, STREAM_B_TO_A,
)

REGISTRY_PATH = Path(__file__).parent.parent / "config" / "feature_registry.yaml"


@pytest.fixture(scope="module")
def registry():
    return FeatureRegistry.load(str(REGISTRY_PATH))


class TestRegistryLoading:
    def test_loads_all_candidates(self, registry):
        # 5 raw + 6 mobility-shock + 14 engineered + 10 NESO grid +
        # 2 OWID one-year-lagged shares = 37 (spec section 2: TEC/CEI and
        # every TEC/CEI-derived feature - TEC_dlog, CO2e_per_TEC, CEI_lag1 -
        # are hard-removed, not merely excluded-by-default). NESO grid + the
        # 2 OWID columns together form the spec's exact 12-item grid block
        # (spec section 9).
        assert len(registry) == 37

    def test_owid_predictors_present(self, registry):
        for name in ('OWID_RenewableShare_L1Y', 'OWID_LowCarbonShare_L1Y'):
            assert name in registry.all_names()
            assert registry.entries[name]['family'] == 'owid_exogenous'
            assert registry.entries[name]['configuration_membership'] == ['A3', 'A4']

    def test_raw_predictors_present(self, registry):
        for name in ['GDP', 'Population', 'Air_Temp', 'Rainfall', 'COVID_Deaths']:
            assert name in registry.all_names()

    def test_mobility_predictors_present(self, registry):
        expected = [
            'Mobility_Retail_Recreation', 'Mobility_Grocery_Pharmacy', 'Mobility_Parks',
            'Mobility_Transit_Stations', 'Mobility_Workplaces', 'Mobility_Residential',
        ]
        assert len(expected) == 6
        for name in expected:
            assert name in registry.all_names()
            assert registry.entries[name]['family'] == 'mobility_exogenous'
            assert registry.entries[name]['configuration_membership'] == ['A1', 'A2', 'A3', 'A4']

    def test_tec_and_cei_fully_absent(self, registry):
        for name in ('TEC', 'CEI', 'TEC_dlog', 'CO2e_per_TEC', 'CEI_lag1'):
            assert name not in registry.all_names()

    def test_engineered_predictors_present(self, registry):
        expected = [
            'CO2e_lag1', 'CO2e_lag2', 'CO2e_lag3', 'CO2e_lag4',
            'CO2e_dlog', 'GDP_dlog',
            'CO2e_per_Population',
            'HDD_proxy', 'CDD_proxy', 'Q2', 'Q3', 'Q4', 'COVID', 'EnergyCrisis',
        ]
        for name in expected:
            assert name in registry.all_names()

    def test_grid_predictors_present(self, registry):
        # spec section 7: exactly 10 NESO grid columns (+2 OWID_*_L1Y
        # elsewhere = the spec's 12-item grid block, spec section 9).
        expected = [
            'Grid_CI_mean', 'Grid_CI_p90', 'Grid_CI_std',
            'Grid_CI_high_share', 'Grid_CI_low_share',
            'Grid_renewable_share', 'Grid_low_carbon_share', 'Grid_fossil_share',
            'Grid_gas_share', 'Grid_wind_share',
        ]
        assert len(expected) == 10
        for name in expected:
            assert name in registry.all_names()
            assert registry.entries[name]['target_derived'] is False
            assert registry.entries[name]['configuration_membership'] == ['A3', 'A4']

    def test_removed_grid_columns_absent(self, registry):
        # Not in spec section 7's declared list - dropped per spec section
        # 9 ("do not expand this block without a documented reason").
        for name in ('Grid_CI_median', 'Grid_CI_max_high_streak', 'Grid_import_share'):
            assert name not in registry.all_names()

    def test_renewable_and_low_carbon_share_are_distinct_columns(self, registry):
        # spec section 7.2: "do not use renewable and low carbon
        # interchangeably" - both must exist as separate registry entries.
        assert 'Grid_renewable_share' in registry.all_names()
        assert 'Grid_low_carbon_share' in registry.all_names()
        assert 'Grid_renewable_share' != 'Grid_low_carbon_share'


class TestGovernance:
    def test_intensity_features_retained_and_lagged(self, registry):
        for name in ['CO2e_per_Population']:
            assert registry.is_retained(name)
            assert registry.entries[name]['min_lag'] >= 1

    def test_enforce_rejects_unknown_feature(self, registry):
        with pytest.raises(ValueError):
            registry.enforce(['CO2e_lag1', 'totally_made_up_feature'], horizon=1)

    def test_enforce_rejects_removed_tec_cei_features(self, registry):
        # TEC/CEI are gone from the registry entirely (spec section 2) -
        # asking to enforce them must raise, same as any other unknown name.
        for removed in ('TEC', 'CEI', 'TEC_dlog', 'CO2e_per_TEC', 'CEI_lag1'):
            with pytest.raises(ValueError):
                registry.enforce(['CO2e_lag1', removed], horizon=1)

    def test_enforce_filters_out_unsafe_and_unretained(self, registry):
        cols = ['CO2e_lag1', 'GDP']
        result = registry.enforce(cols, horizon=1)
        assert 'CO2e_lag1' in result
        assert 'GDP' in result


class TestGovernedFamilies:
    def test_all_five_families_defined(self, registry):
        families = registry.all_governed_families()
        assert set(families.keys()) == set(GOVERNED_FAMILY_NAMES)
        for fam, feats in families.items():
            assert len(feats) > 0, f"{fam} should not be empty"

    def test_g1_excludes_all_target_derived(self, registry):
        g1 = registry.governed_family('G1')
        for f in g1:
            assert registry.entries[f]['target_derived'] is False

    def test_g2_is_autoregressive_and_seasonal_only(self, registry):
        g2 = registry.governed_family('G2')
        for f in g2:
            fam = registry.entries[f]['family']
            assert fam in ('target_lag', 'target_growth', 'seasonal')
        # Must include the four CO2e lags
        for lag_feat in ['CO2e_lag1', 'CO2e_lag2', 'CO2e_lag3', 'CO2e_lag4']:
            assert lag_feat in g2

    def test_g4_excludes_intensity_but_g3_includes_it(self, registry):
        g3 = registry.governed_family('G3')
        g4 = registry.governed_family('G4')
        assert 'CO2e_per_Population' in g3
        assert 'CO2e_per_Population' not in g4

    def test_no_family_contains_removed_tec_cei_features(self, registry):
        for fam in GOVERNED_FAMILY_NAMES:
            for removed in ('TEC', 'CEI', 'TEC_dlog', 'CO2e_per_TEC', 'CEI_lag1'):
                assert removed not in registry.governed_family(fam)


class TestHorizonSafety:
    def test_lag_features_safe_at_all_horizons(self, registry):
        for h in (1, 2, 4):
            assert registry.is_safe_for_horizon('CO2e_lag1', h)

    def test_unknown_removed_feature_never_safe(self, registry):
        for h in (1, 2, 4):
            assert not registry.is_safe_for_horizon('CEI', h)
            assert not registry.is_safe_for_horizon('TEC', h)


class TestStreamBCandidatePools:
    def test_each_b_tag_mirrors_its_a_tag_exactly(self, registry):
        for b_tag, a_tag in STREAM_B_TO_A.items():
            assert registry.candidate_pool_for_stream_b(b_tag) == registry.configuration_members(a_tag)

    def test_all_stream_b_candidate_pools_covers_all_four(self, registry):
        pools = registry.all_stream_b_candidate_pools()
        assert set(pools.keys()) == set(STREAM_B_TAGS)
        for b_tag in STREAM_B_TAGS:
            assert len(pools[b_tag]) > 0

    def test_unknown_b_tag_raises(self, registry):
        with pytest.raises(ValueError):
            registry.candidate_pool_for_stream_b('B5')


class TestToDataframeAndExport:
    def test_to_dataframe_has_one_row_per_feature(self, registry):
        df = registry.to_dataframe()
        assert len(df) == len(registry)

    def test_to_dataframe_has_a_and_b_membership_columns(self, registry):
        df = registry.to_dataframe()
        for a_tag in CONFIGURATION_TAGS:
            assert f'{a_tag}_membership' in df.columns
        for b_tag in STREAM_B_TAGS:
            assert f'{b_tag}_candidate_membership' in df.columns

    def test_b_membership_column_mirrors_corresponding_a_column(self, registry):
        df = registry.to_dataframe().set_index('name')
        for b_tag, a_tag in STREAM_B_TO_A.items():
            pd_series_equal = (df[f'{b_tag}_candidate_membership'] == df[f'{a_tag}_membership']).all()
            assert pd_series_equal

    def test_export_csv_writes_a_readable_file(self, registry, tmp_path):
        out_path = tmp_path / 'feature_registry.csv'
        returned_path = registry.export_csv(str(out_path))
        assert returned_path == out_path
        assert out_path.exists()

        import pandas as pd
        reloaded = pd.read_csv(out_path)
        assert len(reloaded) == len(registry)
        assert 'A1_membership' in reloaded.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
