"""
Tests for Phase 6 additions: target-derived feature exclusion (spec section
19.5), cross-model importance comparison (Table 11 / spec 19.3), and
regime-stability summarization (spec 19.4).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.features.registry import FeatureRegistry
from src.features.configurations import exclude_target_derived_features, NAMED_TARGET_DERIVED_FEATURES
from src.interpretability.cross_model_importance import (
    rank_importance, build_cross_model_importance_table, compute_regime_stability,
    compute_family_contribution,
)

REGISTRY_PATH = Path(__file__).parent.parent / "config" / "feature_registry.yaml"


@pytest.fixture(scope="module")
def registry():
    return FeatureRegistry.load(str(REGISTRY_PATH))


def _fake_A2_matrix(n=20, seed=0):
    dates = pd.date_range('2015-01-01', periods=n, freq='QS')
    rng = np.random.RandomState(seed)
    cols = [
        'GDP', 'Population', 'Air_Temp', 'Rainfall', 'COVID_Deaths',
        'CO2e_lag1', 'CO2e_lag2', 'CO2e_lag3', 'CO2e_lag4',
        'CO2e_dlog', 'GDP_dlog',
        'CO2e_per_Population',
        'HDD_proxy', 'CDD_proxy', 'Q2', 'Q3', 'Q4', 'COVID', 'EnergyCrisis',
    ]
    return pd.DataFrame({c: rng.randn(n) for c in cols}, index=dates)


class TestExcludeTargetDerivedFeatures:
    def test_named_mode_drops_exactly_the_named_set(self, registry):
        X = _fake_A2_matrix()
        reduced = exclude_target_derived_features(X, registry, mode='named')
        for col in NAMED_TARGET_DERIVED_FEATURES:
            assert col not in reduced.columns
        assert len(reduced.columns) == len(X.columns) - len(NAMED_TARGET_DERIVED_FEATURES)
        # CO2e_lag1-4 are historical target-derived too but NOT in the
        # "named" set - must survive this narrower mode.
        assert 'CO2e_lag1' in reduced.columns

    def test_all_mode_drops_every_target_derived_column(self, registry):
        X = _fake_A2_matrix()
        reduced = exclude_target_derived_features(X, registry, mode='all')
        for col in reduced.columns:
            assert registry.entries[col]['target_derived'] is False
        # CO2e_lag1-4 ARE target-derived -> must be dropped in 'all' mode.
        assert 'CO2e_lag1' not in reduced.columns
        assert 'CO2e_dlog' not in reduced.columns

    def test_unknown_mode_raises(self, registry):
        X = _fake_A2_matrix()
        with pytest.raises(ValueError):
            exclude_target_derived_features(X, registry, mode='bogus')

    def test_missing_columns_silently_ignored(self, registry):
        # A1 has no intensity ratios/CO2e_dlog at all.
        X = _fake_A2_matrix()[['GDP', 'Population', 'CO2e_dlog']]
        reduced = exclude_target_derived_features(X, registry, mode='named')
        assert 'CO2e_dlog' not in reduced.columns
        assert set(reduced.columns) == {'GDP', 'Population'}


class TestRankImportance:
    def test_ranks_by_absolute_value_descending(self):
        df = pd.DataFrame({'feature': ['a', 'b', 'c'], 'importance': [-5.0, 1.0, 3.0]})
        ranks = rank_importance(df)
        assert ranks['a'] == 1  # |-5| is largest
        assert ranks['c'] == 2
        assert ranks['b'] == 3


class TestCrossModelImportanceTable:
    def test_merges_ranks_across_models(self):
        ridge_imp = pd.DataFrame({'feature': ['f1', 'f2', 'f3'], 'importance': [3.0, 1.0, 2.0]})
        rf_imp = pd.DataFrame({'feature': ['f1', 'f2', 'f3'], 'importance': [0.5, 0.2, 0.3]})
        table = build_cross_model_importance_table({'ridge': ridge_imp, 'random_forest': rf_imp})

        assert 'ridge_rank' in table.columns
        assert 'random_forest_rank' in table.columns
        assert 'average_rank' in table.columns
        f1_row = table[table['feature'] == 'f1'].iloc[0]
        assert f1_row['ridge_rank'] == 1
        assert f1_row['random_forest_rank'] == 1
        assert f1_row['average_rank'] == pytest.approx(1.0)

    def test_feature_missing_from_one_model_gets_nan_not_zero(self):
        ridge_imp = pd.DataFrame({'feature': ['f1', 'f2'], 'importance': [2.0, 1.0]})
        rf_imp = pd.DataFrame({'feature': ['f1'], 'importance': [0.5]})  # f2 not scored by RF
        table = build_cross_model_importance_table({'ridge': ridge_imp, 'random_forest': rf_imp})

        f2_row = table[table['feature'] == 'f2'].iloc[0]
        assert pd.isna(f2_row['random_forest_rank'])
        assert f2_row['n_models_scored'] == 1

    def test_empty_input_returns_empty(self):
        table = build_cross_model_importance_table({})
        assert table.empty

    def test_all_empty_dataframes_returns_empty(self):
        table = build_cross_model_importance_table({'ridge': pd.DataFrame()})
        assert table.empty


class TestRegimeStability:
    def test_consistently_top_feature_gets_stability_one(self):
        regime_importance = {
            'pre_covid': pd.DataFrame({'feature': ['f1', 'f2', 'f3'], 'importance': [3.0, 1.0, 0.5]}),
            'covid': pd.DataFrame({'feature': ['f1', 'f2', 'f3'], 'importance': [2.5, 0.8, 0.4]}),
            'post_covid': pd.DataFrame({'feature': ['f1', 'f2', 'f3'], 'importance': [2.8, 0.9, 0.3]}),
        }
        result = compute_regime_stability(regime_importance, top_k=1)
        f1_row = result[result['feature'] == 'f1'].iloc[0]
        assert f1_row['regime_stability'] == pytest.approx(1.0)

    def test_never_top_feature_gets_stability_zero(self):
        regime_importance = {
            'pre_covid': pd.DataFrame({'feature': ['f1', 'f2'], 'importance': [3.0, 0.1]}),
            'covid': pd.DataFrame({'feature': ['f1', 'f2'], 'importance': [2.5, 0.1]}),
        }
        result = compute_regime_stability(regime_importance, top_k=1)
        f2_row = result[result['feature'] == 'f2'].iloc[0]
        assert f2_row['regime_stability'] == pytest.approx(0.0)

    def test_empty_input_returns_empty(self):
        result = compute_regime_stability({})
        assert result.empty


class TestFamilyContribution:
    """Spec section 23: aggregate |importance| by registry family
    (mobility_exogenous/grid_exogenous/owid_exogenous vs. other)."""

    def _family_map(self):
        return pd.Series({
            'Mobility_Retail_Recreation': 'mobility_exogenous',
            'Mobility_Workplaces': 'mobility_exogenous',
            'Grid_CI_mean': 'grid_exogenous',
            'OWID_RenewableShare_L1Y': 'owid_exogenous',
            'GDP': 'raw_exogenous',
            'CO2e_lag1': 'target_lag',
        })

    def test_shares_sum_to_one(self):
        importance_df = pd.DataFrame({
            'feature': ['Mobility_Retail_Recreation', 'Grid_CI_mean', 'GDP', 'CO2e_lag1'],
            'importance': [4.0, 3.0, 2.0, 1.0],
        })
        result = compute_family_contribution(importance_df, self._family_map())
        assert result['share_of_total'].sum() == pytest.approx(1.0)

    def test_mobility_features_aggregate_together(self):
        importance_df = pd.DataFrame({
            'feature': ['Mobility_Retail_Recreation', 'Mobility_Workplaces', 'GDP'],
            'importance': [2.0, 3.0, 5.0],
        })
        result = compute_family_contribution(importance_df, self._family_map())
        mobility_row = result[result['family'] == 'mobility_exogenous'].iloc[0]
        assert mobility_row['sum'] == pytest.approx(5.0)
        assert mobility_row['count'] == 2
        assert mobility_row['share_of_total'] == pytest.approx(0.5)

    def test_unmapped_feature_falls_into_other_not_dropped(self):
        importance_df = pd.DataFrame({
            'feature': ['Some_Unregistered_Feature', 'GDP'],
            'importance': [1.0, 1.0],
        })
        result = compute_family_contribution(importance_df, self._family_map())
        assert 'other' in result['family'].values
        assert result['sum'].sum() == pytest.approx(2.0)

    def test_empty_input_returns_empty(self):
        result = compute_family_contribution(pd.DataFrame(columns=['feature', 'importance']), self._family_map())
        assert result.empty

    def test_dominant_family_sorted_first(self):
        importance_df = pd.DataFrame({
            'feature': ['GDP', 'Grid_CI_mean', 'OWID_RenewableShare_L1Y'],
            'importance': [1.0, 8.0, 1.0],
        })
        result = compute_family_contribution(importance_df, self._family_map())
        assert result.iloc[0]['family'] == 'grid_exogenous'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
