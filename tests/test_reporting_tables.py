"""
Tests for src/reporting/tables.py (spec section 20): Tables 1-4 and 12,
built from saved outputs (or, for Table 12, a hand-maintained changelog).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from src.features.registry import FeatureRegistry
from src.reporting.tables import (
    build_table1_raw_variable_inventory,
    build_table2_feature_registry,
    build_table3_configuration_definitions,
    build_table4_grid_aggregation_summary,
    build_table5_configuration_level_performance,
    build_table7_fs_membership,
    build_table_fs_performance,
    build_table12_bug_fix_verification,
    build_table3_ab_configuration_definitions,
    build_table6_global_factor_summary,
    build_table8_fs_outputs,
    build_table9_covid_regime_interpretation,
    build_table10_source_data_quality,
)

REGISTRY_PATH = Path(__file__).parent.parent / "config" / "feature_registry.yaml"


@pytest.fixture(scope="module")
def registry():
    return FeatureRegistry.load(str(REGISTRY_PATH))


@pytest.fixture
def fake_quality_report():
    return {
        'summary': {'date_range': {'start': '1999-01-01', 'end': '2025-01-01'}},
        'missing_values': [
            {'column': 'CO2e', 'missing_count': 0},
            {'column': 'GDP', 'missing_count': 0},
            {'column': 'Rainfall', 'missing_count': 1},
        ],
    }


class TestTable1RawVariableInventory:
    def test_includes_target_and_all_five_raw_predictors(self, registry, fake_quality_report):
        table = build_table1_raw_variable_inventory(registry, fake_quality_report)
        variables = set(table['variable'])
        assert 'CO2e' in variables
        for name in ['GDP', 'Population', 'Air_Temp', 'Rainfall', 'COVID_Deaths']:
            assert name in variables

    def test_tec_and_cei_absent(self, registry, fake_quality_report):
        table = build_table1_raw_variable_inventory(registry, fake_quality_report)
        variables = set(table['variable'])
        assert 'TEC' not in variables
        assert 'CEI' not in variables

    def test_excludes_grid_features(self, registry, fake_quality_report):
        table = build_table1_raw_variable_inventory(registry, fake_quality_report)
        assert not any(v.startswith('Grid_') for v in table['variable'])

    def test_target_row_is_flagged_correctly(self, registry, fake_quality_report):
        table = build_table1_raw_variable_inventory(registry, fake_quality_report)
        target_row = table[table['variable'] == 'CO2e'].iloc[0]
        assert target_row['target_or_predictor'] == 'target'
        assert target_row['target_derived'] == False

    def test_missing_count_pulled_from_quality_report(self, registry, fake_quality_report):
        table = build_table1_raw_variable_inventory(registry, fake_quality_report)
        rainfall_row = table[table['variable'] == 'Rainfall'].iloc[0]
        assert rainfall_row['missing_count'] == 1
        assert rainfall_row['target_derived'] == False


class TestTable2FeatureRegistry:
    def test_matches_registry_to_dataframe(self, registry):
        table = build_table2_feature_registry(registry)
        assert len(table) == len(registry)
        assert 'configuration_membership' in table.columns


class TestTable3ConfigurationDefinitions:
    def test_one_row_per_configuration(self):
        manifest = {
            'A1': {'n_raw': 6, 'n_engineered': 1, 'n_grid': 0, 'nominal_max': 7, 'n_features_total': 7},
            'A2': {'n_raw': 6, 'n_engineered': 17, 'n_grid': 0, 'nominal_max': 23, 'n_features_total': 23},
            'A3': {'n_raw': 6, 'n_engineered': 1, 'n_grid': 12, 'nominal_max': None, 'n_features_total': 19},
            'A4': {'n_raw': 6, 'n_engineered': 17, 'n_grid': 12, 'nominal_max': None, 'n_features_total': 35},
        }
        table = build_table3_configuration_definitions(manifest)
        assert len(table) == 4
        assert set(table['configuration']) == {'A1', 'A2', 'A3', 'A4'}
        a4_row = table[table['configuration'] == 'A4'].iloc[0]
        assert a4_row['audited_total'] == 35


class TestTable5ConfigurationLevelPerformance:
    def _fake_stage1_metrics(self):
        return pd.DataFrame([{
            'configuration': 'A1', 'panel': 'panel1_full_period', 'model': 'ridge',
            'mase_h1': 1.0, 'mase_h2': 1.2, 'mase_h4': 1.5,
            'rmse_h1': 10.0, 'rmse_h2': 12.0, 'rmse_h4': 15.0,
            'smape_h1': 5.0, 'smape_h2': 6.0, 'smape_h4': 7.0,
            'r2_h1': 0.9, 'r2_h2': 0.8, 'r2_h4': 0.7,
            'weighted_mase': 1.16, 'weighted_mae': 100.0,
            'stability_score': 0.7, 'total_runtime_seconds': 5.0, 'n_features': 7,
        }])

    def test_produces_one_row_with_weighted_columns(self):
        horizon_weights = {1: 0.5, 2: 0.3, 4: 0.2}
        table = build_table5_configuration_level_performance(self._fake_stage1_metrics(), horizon_weights)
        assert len(table) == 1
        row = table.iloc[0]
        expected_rmse = 0.5 * 10.0 + 0.3 * 12.0 + 0.2 * 15.0
        assert row['weighted_rmse'] == pytest.approx(expected_rmse)
        assert row['h1_mase'] == 1.0

    def test_empty_input_returns_empty(self):
        table = build_table5_configuration_level_performance(pd.DataFrame(), {1: 0.5, 2: 0.3, 4: 0.2})
        assert table.empty


class TestTableFsPerformance:
    def _fake_stage2_metrics(self):
        return pd.DataFrame([
            {'configuration': 'A2', 'panel': 'panel1_full_period', 'fs_option': 'fs_linear', 'model': 'ridge',
             'n_features': 7, 'weighted_mase': 1.1, 'weighted_mae': 90.0, 'worst_horizon_mase': 1.3,
             'stability_score': 0.8, 'total_runtime_seconds': 3.0},
            {'configuration': 'A2', 'panel': 'panel1_full_period', 'fs_option': 'all_features', 'model': 'ridge',
             'n_features': 23, 'weighted_mase': 1.5, 'weighted_mae': 110.0, 'worst_horizon_mase': 1.8,
             'stability_score': 0.6, 'total_runtime_seconds': 4.0},
            {'configuration': 'A4', 'panel': 'panel2_common_period', 'fs_option': 'fs_consensus', 'model': 'lstm',
             'n_features': 10, 'weighted_mase': 0.9, 'weighted_mae': 70.0, 'worst_horizon_mase': 1.1,
             'stability_score': 0.9, 'total_runtime_seconds': 50.0},
        ])

    def test_filters_to_configuration_and_excludes_all_features(self):
        table = build_table_fs_performance(self._fake_stage2_metrics(), 'A2')
        assert len(table) == 1
        assert table.iloc[0]['fs_option'] == 'fs_linear'

    def test_a4_configuration_returns_its_own_rows(self):
        table = build_table_fs_performance(self._fake_stage2_metrics(), 'A4')
        assert len(table) == 1
        assert table.iloc[0]['model'] == 'lstm'

    def test_missing_configuration_returns_empty(self):
        table = build_table_fs_performance(self._fake_stage2_metrics(), 'A3')
        assert table.empty

    def test_completely_empty_input_does_not_crash(self):
        """Regression: scripts/13 crashed with KeyError('configuration') when
        stage2_metrics.csv did not exist yet (Stage 2 still running) -
        pd.read_csv-on-missing-file fallback is a columnless empty
        DataFrame, and indexing a columnless frame by column name raises
        KeyError instead of returning an empty result."""
        table = build_table_fs_performance(pd.DataFrame(), 'A2')
        assert table.empty


class TestTable4GridAggregationSummary:
    def test_one_row_per_grid_feature(self, registry):
        quality_df = pd.DataFrame([
            {'quarter': '2018-01-01', 'inclusion_status': 'excluded_incomplete', 'min_completeness_threshold': 0.95},
            {'quarter': '2018-04-01', 'inclusion_status': 'included', 'min_completeness_threshold': 0.95},
            {'quarter': '2018-07-01', 'inclusion_status': 'included', 'min_completeness_threshold': 0.95},
        ])
        table = build_table4_grid_aggregation_summary(registry, quality_df)
        assert len(table) == 10  # 10 NESO grid features (spec section 7; +2 OWID = spec's 12-item block)
        assert (table['missing_count_quarters'] == 1).all()
        assert (table['completeness_threshold'] == 0.95).all()

    def test_empty_quality_report_handled(self, registry):
        quality_df = pd.DataFrame(columns=['quarter', 'inclusion_status', 'min_completeness_threshold'])
        table = build_table4_grid_aggregation_summary(registry, quality_df)
        assert len(table) == 10
        assert table['first_available_quarter'].isna().all()


class TestTable7FsMembership:
    def _fake_fs_results(self):
        return {
            'fs_linear': {'selected_features': ['GDP', 'CO2e_lag1']},
            'fs_wrapper': {'selected_features': ['GDP', 'Population']},
            'fs_xgboost_shap': {'selected_features': ['GDP', 'CO2e_lag1', 'TEC']},
            'fs_permutation_stability': {'selected_features': ['GDP']},
            'fs_consensus': {'selected_features': ['GDP', 'CO2e_lag1']},
        }

    def test_gdp_selected_by_all_four_primary_methods(self):
        table = build_table7_fs_membership(self._fake_fs_results(), ['GDP', 'Population', 'CO2e_lag1', 'TEC'])
        gdp_row = table[table['feature'] == 'GDP'].iloc[0]
        assert gdp_row['vote_count'] == 4
        assert gdp_row['selection_frequency'] == pytest.approx(1.0)
        assert gdp_row['fs_consensus'] == 1

    def test_never_selected_feature_has_zero_votes(self):
        table = build_table7_fs_membership(
            self._fake_fs_results(), ['GDP', 'Population', 'CO2e_lag1', 'TEC', 'Rainfall']
        )
        rainfall_row = table[table['feature'] == 'Rainfall'].iloc[0]
        assert rainfall_row['vote_count'] == 0
        assert rainfall_row['fs_consensus'] == 0

    def test_sorted_by_vote_count_descending(self):
        table = build_table7_fs_membership(self._fake_fs_results(), ['GDP', 'Population', 'CO2e_lag1', 'TEC'])
        assert list(table['vote_count']) == sorted(table['vote_count'], reverse=True)


class TestTable12BugFixVerification:
    def test_all_rows_have_required_columns(self):
        table = build_table12_bug_fix_verification()
        required = {'issue', 'previous_behaviour', 'corrected_behaviour', 'validation_test', 'status'}
        assert required <= set(table.columns)
        assert len(table) >= 10  # at least the mandatory spec 3.x bugs + this session's finds
        assert (table['status'] == 'fixed').all()

    def test_referenced_test_files_actually_exist(self):
        """Every validation_test that names a concrete test file must point
        somewhere real - a broken pointer here would undermine the whole
        table's credibility."""
        table = build_table12_bug_fix_verification()
        tests_dir = Path(__file__).parent
        for validation_test in table['validation_test']:
            if not validation_test.startswith('tests/'):
                continue  # narrative entries (e.g. "verified via smoke test") are fine
            file_part = validation_test.split('::')[0].split(' ')[0]
            test_path = tests_dir.parent / file_part
            assert test_path.exists(), f"referenced test file does not exist: {test_path}"


class TestTable3AbConfigurationDefinitions:
    def _manifest(self):
        return {
            'A1': {'nominal_max': 11, 'n_features_total': 11},
            'A2': {'nominal_max': 25, 'n_features_total': 25},
            'A3': {'nominal_max': 23, 'n_features_total': 23},
            'A4': {'nominal_max': 37, 'n_features_total': 37},
        }

    def test_eight_rows_a1_a4_b1_b4(self, registry):
        table = build_table3_ab_configuration_definitions(self._manifest(), registry)
        assert len(table) == 8
        assert set(table['row']) == {'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4'}

    def test_b_rows_mirror_a_rows_counts(self, registry):
        table = build_table3_ab_configuration_definitions(self._manifest(), registry)
        a2 = table[table['row'] == 'A2'].iloc[0]
        b2 = table[table['row'] == 'B2'].iloc[0]
        assert a2['audited_count'] == b2['audited_count']
        assert a2['fs_applied'] == 'none'
        assert b2['fs_applied'] == 'FS1-FS5'

    def test_stream_column_distinguishes_a_from_b(self, registry):
        table = build_table3_ab_configuration_definitions(self._manifest(), registry)
        assert set(table[table['row'].str.startswith('A')]['stream']) == {'A'}
        assert set(table[table['row'].str.startswith('B')]['stream']) == {'B'}


class TestTable6GlobalFactorSummary:
    def _fake_ranked(self, stream_prefix, configs, fs_options, models, n_per=1):
        rows = []
        for c in configs:
            for fs in fs_options:
                for m in models:
                    rows.append({
                        'configuration': c, 'fs_option': fs, 'model': m,
                        'weighted_mase': 1.0, 'vikor_rank': 1, 'total_runtime_seconds': 2.0,
                        'n_features': 10,
                    })
        return pd.DataFrame(rows)

    def test_row_set_covers_all_factor_levels(self):
        stream_a = self._fake_ranked('A', ['A1', 'A2', 'A3', 'A4'], ['all_features'], ['ridge'])
        stream_b = self._fake_ranked(
            'B', ['A1', 'A2', 'A3', 'A4'],
            ['fs_linear', 'fs_wrapper', 'fs_xgboost_shap', 'fs_permutation_stability', 'fs_consensus'],
            ['ridge']
        )
        table = build_table6_global_factor_summary(stream_a, stream_b, models=['ridge'])
        expected_rows = {'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4',
                          'fs_linear', 'fs_wrapper', 'fs_xgboost_shap', 'fs_permutation_stability',
                          'fs_consensus', 'ridge'}
        assert expected_rows.issubset(set(table['row']))

    def test_complete_grid_has_zero_failure_rate(self):
        stream_a = self._fake_ranked('A', ['A1', 'A2', 'A3', 'A4'], ['all_features'], ['ridge'])
        stream_b = self._fake_ranked(
            'B', ['A1', 'A2', 'A3', 'A4'],
            ['fs_linear', 'fs_wrapper', 'fs_xgboost_shap', 'fs_permutation_stability', 'fs_consensus'],
            ['ridge']
        )
        table = build_table6_global_factor_summary(stream_a, stream_b, models=['ridge'])
        a1_row = table[table['row'] == 'A1'].iloc[0]
        assert a1_row['failure_rate'] == pytest.approx(0.0)

    def test_empty_streams_return_all_missing_rows(self):
        table = build_table6_global_factor_summary(pd.DataFrame(), pd.DataFrame(), models=['ridge'])
        assert (table['n_candidates'] == 0).all()
        assert (table['failure_rate'] == 1.0).all()


class TestTable8FsOutputs:
    def _fs_results(self):
        return {
            'fs_linear': {'selected_features': ['GDP', 'Population']},
            'fs_wrapper': {'selected_features': ['GDP']},
            'fs_xgboost_shap': {'selected_features': ['GDP', 'Population', 'Air_Temp']},
            'fs_permutation_stability': {'selected_features': ['GDP']},
            'fs_consensus': {'selected_features': ['GDP']},
        }

    def test_one_row_per_config_per_feature(self):
        fs_by_config = {'B1': self._fs_results()}
        pool = {'B1': ['GDP', 'Population', 'Air_Temp', 'Rainfall']}
        table = build_table8_fs_outputs(fs_by_config, pool)
        assert len(table) == 4
        assert set(table['feature']) == set(pool['B1'])

    def test_vote_count_matches_fs1_to_fs4_membership(self):
        fs_by_config = {'B1': self._fs_results()}
        pool = {'B1': ['GDP', 'Population']}
        table = build_table8_fs_outputs(fs_by_config, pool)
        gdp_row = table[table['feature'] == 'GDP'].iloc[0]
        assert gdp_row['vote_count'] == 4  # selected by all 4 primary FS methods
        pop_row = table[table['feature'] == 'Population'].iloc[0]
        assert pop_row['vote_count'] == 2  # fs_linear, fs_xgboost_shap only

    def test_multiple_b_configs_kept_separate(self):
        fs_by_config = {'B1': self._fs_results(), 'B2': self._fs_results()}
        pool = {'B1': ['GDP'], 'B2': ['GDP']}
        table = build_table8_fs_outputs(fs_by_config, pool)
        assert set(table['b_configuration']) == {'B1', 'B2'}

    def test_empty_input_returns_empty(self):
        table = build_table8_fs_outputs({}, {})
        assert table.empty


class TestTable9CovidRegimeInterpretation:
    def test_one_row_per_winner_per_regime(self):
        regime_df = pd.DataFrame({
            'feature': ['CO2e_lag1', 'GDP', 'Mobility_Workplaces'],
            'importance': [5.0, 2.0, 1.0],
            'regime': ['covid', 'covid', 'covid'],
            'n_samples': [8, 8, 8],
        })
        table = build_table9_covid_regime_interpretation({'Best_A': regime_df})
        assert len(table) == 1
        assert table.iloc[0]['winner'] == 'Best_A'
        assert table.iloc[0]['n_samples'] == 8

    def test_top_predictors_ranked_by_absolute_importance(self):
        regime_df = pd.DataFrame({
            'feature': ['low_imp', 'high_imp'],
            'importance': [0.1, 9.0],
            'regime': ['pre_covid', 'pre_covid'],
            'n_samples': [10, 10],
        })
        table = build_table9_covid_regime_interpretation({'Best_B': regime_df}, top_k=2)
        assert table.iloc[0]['top_predictors'].startswith('high_imp')

    def test_missing_winner_data_reported_not_dropped(self):
        table = build_table9_covid_regime_interpretation({'Best_Overall': pd.DataFrame()})
        assert len(table) == 1
        assert table.iloc[0]['winner'] == 'Best_Overall'
        assert table.iloc[0]['n_samples'] == 0


class TestTable10SourceDataQuality:
    def test_includes_mobility_row_when_metadata_given(self):
        table = build_table10_source_data_quality(
            mobility_metadata={'retrieval_date': '2026-01-01', 'checksum': 'abc123',
                                'first_observation': '2020-02-15', 'last_observation': '2022-10-15',
                                'n_daily_records': 974}
        )
        assert len(table) == 1
        assert table.iloc[0]['source'] == 'Google COVID-19 mobility'
        assert table.iloc[0]['n_records'] == 974

    def test_includes_grid_row_when_quality_report_given(self):
        quality_df = pd.DataFrame({
            'quarter': ['2018Q1', '2018Q2', '2018Q3'],
            'inclusion_status': ['included', 'included', 'excluded_incomplete'],
        })
        table = build_table10_source_data_quality(grid_quality_report=quality_df)
        row = table[table['source'] == 'NESO GB Carbon Intensity API'].iloc[0]
        assert row['n_records'] == 2

    def test_no_sources_given_returns_empty(self):
        table = build_table10_source_data_quality()
        assert table.empty

    def test_owid_rows_included_when_metadata_given(self):
        table = build_table10_source_data_quality(
            owid_renewable_metadata={'retrieved_date': '2026-01-01'},
            owid_low_carbon_metadata={'retrieved_date': '2026-01-01'},
        )
        assert len(table) == 2
        assert 'OWID renewable electricity share' in table['source'].values


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
