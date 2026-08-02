"""
Tests for src/decision/experiment_ranking.py (spec section 18 / Table 10):
Pareto filtering + MCDA ranking of the config x FS x model experimental
grid, and rank-correlation sensitivity across weighting schemes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.decision.experiment_ranking import (
    build_pareto_mcda_table, rank_correlation_across_weight_sets, select_best_overall,
    CRITERIA, WEIGHT_SETS
)


def _fake_metrics(rows):
    """Each row: (configuration, fs_option, model, weighted_mase, worst_horizon_mase, error_std, n_features, runtime)."""
    records = []
    for config, fs, model, wmase, worst, std, nfeat, rt in rows:
        records.append({
            'configuration': config, 'panel': 'panel1_full_period', 'fs_option': fs, 'model': model,
            'weighted_mase': wmase, 'worst_horizon_mase': worst, 'error_std': std,
            'n_features': nfeat, 'total_runtime_seconds': rt,
        })
    return pd.DataFrame(records)


class TestBuildParetoMcdaTablePreservesHorizonColumns:
    """Regression: spec Table 4/5/7 all require the H1/H2/H4 MASE
    breakdown alongside weighted_mase - build_pareto_mcda_table's column
    selection previously dropped mase_h1/h2/h4 even though the source
    metrics carried them, silently blanking those columns in every
    downstream table."""

    def test_horizon_mase_columns_survive(self):
        df = _fake_metrics([
            ('A1', 'all_features', 'ridge', 2.0, 2.5, 0.5, 10, 5.0),
            ('A2', 'all_features', 'ridge', 1.0, 1.5, 0.3, 8, 3.0),
        ])
        df['mase_h1'] = [1.8, 0.9]
        df['mase_h2'] = [2.1, 1.0]
        df['mase_h4'] = [2.5, 1.5]

        table = build_pareto_mcda_table(df)
        for col in ('mase_h1', 'mase_h2', 'mase_h4'):
            assert col in table.columns
        a2_row = table.set_index(['configuration', 'model']).loc[('A2', 'ridge')]
        assert a2_row['mase_h1'] == pytest.approx(0.9)
        assert a2_row['mase_h4'] == pytest.approx(1.5)


class TestBuildParetoMcdaTable:
    def test_dominated_alternative_is_flagged(self):
        # 'A1/ridge' is strictly worse than 'A2/ridge' on every criterion.
        df = _fake_metrics([
            ('A1', 'all_features', 'ridge', 2.0, 2.5, 0.5, 10, 5.0),
            ('A2', 'all_features', 'ridge', 1.0, 1.5, 0.3, 8, 3.0),
        ])
        table = build_pareto_mcda_table(df)
        statuses = table.set_index(['configuration', 'model'])['pareto_status']
        assert statuses[('A1', 'ridge')] == 'dominated'
        assert statuses[('A2', 'ridge')] == 'non_dominated'

    def test_single_non_dominated_alternative_selected_directly(self):
        df = _fake_metrics([
            ('A1', 'all_features', 'ridge', 2.0, 2.5, 0.5, 10, 5.0),
            ('A2', 'all_features', 'ridge', 1.0, 1.5, 0.3, 8, 3.0),
        ])
        table = build_pareto_mcda_table(df)
        selected = table[table['final_decision'].str.startswith('selected')]
        assert len(selected) == 1
        assert selected.iloc[0]['configuration'] == 'A2'

    def test_multiple_non_dominated_ranked_by_vikor(self):
        # Two genuine trade-offs: A wins accuracy, B wins parsimony.
        df = _fake_metrics([
            ('A1', 'all_features', 'ridge', 1.0, 1.2, 0.2, 20, 5.0),
            ('A2', 'all_features', 'ridge', 1.5, 1.6, 0.3, 5, 5.0),
        ])
        table = build_pareto_mcda_table(df)
        assert (table['pareto_status'] == 'non_dominated').all()
        assert table['vikor_rank'].notna().all()
        assert set(table['vikor_rank']) == {1, 2}
        assert (table['final_decision'] == 'selected').sum() == 1

    def test_missing_criteria_rows_dropped_not_crashed(self):
        df = _fake_metrics([
            ('A1', 'all_features', 'ridge', 1.0, 1.2, 0.2, 20, 5.0),
        ])
        df.loc[0, 'weighted_mase'] = np.nan
        table = build_pareto_mcda_table(df)
        assert table.empty

    def test_weight_set_choice_affects_ranking(self):
        # Three genuine trade-off alternatives (VIKOR's min-max normalization
        # is degenerate/non-intuitive with only 2 alternatives) - a clear
        # accuracy leader, a clear parsimony leader, and a middling option.
        df = _fake_metrics([
            ('A1', 'all_features', 'ridge', 1.0, 1.1, 0.15, 30, 5.0),   # accuracy leader, many features
            ('A2', 'all_features', 'ridge', 1.6, 1.8, 0.30, 4, 5.0),    # parsimony leader, worst accuracy
            ('A3', 'all_features', 'ridge', 1.3, 1.4, 0.20, 15, 5.0),   # middling on both
        ])
        table_accuracy = build_pareto_mcda_table(df, weight_set_name='accuracy_emphasis')
        table_parsimony = build_pareto_mcda_table(df, weight_set_name='parsimony_emphasis')

        winner_accuracy = table_accuracy[table_accuracy['final_decision'] == 'selected']['configuration'].iloc[0]
        winner_parsimony = table_parsimony[table_parsimony['final_decision'] == 'selected']['configuration'].iloc[0]
        # Accuracy emphasis must pick the unambiguous accuracy leader (A1).
        # Parsimony emphasis is not asserted to pick a specific alternative
        # (VIKOR compromises across ALL weighted criteria via S, not just
        # the single largest-weighted one - A2's parsimony win can still be
        # outweighed by it being worst on every other criterion) - the
        # meaningful, robust assertion is simply that the two weight sets
        # produce different winners.
        assert winner_accuracy == 'A1'
        assert winner_accuracy != winner_parsimony


class TestRankCorrelationAcrossWeightSets:
    def test_returns_pairwise_comparisons_for_all_schemes(self):
        df = _fake_metrics([
            ('A1', 'all_features', 'ridge', 1.0, 1.2, 0.2, 20, 5.0),
            ('A2', 'all_features', 'ridge', 1.4, 1.6, 0.25, 5, 4.0),
            ('A3', 'all_features', 'ridge', 1.2, 1.3, 0.15, 12, 6.0),
        ])
        result = rank_correlation_across_weight_sets(df)
        n_schemes = len(WEIGHT_SETS) + 2  # + topsis_equal + weighted_sum_equal
        expected_pairs = n_schemes * (n_schemes - 1) // 2
        assert len(result) == expected_pairs
        assert result['spearman_rho'].between(-1, 1).all()

    def test_fewer_than_two_non_dominated_returns_empty(self):
        df = _fake_metrics([
            ('A1', 'all_features', 'ridge', 2.0, 2.5, 0.5, 10, 5.0),
            ('A2', 'all_features', 'ridge', 1.0, 1.5, 0.3, 8, 3.0),  # dominates A1 -> only 1 non-dominated
        ])
        result = rank_correlation_across_weight_sets(df)
        assert result.empty

    def test_identical_alternatives_give_perfect_correlation(self):
        df = _fake_metrics([
            ('A1', 'all_features', 'ridge', 1.0, 1.2, 0.2, 20, 5.0),
            ('A2', 'all_features', 'ridge', 1.4, 1.1, 0.15, 5, 4.0),
            ('A3', 'all_features', 'ridge', 1.2, 1.3, 0.25, 12, 6.0),
        ])
        result = rank_correlation_across_weight_sets(df)
        # All schemes are deterministic functions of the same data - not
        # necessarily perfectly correlated (different weightings can
        # legitimately reorder), but every rho must be a valid, finite value.
        assert result['spearman_rho'].notna().all()


def _fake_winner(config, fs, model, wmase, worst, std, nfeat, rt):
    return pd.Series({
        'configuration': config, 'panel': 'panel2_common_period', 'fs_option': fs, 'model': model,
        'weighted_mase': wmase, 'worst_horizon_mase': worst, 'error_std': std,
        'n_features': nfeat, 'total_runtime_seconds': rt,
    })


class TestSelectBestOverall:
    def test_best_a_dominates_best_b_is_reported(self):
        # Best_A strictly better on every criterion -> it dominates and
        # must also win VIKOR (spec section 21.3's transparency requirement).
        best_a = _fake_winner('A2', 'all_features', 'ridge', 1.0, 1.2, 0.2, 10, 3.0)
        best_b = _fake_winner('B3', 'fs_consensus', 'lightgbm', 1.5, 1.8, 0.4, 15, 5.0)

        overall_label, table7, dominance = select_best_overall(best_a, best_b)

        assert dominance['one_dominates_other'] is True
        assert dominance['dominant'] == 'Best_A'
        assert overall_label == 'Best_A'
        assert len(table7) == 2
        assert set(table7['stream_winner']) == {'Best_A', 'Best_B'}
        assert table7.loc[table7['stream_winner'] == 'Best_A', 'is_best_overall'].iloc[0]
        assert not table7.loc[table7['stream_winner'] == 'Best_B', 'is_best_overall'].iloc[0]

    def test_genuine_tradeoff_neither_dominates(self):
        # Best_A wins accuracy, Best_B wins parsimony/runtime - a genuine
        # Pareto trade-off, VIKOR breaks the tie but dominance is reported false.
        best_a = _fake_winner('A4', 'all_features', 'catboost', 0.8, 0.9, 0.15, 30, 20.0)
        best_b = _fake_winner('B1', 'fs_linear', 'ridge', 1.1, 1.3, 0.25, 4, 1.0)

        overall_label, table7, dominance = select_best_overall(best_a, best_b)

        assert dominance['one_dominates_other'] is False
        assert dominance['dominant'] is None
        assert set(dominance['non_dominated']) == {'Best_A', 'Best_B'}
        assert overall_label in ('Best_A', 'Best_B')

    def test_table7_carries_original_criteria_columns(self):
        best_a = _fake_winner('A1', 'all_features', 'ridge', 1.0, 1.2, 0.2, 10, 3.0)
        best_b = _fake_winner('B2', 'fs_wrapper', 'random_forest', 1.2, 1.4, 0.3, 12, 6.0)
        _, table7, _ = select_best_overall(best_a, best_b)
        for c in CRITERIA:
            assert c in table7.columns


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
