"""
Tests for deterministic VIKOR/TOPSIS tie handling (bug 3.6): must never use
`.rank().astype(int)`, must produce unique 1..n ranks, and must break ties
using documented secondary criteria + stable sort rather than arbitrarily.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.decision.mcda import assign_deterministic_rank, vikor, topsis, pareto_filter


class TestAssignDeterministicRank:
    def test_unique_ranks_even_with_exact_ties(self):
        df = pd.DataFrame({'score': [1.0, 1.0, 1.0, 2.0]}, index=['a', 'b', 'c', 'd'])
        ranks = assign_deterministic_rank(df, 'score', ascending=True)
        assert sorted(ranks.tolist()) == [1, 2, 3, 4]
        assert ranks.nunique() == 4

    def test_reproducible_across_calls(self):
        df = pd.DataFrame({'score': [3.0, 1.0, 1.0, 2.0]}, index=['w', 'x', 'y', 'z'])
        r1 = assign_deterministic_rank(df, 'score', ascending=True)
        r2 = assign_deterministic_rank(df, 'score', ascending=True)
        pd.testing.assert_series_equal(r1, r2)

    def test_tie_break_column_used_before_label(self):
        df = pd.DataFrame({
            'score': [1.0, 1.0],
            'parsimony': [5, 10],  # higher is "better" here
        }, index=['b_alt', 'a_alt'])
        ranks = assign_deterministic_rank(
            df, 'score', ascending=True, tie_break_col='parsimony', tie_break_ascending=False
        )
        # 'a_alt' has higher parsimony (10) so should win the tie (rank 1)
        # despite 'b_alt' sorting first alphabetically.
        assert ranks['a_alt'] == 1
        assert ranks['b_alt'] == 2

    def test_ascending_direction_respected(self):
        df = pd.DataFrame({'score': [10.0, 5.0, 1.0]}, index=['a', 'b', 'c'])
        ranks_asc = assign_deterministic_rank(df, 'score', ascending=True)
        assert ranks_asc['c'] == 1  # smallest score wins when ascending=True
        ranks_desc = assign_deterministic_rank(df, 'score', ascending=False)
        assert ranks_desc['a'] == 1  # largest score wins when ascending=False


class TestVikorTies:
    def _tied_df(self):
        # Two alternatives with identical criteria values -> tied S, R, Q
        return pd.DataFrame({
            'alt': ['A', 'B', 'C'],
            'accuracy': [0.5, 0.5, 0.9],
            'stability': [0.7, 0.7, 0.3],
        })

    def test_vikor_produces_unique_ranks_under_ties(self):
        df = self._tied_df().set_index('alt').reset_index()
        result = vikor(
            df, criteria=['accuracy', 'stability'],
            weights={'accuracy': 0.5, 'stability': 0.5},
            criteria_types={'accuracy': 'cost', 'stability': 'benefit'}
        )
        assert result['vikor_rank'].nunique() == len(result)
        assert set(result['vikor_rank']) == set(range(1, len(result) + 1))

    def test_vikor_raw_score_kept_separate_from_rank(self):
        df = self._tied_df()
        result = vikor(
            df, criteria=['accuracy', 'stability'],
            weights={'accuracy': 0.5, 'stability': 0.5},
            criteria_types={'accuracy': 'cost', 'stability': 'benefit'}
        )
        assert 'vikor_Q' in result.columns
        assert 'vikor_rank' in result.columns
        # A and B are tied on vikor_Q but must have different ranks
        q_a = result.loc[result['alt'] == 'A', 'vikor_Q'].iloc[0]
        q_b = result.loc[result['alt'] == 'B', 'vikor_Q'].iloc[0]
        rank_a = result.loc[result['alt'] == 'A', 'vikor_rank'].iloc[0]
        rank_b = result.loc[result['alt'] == 'B', 'vikor_rank'].iloc[0]
        assert q_a == pytest.approx(q_b)
        assert rank_a != rank_b


class TestVikorCostCriteriaDenominatorBug:
    """Regression for a real, pre-existing bug found while building Table 10
    (src/decision/experiment_ranking.py, Phase 5): vikor()'s S/R computation
    used the signed `f_star[j] - f_minus[j]` as the shared denominator for
    both benefit AND cost criteria. For 'cost' criteria f_star=min and
    f_minus=max (reversed relative to 'benefit'), so that denominator was
    NEGATIVE - silently flipping the sign of every normalized distance for
    any cost criterion. This corrupted S (which went negative - a genuine
    group-utility distance can never be negative) and, worse, left R stuck
    at exactly 0.0 for every alternative (R is initialized to 0 and only
    ever updated via `max(R[i], w[j]*normalized)`; a negative normalized
    value can never win that max against the 0 initial value). Since every
    real use of VIKOR in this codebase includes at least one cost criterion
    (accuracy/MAE is always cost), this silently degraded VIKOR to ranking
    by S alone in scripts/02 and scripts/05's existing FS/model selection,
    not just Table 10. Fixed by taking abs() of the denominator (a no-op
    for benefit criteria, which were already positive)."""

    def test_s_and_r_are_never_negative_with_cost_criteria(self):
        df = pd.DataFrame({
            'alt': ['A', 'B', 'C'],
            'accuracy_cost': [1.0, 1.6, 1.3],  # A best (lowest), B worst
            'parsimony_cost': [30, 4, 15],      # B best (fewest), A worst
        })
        result = vikor(
            df, criteria=['accuracy_cost', 'parsimony_cost'],
            weights={'accuracy_cost': 0.8, 'parsimony_cost': 0.2},
            criteria_types={'accuracy_cost': 'cost', 'parsimony_cost': 'cost'}
        )
        assert (result['vikor_S'] >= 0).all()
        assert (result['vikor_R'] >= 0).all()
        # R must actually vary across alternatives, not be stuck at 0 for all.
        assert result['vikor_R'].nunique() > 1

    def test_heavily_weighted_cost_criterion_correctly_picks_its_winner(self):
        # With accuracy_cost weighted at 0.9, the alternative that is best
        # on accuracy_cost (A, lowest value) must win, despite being worst
        # on the lightly-weighted parsimony_cost criterion.
        df = pd.DataFrame({
            'alt': ['A', 'B', 'C'],
            'accuracy_cost': [1.0, 1.6, 1.3],
            'parsimony_cost': [30, 4, 15],
        })
        result = vikor(
            df, criteria=['accuracy_cost', 'parsimony_cost'],
            weights={'accuracy_cost': 0.9, 'parsimony_cost': 0.1},
            criteria_types={'accuracy_cost': 'cost', 'parsimony_cost': 'cost'}
        )
        winner = result.sort_values('vikor_rank').iloc[0]['alt']
        assert winner == 'A'

    def test_all_cost_criteria_matches_manual_computation(self):
        """Directly checks S/R/Q against a hand-derived expectation for a
        simple 2-criterion, 2-alternative all-cost case."""
        df = pd.DataFrame({
            'alt': ['A', 'B'],
            'c1': [1.0, 3.0],  # A best (min=1.0=f_star, max=3.0=f_minus)
            'c2': [5.0, 5.0],  # tied -> denom degenerates to 0 -> contributes 0
        })
        result = vikor(
            df, criteria=['c1', 'c2'], weights={'c1': 0.5, 'c2': 0.5},
            criteria_types={'c1': 'cost', 'c2': 'cost'}
        )
        a = result[result['alt'] == 'A'].iloc[0]
        b = result[result['alt'] == 'B'].iloc[0]
        # A is best on c1 (normalized distance 0), B is worst (normalized
        # distance 1) -> S_B - S_A should equal the c1 weight (0.5).
        assert (b['vikor_S'] - a['vikor_S']) == pytest.approx(0.5)
        assert a['vikor_S'] == pytest.approx(0.0)
        assert a['vikor_R'] == pytest.approx(0.0)


class TestTopsisTies:
    def test_topsis_produces_unique_ranks_under_ties(self):
        df = pd.DataFrame({
            'alt': ['A', 'B', 'C'],
            'accuracy': [0.5, 0.5, 0.9],
            'stability': [0.7, 0.7, 0.3],
        })
        result = topsis(
            df, criteria=['accuracy', 'stability'],
            weights={'accuracy': 0.5, 'stability': 0.5},
            criteria_types={'accuracy': 'benefit', 'stability': 'benefit'}
        )
        assert result['topsis_rank'].nunique() == len(result)
        assert set(result['topsis_rank']) == set(range(1, len(result) + 1))


class TestParetoUnaffected:
    def test_pareto_filter_still_works_alongside_tie_fix(self):
        df = pd.DataFrame({
            'alt': ['A', 'B', 'C'],
            'accuracy': [0.9, 0.5, 0.5],
            'stability': [0.9, 0.7, 0.7],
        })
        result = pareto_filter(
            df, criteria=['accuracy', 'stability'],
            criteria_types={'accuracy': 'benefit', 'stability': 'benefit'}
        )
        # A dominates both B and C (equal or better on all criteria);
        # B and C are identical to each other so neither dominates.
        assert 'A' in result['alt'].values


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
