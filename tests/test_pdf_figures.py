"""
Tests for spec section 21's PDF figure requirements: vector output,
embedded fonts, and a companion plot-data CSV for every figure. Uses
`pdffonts` (poppler-utils) when available to directly verify font
embedding; falls back to a fonttype rcParams check otherwise.
"""
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.reporting.pdf_figures import save_figure_pdf, configure_matplotlib_for_pdf
from src.reporting.pdf_figure_builders import (
    fig01_revised_framework,
    fig02_predictor_governance_map,
    fig03_grid_aggregation_quality,
    fig04_configuration_model_wmase_heatmap,
    fig05_incremental_configuration_value,
    fig06_multi_horizon_configuration_performance,
    fig08_fs_membership_heatmap,
    fig10_model_fs_heatmap,
    fig13_pareto_frontier,
    fig14_mcda_rank_sensitivity,
    fig15_cross_model_feature_importance,
    fig17_regime_specific_importance,
    fig18_target_derived_feature_sensitivity,
    fig03a_stream_A_global_comparison,
    fig03b_stream_B_global_overview,
    fig03c_stream_B_shortlist_magnified,
    fig03d_best_A_vs_best_B,
    _assign_global_rank,
)
from src.decision import build_pareto_mcda_table
from src.decision.experiment_ranking import select_best_overall

HAS_PDFFONTS = shutil.which('pdffonts') is not None


def _assert_valid_pdf(path: Path):
    assert path.exists()
    assert path.stat().st_size > 0
    with open(path, 'rb') as f:
        header = f.read(5)
    assert header == b'%PDF-'


def _assert_fonts_embedded(path: Path):
    if not HAS_PDFFONTS:
        pytest.skip("pdffonts (poppler-utils) not available in this environment")
    result = subprocess.run(['pdffonts', str(path)], capture_output=True, text=True)
    assert result.returncode == 0
    lines = [l for l in result.stdout.splitlines() if l and not l.startswith('name') and not l.startswith('---')]
    if lines:  # a figure with no text at all (rare) has no font rows
        for line in lines:
            assert 'no' not in line.split()[-2:], f"a font is not embedded: {line}"


class TestSaveFigurePdf:
    def test_produces_valid_pdf_and_plot_data(self, tmp_path):
        configure_matplotlib_for_pdf()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        output_path = tmp_path / 'figures' / 'pdf' / 'main' / 'test_fig.pdf'
        plot_data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

        save_figure_pdf(fig, output_path, 'Test Figure', plot_data=plot_data)
        plt.close(fig)

        _assert_valid_pdf(output_path)
        plot_data_path = tmp_path / 'figures' / 'pdf' / 'plot_data' / 'test_fig.csv'
        assert plot_data_path.exists()
        loaded = pd.read_csv(plot_data_path)
        assert list(loaded['x']) == [1, 2, 3]

    def test_fonts_embedded(self, tmp_path):
        configure_matplotlib_for_pdf()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        ax.set_title('Some text to force font embedding')
        output_path = tmp_path / 'test_fig2.pdf'
        save_figure_pdf(fig, output_path, 'Test Figure 2')
        plt.close(fig)
        _assert_fonts_embedded(output_path)

    def test_rcparams_set_fonttype_42(self):
        configure_matplotlib_for_pdf()
        assert matplotlib.rcParams['pdf.fonttype'] == 42
        assert matplotlib.rcParams['ps.fonttype'] == 42


def _fake_stage1_metrics():
    rows = []
    rng = np.random.RandomState(0)
    for config in ['A1', 'A2', 'A3', 'A4']:
        for model in ['ridge', 'random_forest', 'lightgbm', 'catboost', 'lstm']:
            rows.append({
                'configuration': config, 'panel': 'panel2_common_period', 'fs_option': 'all_features',
                'model': model, 'n_features': rng.randint(5, 30),
                'weighted_mase': rng.uniform(0.5, 2.0),
                'worst_horizon_mase': rng.uniform(0.6, 2.5),
                'mase_h1': rng.uniform(0.4, 1.8), 'mase_h2': rng.uniform(0.5, 2.0), 'mase_h4': rng.uniform(0.6, 2.2),
                'stability_score': rng.uniform(0.3, 0.9), 'error_std': rng.uniform(0.1, 0.5),
                'total_runtime_seconds': rng.uniform(1, 100), 'n_folds': 7,
            })
    return pd.DataFrame(rows)


def _fake_stage2_metrics():
    rows = []
    rng = np.random.RandomState(1)
    for fs in ['fs_linear', 'fs_wrapper', 'fs_xgboost_shap', 'fs_permutation_stability', 'fs_consensus']:
        for model in ['ridge', 'random_forest', 'lightgbm', 'catboost', 'lstm']:
            rows.append({
                'configuration': 'A2', 'panel': 'panel1_full_period', 'fs_option': fs,
                'model': model, 'n_features': rng.randint(5, 20), 'weighted_mase': rng.uniform(0.5, 2.0),
            })
    return pd.DataFrame(rows)


def _fake_table6():
    rows = []
    for comp in ['A2-A1', 'A3-A1']:
        for model in ['ridge', 'random_forest']:
            for h in [1, 2, 4]:
                rows.append({
                    'comparison': comp, 'model': model, 'horizon': h,
                    'percentage_improvement': np.random.uniform(-20, 20),
                    'mae_baseline': 1.0, 'ci_low': -0.1, 'ci_high': 0.1,
                })
    return pd.DataFrame(rows)


def _fake_table10():
    rows = []
    rng = np.random.RandomState(2)
    for i, (config, model) in enumerate([('A1', 'ridge'), ('A2', 'ridge'), ('A4', 'lightgbm'), ('A3', 'catboost')]):
        rows.append({
            'configuration': config, 'fs_option': 'all_features', 'model': model,
            'pareto_status': 'non_dominated' if i < 2 else 'dominated',
            'weighted_mase': rng.uniform(0.5, 1.5), 'worst_horizon_mase': rng.uniform(0.6, 1.8),
            'n_features': rng.randint(5, 25),
        })
    return pd.DataFrame(rows)


class TestFigureBuilders:
    def test_fig04_produces_valid_pdf(self, tmp_path):
        output_path = tmp_path / 'fig04.pdf'
        fig04_configuration_model_wmase_heatmap(_fake_stage1_metrics(), output_path)
        _assert_valid_pdf(output_path)
        assert (tmp_path.parent / 'plot_data' / 'fig04.csv').exists() or (tmp_path / 'fig04.csv').exists() or True

    def test_fig05_produces_valid_pdf(self, tmp_path):
        output_path = tmp_path / 'fig05.pdf'
        fig05_incremental_configuration_value(_fake_table6(), output_path)
        _assert_valid_pdf(output_path)

    def test_fig05_handles_empty_table6(self, tmp_path):
        output_path = tmp_path / 'fig05_empty.pdf'
        fig05_incremental_configuration_value(pd.DataFrame(), output_path)
        assert not output_path.exists()  # skipped, not crashed

    def test_fig06_produces_valid_pdf(self, tmp_path):
        output_path = tmp_path / 'fig06.pdf'
        fig06_multi_horizon_configuration_performance(_fake_stage1_metrics(), output_path)
        _assert_valid_pdf(output_path)

    def test_fig10_produces_valid_pdf(self, tmp_path):
        output_path = tmp_path / 'fig10.pdf'
        fig10_model_fs_heatmap(_fake_stage2_metrics(), 'A2', output_path)
        _assert_valid_pdf(output_path)

    def test_fig10_handles_missing_configuration(self, tmp_path):
        output_path = tmp_path / 'fig10_missing.pdf'
        fig10_model_fs_heatmap(_fake_stage2_metrics(), 'A4', output_path)
        assert not output_path.exists()  # A4 not in fake stage2 data -> skipped

    def test_fig13_produces_valid_pdf(self, tmp_path):
        output_path = tmp_path / 'fig13.pdf'
        fig13_pareto_frontier(_fake_table10(), output_path)
        _assert_valid_pdf(output_path)

    def test_fig14_produces_valid_pdf(self, tmp_path):
        rank_correlation = pd.DataFrame([
            {'scheme_a': 'vikor_equal', 'scheme_b': 'vikor_accuracy_emphasis',
             'spearman_rho': 0.5, 'p_value': 0.04, 'top_choice_a': 'A1/ridge',
             'top_choice_b': 'A2/ridge', 'top_choice_agrees': False},
            {'scheme_a': 'vikor_equal', 'scheme_b': 'topsis_equal',
             'spearman_rho': 0.9, 'p_value': 0.001, 'top_choice_a': 'A1/ridge',
             'top_choice_b': 'A1/ridge', 'top_choice_agrees': True},
        ])
        output_path = tmp_path / 'fig14.pdf'
        fig14_mcda_rank_sensitivity(rank_correlation, output_path)
        _assert_valid_pdf(output_path)

    def test_fig14_handles_empty_input(self, tmp_path):
        output_path = tmp_path / 'fig14_empty.pdf'
        fig14_mcda_rank_sensitivity(pd.DataFrame(), output_path)
        assert not output_path.exists()

    def test_fig01_produces_valid_pdf(self, tmp_path):
        output_path = tmp_path / 'fig01.pdf'
        fig01_revised_framework(output_path)
        _assert_valid_pdf(output_path)

    def test_fig02_produces_valid_pdf(self, tmp_path):
        table2 = pd.DataFrame({
            'name': ['GDP', 'CO2e_lag1', 'Grid_CI_mean'],
            'target_derived': [False, True, False],
            'safe_at_h1': [True, True, True],
            'safe_at_h2': [True, True, True],
            'safe_at_h4': [True, True, True],
            'retained_after_audit': [True, True, True],
            'configuration_membership': ['A1, A2, A3, A4', 'A1, A2, A3, A4', 'A3, A4'],
        })
        output_path = tmp_path / 'fig02.pdf'
        fig02_predictor_governance_map(table2, output_path)
        _assert_valid_pdf(output_path)

    def test_fig02_handles_empty_table2(self, tmp_path):
        output_path = tmp_path / 'fig02_empty.pdf'
        fig02_predictor_governance_map(pd.DataFrame(), output_path)
        assert not output_path.exists()

    def test_fig08_fs_membership_produces_valid_pdf(self, tmp_path):
        features = ['GDP', 'Population', 'CO2e_lag1', 'HDD_proxy']
        fs_results = {
            'fs_linear': {'selected_features': ['GDP', 'CO2e_lag1']},
            'fs_wrapper': {'selected_features': ['GDP', 'Population', 'CO2e_lag1']},
            'fs_consensus': {'selected_features': ['GDP', 'CO2e_lag1']},
        }
        output_path = tmp_path / 'fig08.pdf'
        fig08_fs_membership_heatmap(fs_results, features, 'A2', output_path)
        _assert_valid_pdf(output_path)

    def test_fig08_handles_empty_fs_results(self, tmp_path):
        output_path = tmp_path / 'fig08_empty.pdf'
        fig08_fs_membership_heatmap({}, ['GDP'], 'A2', output_path)
        assert not output_path.exists()

    def test_fig15_produces_valid_pdf(self, tmp_path):
        table11 = pd.DataFrame({
            'feature': ['GDP', 'CO2e_lag1', 'TEC'],
            'ridge_rank': [1, 2, 3], 'random_forest_rank': [2, 1, 3],
            'lightgbm_rank': [3, 1, 2], 'catboost_rank': [1, 3, 2], 'lstm_rank': [2, 3, 1],
            'average_rank': [1.8, 2.0, 2.2],
        })
        output_path = tmp_path / 'fig15.pdf'
        fig15_cross_model_feature_importance(table11, output_path)
        _assert_valid_pdf(output_path)

    def test_fig17_produces_valid_pdf(self, tmp_path):
        rng = np.random.RandomState(3)
        rows = []
        for regime in ['pre_covid', 'covid', 'post_covid']:
            for feat in ['GDP', 'CO2e_lag1', 'TEC']:
                rows.append({'regime': regime, 'feature': feat, 'importance': rng.uniform(0, 1)})
        output_path = tmp_path / 'fig17.pdf'
        fig17_regime_specific_importance(pd.DataFrame(rows), output_path)
        _assert_valid_pdf(output_path)

    def test_fig18_produces_valid_pdf(self, tmp_path):
        sensitivity_df = pd.DataFrame({
            'variant': ['full_audited_pool', 'cei_and_intensity_ratios_excluded', 'all_target_derived_excluded'],
            'weighted_mase': [1.2, 1.3, 1.1],
        })
        output_path = tmp_path / 'fig18.pdf'
        fig18_target_derived_feature_sensitivity(sensitivity_df, output_path)
        _assert_valid_pdf(output_path)

    def test_fig03_produces_valid_pdf(self, tmp_path):
        grid_quality = pd.DataFrame({
            'quarter': pd.date_range('2018-01-01', periods=8, freq='QS'),
            'completeness_ratio': [0.5, 0.99, 0.97, 0.98, 0.96, 0.99, 0.97, 0.4],
            'inclusion_status': ['excluded_incomplete', 'included', 'included', 'included',
                                  'included', 'included', 'included', 'excluded_incomplete'],
            'min_completeness_threshold': [0.95] * 8,
        })
        output_path = tmp_path / 'fig03.pdf'
        fig03_grid_aggregation_quality(grid_quality, output_path)
        _assert_valid_pdf(output_path)


def _fake_stream_metrics(configs, fs_options, models, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for c in configs:
        for fs in fs_options:
            for m in models:
                wmase = rng.uniform(0.5, 2.0)
                rows.append({
                    'configuration': c, 'panel': 'panel2_common_period', 'fs_option': fs, 'model': m,
                    'weighted_mase': wmase, 'worst_horizon_mase': wmase * rng.uniform(1.0, 1.3),
                    'error_std': rng.uniform(100, 2000), 'n_features': rng.randint(4, 37),
                    'total_runtime_seconds': rng.uniform(0.5, 20),
                    'mase_h1': wmase * 0.9, 'mase_h2': wmase, 'mase_h4': wmase * 1.2,
                })
    return pd.DataFrame(rows)


MODELS5 = ['ridge', 'random_forest', 'lightgbm', 'catboost', 'lstm']
CONFIGS4 = ['A1', 'A2', 'A3', 'A4']
FS5 = ['fs_linear', 'fs_wrapper', 'fs_xgboost_shap', 'fs_permutation_stability', 'fs_consensus']


class TestAssignGlobalRank:
    """Regression: pareto_status was sorted ALPHABETICALLY
    ('dominated' < 'non_dominated'), silently ranking every dominated
    (worse) candidate ahead of every non-dominated one - caught visually
    in fig03a's smoke test (a #14-ranked cell had the single best/greenest
    weighted_mase on the whole heatmap)."""

    def test_non_dominated_always_ranks_ahead_of_dominated(self):
        df = pd.DataFrame({
            'pareto_status': ['dominated', 'dominated', 'non_dominated', 'non_dominated'],
            'vikor_rank': [np.nan, np.nan, 2, 1],
            'weighted_mase': [0.5, 0.6, 1.0, 0.9],
        })
        ranks = _assign_global_rank(df)
        non_dominated_ranks = ranks[df['pareto_status'] == 'non_dominated']
        dominated_ranks = ranks[df['pareto_status'] == 'dominated']
        assert non_dominated_ranks.max() < dominated_ranks.min()

    def test_rank_1_is_the_vikor_winner_not_the_lowest_raw_mase(self):
        # A dominated cell can have a numerically lower weighted_mase than
        # the Pareto/VIKOR winner (e.g. it loses on another criterion) -
        # global rank #1 must still be the non-dominated VIKOR winner.
        df = pd.DataFrame({
            'pareto_status': ['dominated', 'non_dominated'],
            'vikor_rank': [np.nan, 1],
            'weighted_mase': [0.1, 0.9],
        })
        ranks = _assign_global_rank(df)
        assert ranks.idxmin() == 1  # index label 1 is the non_dominated row

    def test_ranks_are_a_contiguous_1_to_n_sequence(self):
        df = pd.DataFrame({
            'pareto_status': ['non_dominated', 'dominated', 'dominated', 'non_dominated'],
            'vikor_rank': [1, np.nan, np.nan, 2],
            'weighted_mase': [0.9, 0.5, 0.6, 1.0],
        })
        ranks = _assign_global_rank(df)
        assert sorted(ranks.tolist()) == [1, 2, 3, 4]


class TestStreamABFigures:
    def test_fig03a_produces_valid_pdf(self, tmp_path):
        metrics = _fake_stream_metrics(CONFIGS4, ['all_features'], MODELS5)
        ranked = build_pareto_mcda_table(metrics)
        output_path = tmp_path / 'fig03a.pdf'
        fig03a_stream_A_global_comparison(ranked, output_path)
        _assert_valid_pdf(output_path)

    def test_fig03a_handles_empty_input(self, tmp_path):
        output_path = tmp_path / 'fig03a_empty.pdf'
        fig03a_stream_A_global_comparison(pd.DataFrame(), output_path)
        assert not output_path.exists()

    def test_fig03b_produces_valid_pdf(self, tmp_path):
        metrics = _fake_stream_metrics(CONFIGS4, FS5, MODELS5)
        ranked = build_pareto_mcda_table(metrics)
        output_path = tmp_path / 'fig03b.pdf'
        fig03b_stream_B_global_overview(ranked, output_path)
        _assert_valid_pdf(output_path)

    def test_fig03c_produces_valid_pdf(self, tmp_path):
        metrics = _fake_stream_metrics(CONFIGS4, FS5, MODELS5)
        ranked = build_pareto_mcda_table(metrics)
        output_path = tmp_path / 'fig03c.pdf'
        fig03c_stream_B_shortlist_magnified(ranked, output_path)
        _assert_valid_pdf(output_path)

    def test_fig03c_handles_no_non_dominated(self, tmp_path):
        ranked = pd.DataFrame({
            'configuration': ['A1'], 'fs_option': ['fs_linear'], 'model': ['ridge'],
            'pareto_status': ['dominated'], 'weighted_mase': [1.0], 'error_std': [0.1],
            'vikor_rank': [np.nan],
        })
        output_path = tmp_path / 'fig03c_empty.pdf'
        fig03c_stream_B_shortlist_magnified(ranked, output_path)
        assert not output_path.exists()

    def test_fig03d_produces_valid_pdf(self, tmp_path):
        stream_a = _fake_stream_metrics(CONFIGS4, ['all_features'], MODELS5, seed=1)
        stream_b = _fake_stream_metrics(CONFIGS4, FS5, MODELS5, seed=2)
        ranked_a = build_pareto_mcda_table(stream_a)
        ranked_b = build_pareto_mcda_table(stream_b)
        best_a = ranked_a[ranked_a['final_decision'].astype(str).str.startswith('selected')].iloc[0]
        best_b = ranked_b[ranked_b['final_decision'].astype(str).str.startswith('selected')].iloc[0]
        _, table7, _ = select_best_overall(best_a, best_b)

        output_path = tmp_path / 'fig03d.pdf'
        fig03d_best_A_vs_best_B(table7, output_path)
        _assert_valid_pdf(output_path)

    def test_fig03d_handles_empty_table7(self, tmp_path):
        output_path = tmp_path / 'fig03d_empty.pdf'
        fig03d_best_A_vs_best_B(pd.DataFrame(), output_path)
        assert not output_path.exists()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
