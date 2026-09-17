# Manuscript ↔ repository crosswalk

The pipeline's internal table/figure filenames (`table1_...`, `fig01_...`)
follow an early internal build-spec numbering scheme, not the final
manuscript's numbering - the two drifted apart across manuscript
revisions. **A filename's number is not a reliable guide to which
manuscript table/figure it produces.** This file is the verified crosswalk
(checked against each file's actual row count and column headers, not
guessed from filenames), and is the canonical way to find "what generated
manuscript Table N / Figure N."

If you add or renumber a manuscript table/figure, update this file in the
same commit - a stale crosswalk is worse than none.

## Tables

| Manuscript table | Content | Repository file |
|---|---|---|
| Table 1 | Literature comparison of contemporary frameworks | Hand-authored, not pipeline-generated |
| Table 2 | Predictor blocks, timing rules, roles (5 blocks) | Hand-authored summary of `config/feature_registry.yaml` |
| Table 3 | Two-stream configuration design (A1-A4, B1-B4) | `outputs/tables/table3_ab_configuration_definitions.csv` (verified: 8 rows) |
| Table 4 | Forecasting/optimisation/validation/decision settings | Hand-authored (methods settings, not data-generated) |
| Table 5 | External-source coverage and data-quality controls | `outputs/tables/table10_source_data_quality.csv` (verified: 4 rows, matching columns) |
| Table 6 | Stream A results, all 20 unrestricted candidates | `outputs/tables/table4_stream_A_comparison.csv` (verified: 20 rows) |
| Table 7 | Leading Stream B candidates (top 15 of 100) | Top 15 of `outputs/tables/table5_stream_B_comparison.csv` (verified: 100 rows total) |
| Table 8 | Global factor-level summary | `outputs/tables/table6_global_factor_summary.csv` (verified: 18 rows) |
| Table 9 | Feature-selection agreement (recurring core / rejections) | Hand-authored summary of `outputs/tables/table8_fs_outputs.csv` (verified: 96 rows, per feature × selector) |
| Table 10 | Stream winners and final cross-stream decision | `outputs/tables/table7_stream_winners_and_final.csv` (verified: 2 rows, Best_A/Best_B) |
| Table 11 | Corrected champion interpretation across regimes | `outputs/tables/table9_covid_regime_interpretation.csv` (verified: 9 rows = 3 winners × 3 regimes; manuscript shows the champion's 3 rows) |
| Table 12 | Robustness: pooled paired tests (Panel A) + grid-family ablation (Panel B) | Panel A: `outputs/robustness/grid_signal_paired_tests.csv` (pooled rows); Panel B: `outputs/robustness/grid_ablation_results.csv` (best model per variant) |
| Table 13 *(new, this pass)* | Per-horizon significance alongside the pooled Table 12 result | `outputs/tables/table13_per_horizon_significance.csv`, built by `scripts/23_per_horizon_significance.py` |

**Supplementary tables** (referenced in text, not in the main 1-13 sequence):
- Supplementary Table S1 (full 37-row feature registry) = `outputs/tables/table2_feature_registry.csv` (verified: 37 rows) - note the filename says "table2" but this is a *supplementary* table, not manuscript Table 2.
- Supplementary Table S3 (full 100-candidate Stream B table) = `outputs/tables/table5_stream_B_comparison.csv` (same file as Table 7's source, just unabridged).
- Supplementary Table S4 (96 feature-level FS decisions) = `outputs/tables/table8_fs_outputs.csv`.
- Supplementary Tables S5/S6 (mobility sensitivity / origin diagnostics) = `outputs/robustness/mobility_sensitivity_results.csv` / `outputs/robustness/champion_origin_diagnostics.csv`.

## Figures

Verified against the manuscript's figure descriptions where the filename
alone doesn't make the mapping obvious. Filenames retain their internal
spec numbering (`fig01`, `fig02`, ... - non-contiguous by design, since
not every internally-planned figure made the final manuscript); treat the
description column, not the number in the filename, as authoritative.

| Manuscript figure | Content | Repository file(s) |
|---|---|---|
| Figure 1 | Q-DECEM framework diagram | `outputs/figures/pdf/main/fig01_revised_qdecem_framework.pdf` |
| Figure 2 | Predictor governance/leakage map + grid completeness | `fig02_predictor_governance_map.pdf` (panel a) + `fig03_grid_aggregation_quality.pdf` (panel b) |
| Figure 3 | Global comparison across streams (4 panels) | `fig03a_stream_A_global_comparison.pdf`, `fig03b_stream_B_global_overview.pdf`, `fig03c_stream_B_shortlist_magnified.pdf`, `fig03d_best_A_vs_best_B.pdf` |
| Figure 4 | Feature-selection membership, B1-B4 | `fig08_b1_feature_selection_membership.pdf` .. `fig08_b4_feature_selection_membership.pdf` |
| Figure 5 | Out-of-sample forecasts, champion + finalist comparison (panels a-f, H1/H2/H4) | `outputs/figures/pdf/forecasting/fig_best_overall_forecast_H{1,2,4}.pdf` (left column) + `fig_best_A_vs_B_forecast_H{1,2,4}.pdf` (right column), from `scripts/15_generate_forecast_atlas.py` |
| Figure 6 | Pareto structure and MCDA sensitivity | `fig11a_stream_A_pareto_frontier.pdf`, `fig11b_stream_B_pareto_frontier.pdf`, `fig14a_mcda_rank_sensitivity_stream_A.pdf`, `fig14b_mcda_rank_sensitivity_stream_B.pdf` |
| Figure 7 | Corrected champion SHAP and regime interpretation | `fig_champion_regime_importance.pdf` |
| Figure 8 | Grid-signal ablation | `outputs/figures/pdf/sensitivity/fig_grid_ablation.pdf` |

**Sensitivity/supplementary figures** (referenced in text as
supplementary, not in the main 1-8 sequence): `fig_mobility_robustness.pdf`
(mobility M0-M3), `fig_origin_level_grid_gain.pdf` (per-origin diagnostic),
`fig_ci_decomposition.pdf` and `fig_ci_std_mechanism.pdf` (this pass's
carbon-intensity decomposition/mechanism follow-ups), `fig_per_horizon_significance.pdf`
(this pass's Table 13 companion).

**New, not yet in any manuscript draft**: `fig_policy_capacity_vs_flexibility.pdf`,
`fig_policy_stakeholder_map.pdf`, and `fig_champion_prediction_intervals.pdf`
- candidates for a Discussion 5.5 "policy and economic implications"
section, per `RESEARCH_OVERVIEW.md`.

## Known gaps

- Tables 2 and 9 in the manuscript are hand-authored summaries derived
  from a pipeline output, not 1:1 exports - if the manuscript text is
  edited without re-deriving from the underlying CSV, the two can drift.
  Cross-check before submission.
