# Known limitations

This file consolidates limitations that were previously scattered across
`outputs/final_rerun_2026/FINAL_RERUN_REPORT.md`, `outputs/DATA_ANALYSIS_REPORT.md`
and `outputs/audit/champion_shap_diagnosis.md`. It is the single place the
manuscript's Limitations section (5.4) and any future replication should
point to. Update this file, not just the manuscript text, whenever a new
limitation is found or an old one is resolved.

## Sample size

- The common grid-covered evaluation panel is 28 quarters (2018Q2-2025Q1),
  giving 11 out-of-sample H1 forecasts and 8 each at H2/H4 - 27 paired
  horizon-origin observations when pooled.
- Regime-specific SHAP subsamples are as small as n=7 (pre-COVID). Spearman
  rank-correlation claims across regimes (0.915-0.965) are evocative but
  drawn from very few points; treat as suggestive, not conclusive.
- "Pre-COVID" in the regime analysis means 2018Q1-2019Q4 only, because grid
  data starts in 2018 - it is a narrow, already-mid-transition reference
  period, not a claim spanning the UK grid's full decarbonisation history.

## Statistical testing

- The manuscript's headline significance test pools H1+H2+H4 into n=27 for
  a single Wilcoxon test per comparison. Forecasts sharing an origin or
  overlapping training windows are not fully independent - pooling should
  be read as a family-level result, not as 27 independent trials.
- Per-horizon significance (now in `outputs/tables/table13_per_horizon_significance.csv`,
  built by `scripts/23_per_horizon_significance.py`) shows that **no single
  horizon individually survives Holm correction** for any comparison,
  including A3 vs A2 - only the pooled test does. Report both, not just
  the pooled number.
- 120 total candidates (20 Stream A + 100 Stream B) were searched over an
  11-28 quarter panel. Nested CV prevents leakage within a candidate's own
  fit, but does not eliminate selection-multiplicity risk across 120
  candidates at this sample size.
- Pareto filtering removes only 4/20 (Stream A) and 52/100 (Stream B)
  candidates before VIKOR - with 5 simultaneous decision criteria, high
  non-domination rates are expected (curse of dimensionality), so VIKOR is
  doing most of the actual selection work, not Pareto filtering.

## The carbon-intensity decomposition nuance (important - read before citing "dispersion dominates")

`scripts/22_ci_decomposition_ablation.py` (added in this pass, addressing the
explicit "not yet run" follow-up from `FINAL_RERUN_REPORT.md` section G)
isolates each Grid_CI_* statistic alone against the raw+mobility baseline:

| Variant (best model)     | Weighted MASE |
|---------------------------|--------------:|
| Mean only                 | 0.598 |
| Low-intensity share only  | 0.614 |
| 90th percentile only      | 0.621 |
| All 5 CI statistics       | 0.632 |
| No grid data (baseline)   | 0.637 |
| **Dispersion (std) only** | **0.662** |
| High-intensity share only | 0.709 |

**Tested alone, the mean outperforms dispersion, and dispersion alone is
worse than using no grid data at all** - despite dispersion (`Grid_CI_std`)
carrying the largest SHAP attribution share (31.8%) in the full 23-feature
joint model. This is not a contradiction: TreeSHAP reports a feature's
marginal contribution *given every other feature already in the model*,
not its value as a standalone predictor. The correct, careful claim is:

> Within the full A3 feature set, dispersion does the most explanatory
> work on top of the mean and upper tail already being present. It is
> not, by itself, the single best grid statistic - the full combination
> is what drives the forecasting gain, not any one summary statistic.

Any manuscript or policy text (including the "capacity vs. flexibility"
framing in Discussion 5.5) that says "dispersion, not the average, is what
matters" should be qualified with this finding. See
`outputs/figures/pdf/main/fig_policy_capacity_vs_flexibility.pdf` and
`outputs/figures/pdf/sensitivity/fig_ci_decomposition.pdf` for the figures
built to make this distinction explicit.

## Reduced-PSO-budget sensitivity sweeps

`scripts/18_mobility_sensitivity.py`, `scripts/19_grid_ablation.py`, and
`scripts/22_ci_decomposition_ablation.py` all use a reduced PSO budget
(`n_particles=8, n_iterations=10` vs. the primary run's 20/30 with nested
per-outer-fold retuning). This is a deliberate, documented compute
trade-off for sensitivity/mechanism sweeps, not the champion-selection
pipeline. Two consequences to keep in mind:

1. **The `A3_full` variant is not re-fit under the reduced budget.**
   `scripts/19_grid_ablation.py` now explicitly reconciles it with the
   full-budget headline numbers from `outputs/tables/table4_stream_A_comparison.csv`
   (fixed 2026-09 - previously the ablation table silently reported a
   different number for "A3/LightGBM, 23 features" than every other table
   in the manuscript, off by 6-10% at H2/H4, with no footnote).
2. **The other ablation/decomposition bookend variants are somewhat
   sensitive to the specific reduced-budget PSO draw.** Re-running
   `scripts/19_grid_ablation.py` reproduces the qualitative ranking (full
   A3 best; generation-mix-only worse than no grid data) but the exact
   ordering of two close middle variants (`A3_CI_only` vs `A3_no_genmix`)
   is not perfectly stable run-to-run. Treat exact values for non-champion
   ablation variants as indicative, not to three-decimal precision, unless
   re-run at full budget.

## Interpretation boundaries

- SHAP values are predictive attribution, not causal effects. A variable
  can improve a forecast without being a policy lever.
- Grid carbon intensity represents the electricity system, not all
  sources of national CO2 - it is an external predictor of economy-wide
  emissions, not a substitute target.
- None of the government/sector use cases in the policy map or
  Discussion 5.5 imply the model should override the reconciled national
  inventory. They are framed as monitoring inputs.

## Mobility data

- The Google COVID-19 Community Mobility archive ends 15 October 2022.
  Outside that window, mobility variables are set to a neutral zero by
  convention, not an observed value. `scripts/18_mobility_sensitivity.py`
  confirms the A3 ranking is robust to this convention (M0-M3 scenarios),
  but the series cannot serve as a permanent real-time input going
  forward without a replacement source.

## Cross-country / cross-sector generalisation

All results are for the UK only, 1999-2025 (grid-covered common period
2018-2025). The feature registry and pipeline are config-driven rather
than UK-hardcoded, so the lowest-cost replication path is one comparator
grid with a public carbon-intensity API of similar granularity, reusing
the existing scripts largely unchanged - not yet attempted.
