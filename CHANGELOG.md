# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## [0.5.0] - 2026-09-17

### Added
- `scripts/22_ci_decomposition_ablation.py`: decomposes the carbon-intensity
  family into its five individual statistics (mean, p90, std, high/low
  share) against the no-grid baseline - the follow-up explicitly flagged
  as "not yet run" in `outputs/final_rerun_2026/FINAL_RERUN_REPORT.md`
  section G. Result: tested alone, the mean outperforms dispersion, and
  dispersion alone underperforms the no-grid baseline - see `LIMITATIONS.md`
  for the full write-up of why this qualifies (not contradicts) the SHAP
  finding that dispersion carries the largest attribution share in the
  full joint model.
- `scripts/23_per_horizon_significance.py`: surfaces per-horizon (H1/H2/H4)
  Wilcoxon significance alongside the pooled n=27 test already reported in
  Table 12/Section 4.6, as `outputs/tables/table13_per_horizon_significance.csv`.
- `scripts/24_policy_figures.py`: two new manuscript-ready figures for the
  policy/managerial-implications section -
  `fig_policy_capacity_vs_flexibility.pdf` (family-level vs. single-statistic
  ablation, explicitly captioned with the decomposition nuance above) and
  `fig_policy_stakeholder_map.pdf` (static vector redraw of the
  finding -> policy lever -> institution map).
- `.github/workflows/tests.yml`: CI running the full pytest suite plus a
  dedicated leakage/nested-CV-isolation job on every push/PR to `main`.
  Test coverage already existed (27 files, including a leakage sentinel and
  SHAP integrity checks) but nothing ran it automatically before this.
- `LICENSE` (MIT, matching the license already declared in `pyproject.toml`
  but not previously materialised as a file) and `LIMITATIONS.md`
  (consolidates limitations previously scattered across three separate
  audit documents into one canonical, citable reference).

### Fixed
- `scripts/19_grid_ablation.py`: the `A3_full` ablation variant is now
  reconciled with the full-PSO-budget headline champion numbers from
  `outputs/tables/table4_stream_A_comparison.csv`, instead of being
  silently re-fit under the ablation sweep's reduced PSO budget. Previously
  Table 12/Fig. 8's "Full A3" row (WMASE 0.455, H2 0.609, H4 0.419) did not
  match the same configuration reported everywhere else in the manuscript
  (WMASE 0.4541, H2 0.575, H4 0.467) - a 6-10% relative gap at H2/H4 with
  no footnote explaining it. Fixed at the source rather than papered over
  with a caveat.
- Colour-coding bug in the (now superseded) first pass of
  `scripts/19_grid_ablation.py`'s sibling script: the carbon-intensity
  decomposition figure originally highlighted `A3_CI_std_only` in a
  "winner" dark-green colour despite it being the second-worst-performing
  single-statistic variant. Now highlights the actual best standalone
  statistic (mean) and flags dispersion's below-baseline standalone result
  with an explicit callout instead.

## Notes for the manuscript

See `LIMITATIONS.md` for the full detail. In brief, before the next
submission pass:
1. Update Table 12 / Fig. 8 to the reconciled `A3_full` numbers (now fixed
   in the underlying CSV/figure; manuscript prose numbers were already
   correct, only the ablation table's own copy was stale).
2. Add the per-horizon significance breakdown (Table 13) alongside the
   pooled Table 12 result.
3. Add the carbon-intensity decomposition result and its qualifying
   language wherever the manuscript currently implies dispersion is
   unconditionally the best single grid signal (Discussion 5.1/5.5, and
   the Abstract's "dispersion is the dominant predictor" framing should
   gain one clause: "...within the full grid feature set").
