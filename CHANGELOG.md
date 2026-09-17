# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## [0.7.0] - 2026-09-18

Repository restructuring for publishability - no forecasting results
changed; this is entirely documentation/organisation.

### Added
- `PAPER_MAPPING.md`: verified crosswalk from manuscript Table/Figure
  numbers to the repository file that actually produced each one. The
  pipeline's internal `table1`-`table10`/`fig01`-`fig18` filenames follow
  an early internal build-spec numbering scheme that drifted from the
  manuscript's final numbering across revisions (verified by row count
  and column headers, not guessed from filenames) - e.g.
  `table6_global_factor_summary.csv` is actually manuscript Table 8, and
  `table10_source_data_quality.csv` is actually manuscript Table 5. This
  was previously undocumented anywhere.
- `CONTRIBUTING.md`, `.github/PULL_REQUEST_TEMPLATE.md`,
  `.github/ISSUE_TEMPLATE/bug_report.md`.
- A "Documentation map" section at the top of `README.md` indexing all
  six root-level docs by the question each one answers.

### Changed
- Archived the original single-flat-matrix pipeline (`scripts/01_run_fs.py`
  through `06_interpret_champion.py`, superseded by the current A1-A4/
  Stream-A-B design and never covered by the test suite) to
  `scripts/legacy/`, with a README explaining why and fixed `sys.path`
  depth so they still import correctly if run from their new location.
- Trimmed `outputs/DATA_ANALYSIS_REPORT.md`: removed its Section 10
  (added 2026-08-24), which fully duplicated content now maintained
  canonically in `LIMITATIONS.md`/`CHANGELOG.md`/`outputs/audit/`/
  `outputs/final_rerun_2026/`. Added a banner to the remaining sections
  1-9 flagging them as an early-stage EDA snapshot (still references
  `TEC`/`CEI`, since removed) and pointing to current docs.
- Removed `outputs/runs/full_run_stage1/` - an incomplete, orphaned early
  run (Stream A only, no Stream B/MCDA/tables) added in the initial
  commit and referenced nowhere in any script, test, or doc.

### Known, not fixed this pass (flagged for a deliberate future decision)
- `[tool.setuptools.packages.find] include = ["src*"]` in `pyproject.toml`
  means `pip install .` installs a package literally named `src` - fixing
  this properly means renaming the `src/` directory, which touches every
  `from src.x import y` across ~30 scripts and 28 test files. Not
  attempted here: high mechanical-rename risk for a codebase that is
  currently run via `python scripts/N.py`, not imported as an installed
  library, so the practical benefit is low relative to the risk.
- `config/` (singular, `feature_registry.yaml`) vs `configs/` (plural, run
  configs) is a confusing near-duplicate top-level name. Not renamed this
  pass - `config/feature_registry.yaml`'s path is referenced by both code
  and the manuscript's spec-section citations; renaming it has a wider
  blast radius than the clarity gained. Documented instead in the
  Documentation map.

## [0.6.0] - 2026-09-17

### Added
- `scripts/20_statistical_robustness.py`: now also runs the Diebold-Mariano
  test (already implemented in `src/evaluation/statistical_tests.py` and
  used elsewhere, but not previously wired into this robustness check)
  alongside Wilcoxon/paired-t. Finding: **A3 significantly beats A2 at H1
  individually** (DM statistic -3.63, Holm-corrected p=0.037) - stronger,
  convergent per-horizon evidence alongside the existing pooled result. DM
  correctly returns NaN (not a fabricated number) at H4, where n=8 and
  lag truncation h-1=3 make the long-run-variance estimator unstable.
  Surfaced in `outputs/tables/table13_per_horizon_significance.csv` too.
- `scripts/25_ci_std_mechanism_investigation.py`: tests two candidate
  mechanisms for why carbon-intensity dispersion dominates (forecast-error
  "system stress", wind intermittency) using data already cached locally.
  Honest result: neither correlates strongly with `Grid_CI_std` (0.03-0.16
  and 0.37 respectively) - **what the dispersion signal mechanistically
  represents remains an open question**, documented as such rather than
  forced into either story.
- `scripts/26_prediction_intervals.py`: empirical, leave-one-out prediction
  intervals around the champion's point forecasts, with an honest
  calibration check. Finding: nominal 80% intervals achieve only 62-64%
  empirical coverage at this sample size - reported as an indicative range,
  not a validated/calibrated interval.
- `scripts/27_live_nowcast_demo.py`: runs the champion's actual direct
  H1/H2/H4 models from the true current origin (2025Q1) to produce real
  (unscored - no ground truth exists yet) forecasts for 2025Q2/Q3 and
  2026Q1, using exclusively real inputs (the direct-horizon design means
  no future-quarter placeholder values are ever needed). Also documents
  that real grid data already exists five quarters beyond the current
  origin (through 2026Q2), concrete evidence for the "grid data updates
  before the macro/inventory data" claim behind the early-warning framing.
- `Makefile` (common commands), `.github/dependabot.yml` (monthly pip +
  GitHub Actions update PRs, checked by the existing CI before merge),
  and a coverage report (`--cov=src`, uploaded as a CI artifact) added to
  `.github/workflows/tests.yml`.
- Verified the README's documented `venv` + `pip install -r requirements.txt`
  install path against a genuinely fresh Python 3.10 environment (all 379
  tests pass) - no bug found, but previously unverified.

### Fixed
- `.gitignore`: added `.coverage`/`coverage.xml`/`htmlcov/`.

## [0.5.0] - 2026-09-17

Archived on Zenodo: [10.5281/zenodo.22820748](https://doi.org/10.5281/zenodo.22820748).
This is the version-specific DOI to cite - see `CITATION.cff` / README's
"Citing this work" section.

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
