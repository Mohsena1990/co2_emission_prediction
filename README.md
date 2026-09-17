# Q-DECEM: Quarterly UK CO2e Forecasting Framework

![tests](https://github.com/Mohsena1990/co2_emission_prediction/actions/workflows/tests.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22820748.svg)](https://doi.org/10.5281/zenodo.22820748)

A modular Python framework for quarterly UK CO2e emissions forecasting across four
leakage-controlled data configurations (raw / +engineered / +grid-fusion /
+engineered+grid-fusion), with nested walk-forward validation, five feature-selection
strategies, PSO hyperparameter tuning, and Pareto/MCDA decision support.

## Documentation map

Six documents, each answering a different question - read the one that
matches yours rather than searching all of them:

| Document | Answers |
|---|---|
| [`RESEARCH_OVERVIEW.md`](RESEARCH_OVERVIEW.md) | "What is this, conceptually?" - core idea, algorithm design, policy layer |
| [`LIMITATIONS.md`](LIMITATIONS.md) | "Can I trust this specific claim?" - sample size, statistical caveats, open questions |
| [`PAPER_MAPPING.md`](PAPER_MAPPING.md) | "Which script produced Table/Figure N in the paper?" |
| [`CHANGELOG.md`](CHANGELOG.md) | "What changed, and when?" |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | "How do I make a change safely?" |
| This file | "How do I install and run it?" |

## Overview

**Research question:** within a leakage-controlled quarterly UK CO2e forecasting
framework, how much predictive value is contributed by feature engineering,
electricity-grid data fusion, feature-selection strategy, and forecasting-model
architecture?

**Four data configurations** (spec section 8), built purely from
`config/feature_registry.yaml`'s `configuration_membership` tags - no hardcoded
feature lists:

| Config | Contents | Nominal max features |
|---|---|---|
| A1 | 5 audited raw quarterly predictors + 6 Google COVID mobility-shock predictors (TEC/CEI fully removed) | 11 |
| A2 | A1 + 14 engineered predictors (lags, growth rates, a historical population-scaled intensity ratio, weather, seasonal, disruption dummies) | 25 |
| A3 | A1 + 12 audited GB electricity-grid quarterly features | 11+K |
| A4 | A2 + the same 12 grid features | 25+K |

**Two experimental panels** (spec section 9), since grid data is only available from
2018 onward while the raw macro data spans 1999-2025:
- **Panel 1** (full period): A1 vs. A2 over the longest valid window - isolates the
  value of feature engineering alone.
- **Panel 2** (common period): all four configurations over the grid-covered common
  window, with identical outer folds - isolates the value of grid fusion, and grid
  fusion's interaction with feature engineering.

**Other core capabilities:**

- **Five Feature Selection Strategies**: FS1 linear stability (VIF + Ridge/ElasticNet),
  FS2 wrapper (RFE/SFS/SBS), FS3 XGBoost-SHAP stability, FS4 permutation stability, FS5
  consensus (vote-based across FS1-FS4) - all fit strictly inside inner expanding-window
  folds of the outer training data, never the outer test fold.
- **Predictor Governance**: every feature (raw, engineered, and grid) is declared in
  `config/feature_registry.yaml` (family, target-derived status, minimum lag, per-horizon
  safety, configuration membership) and enforced at feature-matrix build time.
- **Nested Walk-Forward Validation**: outer expanding-window folds for reported
  performance; inner expanding-window folds (or single-cutoff isolated tuning plans, in
  reduced-budget "sweep" mode) for feature selection and PSO tuning - never the same data.
- **PSO Hyperparameter Tuning**: reduced budget for the full config x FS x model sweep,
  full budget + multiple seeds reserved for re-running the Pareto-shortlisted configurations.
- **Pareto + MCDA Decision Support**: VIKOR/TOPSIS with deterministic tie-breaking, 5
  weight-sensitivity schemes, rank-correlation comparison across schemes.
- **Statistical Comparison**: paired fold-level bootstrap CIs, Diebold-Mariano test,
  Wilcoxon signed-rank, Bonferroni/Benjamini-Hochberg multiple-comparison correction.
- **Cross-Model Interpretability**: Ridge coefficients, TreeSHAP (RF/LightGBM/CatBoost),
  permutation importance (LSTM), normalized and ranked for cross-model comparison.
- **Regime & Target-Derived Sensitivity**: pre/COVID/post-COVID importance stability;
  re-runs with CEI/intensity-ratio/CO2e_dlog (and, optionally, all historical
  target-derived) features excluded.
- **Annual Consistency Safeguards**: verify quarterly predictions aggregate correctly to
  annual totals.
- **PDF Figures**: vector output, embedded TrueType fonts, journal-ready dimensions, a
  companion plot-data CSV for every figure.

## Models

- Ridge Regression
- Random Forest
- LightGBM
- CatBoost
- LSTM (PyTorch)

Baselines: previous-quarter naive, seasonal naive (y_t = y_{t-4}).

## Installation

Clean-environment setup (Linux/Mac):

```bash
git clone <repo-url> co2_emission_prediction
cd co2_emission_prediction

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
git clone <repo-url> co2_emission_prediction
cd co2_emission_prediction

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Verify the install:

```bash
pytest tests/ -q
```

## Project Structure

```
.
├── config/
│   └── feature_registry.yaml   # Single source of truth for every raw/engineered/grid feature
├── configs/                    # Run configurations
│   ├── default_config.yaml
│   ├── fast_validation_config.yaml   # Reduced PSO budget, for fast end-to-end validation
│   └── sweep_config.yaml             # Reduced budget for the FULL config x FS x model sweep
├── data/
│   ├── raw/
│   │   └── grid_cache/         # Cached raw GB Carbon Intensity API responses
│   └── processed/              # X_A1..X_A4.parquet, y, cv_plan, panel2_cv_plan (generated)
├── outputs/
│   ├── audit/                  # Repository audit (pipeline_audit.md/json)
│   ├── runs/run_<id>/          # Per-run intermediate outputs (see below)
│   ├── tables/                 # Final Tables 1-12 (spec section 20)
│   └── figures/pdf/main/       # Final PDF figures + plot_data/ (spec section 21/22)
├── scripts/
│   ├── 00_make_dataset.py                  # Load, clean, engineer features, build A1-A4 matrices, Panel 1/2 CV plans
│   ├── fetch_grid_data.py                  # Backfill the GB Carbon Intensity API cache (idempotent)
│   ├── fetch_mobility_data.py              # Cache Google's UK-national COVID mobility CSV (idempotent)
│   ├── fetch_owid_data.py                  # Cache OWID UK renewable/low-carbon electricity share CSVs (idempotent)
│   ├── 01_run_fs.py .. 06_interpret_champion.py   # Legacy single-configuration pipeline (still functional)
│   ├── 10_run_experiment_grid.py           # Stage 1 (config x model) + Stage 2 (FS x model) grid, spec section 12
│   ├── 11_pareto_mcda_and_incremental.py   # Table 6 (incremental value) + Table 10 (Pareto/MCDA)
│   ├── 12_interpretability_and_sensitivity.py  # Table 11 (cross-model importance) + regime + target-derived sensitivity
│   ├── 13_generate_tables.py               # All 12 tables -> outputs/tables/
│   ├── 14_generate_figures.py              # PDF figures -> outputs/figures/pdf/main/
│   ├── 18_mobility_sensitivity.py          # M0-M3 mobility-encoding robustness sweep
│   ├── 19_grid_ablation.py                 # Grid-family ablation (CI vs. generation-mix vs. full A3)
│   ├── 20_statistical_robustness.py        # Paired significance tests (A3 vs A1/A2/A4/naive)
│   ├── 21_origin_diagnostics.py            # Per-origin forecast/error diagnostics
│   ├── 22_ci_decomposition_ablation.py     # Single-statistic decomposition within the CI family
│   ├── 23_per_horizon_significance.py      # Per-horizon significance (incl. Diebold-Mariano) alongside the pooled test
│   ├── 24_policy_figures.py                # Policy/managerial-implications figures
│   ├── 25_ci_std_mechanism_investigation.py # What Grid_CI_std proxies for (open question - see LIMITATIONS.md)
│   ├── 26_prediction_intervals.py          # Empirical prediction intervals + calibration check for the champion
│   └── 27_live_nowcast_demo.py             # Runs the deployed mechanism from the current real origin (demo, not scored)
├── src/
│   ├── core/                  # Config, logging, utilities
│   ├── data_io/                # Data loading, schema
│   ├── quality/                # Data quality checks
│   ├── features/                # Feature engineering, registry, A1-A4 matrix builder
│   ├── grid/                    # GB Carbon Intensity API fetch + quarterly aggregation
│   ├── mobility/                 # Google COVID mobility fetch + quarterly aggregation (neutral-zero convention)
│   ├── owid/                     # OWID renewable/low-carbon annual share fetch + one-year-lag alignment
│   ├── splits/                  # Walk-forward CV, nested CV, Panel 2 common-period CV
│   ├── fs/                      # Feature selection (FS1-FS5)
│   ├── models/                  # Forecasting models
│   ├── optimization/            # PSO/GWO, nested per-outer-fold retuning
│   ├── pipeline/                # Single-cell experiment orchestrator (run_configuration_model)
│   ├── evaluation/               # Metrics, statistical tests, incremental value
│   ├── safeguards/               # Annual consistency
│   ├── decision/                 # MCDA (VIKOR/TOPSIS), Pareto filter, experiment ranking
│   ├── interpretability/          # SHAP analysis, cross-model importance, regime stability
│   └── reporting/                  # Tables, PDF figures
├── tests/
├── requirements.txt
└── README.md
```

## Usage

### One-command run (Full A1-A4 Pipeline)

```bash
scripts/run_full_pipeline.sh [run_id] [data_config] [sweep_config]

# e.g.
scripts/run_full_pipeline.sh my_run
```

Runs grid-cache refresh -> dataset/matrix build -> Stage 1+2 experimental grid ->
Pareto/MCDA + incremental-value tables -> interpretability/regime/sensitivity ->
tables -> PDF figures, all under one `run_id`, and stops on the first failing step
(`set -euo pipefail`). Defaults: `run_id=run_<timestamp>`,
`data_config=configs/default_config.yaml`, `sweep_config=configs/sweep_config.yaml`.
Expect a multi-hour runtime for the full config x FS x model grid - run it with
`nohup`/`tmux`/`screen` for anything beyond a quick smoke test.

GPU is used automatically wherever the installed backend supports it (CatBoost,
LSTM/PyTorch - LightGBM's pip wheel is CPU-only, see **GPU support** below); set
`model.use_gpu: false` in the config file to force CPU everywhere.

### Step-by-step equivalent

```bash
# Step 0: backfill/refresh the mobility + grid + OWID data caches, then build A1-A4 matrices + Panel 1/2 CV plans
python scripts/fetch_mobility_data.py --config configs/default_config.yaml
python scripts/fetch_grid_data.py --config configs/default_config.yaml
python scripts/fetch_owid_data.py --config configs/default_config.yaml
python scripts/00_make_dataset.py --config configs/default_config.yaml --run-id my_run

# Step 1: Run the full config x model (Stage 1) + FS x model (Stage 2) experimental grid
python scripts/10_run_experiment_grid.py --config configs/sweep_config.yaml --run-id my_run --stage both

# Step 2: Incremental-value comparisons (Table 6) + Pareto/MCDA ranking (Table 10)
python scripts/11_pareto_mcda_and_incremental.py --config configs/sweep_config.yaml --run-id my_run

# Step 3: Cross-model interpretability (Table 11), regime + target-derived sensitivity
python scripts/12_interpretability_and_sensitivity.py --config configs/sweep_config.yaml --run-id my_run

# Step 4: Generate all 12 tables and the PDF figures
python scripts/13_generate_tables.py --config configs/sweep_config.yaml --run-id my_run
python scripts/14_generate_figures.py --config configs/sweep_config.yaml --run-id my_run
```

`configs/sweep_config.yaml` is a reduced-PSO-budget profile intended for the full
config x FS x model sweep (Stage 1/2); re-run a specific Pareto-shortlisted cell at
full budget (`configs/default_config.yaml`, with `optimization.nested_retuning: true`
for true per-outer-fold PSO retuning) for the numbers that go in a final report.

### GPU support

`model.use_gpu` (default `true`) is a process-wide toggle, not a per-model
hyperparameter - each backend is probed once per process and falls back to CPU
automatically if the installed build doesn't support it:

| Model | GPU backend | Notes |
|---|---|---|
| CatBoost | `task_type: GPU` | Works out of the box with the standard `catboost` pip package. |
| LSTM (PyTorch) | CUDA | Works out of the box if `torch.cuda.is_available()`. |
| LightGBM | CPU only | The pip wheel ships without GPU support; install a CUDA/OpenCL build (source build with `-DUSE_CUDA=1`, or a conda-forge `lightgbm=*=cuda*` package) to enable it. |
| Ridge, Random Forest | CPU only | No GPU implementation in scikit-learn; Random Forest uses all CPU cores (`n_jobs=-1`). |

Set `model.use_gpu: false` in a config file (or override at the config layer) to
force CPU everywhere, e.g. for reproducibility comparisons or to avoid contending
with another GPU job.

### Legacy pipeline (archived)

The project's original single-flat-matrix pipeline (predates the A1-A4
configuration split) is archived under [`scripts/legacy/`](scripts/legacy/README.md)
for historical reference. It is not maintained, not tested, and not used
by any current result - use the pipeline above instead.

## Configuration

Edit `configs/default_config.yaml` (or pass `--config` with a variant) to customize:

```yaml
splits:
  min_train_size: 40          # Panel 1 outer-fold sizing
  panel2_min_train_size: 16   # Panel 2 (common period) is much shorter - its own sizing
  horizons: [1, 2, 4]
  horizon_weights: {1: 0.5, 2: 0.3, 4: 0.2}

optimization:
  optimizer: "pso"             # or "gwo"
  n_particles: 20
  n_iterations: 30
  nested_retuning: false       # true = real per-outer-fold PSO retuning (spec 15/16)

grid:
  enabled: true
  cache_dir: "data/raw/grid_cache"
  minimum_quarter_completeness: 0.95

mcda:
  method: "vikor"               # or "topsis"
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific area
pytest tests/test_leakage_sentinel.py -v
pytest tests/test_configurations.py -v
pytest tests/test_grid.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Key Constraints

1. **No Random Splits**: Only walk-forward/expanding-window CV, outer and inner.
2. **No Leakage**: Feature selection and PSO tuning never see outer-test-fold data;
   scalers fit on the training fold only; lags/growth/intensity features are computed
   without contemporaneous target values.
3. **Panel Discipline**: full-period A1/A2 results are never compared directly against
   shorter-period A3/A4 results when attributing performance to grid fusion - use Panel
   2's common-period, identical-fold comparison for that claim.
4. **LSTM Constraints**: small model (limited lookback, dropout, early stopping);
   `build_predict_input` enforces exact prediction-length/date alignment.
5. **Annual Consistency**: quarterly predictions must aggregate to sensible annual totals.

## Dependencies

- Python 3.8+
- numpy, pandas, scipy
- scikit-learn, lightgbm, catboost, xgboost
- torch (for LSTM)
- shap, matplotlib
- requests (grid-data ingestion)
- See `requirements.txt` for the full pinned list.

## Citing this work

This repository is archived on Zenodo with a version-specific DOI, so a
citation always points to the exact code and results a paper used, rather
than a `main` branch that can change underneath it:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22820748.svg)](https://doi.org/10.5281/zenodo.22820748)

```bibtex
@software{asghari_ilani_qdecem_2026,
  author  = {Asghari Ilani, Mohsen},
  title   = {{Q-DECEM: Quarterly UK CO2e Forecasting Framework}},
  year    = {2026},
  version = {0.5.0},
  doi     = {10.5281/zenodo.22820748},
  url     = {https://github.com/Mohsena1990/co2_emission_prediction}
}
```

See [`CITATION.cff`](CITATION.cff) for the machine-readable version (GitHub
surfaces this automatically via the "Cite this repository" button).

## License

See [`LICENSE`](LICENSE) (MIT).
