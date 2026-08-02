#!/usr/bin/env bash
# Q-DECEM: run the complete A1-A4 pipeline end-to-end with a single command.
#
#   scripts/run_full_pipeline.sh [run_id] [data_config] [sweep_config]
#
# Defaults: run_id=run_<timestamp>, data_config=configs/default_config.yaml,
# sweep_config=configs/sweep_config.yaml (reduced-PSO-budget sweep profile -
# re-run a Pareto-shortlisted cell at configs/default_config.yaml with
# optimization.nested_retuning: true for full-budget final numbers).
#
# Steps: refresh the Google mobility + GB Carbon Intensity grid + OWID
# electricity-share caches -> build A1-A4 matrices + Panel 1/2 CV plans ->
# run the Stream A (no FS) + Stream B (FS1-5) config x model grid -> two-stage
# Pareto/MCDM (Best_A/Best_B/Best_Overall) -> interpretability/regime/sensitivity
# -> tables -> PDF figures -> forecast atlas. Every step reuses the same run_id,
# so all output lands in outputs/runs/<run_id>/.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_ID="${1:-run_$(date +%Y%m%d_%H%M)}"
DATA_CONFIG="${2:-configs/default_config.yaml}"
SWEEP_CONFIG="${3:-configs/sweep_config.yaml}"

echo "=========================================================="
echo "Q-DECEM full pipeline — run_id=${RUN_ID}"
echo "  data config:  ${DATA_CONFIG}"
echo "  sweep config: ${SWEEP_CONFIG}"
echo "=========================================================="

echo; echo "[0/10] Refreshing Google mobility data cache..."
python scripts/fetch_mobility_data.py --config "${DATA_CONFIG}"

echo; echo "[1/10] Refreshing GB Carbon Intensity grid-data cache..."
python scripts/fetch_grid_data.py --config "${DATA_CONFIG}"

echo; echo "[2/10] Refreshing OWID electricity-share data cache..."
python scripts/fetch_owid_data.py --config "${DATA_CONFIG}"

echo; echo "[3/10] Building A1-A4 feature matrices + Panel 1/2 CV plans..."
python scripts/00_make_dataset.py --config "${DATA_CONFIG}" --run-id "${RUN_ID}"

echo; echo "[4/10] Running Stream A (config x model) + Stream B (FS x model) grid..."
python scripts/10_run_experiment_grid.py --config "${SWEEP_CONFIG}" --run-id "${RUN_ID}" --stream both

echo; echo "[5/10] Pareto/MCDA ranking + incremental-value comparisons..."
python scripts/11_pareto_mcda_and_incremental.py --config "${SWEEP_CONFIG}" --run-id "${RUN_ID}"

echo; echo "[6/10] Cross-model interpretability + regime + target-derived sensitivity..."
python scripts/12_interpretability_and_sensitivity.py --config "${SWEEP_CONFIG}" --run-id "${RUN_ID}"

echo; echo "[7/10] Generating tables..."
python scripts/13_generate_tables.py --config "${SWEEP_CONFIG}" --run-id "${RUN_ID}"

echo; echo "[8/10] Generating PDF figures..."
python scripts/14_generate_figures.py --config "${SWEEP_CONFIG}" --run-id "${RUN_ID}"

echo; echo "[9/10] Generating forecast atlas (36 PDFs)..."
python scripts/15_generate_forecast_atlas.py --config "${SWEEP_CONFIG}" --run-id "${RUN_ID}"

echo; echo "[10/10] Done."
echo "Outputs: outputs/runs/${RUN_ID}/"
