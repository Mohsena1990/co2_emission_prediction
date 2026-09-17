#!/usr/bin/env python
"""
Script 27: live-nowcast demonstration - the mechanism, not a new headline
result.

Produces the champion's actual next H1/H2/H4 forecasts from the true
current origin (2025Q1, the last quarter with a real published target),
using the exact direct-horizon methodology the manuscript validated
(src/features/engineering.py::create_direct_horizon_targets: horizon-h
models are trained on (X_t, y_{t+h}) pairs, so a forecast for origin t
uses ONLY features already known at t - never a future quarter's own
covariates). Because 2025Q1 is itself a fully real, already-observed
row of X_A3 (every predictor, including grid data, was genuinely
available by then), this nowcast needs zero placeholder inputs - a
cleaner and more defensible demonstration than an earlier draft of this
script that (incorrectly) tried to forecast using synthesized future
quarters' own macro/weather placeholders.

What this DOES show concretely: data/processed/grid_quarterly_features.parquet
already contains real grid data through 2026Q2 (fetched for the source-
validation check behind manuscript Table 5), i.e. five quarters ahead of
where the macro/target series currently ends (2025Q1, bounded by
data/raw/data 1999-2025Q1.xlsx). The moment ONS publishes 2025Q2's
figures and this pipeline's origin advances, the grid predictor for that
new origin is already sitting in the cache - concrete evidence for the
"grid data updates faster than the macro/inventory data this pipeline
also depends on" claim in RESEARCH_OVERVIEW.md, without needing a single
placeholder value.

This is a demonstration of the deployed mechanism, not a new accuracy
claim - there is no ground truth yet for 2025Q2/Q3/2026Q1 to score
against, and it must not be cited as validated accuracy.

Outputs:
    outputs/robustness/live_nowcast_demo.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.core import Config, setup_logging
from src.data_io import load_processed_data
from src.features.engineering import create_direct_horizon_targets
from src.models import ModelRegistry

PRIMARY_RUN_ID = 'my_run'
CONFIGURATION, FS_OPTION, MODEL_NAME = 'A3', 'all_features', 'lightgbm'
HORIZONS = [1, 2, 4]


def load_horizon_hyperparameters(run_dir: Path, configuration: str, fs_option: str,
                                  model_name: str, horizon: int, logger) -> dict:
    """Same provenance-recovery principle as scripts/12's
    _load_cell_hyperparameters (see outputs/audit/champion_shap_diagnosis.md
    for why refitting with library defaults is unsafe), but filtered by
    HORIZON too, since that function does not take a horizon argument and
    would otherwise pick whichever row has the latest train_end regardless
    of which horizon it belongs to."""
    rows = []
    for stream in ('A', 'B'):
        prov_path = run_dir / 'fold_predictions' / f'stream_{stream}' / 'provenance.json'
        if not prov_path.exists():
            continue
        with open(prov_path) as f:
            prov = json.load(f)
        rows.extend([
            r for r in prov
            if r['configuration'] == configuration and r['fs_option'] == fs_option
            and r['model'] == model_name and r['horizon'] == horizon
        ])
    if not rows:
        raise ValueError(
            f"No fold provenance found for {configuration}/{fs_option}/{model_name}/h={horizon} - "
            f"refusing to silently fall back to library-default hyperparameters."
        )
    rows.sort(key=lambda r: r['train_end'], reverse=True)
    chosen = rows[0]
    logger.info(f"  h={horizon}: tuned hyperparameters from fold {chosen['fold_id']} "
                f"(train_end={chosen['train_end']}): {chosen['hyperparameters']}")
    return chosen['hyperparameters']


def main():
    config = Config()
    config.run_id = PRIMARY_RUN_ID
    logger = setup_logging(log_dir=Path('outputs/runs') / PRIMARY_RUN_ID / 'logs', run_id='live_nowcast_demo')
    logger.info("=" * 60)
    logger.info("Script 27: live-nowcast demonstration (mechanism, not a new accuracy claim)")
    logger.info("=" * 60)

    processed_dir = Path('data/processed')
    X_A3 = load_processed_data(processed_dir / 'X_A3')
    y_log = load_processed_data(processed_dir / 'y')['target']  # log-transformed (src/features/engineering.py)
    common_idx = X_A3.index.intersection(y_log.index)
    X_A3, y_log = X_A3.loc[common_idx].sort_index(), y_log.loc[common_idx].sort_index()

    origin_quarter = X_A3.index.max()
    logger.info(f"Current forecast origin (last quarter with a real published target): {origin_quarter.date()}")
    logger.info(f"Origin row's own features are all real (GDP, grid, mobility, weather - fully observed by then)")

    grid_features = pd.read_parquet(processed_dir / 'grid_quarterly_features.parquet')
    future_grid_available = grid_features.index[grid_features.index > origin_quarter]
    logger.info(f"Grid data already cached BEYOND the current origin (available before the macro/target series "
                f"catches up): {[q.date() for q in future_grid_available]}")

    y_by_horizon = create_direct_horizon_targets(y_log, HORIZONS)

    results = []
    for h in HORIZONS:
        y_h = y_by_horizon[h].dropna()
        train_idx = X_A3.index.intersection(y_h.index)
        X_train, y_train = X_A3.loc[train_idx], y_h.loc[train_idx]

        hyperparams = load_horizon_hyperparameters(config.run_dir, CONFIGURATION, FS_OPTION, MODEL_NAME, h, logger)
        model = ModelRegistry.create(MODEL_NAME, hyperparams)
        model.fit(X_train, y_train)

        X_origin = X_A3.loc[[origin_quarter]]
        pred_log = float(model.predict(X_origin)[0])
        pred_raw = float(np.exp(pred_log))  # invert create_target_variable's np.log (not log1p)
        target_quarter = origin_quarter + pd.DateOffset(months=3 * h)

        results.append({
            'origin_quarter': origin_quarter.date(), 'horizon': h,
            'target_quarter': target_quarter.date(),
            'point_forecast_co2e_thousand_tonnes': pred_raw,
            'n_training_pairs': len(X_train),
            'all_origin_inputs_real': True,
        })
        logger.info(f"  H{h} forecast for {target_quarter.date()}: {pred_raw:,.0f} thousand tonnes CO2e "
                    f"(trained on {len(X_train)} direct (X_t, y_t+{h}) pairs)")

    result_df = pd.DataFrame(results)
    Path('outputs/robustness').mkdir(parents=True, exist_ok=True)
    result_df.to_csv('outputs/robustness/live_nowcast_demo.csv', index=False)

    print("\nLive-nowcast demonstration - NOT a scored accuracy result (no ground truth exists yet "
          "for these target quarters):")
    print(result_df.to_string(index=False))
    print(f"\nEvery input used is real (origin quarter {origin_quarter.date()} is fully observed) - "
          f"no placeholder values were needed, because this pipeline forecasts DIRECTLY from a known "
          f"origin to a future target (X_t -> y_t+h), never recursively through synthesized future rows.")
    if len(future_grid_available):
        print(f"\nSeparately: real grid data already exists for {len(future_grid_available)} quarter(s) beyond "
              f"the current origin ({[q.date() for q in future_grid_available]}) - concrete evidence that grid "
              f"data updates before the macro/target series does, which is what motivates the early-warning "
              f"framing in RESEARCH_OVERVIEW.md/Discussion 5.5, not a claim used in the forecasts above.")
    print("\nWrote outputs/robustness/live_nowcast_demo.csv")


if __name__ == '__main__':
    main()
