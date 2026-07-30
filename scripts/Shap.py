#!/usr/bin/env python
"""
Script 06: Multi-Model SHAP Interpretation
==========================================
Run SHAP interpretation, regime analysis, and seasonal analysis for
tree-based models across selected FS options.

Target models:
    - random_forest
    - catboost
    - lightgbm

Target FS options:
    - fs_wrapper
    - fs_linear

Regimes:
    - pre_covid  : before 2020-01-01
    - covid      : 2020-01-01 to 2021-12-31
    - post_covid : from 2022-01-01 onward

Usage:
    python scripts/06_interpret_champion.py [--config CONFIG_PATH] [--run-id RUN_ID]

Outputs:
    - outputs/runs/<run_id>/tables/multi_model_shap_interpretation.json
    - outputs/runs/<run_id>/tables/top_drivers_<fs>_<model>.csv
    - outputs/runs/<run_id>/tables/regime_importance_<fs>_<model>_<regime>.csv
    - outputs/runs/<run_id>/tables/seasonal_importance_<fs>_<model>.csv
    - outputs/runs/<run_id>/figures/shap_summary_<fs>_<model>.png
    - outputs/runs/<run_id>/figures/feature_importance_<fs>_<model>.png
    - outputs/runs/<run_id>/figures/shap_regime_compare_<fs>_<model>.png
    - outputs/runs/<run_id>/figures/seasonal_leverage_<fs>_<model>.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.core import (
    Config,
    create_run_directories,
    setup_logging,
    set_seed,
    save_json_numpy,
    load_json,
    get_latest_run_id,
)
from src.data_io import load_processed_data
from src.models import ModelRegistry
from src.interpretability import (
    compute_shap_values,
    get_feature_importance_from_shap,
    analyze_regime_shap,
    analyze_seasonal_shap,
    compute_permutation_importance,
)
from src.reporting import (
    plot_feature_importance,
    plot_regime_comparison,
    set_plot_style,
)


# =========================================================
# Helpers
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-model SHAP interpretation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def resolve_processed_dir():
    """
    Resolve processed data directory using known project path first,
    then fallback to relative path.
    """
    candidates = [
        Path(r"C:\Users\Admin\energy_consumption\data\processed"),
        Path("data/processed"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Processed data directory not found. Checked:\n" +
        "\n".join(str(p) for p in candidates)
    )


def resolve_multi_fs_path(config, dirs):
    """
    Resolve multi_fs_optimization_results.json.
    Priority:
      1) current run tables dir
      2) known run from your project
      3) latest available run containing the file
    """
    candidates = [
        dirs["tables"] / "multi_fs_optimization_results.json",
        Path("outputs") / "runs" / "run_20260203_1347" / "tables" / "multi_fs_optimization_results.json",
    ]

    for p in candidates:
        if p.exists():
            return p

    runs_dir = Path(config.output.base_dir)
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.glob("run_*"), reverse=True):
            candidate = run_dir / "tables" / "multi_fs_optimization_results.json"
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "multi_fs_optimization_results.json not found in current run, known run, or latest runs."
    )


def infer_run_id_from_path(path_obj: Path):
    """
    Extract run_id from a path like outputs/runs/run_20260203_1347/tables/...
    """
    for part in path_obj.parts:
        if part.startswith("run_"):
            return part
    return None


def resolve_model_path(run_id: str, fs_option: str, model_name: str):
    """
    Build and validate model path.
    """
    model_path = Path("outputs") / "runs" / run_id / "models" / fs_option / f"{model_name}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path


def load_y_series(y_obj):
    """
    Robustly extract target series from loaded y object.
    """
    if isinstance(y_obj, pd.Series):
        return y_obj
    if isinstance(y_obj, pd.DataFrame):
        if "target" in y_obj.columns:
            return y_obj["target"]
        if y_obj.shape[1] == 1:
            return y_obj.iloc[:, 0]
        raise ValueError("Could not infer target column from y DataFrame.")
    raise TypeError(f"Unsupported y object type: {type(y_obj)}")


def quarter_str_to_timestamp(value):
    """
    Convert strings like '2020Q1' or '2020-Q1' to pandas Timestamp at quarter start.
    """
    if pd.isna(value):
        return pd.NaT

    s = str(value).strip().upper().replace("-", "")
    if "Q" in s:
        try:
            year = int(s[:4])
            q = int(s.split("Q")[1])
            month_map = {1: 1, 2: 4, 3: 7, 4: 10}
            return pd.Timestamp(year=year, month=month_map[q], day=1)
        except Exception:
            return pd.NaT

    try:
        return pd.to_datetime(value)
    except Exception:
        return pd.NaT


def attach_datetime_index(X, y, processed_dir, logger):
    """
    Ensure X and y share a datetime-like index suitable for regime splitting.
    Strategy:
      1) If X index already datetime -> use it
      2) Else if X has a Quarter column -> parse it
      3) Else if df_clean has a Quarter column and matching length -> use it
      4) Else try converting current index directly
    """
    X = X.copy()
    y = y.copy()

    # Case 1: X index already datetime
    if isinstance(X.index, pd.DatetimeIndex):
        return X, y

    # Case 2: Quarter column inside X
    if "Quarter" in X.columns:
        dt_index = X["Quarter"].apply(quarter_str_to_timestamp)
        X.index = pd.DatetimeIndex(dt_index)
        y.index = X.index
        X = X.drop(columns=["Quarter"], errors="ignore")
        return X, y

    # Case 3: use df_clean if available
    try:
        df_clean = load_processed_data(processed_dir / "df_clean")
        if isinstance(df_clean, pd.DataFrame) and "Quarter" in df_clean.columns and len(df_clean) >= len(X):
            dt_index = df_clean["Quarter"].iloc[:len(X)].apply(quarter_str_to_timestamp)
            X.index = pd.DatetimeIndex(dt_index)
            y.index = X.index
            return X, y
    except Exception as e:
        logger.warning(f"Could not use df_clean to construct datetime index: {e}")

    # Case 4: try converting existing index
    try:
        dt_index = pd.to_datetime(X.index)
        X.index = pd.DatetimeIndex(dt_index)
        y.index = X.index
        return X, y
    except Exception:
        raise ValueError(
            "Could not construct a datetime index for regime splitting. "
            "Please verify X_full / df_clean contain quarterly time information."
        )


def safe_to_records(df):
    if df is None:
        return []
    return df.replace({np.nan: None}).to_dict(orient="records")


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config()

    if args.run_id:
        config.run_id = args.run_id
    else:
        latest = get_latest_run_id(config.output.base_dir)
        if latest:
            config.run_id = latest

    set_seed(config.seed)
    dirs = create_run_directories(config)

    logger = setup_logging(log_dir=dirs["logs"], run_id=config.run_id)
    set_plot_style()

    logger.info("=" * 70)
    logger.info("Script 06: Multi-Model SHAP Interpretation")
    logger.info("=" * 70)

    # -----------------------------------------------------
    # Load metadata
    # -----------------------------------------------------
    processed_dir = resolve_processed_dir()
    logger.info(f"Using processed data directory: {processed_dir}")

    multi_fs_path = resolve_multi_fs_path(config, dirs)
    logger.info(f"Using FS metadata file: {multi_fs_path}")

    multi_fs_info = load_json(multi_fs_path)
    source_run_id = infer_run_id_from_path(multi_fs_path)
    if source_run_id is None:
        raise ValueError("Could not infer source run_id from multi_fs_optimization_results.json path.")
    logger.info(f"Using model run: {source_run_id}")

    fs_results = multi_fs_info.get("fs_results", {})
    if not fs_results:
        raise ValueError("No fs_results found in multi_fs_optimization_results.json")

    # Enforce requested order
    fs_order = ["fs_wrapper", "fs_linear"]
    model_order = ["random_forest", "catboost", "lightgbm"]

    available_fs = [fs for fs in fs_order if fs in fs_results]
    if not available_fs:
        raise ValueError("Neither fs_wrapper nor fs_linear found in fs_results.")

    logger.info(f"FS options to process: {available_fs}")
    logger.info(f"Models to process (priority order): {model_order}")

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    logger.info("Loading processed data...")
    X_full = load_processed_data(processed_dir / "X_full")
    y_loaded = load_processed_data(processed_dir / "y")
    y = load_y_series(y_loaded)

    if not isinstance(X_full, pd.DataFrame):
        raise TypeError(f"X_full must be a DataFrame, got {type(X_full)}")

    # Align once
    common_idx = X_full.index.intersection(y.index)
    X_full = X_full.loc[common_idx].copy()
    y = y.loc[common_idx].copy()

    # Attach datetime index for regime splitting
    X_full, y = attach_datetime_index(X_full, y, processed_dir, logger)

    # Drop rows with missing datetime index
    valid_time_mask = ~pd.isna(X_full.index)
    X_full = X_full.loc[valid_time_mask].copy()
    y = y.loc[valid_time_mask].copy()

    logger.info(f"Base dataset: {len(X_full)} rows, {len(X_full.columns)} columns")

    all_results = {
        "source_run_id": source_run_id,
        "processed_dir": str(processed_dir),
        "fs_options_processed": available_fs,
        "models_processed": model_order,
        "regimes": {
            "pre_covid": {"start": None, "end": "2020-01-01"},
            "covid": {"start": "2020-01-01", "end": "2022-01-01"},
            "post_covid": {"start": "2022-01-01", "end": None},
        },
        "results": {}
    }

    # -----------------------------------------------------
    # Loop over FS options
    # -----------------------------------------------------
    for fs_option in available_fs:
        logger.info("-" * 70)
        logger.info(f"Processing FS option: {fs_option}")

        fs_info = fs_results[fs_option]
        selected_features = fs_info.get("features", [])
        trained_models = fs_info.get("trained_models", [])

        available_features = [f for f in selected_features if f in X_full.columns]
        missing_features = [f for f in selected_features if f not in X_full.columns]

        if not available_features:
            logger.warning(f"No available features found for {fs_option}; skipping.")
            continue

        if missing_features:
            logger.warning(f"{fs_option}: missing features ignored: {missing_features}")

        X_fs = X_full[available_features].copy()

        # Drop NA rows for this FS
        valid_mask = ~(X_fs.isnull().any(axis=1) | y.isnull())
        X_fs = X_fs.loc[valid_mask].copy()
        y_fs = y.loc[valid_mask].copy()

        logger.info(f"{fs_option}: {len(X_fs)} samples, {len(X_fs.columns)} features")

        all_results["results"][fs_option] = {
            "selected_features": available_features,
            "missing_features": missing_features,
            "models": {}
        }

        # -------------------------------------------------
        # Loop over target models
        # -------------------------------------------------
        for model_name in model_order:
            logger.info("." * 70)
            logger.info(f"Processing model: {model_name} under {fs_option}")

            model_result = {
                "model_name": model_name,
                "fs_option": fs_option,
                "n_samples": len(X_fs),
                "n_features": len(X_fs.columns),
                "selected_features": available_features,
                "status": "not_run",
            }

            if model_name not in trained_models:
                logger.warning(f"{model_name} was not trained for {fs_option}; skipping.")
                model_result["status"] = "skipped_not_trained"
                all_results["results"][fs_option]["models"][model_name] = model_result
                continue

            try:
                model_path = resolve_model_path(source_run_id, fs_option, model_name)
                logger.info(f"Loading model from: {model_path}")

                model = ModelRegistry.get(model_name).load(model_path)
                model_result["model_path"] = str(model_path)
                model_result["status"] = "loaded"

                # -----------------------------------------
                # SHAP analysis
                # -----------------------------------------
                try:
                    import shap

                    logger.info("Computing SHAP values...")
                    shap_values, explainer = compute_shap_values(model, X_fs, "tree")
                    model_result["interpretation_method"] = "shap"

                    importance_df = get_feature_importance_from_shap(shap_values, list(X_fs.columns))
                    model_result["shap_importance"] = safe_to_records(importance_df)

                    # Save top drivers
                    top_drivers_path = dirs["tables"] / f"top_drivers_{fs_option}_{model_name}.csv"
                    importance_df.to_csv(top_drivers_path, index=False)

                    # SHAP beeswarm
                    shap_summary_path = dirs["figures"] / f"shap_summary_{fs_option}_{model_name}.png"
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(shap_values, X_fs, max_display=15, show=False)
                    plt.title(f"SHAP Summary - {fs_option}/{model_name}", fontweight="bold")
                    plt.tight_layout()
                    plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
                    plt.close()

                    # Feature importance plot
                    feature_importance_path = dirs["figures"] / f"feature_importance_{fs_option}_{model_name}.png"
                    plot_feature_importance(
                        importance_df,
                        title=f"Feature Importance (SHAP) - {fs_option}/{model_name}",
                        output_path=feature_importance_path
                    )

                    model_result["top_drivers_csv"] = str(top_drivers_path)
                    model_result["shap_summary_figure"] = str(shap_summary_path)
                    model_result["feature_importance_figure"] = str(feature_importance_path)

                except Exception as e:
                    logger.warning(f"SHAP failed for {fs_option}/{model_name}: {e}")
                    logger.info("Falling back to permutation importance...")

                    perm_df = compute_permutation_importance(model, X_fs, y_fs)
                    model_result["interpretation_method"] = "permutation"
                    model_result["permutation_importance"] = safe_to_records(perm_df)

                    perm_path = dirs["tables"] / f"top_drivers_{fs_option}_{model_name}.csv"
                    perm_df.to_csv(perm_path, index=False)

                    plot_feature_importance(
                        perm_df.rename(columns={"importance_mean": "importance"}),
                        title=f"Permutation Importance - {fs_option}/{model_name}",
                        output_path=dirs["figures"] / f"feature_importance_{fs_option}_{model_name}.png"
                    )

                    model_result["top_drivers_csv"] = str(perm_path)
                    model_result["feature_importance_figure"] = str(
                        dirs["figures"] / f"feature_importance_{fs_option}_{model_name}.png"
                    )

                # -----------------------------------------
                # Regime analysis
                # -----------------------------------------
                try:
                    logger.info("Running regime analysis...")
                    regime_results = analyze_regime_shap(
                        model,
                        X_fs,
                        regime_periods={
                            "pre_covid": (None, "2020-01-01"),
                            "covid": ("2020-01-01", "2022-01-01"),
                            "post_covid": ("2022-01-01", None),
                        },
                        model_type="tree"
                    )

                    model_result["regime_analysis"] = {}

                    if regime_results:
                        for regime_name, regime_df in regime_results.items():
                            regime_csv = dirs["tables"] / f"regime_importance_{fs_option}_{model_name}_{regime_name}.csv"
                            regime_df.to_csv(regime_csv, index=False)
                            model_result["regime_analysis"][regime_name] = {
                                "csv": str(regime_csv),
                                "data": safe_to_records(regime_df)
                            }

                        regime_fig = dirs["figures"] / f"shap_regime_compare_{fs_option}_{model_name}.png"
                        plot_regime_comparison(
                            regime_results,
                            title=f"Feature Importance by Regime - {fs_option}/{model_name}",
                            output_path=regime_fig
                        )
                        model_result["regime_figure"] = str(regime_fig)

                except Exception as e:
                    logger.warning(f"Regime analysis failed for {fs_option}/{model_name}: {e}")
                    model_result["regime_analysis_error"] = str(e)

                # -----------------------------------------
                # Seasonal analysis
                # -----------------------------------------
                try:
                    logger.info("Running seasonal analysis...")
                    seasonal_df = analyze_seasonal_shap(model, X_fs, model_type="tree")

                    if isinstance(seasonal_df, pd.DataFrame) and len(seasonal_df) > 0:
                        seasonal_csv = dirs["tables"] / f"seasonal_importance_{fs_option}_{model_name}.csv"
                        seasonal_df.to_csv(seasonal_csv, index=False)

                        model_result["seasonal_analysis"] = {
                            "csv": str(seasonal_csv),
                            "data": safe_to_records(seasonal_df)
                        }

                        if {"feature", "quarter", "importance"}.issubset(set(seasonal_df.columns)):
                            fig, ax = plt.subplots(figsize=(12, 6))

                            pivot = seasonal_df.pivot(
                                index="feature",
                                columns="quarter",
                                values="importance"
                            )

                            top_features = (
                                seasonal_df.groupby("feature")["importance"]
                                .mean()
                                .nlargest(10)
                                .index
                            )

                            pivot.loc[top_features].plot(kind="bar", ax=ax, width=0.8)
                            ax.set_xlabel("Feature")
                            ax.set_ylabel("Importance")
                            ax.set_title(
                                f"Seasonal Feature Leverage (Top 10 Features) - {fs_option}/{model_name}",
                                fontweight="bold"
                            )
                            ax.legend(title="Quarter")
                            plt.xticks(rotation=45, ha="right")
                            plt.tight_layout()

                            seasonal_fig = dirs["figures"] / f"seasonal_leverage_{fs_option}_{model_name}.png"
                            plt.savefig(seasonal_fig, dpi=300, bbox_inches="tight")
                            plt.close()

                            model_result["seasonal_figure"] = str(seasonal_fig)

                except Exception as e:
                    logger.warning(f"Seasonal analysis failed for {fs_option}/{model_name}: {e}")
                    model_result["seasonal_analysis_error"] = str(e)

                model_result["status"] = "completed"
                all_results["results"][fs_option]["models"][model_name] = model_result

            except Exception as e:
                logger.error(f"Failed for {fs_option}/{model_name}: {e}")
                model_result["status"] = "failed"
                model_result["error"] = str(e)
                all_results["results"][fs_option]["models"][model_name] = model_result

    # -----------------------------------------------------
    # Save master summary
    # -----------------------------------------------------
    summary_path = dirs["tables"] / "multi_model_shap_interpretation.json"
    save_json_numpy(all_results, summary_path)

    logger.info("=" * 70)
    logger.info("Multi-model SHAP interpretation complete.")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Tables saved to: {dirs['tables']}")
    logger.info(f"Figures saved to: {dirs['figures']}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())