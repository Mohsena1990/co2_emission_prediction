# Diagnosis: all-zero, all-identical regime SHAP for the A3/LightGBM champion

## Summary

The saved regime-specific SHAP output for the A3/LightGBM champion
(`outputs/runs/my_run/interpretability/best_A/regime_importance.csv`) contained
exactly `0.0` for every feature in every regime (Pre-COVID, COVID, Post-COVID), and
the "top predictors" reported in `outputs/tables/table9_covid_regime_interpretation.csv`
were identical across all three regimes (`GDP, Population, Air_Temp, Rainfall,
COVID_Deaths` in that exact order, every time).

**Root cause: the interpretability refit used library-default hyperparameters
instead of the cell's actual PSO-tuned hyperparameters, which — for this small
sample — makes LightGBM collapse to a constant predictor.** This is not a SHAP bug,
not a data/index bug, and not a saving/aggregation bug: SHAP correctly reported
zero attribution because the model genuinely did not use any feature.

## How this was diagnosed

The audit worked through the failure-mode checklist explicitly rather than
guessing:

| Candidate cause | Checked | Verdict |
|---|---|---|
| Wrong feature matrix | Loaded the exact matrix `_load_winner_matrix` builds (A3, common period, 28 rows × 23 features) and confirmed columns/values match `data/processed/X_A3.parquet` after `slice_configurations_to_common_period`. | Ruled out |
| Transformed vs untransformed feature mismatch | Confirmed LightGBM/RandomForest/CatBoost wrappers fit directly on raw `X.values` (no internal scaler) — `src/models/traditional.py`. So this specific mismatch does not apply to the LightGBM champion. **However, a real (latent, not-yet-triggered) instance of exactly this bug was found for Ridge**: `RidgeModel.fit` scales X via an internal `StandardScaler` before fitting, but `compute_shap_values`'s `'linear'` branch passed the *raw* `X` to `shap.LinearExplainer(model.model, X)`. Ridge is not the current champion so this did not produce the reported zeros, but it was fixed anyway (see Fixes below) since Ridge could become a future Stream A/B winner. |
| Using the wrong model object | **This is the actual root cause** — see below. |
| Wrong SHAP explainer | `compute_shap_values` correctly dispatches LightGBM/RandomForest/CatBoost to `shap.TreeExplainer(model.model)`. Confirmed correct explainer class was used. | Ruled out |
| Incorrect background/reference data | TreeExplainer's path-dependent default does not require a background set; not applicable here. | Ruled out |
| Misaligned feature names | Confirmed `X.columns` order into `compute_shap_values` matches `model.feature_names` set during `.fit()`. | Ruled out |
| Prediction-horizon mismatch | The interpretability refit is a single full-sample fit, not horizon-specific; `y` is the untransformed-index-aligned CO2e series in both the champion evaluation and this refit. | Ruled out |
| Saving/loading corruption | The bug reproduces live, in-memory, before anything is written to disk (see repro below). | Ruled out |
| Zero-filled dataframe initialization | `get_feature_importance_from_shap` computes `np.mean(np.abs(shap_values), axis=0)` from the real SHAP array — it is not pre-filled with zeros; the *input* SHAP array itself is all zero. | Ruled out (symptom, not cause) |
| Incorrect aggregation | Same as above — aggregation is arithmetically correct given all-zero input. | Ruled out |
| Regime-filtering after wrong index reset | `analyze_regime_shap`'s date-mask filtering (`X.index >= start`) was checked against `X_config`'s DatetimeIndex and produces the correct row counts per regime (7 / 8 / 13, matching `regime_importance.csv`'s own `n_samples` column). | Ruled out |
| SHAP output shape mismatch / multiclass-style indexing | `shap.TreeExplainer(...).shap_values(X)` returned a plain `(n_samples, n_features)` ndarray for this regressor — no multiclass list-of-arrays behavior in this SHAP/LightGBM version pairing (shap 0.49.1, lightgbm 4.6.0). | Ruled out |
| Fold/model overwrite | Each of the three winners (Best_A/Best_B/Best_Overall) gets its own model instance inside `interpret_winner`'s per-winner loop; no shared mutable state across winners. | Ruled out |

## Root cause, in detail

`scripts/12_interpretability_and_sensitivity.py::interpret_winner` (pre-fix, line
190) created every one of the 5 models for the cross-model importance table via:

```python
model = ModelRegistry.create(m, {})   # <-- empty params dict
model.fit(X_config, y)
```

This ignores the actual PSO-tuned hyperparameters that produced the champion's
real, reported WMASE = 0.454 during Stream A evaluation (`run_configuration_model`
tunes hyperparameters per outer fold via `optimize_model_nested`, and those tuned
values are exactly what generated the walk-forward forecasts underlying Table 7).
Instead, `interpret_winner` silently refit with **library defaults**.

For LightGBM, the default `min_child_samples=20` combined with the A3
common-period sample (28 rows total; regime subsets of 7/8/13 rows) means **no
split in any tree can satisfy the min-leaf-size constraint** (any split of a
28-row node puts fewer than 20 rows on at least one side). Every tree in the
ensemble degenerates to a single root leaf — i.e. a constant. Live repro:

```
model = ModelRegistry.create('lightgbm', {})   # default params
model.fit(X_A3, y)
preds = model.predict(X_A3)
# preds == [11.58206674, 11.58206674, ...]  (std ≈ 1.77e-15)

shap_values = shap.TreeExplainer(model.model).shap_values(X_A3)
# np.abs(shap_values).mean() == 0.0  exactly, for every feature
```

Since the model's output does not depend on any input feature, TreeSHAP correctly
attributes exactly `0.0` to every feature — SHAP is not malfunctioning, it is
faithfully reporting that a broken (untuned) model uses no information. Because
every value is identically `0.0`, `get_feature_importance_from_shap`'s
`sort_values('importance', ascending=False)` is a no-op stable sort that leaves
features in their original `X.columns` insertion order — which is exactly why the
"top predictors" appeared identical, in the identical order, across all three
regimes: it isn't a ranking at all, it's just the DataFrame's column order leaking
through a degenerate sort.

## Fix applied

1. **`scripts/12_interpretability_and_sensitivity.py`**: new helper
   `_load_cell_hyperparameters(configuration, fs_option, model_name, config,
   logger)` reads the real tuned hyperparameters back out of
   `fold_predictions/stream_{A,B}/provenance.json` (written by
   `run_configuration_model`'s per-fold `fold_provenance` records, which already
   include a `hyperparameters` field per outer fold). Since `nested_retuning=True`
   tunes independently per outer fold, there is no single canonical "cell
   best_params" — the fix picks the fold with the latest `train_end` (the most
   historical data, closest to a deployment/interpretability fit) as the
   representative setting. `interpret_winner` now calls this for **every** model
   (not just the champion), so Table 11's cross-model comparison is also on tuned
   models, not defaults. If no provenance row exists for a requested cell, the
   function raises loudly instead of silently falling back to `{}` (the original
   silent fallback is exactly what caused this bug).
2. **`src/interpretability/shap_analysis.py`**: fixed the related-but-not-yet-
   triggered Ridge/LinearExplainer scale mismatch — `compute_shap_values`'s
   `'linear'` branch now transforms `X` through the wrapper's own `scaler` before
   building/calling `LinearExplainer`, matching what `RidgeModel.fit` actually
   trained on.
3. **`src/interpretability/integrity_checks.py`** (new): `compute_regime_shap_with_integrity`
   computes regime SHAP and, in the same pass, checks for exactly the failure
   modes above (all-zero, identical-across-regimes, row/feature-count mismatches,
   NaN/Inf, and an additivity reconstruction check against the explainer's
   `expected_value` where supported) — see `outputs/audit/shap_integrity_checks.csv`
   and section 5 of this audit for results on the corrected rerun.

## Verdict

Category, per the audit checklist in the task brief: **"wrong model object"** —
specifically, an untuned/default-hyperparameter model instance was substituted for
the actual validated champion model at the interpretability stage, without the
scale-of-damage (complete constant-prediction collapse) being caught because the
resulting SHAP values were technically valid (correctly zero for a model that uses
no information) rather than throwing an exception.
