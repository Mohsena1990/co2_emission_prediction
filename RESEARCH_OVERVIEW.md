# Research overview: core idea, algorithm design, policy interpretability

This file gives a top-down reading of the project for anyone who lands in
the repository without having read the manuscript first: what problem it
solves, how the pipeline is built to solve it honestly, and what the
result is actually good for outside a forecasting-accuracy leaderboard.

## 1. Core idea

National CO2 inventories are annual and lag reality by up to a year.
Quarterly forecasting is more timely, but conventional quarterly predictors
(GDP, population, weather, historical emissions) describe the *scale* of
economic activity, not the *operating state* of the energy system that
generates most of those emissions. Half-hourly electricity-grid data
(carbon intensity, generation mix) is available in near-real time and
describes exactly that operating state - but adding it is not automatically
an improvement: a richer predictor pool can leak information, encode
short-lived correlations, or simply add noise.

The core idea is therefore a **paired test**, not a single model: does
grid information improve quarterly UK CO2 forecasting once every
predictor's timing, lag and target-dependence is audited (so no result can
be a disguised copy of the target), and once the comparison is made against
both a plain baseline and an engineered-features-only alternative on
*identical* forecast dates? The A1-A4 configuration design (Table 3) exists
specifically to isolate this question from "more features are usually
better" - A3 (baseline + grid) beating A2 (baseline + engineering) is the
single comparison the whole project is built to make trustworthy.

## 2. Algorithm design

Three design choices follow directly from the core idea, not from wanting
a bigger model:

**Two independent streams, not one pipeline.** Stream A asks whether an
information source helps *before* any feature selection touches it; Stream
B asks whether five different selection philosophies (linear stability,
wrapper, XGBoost-SHAP, permutation, consensus) can improve on that. Running
them separately means a good B-stream result can't be mistaken for
evidence about the raw information content of a pool, and vice versa.

**Nesting, not just holding out a test set.** Every operation that learns
from data - scaling, feature selection, PSO hyperparameter search - is
re-estimated inside each outer expanding-window training fold, never once
on the full series. This is the difference between "the model would have
forecast well in real time" and "the model forecasts the past well because
it partially saw the future during tuning." `tests/test_leakage_sentinel.py`,
`tests/test_nested_cv_isolation.py` and `tests/test_pso_isolation.py` exist
to make this a checked invariant, not a documentation claim.

**Decision by Pareto + VIKOR, not by the single best accuracy number.**
Twenty (Stream A) and 100 (Stream B) candidates are scored on accuracy,
worst-horizon robustness, fold-to-fold stability, parsimony and runtime
simultaneously. Pareto filtering removes strictly dominated candidates;
VIKOR then picks a transparent compromise among what remains, under
weighting schemes that are themselves tested for sensitivity (Fig. 6).
This is what stops "pick whichever model has the lowest WMASE" from being
an implicit, unexamined value judgement.

The result of this design, not a model choice made in isolation, is what
makes the champion (A3/LightGBM) and its interpretation defensible: it won
under an audited feature set, inside nested validation, against a
multi-criterion decision rule - and every one of those three properties is
independently checked in `tests/`.

## 3. What the algorithm's output adds for policy and economic interpretability

A forecasting pipeline's accuracy metrics answer "does this work." They
don't answer "who should care and why." That gap is where this project's
interpretability layer (TreeSHAP + regime analysis + the ablation studies
in `scripts/19`, `22`, `23`, `24`) earns its place, and where the added
policy material sits:

- **From attribution to mechanism.** TreeSHAP alone says which features the
  model uses. The family-level ablation (`scripts/19`) and the
  single-statistic decomposition (`scripts/22`, added in this pass) turn
  that into a mechanism claim: the electricity-grid gain comes from the
  *combination* of carbon-intensity statistics, not from any one number,
  and specifically not from generation-mix shares alone (which
  underperform having no grid data at all). That distinction is what
  makes a "capacity vs. flexibility" policy framing defensible rather than
  a post-hoc story fitted to a single SHAP bar chart - see
  `LIMITATIONS.md`'s decomposition section for exactly how far the
  dispersion claim can and cannot be pushed.
- **From mechanism to institution.** `outputs/figures/pdf/main/fig_policy_stakeholder_map.pdf`
  and `fig_policy_capacity_vs_flexibility.pdf` (§4 below) translate that
  mechanism into who would use it: a system operator watching flexibility
  adequacy, a climate-delivery body watching in-year divergence from a
  carbon budget, a market participant pricing transition risk, a
  statistical body cross-checking an in-year estimate. None of this
  changes the forecasting result; it states plainly what the result is
  *for*, which the accuracy tables alone do not.
- **From institution back to honesty about limits.** Every use case above
  is stated as a monitoring input, not a compliance or investment
  instrument, and is paired with the same caveats the manuscript already
  applies to SHAP (predictive attribution, not causal effect) and to
  sample size (27 paired forecast origins). The policy layer is only as
  credible as the statistics underneath it - which is why `LIMITATIONS.md`
  and this file are meant to be read together, not the policy figures in
  isolation.

## 4. Where this lives in the repository

| Artifact | What it is |
|---|---|
| `scripts/19_grid_ablation.py` | Family-level ablation (CI vs. generation-mix vs. full A3), now reconciled with the headline champion numbers |
| `scripts/22_ci_decomposition_ablation.py` | Single-statistic decomposition within the carbon-intensity family |
| `scripts/23_per_horizon_significance.py` | Per-horizon significance surfaced alongside the pooled Table 12 result |
| `scripts/24_policy_figures.py` | The two policy/managerial figures described above |
| `LIMITATIONS.md` | Canonical, consolidated list of what these results do and do not support |
| `CHANGELOG.md` | What changed, when, and why |
