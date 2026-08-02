"""
Nested (outer/inner) expanding-window cross-validation.

STRUCTURAL FIX (bug 3.5 / spec section 11): the pre-existing pipeline had
only a single, flat CVPlan. Feature selection and PSO hyperparameter tuning
were evaluated on the *same* folds later used to report "final" performance
(see scripts/03_optimize_models.py + scripts/04_evaluate_and_safeguards.py,
both consuming the one shared cv_plan produced by scripts/00_make_dataset.py)
- i.e. there was no separation between model-selection data and held-out
evaluation data at all.

This module adds that separation explicitly:

  - OUTER folds: expanding-window splits of the full series, used ONLY to
    estimate final out-of-sample performance. Built with
    `create_walk_forward_splits` (unchanged - existing tests keep passing).
  - INNER folds: for each outer fold, an independent expanding-window CVPlan
    built strictly from `X.iloc[outer_fold.train_indices]` - i.e. the inner
    plan's index space is local to the outer training slice and can never
    reference an outer test row. Inner folds are used ONLY for feature
    selection, PSO tuning, threshold selection, and other model-selection
    choices.

No outer test row is ever visible to code that only sees a NestedFold's
`inner_cv_plan` and the corresponding `X_outer_train`/`y_outer_train` slices.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from ..core.config import SplitConfig
from ..core.logging_utils import get_logger
from .walk_forward import CVFold, CVPlan, create_walk_forward_splits


@dataclass
class NestedFold:
    """One outer fold together with its self-contained inner CV plan."""
    outer_fold_id: int
    outer: CVFold
    inner_cv_plan: CVPlan

    def __repr__(self):
        return (f"NestedFold(outer_fold={self.outer_fold_id}, "
                f"outer={self.outer!r}, inner_folds={self.inner_cv_plan.n_folds})")


def create_nested_walk_forward_splits(
    X: pd.DataFrame,
    y: pd.Series,
    config: SplitConfig,
    horizons: Optional[List[int]] = None
) -> List[NestedFold]:
    """
    Build nested outer/inner expanding-window CV.

    Args:
        X: Full feature DataFrame (datetime index).
        y: Full target Series (same index as X).
        config: SplitConfig - uses `min_train_size`/`test_size` for the
            OUTER split and `inner_min_train_size`/`inner_test_size` for each
            outer fold's INNER split.
        horizons: Horizons to build outer folds for (default: config.horizons).

    Returns:
        List of NestedFold, one per outer fold. Each NestedFold's
        `inner_cv_plan` indices are relative to
        `X.iloc[outer.train_indices]` / `y.iloc[outer.train_indices]` - NOT
        the original X/y - callers must slice the outer-training data first:

            X_outer_train = X.iloc[nested_fold.outer.train_indices]
            y_outer_train = y.iloc[nested_fold.outer.train_indices]
            for X_in_tr, y_in_tr, X_in_te, y_in_te, inner_fold in \\
                    generate_cv_folds(X_outer_train, y_outer_train, nested_fold.inner_cv_plan):
                ...
    """
    logger = get_logger()
    outer_plan = create_walk_forward_splits(X, y, config, horizons)

    inner_config = SplitConfig(
        method=config.inner_scheme,
        min_train_size=config.inner_min_train_size,
        test_size=config.inner_test_size,
        horizons=config.horizons,
        horizon_weights=config.horizon_weights,
    )

    nested_folds = []
    skipped = 0
    for outer_fold in outer_plan.folds:
        X_outer_train = X.iloc[outer_fold.train_indices]
        y_outer_train = y.iloc[outer_fold.train_indices]

        # Inner folds are only built for this outer fold's own horizon -
        # PSO/FS tuning for an H2 outer fold should validate at H2 too.
        inner_plan = create_walk_forward_splits(
            X_outer_train, y_outer_train, inner_config, horizons=[outer_fold.horizon]
        )

        if inner_plan.n_folds == 0:
            # Outer training window too short for even one inner fold at
            # this horizon - skip rather than silently running PSO/FS with
            # zero validation folds (which would fall back to full-sample
            # fitting and reintroduce the isolation bug this module fixes).
            skipped += 1
            logger.warning(
                f"Outer fold {outer_fold.fold_id} (h={outer_fold.horizon}): "
                f"outer training window ({len(outer_fold.train_indices)} rows) "
                f"too short to build any inner fold with "
                f"inner_min_train_size={config.inner_min_train_size}, "
                f"inner_test_size={config.inner_test_size}. Skipping this "
                f"outer fold for inner-tuned methods."
            )
            continue

        nested_folds.append(NestedFold(
            outer_fold_id=outer_fold.fold_id,
            outer=outer_fold,
            inner_cv_plan=inner_plan
        ))

    logger.info(
        f"Built nested CV: {len(nested_folds)} outer folds with valid inner "
        f"plans ({skipped} outer folds skipped - insufficient training window)"
    )
    return nested_folds


def build_tuning_cv_plan(
    X: pd.DataFrame,
    y: pd.Series,
    eval_cv_plan: CVPlan,
    split_config: SplitConfig
) -> CVPlan:
    """
    Build a single hyperparameter-tuning CVPlan that is strictly isolated
    from every fold `eval_cv_plan` will later use to report "final"
    out-of-sample performance (bug 3.5 / spec section 11).

    Rather than the full per-outer-fold nested retuning `NestedFold` was
    built for (statistically ideal, but ~15-40x more PSO/GWO evaluations
    for a full FS x model grid), this takes the cheaper, still leakage-free
    route: find the earliest date any evaluation fold ever uses as a test
    target, and restrict tuning to a single inner expanding-window CVPlan
    built only from data strictly before that date. No row scored by this
    tuning plan can ever also be scored as a held-out target by
    `eval_cv_plan`.

    Args:
        X: Full feature DataFrame (datetime index).
        y: Full target Series (same index as X).
        eval_cv_plan: The flat CVPlan that will be used for final
            walk-forward evaluation (e.g. produced by scripts/00 and reused
            by scripts/03 and 04 today).
        split_config: SplitConfig - uses `inner_min_train_size`/
            `inner_test_size` (already present, previously unused by any
            script) for the tuning-only CVPlan.

    Returns:
        A CVPlan built from `X[X.index < cutoff]` - use this in place of
        `eval_cv_plan` wherever hyperparameters are selected (PSO/GWO
        objectives), never for final reported metrics.
    """
    logger = get_logger()

    cutoff = min(fold.test_start for fold in eval_cv_plan.folds)
    X_tune = X[X.index < cutoff]
    y_tune = y[y.index < cutoff]

    tuning_config = SplitConfig(
        method=split_config.method,
        min_train_size=split_config.inner_min_train_size,
        test_size=split_config.inner_test_size,
        horizons=split_config.horizons,
        horizon_weights=split_config.horizon_weights,
    )

    tuning_plan = create_walk_forward_splits(X_tune, y_tune, tuning_config)

    if tuning_plan.n_folds == 0:
        raise ValueError(
            f"build_tuning_cv_plan: no tuning folds could be built from the "
            f"{len(X_tune)} rows before the evaluation cutoff {cutoff} with "
            f"inner_min_train_size={split_config.inner_min_train_size}, "
            f"inner_test_size={split_config.inner_test_size}. Hyperparameter "
            f"tuning cannot proceed without at least one isolated fold."
        )

    max_tuning_date = max(
        max(fold.train_end, fold.test_end) for fold in tuning_plan.folds
    )
    assert max_tuning_date < cutoff, (
        f"Isolation violated: tuning plan reaches {max_tuning_date}, which "
        f"is not before the evaluation cutoff {cutoff}"
    )

    logger.info(
        f"Built tuning CV plan: {tuning_plan.n_folds} folds from "
        f"{len(X_tune)} rows before {cutoff.strftime('%Y-Q%q')} "
        f"(evaluation cutoff) - isolation OK (max tuning date "
        f"{max_tuning_date.strftime('%Y-Q%q')})"
    )

    return tuning_plan


def validate_nested_isolation(X: pd.DataFrame, nested_folds: List[NestedFold]) -> bool:
    """
    Sanity-check that no inner fold of any NestedFold ever touches a date at
    or after its outer fold's test start - i.e. the outer test period is
    completely invisible to inner-fold model selection.

    Returns True if isolation holds for every nested fold; raises
    AssertionError with details otherwise (used by tests, not silently
    returning False).
    """
    for nf in nested_folds:
        X_outer_train = X.iloc[nf.outer.train_indices]
        outer_test_start = nf.outer.test_start

        max_inner_date = None
        for inner_fold in nf.inner_cv_plan.folds:
            candidate = max(inner_fold.train_end, inner_fold.test_end)
            if max_inner_date is None or candidate > max_inner_date:
                max_inner_date = candidate

        if max_inner_date is not None:
            assert max_inner_date < outer_test_start, (
                f"Isolation violated for outer fold {nf.outer_fold_id}: an "
                f"inner fold reaches {max_inner_date}, which is not before "
                f"the outer test start {outer_test_start}"
            )
    return True
