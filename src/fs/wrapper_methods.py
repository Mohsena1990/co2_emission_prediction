"""
Wrapper-based feature selection methods.

Wrapper methods use a machine learning model to evaluate
feature subsets by training and scoring the model.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from ..core.logging_utils import get_logger
from ..core.config import Config
from ..splits.walk_forward import CVPlan


def _unique_folds(cv_plan: CVPlan) -> List:
    """
    Deduplicate a CVPlan's folds down to unique (train_indices, test_indices)
    pairs - a CVPlan typically repeats the same split once per horizon, and
    wrapper-method fold iteration only needs each split once (mirrors the
    dedup pattern already used in linear.py/embedded_methods.py).
    """
    seen = set()
    folds = []
    for fold in cv_plan.folds:
        key = (tuple(fold.train_indices), tuple(fold.test_indices))
        if key not in seen:
            seen.add(key)
            folds.append(fold)
    return folds


def _cv_splits(cv_plan: CVPlan) -> List[Tuple[List[int], List[int]]]:
    """
    Build an explicit (train_idx, test_idx) list from a CVPlan's unique
    folds, suitable for sklearn's `cv=` parameter. Bug 3.4/spec section 11:
    these are the project's own expanding-window splits (chronologically
    ordered, no shuffling), so passing them to sklearn's wrapper selectors
    replaces the previous plain `KFold`/no-CV full-sample fits - which,
    being contiguous but non-time-aware, could train on a later block and
    test on an earlier one (look-ahead leakage).
    """
    return [(list(f.train_indices), list(f.test_indices)) for f in _unique_folds(cv_plan)]


def rfe_selection(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    n_features_to_select: int = 10,
    estimator_type: str = 'ridge',
    step: int = 1,
    stability_threshold: float = 0.5,
    min_features: int = 3,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Recursive Feature Elimination (RFE), made time-series-safe (spec section
    11 / bug 3.4): sklearn's `RFE` has no native CV concept, so the previous
    implementation fit it once on the *entire* sample (train and outer-test
    rows together) - a direct leakage bug, not just an ordinary-KFold one.

    Instead, RFE is now run independently within each fold's *training*
    slice only (`X.iloc[fold.train_indices]`, never touching `test_indices`),
    and the final feature set is the stability-selected union across folds -
    the same selection-frequency pattern already used by
    `linear.ridge_stability_selection`.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan - RFE is fit only on each fold's training indices
        n_features_to_select: Number of features to keep per fold
        estimator_type: 'ridge' or 'random_forest'
        step: Number of features to remove at each iteration
        stability_threshold: Minimum fraction of folds a feature must be
            selected in to be retained
        min_features: Minimum number of features to select (fallback)
        seed: Random seed

    Returns:
        Tuple of (selected feature names, per-feature stability DataFrame)
    """
    logger = get_logger()
    logger.info(f"Running RFE feature selection (target: {n_features_to_select} features)...")

    folds = _unique_folds(cv_plan)
    if not folds:
        raise ValueError("rfe_selection: cv_plan has no folds to fit RFE on.")

    selected_counts = {col: 0 for col in X.columns}
    ranking_sums = {col: 0.0 for col in X.columns}

    for fold in folds:
        X_train = X.iloc[fold.train_indices]
        y_train = y.iloc[fold.train_indices]

        if estimator_type == 'ridge':
            estimator = Ridge(alpha=1.0)
        else:
            estimator = RandomForestRegressor(n_estimators=50, max_depth=5,
                                              random_state=seed, n_jobs=-1)

        rfe = RFE(estimator, n_features_to_select=min(n_features_to_select, len(X.columns)),
                  step=step)
        rfe.fit(X_train, y_train)

        for col, sel, rank in zip(X.columns, rfe.support_, rfe.ranking_):
            if sel:
                selected_counts[col] += 1
            ranking_sums[col] += rank

    n_folds = len(folds)
    scores_df = pd.DataFrame([
        {
            'feature': col,
            'selection_frequency': selected_counts[col] / n_folds,
            'n_folds_selected': selected_counts[col],
            'n_folds_total': n_folds,
            'avg_ranking': ranking_sums[col] / n_folds
        }
        for col in X.columns
    ]).sort_values(['selection_frequency', 'avg_ranking'], ascending=[False, True])

    selected = scores_df[scores_df['selection_frequency'] >= stability_threshold]['feature'].tolist()

    if len(selected) < min_features:
        remaining = scores_df[~scores_df['feature'].isin(selected)].head(min_features - len(selected))
        selected.extend(remaining['feature'].tolist())

    scores_df['selected'] = scores_df['feature'].isin(selected)

    logger.info(f"RFE: Selected {len(selected)} features using {estimator_type} "
                f"(stability >= {stability_threshold} across {n_folds} folds)")
    return selected, scores_df


def sequential_forward_selection(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    n_features_to_select: int = 10,
    estimator_type: str = 'ridge',
    scoring: str = 'neg_mean_absolute_error',
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Sequential Forward Selection (SFS), evaluated on the project's own
    expanding-window folds (spec section 11 / bug 3.4) rather than sklearn's
    default shuffled/contiguous-block `KFold` - the previous `cv=5` int
    let SequentialFeatureSelector score candidate subsets on blocks that are
    not time-ordered, so a later (future) block could serve as training data
    for an earlier block's validation score.

    Starts with empty set and adds features one by one
    that maximize model performance.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan - converted to explicit expanding-window
            (train_idx, test_idx) splits and passed as sklearn's `cv=`
        n_features_to_select: Number of features to select
        estimator_type: 'ridge' or 'random_forest'
        scoring: Scoring metric for CV
        seed: Random seed

    Returns:
        Tuple of (selected feature names, selection DataFrame)
    """
    logger = get_logger()
    logger.info(f"Running Sequential Forward Selection (target: {n_features_to_select} features)...")

    # Select estimator
    if estimator_type == 'ridge':
        estimator = Ridge(alpha=1.0)
    else:
        estimator = RandomForestRegressor(n_estimators=50, max_depth=5,
                                          random_state=seed, n_jobs=-1)

    cv_splits = _cv_splits(cv_plan)
    if not cv_splits:
        raise ValueError("sequential_forward_selection: cv_plan has no folds to validate on.")

    # Run SFS
    n_select = min(n_features_to_select, len(X.columns))
    sfs = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_select,
        direction='forward',
        cv=cv_splits,
        scoring=scoring,
        n_jobs=-1
    )
    sfs.fit(X, y)

    selected = list(X.columns[sfs.get_support()])

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'selected': sfs.get_support(),
        'selection_order': [selected.index(f) + 1 if f in selected else 0
                           for f in X.columns]
    })

    logger.info(f"SFS: Selected {len(selected)} features using {estimator_type}")
    return selected, scores_df


def sequential_backward_selection(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    n_features_to_select: int = 10,
    estimator_type: str = 'ridge',
    scoring: str = 'neg_mean_absolute_error',
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Sequential Backward Selection (SBS), evaluated on the project's own
    expanding-window folds (spec section 11 / bug 3.4) - see
    `sequential_forward_selection` docstring for why the previous `cv=5` int
    was a leakage risk.

    Starts with all features and removes features one by one
    that have the least impact on model performance.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan - converted to explicit expanding-window
            (train_idx, test_idx) splits and passed as sklearn's `cv=`
        n_features_to_select: Number of features to keep
        estimator_type: 'ridge' or 'random_forest'
        scoring: Scoring metric for CV
        seed: Random seed

    Returns:
        Tuple of (selected feature names, selection DataFrame)
    """
    logger = get_logger()
    logger.info(f"Running Sequential Backward Selection (target: {n_features_to_select} features)...")

    # Select estimator
    if estimator_type == 'ridge':
        estimator = Ridge(alpha=1.0)
    else:
        estimator = RandomForestRegressor(n_estimators=50, max_depth=5,
                                          random_state=seed, n_jobs=-1)

    cv_splits = _cv_splits(cv_plan)
    if not cv_splits:
        raise ValueError("sequential_backward_selection: cv_plan has no folds to validate on.")

    # Run SBS
    n_select = min(n_features_to_select, len(X.columns))
    sbs = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_select,
        direction='backward',
        cv=cv_splits,
        scoring=scoring,
        n_jobs=-1
    )
    sbs.fit(X, y)

    selected = list(X.columns[sbs.get_support()])

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'selected': sbs.get_support()
    })

    logger.info(f"SBS: Selected {len(selected)} features using {estimator_type}")
    return selected, scores_df


def exhaustive_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    min_features: int = 1,
    max_features: int = 5,
    cv: int = 5,
    scoring: str = 'neg_mean_absolute_error',
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Exhaustive feature selection (for small feature sets).

    Tests all possible combinations within the size range.
    Warning: Computationally expensive for large feature sets!

    Args:
        X: Feature DataFrame (should be small, <15 features)
        y: Target Series
        min_features: Minimum subset size
        max_features: Maximum subset size
        cv: Cross-validation folds
        scoring: Scoring metric
        seed: Random seed

    Returns:
        Tuple of (best feature subset, results DataFrame)
    """
    from itertools import combinations

    logger = get_logger()

    if len(X.columns) > 15:
        logger.warning("Exhaustive search not recommended for >15 features. Using top 15.")
        # Use correlation to pre-filter
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        X = X[correlations.head(15).index]

    logger.info(f"Running exhaustive feature selection ({len(X.columns)} features)...")

    estimator = Ridge(alpha=1.0)
    results = []

    for size in range(min_features, min(max_features + 1, len(X.columns) + 1)):
        for combo in combinations(X.columns, size):
            X_subset = X[list(combo)]
            scores = cross_val_score(estimator, X_subset, y, cv=cv, scoring=scoring)
            results.append({
                'features': combo,
                'n_features': size,
                'mean_score': scores.mean(),
                'std_score': scores.std()
            })

    results_df = pd.DataFrame(results).sort_values('mean_score', ascending=False)

    # Best subset
    best_row = results_df.iloc[0]
    selected = list(best_row['features'])

    logger.info(f"Exhaustive: Selected {len(selected)} features (score: {best_row['mean_score']:.4f})")
    return selected, results_df


def run_all_wrapper_methods(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    n_features: int = 10,
    stability_threshold: float = 0.5,
    min_features: int = 3,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Run all wrapper-based feature selection methods, all evaluated strictly
    on `cv_plan`'s expanding-window folds (spec section 11 / bug 3.4).

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan - passed through to RFE/SFS/SBS instead of an
            ordinary `cv=int` fold count
        n_features: Target number of features
        stability_threshold: RFE fold-selection-frequency threshold
        min_features: Minimum features to select (fallback)
        seed: Random seed

    Returns:
        Dictionary with results from all wrapper methods
    """
    logger = get_logger()
    logger.info("=" * 50)
    logger.info("Running ALL wrapper-based feature selection methods...")

    results = {}

    # RFE with Ridge
    try:
        rfe_selected, rfe_scores = rfe_selection(X, y, cv_plan, n_features_to_select=n_features,
                                                  estimator_type='ridge',
                                                  stability_threshold=stability_threshold,
                                                  min_features=min_features, seed=seed)
        results['rfe_ridge'] = {
            'method': 'wrapper',
            'type': 'rfe',
            'estimator': 'ridge',
            'selected_features': rfe_selected,
            'n_selected': len(rfe_selected),
            'scores': rfe_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"RFE (Ridge) failed: {e}")

    # RFE with Random Forest
    try:
        rfe_rf_selected, rfe_rf_scores = rfe_selection(X, y, cv_plan, n_features_to_select=n_features,
                                                        estimator_type='random_forest',
                                                        stability_threshold=stability_threshold,
                                                        min_features=min_features, seed=seed)
        results['rfe_rf'] = {
            'method': 'wrapper',
            'type': 'rfe',
            'estimator': 'random_forest',
            'selected_features': rfe_rf_selected,
            'n_selected': len(rfe_rf_selected),
            'scores': rfe_rf_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"RFE (RF) failed: {e}")

    # Sequential Forward Selection
    try:
        sfs_selected, sfs_scores = sequential_forward_selection(X, y, cv_plan, n_features_to_select=n_features,
                                                                 seed=seed)
        results['sfs'] = {
            'method': 'wrapper',
            'type': 'sequential_forward',
            'selected_features': sfs_selected,
            'n_selected': len(sfs_selected),
            'scores': sfs_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"SFS failed: {e}")

    # Sequential Backward Selection
    try:
        sbs_selected, sbs_scores = sequential_backward_selection(X, y, cv_plan, n_features_to_select=n_features,
                                                                   seed=seed)
        results['sbs'] = {
            'method': 'wrapper',
            'type': 'sequential_backward',
            'selected_features': sbs_selected,
            'n_selected': len(sbs_selected),
            'scores': sbs_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"SBS failed: {e}")

    logger.info(f"Wrapper methods complete: {len(results)} methods run")
    return results


def fs_wrapper(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    config: Config
) -> Dict[str, Any]:
    """
    FS2: Wrapper selection pipeline (RFE + SFS + SBS union).

    Bug 3.4 fix: `cv_plan` was previously accepted but ignored - RFE fit on
    the full sample and SFS/SBS used plain sklearn `KFold`. All three now
    evaluate candidate subsets strictly on `cv_plan`'s expanding-window
    folds (see `run_all_wrapper_methods`), matching how FS1/FS3/FS4 already
    respect fold boundaries.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan - propagated to every wrapper sub-method
        config: Configuration

    Returns:
        Dictionary with selected features and scores, same shape as fs_linear.
    """
    logger = get_logger()
    logger.info("Running wrapper feature selection (FS_wrapper)...")

    top_k = getattr(config.fs, 'top_k_features', 10)
    min_features = getattr(config.fs, 'min_features', 3)
    stability_threshold = getattr(config.fs, 'stability_threshold', 0.5)

    wrapper_results = run_all_wrapper_methods(
        X, y, cv_plan, n_features=top_k, stability_threshold=stability_threshold,
        min_features=min_features, seed=config.seed
    )

    all_selected = set()
    for method_result in wrapper_results.values():
        all_selected.update(method_result.get('selected_features', []))

    final_selected = [f for f in X.columns if f in all_selected]

    if len(final_selected) < min_features:
        logger.info(f"FS_wrapper: Only {len(final_selected)} features, adding top-voted features")
        remaining = [f for f in X.columns if f not in final_selected][:min_features - len(final_selected)]
        final_selected.extend(remaining)

    logger.info(f"FS_wrapper final: {len(final_selected)} features selected")

    return {
        'method': 'wrapper',
        'steps': list(wrapper_results.values()),
        'sub_methods': wrapper_results,
        'selected_features': final_selected,
        'n_selected': len(final_selected)
    }
