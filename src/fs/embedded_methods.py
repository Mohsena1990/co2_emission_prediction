"""
Embedded feature selection methods.

Embedded methods perform feature selection as part of the
model training process (e.g., L1 regularization, tree importance).
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from ..core.logging_utils import get_logger
from ..core.config import Config
from ..splits.walk_forward import CVPlan


def lasso_selection(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = None,
    cv: int = 5,
    max_iter: int = 10000,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    LASSO-based feature selection (L1 regularization).

    LASSO shrinks some coefficients to exactly zero,
    effectively performing feature selection.

    Args:
        X: Feature DataFrame
        y: Target Series
        alpha: Regularization strength (None = use CV)
        cv: Cross-validation folds for alpha selection
        max_iter: Maximum iterations
        seed: Random seed

    Returns:
        Tuple of (selected feature names, coefficients DataFrame)
    """
    logger = get_logger()
    logger.info("Running LASSO feature selection...")

    # Scale features for fair regularization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use cross-validation to find optimal alpha if not specified
    if alpha is None:
        lasso_cv = LassoCV(cv=cv, random_state=seed, max_iter=max_iter)
        lasso_cv.fit(X_scaled, y)
        alpha = lasso_cv.alpha_
        logger.info(f"LASSO optimal alpha via CV: {alpha:.6f}")

    # Fit LASSO with selected alpha
    lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=seed)
    lasso.fit(X_scaled, y)

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': lasso.coef_,
        'abs_coefficient': np.abs(lasso.coef_),
        'selected': np.abs(lasso.coef_) > 1e-10
    }).sort_values('abs_coefficient', ascending=False)

    selected = scores_df[scores_df['selected']]['feature'].tolist()

    logger.info(f"LASSO: Selected {len(selected)} features (alpha={alpha:.6f})")
    return selected, scores_df


def elasticnet_selection(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = None,
    l1_ratio: float = 0.5,
    cv: int = 5,
    max_iter: int = 10000,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    ElasticNet feature selection (L1 + L2 regularization).

    Combines LASSO (L1) and Ridge (L2) penalties.
    l1_ratio controls the mix (1 = pure LASSO, 0 = pure Ridge).

    Args:
        X: Feature DataFrame
        y: Target Series
        alpha: Regularization strength (None = use CV)
        l1_ratio: Balance between L1 and L2 (0-1)
        cv: Cross-validation folds
        max_iter: Maximum iterations
        seed: Random seed

    Returns:
        Tuple of (selected feature names, coefficients DataFrame)
    """
    logger = get_logger()
    logger.info(f"Running ElasticNet feature selection (l1_ratio={l1_ratio})...")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use CV for alpha if not specified
    if alpha is None:
        en_cv = ElasticNetCV(l1_ratio=l1_ratio, cv=cv, random_state=seed, max_iter=max_iter)
        en_cv.fit(X_scaled, y)
        alpha = en_cv.alpha_
        logger.info(f"ElasticNet optimal alpha: {alpha:.6f}")

    # Fit ElasticNet
    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=seed)
    en.fit(X_scaled, y)

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': en.coef_,
        'abs_coefficient': np.abs(en.coef_),
        'selected': np.abs(en.coef_) > 1e-10
    }).sort_values('abs_coefficient', ascending=False)

    selected = scores_df[scores_df['selected']]['feature'].tolist()

    logger.info(f"ElasticNet: Selected {len(selected)} features")
    return selected, scores_df


def gradient_boosting_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 10,
    threshold: float = None,
    n_estimators: int = 100,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Gradient Boosting feature importance selection.

    Uses LightGBM (if available) or sklearn GradientBoosting
    to compute feature importances.

    Args:
        X: Feature DataFrame
        y: Target Series
        top_k: Number of top features to select
        threshold: Alternative - importance threshold
        n_estimators: Number of boosting iterations
        seed: Random seed

    Returns:
        Tuple of (selected feature names, importance DataFrame)
    """
    logger = get_logger()
    logger.info("Running Gradient Boosting feature selection...")

    # Try LightGBM first, fallback to sklearn
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=5,
            verbosity=-1,
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X, y)
        importances = model.feature_importances_
        logger.info("Using LightGBM for feature importance")
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=5,
            random_state=seed
        )
        model.fit(X, y)
        importances = model.feature_importances_
        logger.info("Using sklearn GradientBoosting for feature importance")

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Normalize
    total_importance = scores_df['importance'].sum()
    if total_importance > 0:
        scores_df['importance_normalized'] = scores_df['importance'] / total_importance
    else:
        scores_df['importance_normalized'] = 0

    # Select features
    if threshold is not None:
        selected = scores_df[scores_df['importance_normalized'] >= threshold]['feature'].tolist()
    else:
        selected = scores_df.head(top_k)['feature'].tolist()

    logger.info(f"Gradient Boosting: Selected {len(selected)} features")
    return selected, scores_df


def random_forest_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 10,
    threshold: float = None,
    n_estimators: int = 100,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Random Forest feature importance selection.

    Uses mean decrease in impurity (MDI) importance.

    Args:
        X: Feature DataFrame
        y: Target Series
        top_k: Number of top features to select
        threshold: Alternative - importance threshold
        n_estimators: Number of trees
        seed: Random seed

    Returns:
        Tuple of (selected feature names, importance DataFrame)
    """
    from sklearn.ensemble import RandomForestRegressor

    logger = get_logger()
    logger.info("Running Random Forest feature selection...")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X, y)

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Normalize
    total = scores_df['importance'].sum()
    scores_df['importance_normalized'] = scores_df['importance'] / total if total > 0 else 0

    # Select features
    if threshold is not None:
        selected = scores_df[scores_df['importance_normalized'] >= threshold]['feature'].tolist()
    else:
        selected = scores_df.head(top_k)['feature'].tolist()

    logger.info(f"Random Forest: Selected {len(selected)} features")
    return selected, scores_df


def catboost_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 10,
    threshold: float = None,
    iterations: int = 100,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    CatBoost feature importance selection.

    Args:
        X: Feature DataFrame
        y: Target Series
        top_k: Number of top features to select
        threshold: Alternative - importance threshold
        iterations: Number of boosting iterations
        seed: Random seed

    Returns:
        Tuple of (selected feature names, importance DataFrame)
    """
    logger = get_logger()
    logger.info("Running CatBoost feature selection...")

    try:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=0.1,
            depth=5,
            verbose=False,
            random_seed=seed,
            allow_writing_files=False
        )
        model.fit(X, y)
        importances = model.feature_importances_
    except ImportError:
        logger.warning("CatBoost not available. Using Gradient Boosting fallback.")
        return gradient_boosting_selection(X, y, top_k, threshold, n_estimators=iterations, seed=seed)

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Normalize
    total = scores_df['importance'].sum()
    scores_df['importance_normalized'] = scores_df['importance'] / total if total > 0 else 0

    # Select features
    if threshold is not None:
        selected = scores_df[scores_df['importance_normalized'] >= threshold]['feature'].tolist()
    else:
        selected = scores_df.head(top_k)['feature'].tolist()

    logger.info(f"CatBoost: Selected {len(selected)} features")
    return selected, scores_df


def permutation_stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    evaluator_model: str = 'lightgbm',
    top_k: int = 10,
    stability_threshold: float = 0.5,
    min_features: int = 3,
    n_repeats: int = 10,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    FS4: model-agnostic permutation-importance stability selection across CV
    folds. Permutation importance only requires a fitted estimator's
    `.predict`, not model-internal attributes (coefficients/gain/etc.), so
    the same procedure works regardless of which estimator is used to
    probe importance.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        evaluator_model: Estimator used to probe importance ('lightgbm' or
            'rf' - falls back to 'rf' if LightGBM is unavailable)
        top_k: Number of top-permutation-importance features per fold
        stability_threshold: Minimum fraction of folds a feature must be in
            the top-K to be selected
        min_features: Minimum number of features to select (fallback)
        n_repeats: Number of permutation repeats per fold
        seed: Random seed

    Returns:
        Tuple of (selected features, stability scores)
    """
    logger = get_logger()

    top_k_counts = {col: 0 for col in X.columns}
    importance_sums = {col: 0.0 for col in X.columns}

    # Get unique folds
    seen_folds = set()
    unique_folds = []
    for fold in cv_plan.folds:
        fold_key = (tuple(fold.train_indices), tuple(fold.test_indices))
        if fold_key not in seen_folds:
            seen_folds.add(fold_key)
            unique_folds.append(fold)

    n_folds = 0
    for fold in unique_folds:
        X_train = X.iloc[fold.train_indices]
        y_train = y.iloc[fold.train_indices]

        model_used = evaluator_model
        if model_used == 'lightgbm':
            try:
                import lightgbm as lgb
                model = lgb.LGBMRegressor(
                    n_estimators=100, learning_rate=0.1,
                    verbosity=-1, random_state=seed
                )
                model.fit(X_train, y_train)
            except ImportError:
                model_used = 'rf'

        if model_used == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10,
                random_state=seed, n_jobs=-1
            )
            model.fit(X_train, y_train)

        result = permutation_importance(
            model, X_train, y_train,
            n_repeats=n_repeats, random_state=seed,
            scoring='neg_mean_absolute_error'
        )
        importances = result.importances_mean

        importance_order = np.argsort(importances)[::-1]
        top_k_features = [X.columns[i] for i in importance_order[:top_k]]

        for col, imp in zip(X.columns, importances):
            importance_sums[col] += imp
        for col in top_k_features:
            top_k_counts[col] += 1

        n_folds += 1

    stability_df = pd.DataFrame([
        {
            'feature': col,
            'top_k_count': top_k_counts[col],
            'stability': top_k_counts[col] / n_folds if n_folds > 0 else 0,
            'mean_permutation_importance': importance_sums[col] / n_folds if n_folds > 0 else 0
        }
        for col in X.columns
    ]).sort_values('stability', ascending=False)

    selected = stability_df[stability_df['stability'] >= stability_threshold]['feature'].tolist()

    if len(selected) < min_features:
        logger.info(f"Permutation stability: Only {len(selected)} features passed threshold, adding top features to reach {min_features}")
        remaining = stability_df[~stability_df['feature'].isin(selected)].head(min_features - len(selected))
        selected.extend(remaining['feature'].tolist())

    logger.info(f"Permutation stability: Selected {len(selected)} features (stability >= {stability_threshold})")

    return selected, stability_df


def fs_permutation_stability(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    config: Config
) -> Dict[str, Any]:
    """
    FS4: model-agnostic permutation stability selection pipeline.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        config: Configuration

    Returns:
        Dictionary with selected features and scores, same shape as fs_linear.
    """
    logger = get_logger()
    logger.info("Running permutation stability feature selection (FS_permutation_stability)...")

    top_k = getattr(config.fs, 'top_k_features', 10)
    min_features = getattr(config.fs, 'min_features', 3)

    results = {
        'method': 'permutation_stability',
        'steps': []
    }

    selected, stability_df = permutation_stability_selection(
        X, y, cv_plan,
        evaluator_model=config.fs.evaluator_model,
        top_k=top_k,
        stability_threshold=config.fs.stability_threshold,
        min_features=min_features,
        seed=config.seed
    )
    results['steps'].append({
        'name': 'permutation_stability',
        'selected': selected,
        'n_selected': len(selected),
        'scores': stability_df.to_dict(orient='records')
    })

    final_selected = [f for f in X.columns if f in selected]

    results['selected_features'] = final_selected
    results['n_selected'] = len(final_selected)

    logger.info(f"FS_permutation_stability final: {len(final_selected)} features selected")

    return results


def run_all_embedded_methods(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 10,
    cv: int = 5,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Run all embedded feature selection methods.

    Args:
        X: Feature DataFrame
        y: Target Series
        top_k: Target number of features
        cv: Cross-validation folds
        seed: Random seed

    Returns:
        Dictionary with results from all embedded methods
    """
    logger = get_logger()
    logger.info("=" * 50)
    logger.info("Running ALL embedded feature selection methods...")

    results = {}

    # LASSO
    try:
        lasso_selected, lasso_scores = lasso_selection(X, y, cv=cv, seed=seed)
        results['lasso'] = {
            'method': 'embedded',
            'type': 'lasso',
            'selected_features': lasso_selected,
            'n_selected': len(lasso_selected),
            'scores': lasso_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"LASSO failed: {e}")

    # ElasticNet
    try:
        en_selected, en_scores = elasticnet_selection(X, y, cv=cv, seed=seed)
        results['elasticnet'] = {
            'method': 'embedded',
            'type': 'elasticnet',
            'selected_features': en_selected,
            'n_selected': len(en_selected),
            'scores': en_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"ElasticNet failed: {e}")

    # Gradient Boosting
    try:
        gb_selected, gb_scores = gradient_boosting_selection(X, y, top_k=top_k, seed=seed)
        results['gradient_boosting'] = {
            'method': 'embedded',
            'type': 'gradient_boosting',
            'selected_features': gb_selected,
            'n_selected': len(gb_selected),
            'scores': gb_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"Gradient Boosting failed: {e}")

    # Random Forest
    try:
        rf_selected, rf_scores = random_forest_selection(X, y, top_k=top_k, seed=seed)
        results['random_forest'] = {
            'method': 'embedded',
            'type': 'random_forest',
            'selected_features': rf_selected,
            'n_selected': len(rf_selected),
            'scores': rf_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"Random Forest failed: {e}")

    # CatBoost
    try:
        cb_selected, cb_scores = catboost_selection(X, y, top_k=top_k, seed=seed)
        results['catboost'] = {
            'method': 'embedded',
            'type': 'catboost',
            'selected_features': cb_selected,
            'n_selected': len(cb_selected),
            'scores': cb_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"CatBoost failed: {e}")

    logger.info(f"Embedded methods complete: {len(results)} methods run")
    return results
