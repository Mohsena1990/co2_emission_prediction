"""
Nonlinear feature selection methods: RF importance, Boruta, LightGBM/CatBoost importance.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ..core.logging_utils import get_logger
from ..core.config import Config
from ..splits.walk_forward import CVPlan


def random_forest_importance(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    n_estimators: int = 100,
    max_depth: int = 10,
    top_k: Optional[int] = None,
    threshold: float = 0.01,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features based on Random Forest feature importance.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        top_k: Select top K features (if None, use threshold)
        threshold: Minimum importance threshold
        seed: Random seed

    Returns:
        Tuple of (selected features, importance scores)
    """
    logger = get_logger()

    importance_scores = {col: [] for col in X.columns}

    # Get unique folds
    seen_folds = set()
    unique_folds = []
    for fold in cv_plan.folds:
        fold_key = (tuple(fold.train_indices), tuple(fold.test_indices))
        if fold_key not in seen_folds:
            seen_folds.add(fold_key)
            unique_folds.append(fold)

    for fold in unique_folds:
        X_train = X.iloc[fold.train_indices]
        y_train = y.iloc[fold.train_indices]

        # Fit Random Forest
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Store importances
        for col, imp in zip(X.columns, model.feature_importances_):
            importance_scores[col].append(imp)

    # Aggregate importance scores
    scores_df = pd.DataFrame([
        {
            'feature': col,
            'mean_importance': np.mean(scores),
            'std_importance': np.std(scores),
            'min_importance': np.min(scores),
            'max_importance': np.max(scores)
        }
        for col, scores in importance_scores.items()
    ]).sort_values('mean_importance', ascending=False)

    # Select features
    if top_k is not None:
        selected = scores_df.head(top_k)['feature'].tolist()
    else:
        selected = scores_df[scores_df['mean_importance'] >= threshold]['feature'].tolist()

    logger.info(f"RF importance: Selected {len(selected)} features")

    return selected, scores_df


def lightgbm_importance(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    params: Optional[Dict] = None,
    top_k: Optional[int] = None,
    threshold: float = 0.01,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features based on LightGBM feature importance.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        params: LightGBM parameters
        top_k: Select top K features
        threshold: Minimum importance threshold
        seed: Random seed

    Returns:
        Tuple of (selected features, importance scores)
    """
    logger = get_logger()

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not installed, skipping LightGBM importance")
        return [], pd.DataFrame()

    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': seed
        }

    importance_scores = {col: [] for col in X.columns}

    # Get unique folds
    seen_folds = set()
    unique_folds = []
    for fold in cv_plan.folds:
        fold_key = (tuple(fold.train_indices), tuple(fold.test_indices))
        if fold_key not in seen_folds:
            seen_folds.add(fold_key)
            unique_folds.append(fold)

    for fold in unique_folds:
        X_train = X.iloc[fold.train_indices]
        y_train = y.iloc[fold.train_indices]

        # Fit LightGBM
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        # Store importances (gain-based)
        importances = model.feature_importances_
        for col, imp in zip(X.columns, importances):
            importance_scores[col].append(imp)

    # Aggregate importance scores
    scores_df = pd.DataFrame([
        {
            'feature': col,
            'mean_importance': np.mean(scores),
            'std_importance': np.std(scores),
            'min_importance': np.min(scores),
            'max_importance': np.max(scores)
        }
        for col, scores in importance_scores.items()
    ]).sort_values('mean_importance', ascending=False)

    # Normalize to 0-1 range
    max_imp = scores_df['mean_importance'].max()
    if max_imp > 0:
        scores_df['normalized_importance'] = scores_df['mean_importance'] / max_imp
    else:
        scores_df['normalized_importance'] = 0

    # Select features
    if top_k is not None:
        selected = scores_df.head(top_k)['feature'].tolist()
    else:
        selected = scores_df[scores_df['normalized_importance'] >= threshold]['feature'].tolist()

    logger.info(f"LightGBM importance: Selected {len(selected)} features")

    return selected, scores_df


def catboost_importance(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    params: Optional[Dict] = None,
    top_k: Optional[int] = None,
    threshold: float = 0.01,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features based on CatBoost feature importance.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        params: CatBoost parameters
        top_k: Select top K features
        threshold: Minimum importance threshold
        seed: Random seed

    Returns:
        Tuple of (selected features, importance scores)
    """
    logger = get_logger()

    try:
        from catboost import CatBoostRegressor
    except ImportError:
        logger.warning("CatBoost not installed, skipping CatBoost importance")
        return [], pd.DataFrame()

    if params is None:
        params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'verbose': False,
            'random_seed': seed,
            'allow_writing_files': False
        }

    importance_scores = {col: [] for col in X.columns}

    # Get unique folds
    seen_folds = set()
    unique_folds = []
    for fold in cv_plan.folds:
        fold_key = (tuple(fold.train_indices), tuple(fold.test_indices))
        if fold_key not in seen_folds:
            seen_folds.add(fold_key)
            unique_folds.append(fold)

    for fold in unique_folds:
        X_train = X.iloc[fold.train_indices]
        y_train = y.iloc[fold.train_indices]

        # Fit CatBoost
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)

        # Store importances
        importances = model.feature_importances_
        for col, imp in zip(X.columns, importances):
            importance_scores[col].append(imp)

    # Aggregate importance scores
    scores_df = pd.DataFrame([
        {
            'feature': col,
            'mean_importance': np.mean(scores),
            'std_importance': np.std(scores),
            'min_importance': np.min(scores),
            'max_importance': np.max(scores)
        }
        for col, scores in importance_scores.items()
    ]).sort_values('mean_importance', ascending=False)

    # Normalize
    max_imp = scores_df['mean_importance'].max()
    if max_imp > 0:
        scores_df['normalized_importance'] = scores_df['mean_importance'] / max_imp
    else:
        scores_df['normalized_importance'] = 0

    # Select features
    if top_k is not None:
        selected = scores_df.head(top_k)['feature'].tolist()
    else:
        selected = scores_df[scores_df['normalized_importance'] >= threshold]['feature'].tolist()

    logger.info(f"CatBoost importance: Selected {len(selected)} features")

    return selected, scores_df


def xgboost_shap_stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    top_k: int = 10,
    stability_threshold: float = 0.5,
    min_features: int = 3,
    params: Optional[Dict] = None,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    FS3: select features based on XGBoost + SHAP importance stability
    across CV folds (mean |SHAP value| per feature, top-K per fold, then
    the fraction of folds where a feature makes the top-K).

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        top_k: Number of top-SHAP features considered "important" per fold
        stability_threshold: Minimum fraction of folds a feature must be
            in the top-K to be selected
        min_features: Minimum number of features to select (fallback)
        params: XGBoost parameters (defaults used if None)
        seed: Random seed

    Returns:
        Tuple of (selected features, stability scores)
    """
    logger = get_logger()

    import xgboost as xgb

    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': seed,
            'verbosity': 0
        }

    top_k_counts = {col: 0 for col in X.columns}
    shap_importance_sums = {col: 0.0 for col in X.columns}

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

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Native TreeSHAP contributions straight from the booster (exact
        # SHAP values, same algorithm shap.TreeExplainer uses) - avoids a
        # shap 0.49.1 / xgboost 3.x incompatibility where shap can't parse
        # this xgboost version's serialized base_score
        # (e.g. '[9.987916E2]') when loading the booster itself. The last
        # column is the expected-value/bias term, dropped here since only
        # per-feature contributions are needed.
        contribs = model.get_booster().predict(
            xgb.DMatrix(X_train), pred_contribs=True
        )
        shap_values = contribs[:, :-1]
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        importance_order = np.argsort(mean_abs_shap)[::-1]
        top_k_features = [X.columns[i] for i in importance_order[:top_k]]

        for col, imp in zip(X.columns, mean_abs_shap):
            shap_importance_sums[col] += imp
        for col in top_k_features:
            top_k_counts[col] += 1

        n_folds += 1

    stability_df = pd.DataFrame([
        {
            'feature': col,
            'top_k_count': top_k_counts[col],
            'stability': top_k_counts[col] / n_folds if n_folds > 0 else 0,
            'mean_abs_shap': shap_importance_sums[col] / n_folds if n_folds > 0 else 0
        }
        for col in X.columns
    ]).sort_values('stability', ascending=False)

    selected = stability_df[stability_df['stability'] >= stability_threshold]['feature'].tolist()

    if len(selected) < min_features:
        logger.info(f"XGBoost-SHAP: Only {len(selected)} features passed threshold, adding top features to reach {min_features}")
        remaining = stability_df[~stability_df['feature'].isin(selected)].head(min_features - len(selected))
        selected.extend(remaining['feature'].tolist())

    logger.info(f"XGBoost-SHAP stability: Selected {len(selected)} features (stability >= {stability_threshold})")

    return selected, stability_df


def fs_xgboost_shap(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    config: Config
) -> Dict[str, Any]:
    """
    FS3: XGBoost-SHAP stability selection pipeline.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        config: Configuration

    Returns:
        Dictionary with selected features and scores, same shape as fs_linear.
    """
    logger = get_logger()
    logger.info("Running XGBoost-SHAP stability feature selection (FS_xgboost_shap)...")

    top_k = getattr(config.fs, 'top_k_features', 10)
    min_features = getattr(config.fs, 'min_features', 3)

    results = {
        'method': 'xgboost_shap',
        'steps': []
    }

    selected, stability_df = xgboost_shap_stability_selection(
        X, y, cv_plan,
        top_k=top_k,
        stability_threshold=config.fs.stability_threshold,
        min_features=min_features,
        seed=config.seed
    )
    results['steps'].append({
        'name': 'xgboost_shap_stability',
        'selected': selected,
        'n_selected': len(selected),
        'scores': stability_df.to_dict(orient='records')
    })

    final_selected = [f for f in X.columns if f in selected]

    results['selected_features'] = final_selected
    results['n_selected'] = len(final_selected)

    logger.info(f"FS_xgboost_shap final: {len(final_selected)} features selected")

    return results


def fs_nonlinear(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    config: Config
) -> Dict[str, Any]:
    """
    Full nonlinear feature selection pipeline.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        config: Configuration

    Returns:
        Dictionary with selected features and scores
    """
    logger = get_logger()
    logger.info("Running nonlinear feature selection (FS_nonlinear)...")

    results = {
        'method': 'nonlinear',
        'steps': []
    }

    # Step 1: Random Forest importance
    rf_selected, rf_scores = random_forest_importance(
        X, y, cv_plan, seed=config.seed
    )
    results['steps'].append({
        'name': 'random_forest',
        'selected': rf_selected,
        'n_selected': len(rf_selected),
        'scores': rf_scores.to_dict(orient='records')
    })

    # Step 2: LightGBM importance
    lgb_selected, lgb_scores = lightgbm_importance(
        X, y, cv_plan, seed=config.seed
    )
    results['steps'].append({
        'name': 'lightgbm',
        'selected': lgb_selected,
        'n_selected': len(lgb_selected),
        'scores': lgb_scores.to_dict(orient='records') if len(lgb_scores) > 0 else []
    })

    # Step 3: CatBoost importance
    cat_selected, cat_scores = catboost_importance(
        X, y, cv_plan, seed=config.seed
    )
    results['steps'].append({
        'name': 'catboost',
        'selected': cat_selected,
        'n_selected': len(cat_selected),
        'scores': cat_scores.to_dict(orient='records') if len(cat_scores) > 0 else []
    })

    # Final selection: union of tree-based methods
    all_selected = set(rf_selected) | set(lgb_selected) | set(cat_selected)
    final_selected = [f for f in X.columns if f in all_selected]

    results['selected_features'] = final_selected
    results['n_selected'] = len(final_selected)

    logger.info(f"FS_nonlinear final: {len(final_selected)} features selected")

    return results
