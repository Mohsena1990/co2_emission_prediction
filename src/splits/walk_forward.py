"""
Walk-forward cross-validation for CO2 forecasting framework.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Generator, Optional
from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import Config, SplitConfig
from ..core.logging_utils import get_logger


@dataclass
class CVFold:
    """Represents a single cross-validation fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_indices: np.ndarray
    test_indices: np.ndarray
    horizon: int = 1

    def __repr__(self):
        return (f"CVFold(fold={self.fold_id}, "
                f"train={self.train_start.strftime('%Y-Q%q')} to {self.train_end.strftime('%Y-Q%q')}, "
                f"test={self.test_start.strftime('%Y-Q%q')} to {self.test_end.strftime('%Y-Q%q')}, "
                f"horizon={self.horizon})")


@dataclass
class CVPlan:
    """Complete cross-validation plan."""
    folds: List[CVFold] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    # Direct per-horizon targets (spec section 12 / bug: H2/H4 folds must
    # predict y at origin+horizon from ORIGIN features, not the row-aligned
    # y at the shifted test window - see create_walk_forward_splits and
    # features.engineering.create_direct_horizon_targets). Keyed by
    # horizon; each series shares X's index (origin-aligned), with
    # y_by_horizon[h].loc[t] == y.loc[t + h].
    y_by_horizon: Dict[int, pd.Series] = field(default_factory=dict)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def get_folds_by_horizon(self, horizon: int) -> List[CVFold]:
        """Get folds for a specific horizon."""
        return [f for f in self.folds if f.horizon == horizon]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert CV plan to DataFrame."""
        records = []
        for fold in self.folds:
            records.append({
                'fold_id': fold.fold_id,
                'horizon': fold.horizon,
                'train_start': fold.train_start,
                'train_end': fold.train_end,
                'test_start': fold.test_start,
                'test_end': fold.test_end,
                'train_size': len(fold.train_indices),
                'test_size': len(fold.test_indices)
            })
        return pd.DataFrame(records)


def create_walk_forward_splits(
    X: pd.DataFrame,
    y: pd.Series,
    config: SplitConfig,
    horizons: Optional[List[int]] = None
) -> CVPlan:
    """
    Create walk-forward cross-validation splits.

    Both train and test indices are always ORIGIN positions into X (never
    shifted forward by the horizon): a fold's test window is the set of
    forecast origins immediately following its training window, walked
    forward `test_size` positions at a time, identically for every horizon.
    What differs per horizon is only which target value each origin is
    paired with - `cv_plan.y_by_horizon[h]` (see `CVPlan`) holds
    y_by_horizon[h].loc[t] == y.loc[t + h], so a "horizon=4" fold trains and
    evaluates a genuine 4-step-ahead direct forecast from origin features,
    rather than (the previous, buggy behaviour) evaluating 1-step-ahead-style
    predictions on a test window shifted 4 rows into the future - which let
    near-term lag features at the shifted window leak information a real
    4-quarter-ahead forecast would never have at its true origin.

    Args:
        X: Feature DataFrame with datetime index
        y: Target Series
        config: Split configuration
        horizons: List of forecast horizons (default from config)

    Returns:
        CVPlan with all folds and `y_by_horizon`
    """
    logger = get_logger()

    if horizons is None:
        horizons = config.horizons

    # Ensure aligned data
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    # Drop rows with NaN values (from lags)
    valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    n_samples = len(X)
    logger.info(f"Creating walk-forward splits for {n_samples} valid samples")

    from ..features.engineering import create_direct_horizon_targets
    y_by_horizon = create_direct_horizon_targets(y, horizons)

    folds = []
    fold_id = 0

    for horizon in horizons:
        # Walk-forward: start from min_train_size, expand training set.
        # Test origins must leave room for the horizon-shifted target to
        # exist (origin + horizon <= last valid index).
        max_origin_idx = n_samples - 1 - horizon
        train_end_idx = config.min_train_size - 1

        while train_end_idx + config.test_size <= max_origin_idx + 1:
            # Training indices: 0 to train_end_idx (inclusive)
            train_indices = np.arange(0, train_end_idx + 1)

            # Test indices: origins immediately following training,
            # NOT shifted forward by horizon - the horizon only changes
            # which target (y_by_horizon[horizon]) these origins are
            # paired with.
            test_start_idx = train_end_idx + 1
            test_end_idx = min(test_start_idx + config.test_size, max_origin_idx + 1)
            test_indices = np.arange(test_start_idx, test_end_idx)

            if len(test_indices) == 0:
                break

            fold = CVFold(
                fold_id=fold_id,
                train_start=X.index[train_indices[0]],
                train_end=X.index[train_indices[-1]],
                test_start=X.index[test_indices[0]],
                test_end=X.index[test_indices[-1]],
                train_indices=train_indices,
                test_indices=test_indices,
                horizon=horizon
            )

            folds.append(fold)
            fold_id += 1

            # Move forward by test_size
            train_end_idx += config.test_size

    cv_plan = CVPlan(
        folds=folds,
        config={
            'method': config.method,
            'min_train_size': config.min_train_size,
            'test_size': config.test_size,
            'horizons': horizons,
            'n_folds': len(folds),
            'n_samples': n_samples
        },
        y_by_horizon=y_by_horizon
    )

    logger.info(f"Created {len(folds)} CV folds across horizons {horizons}")

    return cv_plan


def create_expanding_window_splits(
    X: pd.DataFrame,
    y: pd.Series,
    config: SplitConfig,
    horizons: Optional[List[int]] = None
) -> CVPlan:
    """
    Create expanding window cross-validation splits.
    Similar to walk-forward but training window always starts from beginning.

    Args:
        X: Feature DataFrame
        y: Target Series
        config: Split configuration
        horizons: List of forecast horizons

    Returns:
        CVPlan with all folds
    """
    # Expanding window is essentially walk-forward with fixed start
    return create_walk_forward_splits(X, y, config, horizons)


def generate_cv_folds(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan
) -> Generator[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, CVFold], None, None]:
    """
    Generator that yields train/test data for each fold.

    Uses `cv_plan.y_by_horizon[fold.horizon]` (direct per-horizon targets)
    for y_train/y_test when present - i.e. whenever `cv_plan` was built by
    `create_walk_forward_splits` - so X_train/X_test stay origin-aligned
    features while y_train/y_test are the true horizon-shifted targets.
    Falls back to the plain `y` argument only for a `cv_plan` built some
    other way (e.g. hand-constructed in a test) without `y_by_horizon`.

    `y_test`'s index is relabelled from the origin date to the actual
    forecast target date (origin + horizon), since that is the date any
    caller doing time-series reporting (predictions_df, annual aggregation)
    actually cares about.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan

    Yields:
        Tuple of (X_train, y_train, X_test, y_test, fold)
    """
    for fold in cv_plan.folds:
        y_h = cv_plan.y_by_horizon.get(fold.horizon) if cv_plan.y_by_horizon else None
        if y_h is None:
            y_h = y

        X_train = X.iloc[fold.train_indices]
        y_train = y_h.iloc[fold.train_indices]
        X_test = X.iloc[fold.test_indices]
        y_test = y_h.iloc[fold.test_indices]

        if y_h is not y:
            target_positions = fold.test_indices + fold.horizon
            assert target_positions.max() < len(X.index), (
                f"generate_cv_folds: fold {fold.fold_id} (horizon={fold.horizon}) "
                f"target position {target_positions.max()} is out of bounds for "
                f"X of length {len(X.index)} - this should be unreachable given "
                f"create_walk_forward_splits' own bounds check."
            )
            y_test = y_test.copy()
            y_test.index = X.index[target_positions]

        yield X_train, y_train, X_test, y_test, fold


def validate_no_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> bool:
    """
    Validate that there is no temporal leakage between train and test sets.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        True if no leakage detected
    """
    # Check that all training dates are before test dates
    train_max = X_train.index.max()
    test_min = X_test.index.min()

    if train_max >= test_min:
        return False

    # Check for overlapping indices
    overlap = X_train.index.intersection(X_test.index)
    if len(overlap) > 0:
        return False

    return True


def save_cv_plan(cv_plan: CVPlan, output_path: Path):
    """Save CV plan to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as DataFrame
    df = cv_plan.to_dataframe()
    df.to_csv(output_path.with_suffix('.csv'), index=False)

    # Save as pickle for indices
    import pickle
    with open(output_path.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(cv_plan, f)


def load_cv_plan(path: Path) -> CVPlan:
    """Load CV plan from disk."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)
