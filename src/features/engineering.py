"""
Feature engineering for CO2 forecasting framework.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from ..core.config import Config, FeatureConfig
from ..core.logging_utils import get_logger


def create_target_variable(
    df: pd.DataFrame,
    target_col: str,
    transform: str = 'log'
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Create target variable with transformation.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        transform: Transformation type ('log' or 'delta_log')

    Returns:
        Tuple of (transformed target series, transform metadata)
    """
    logger = get_logger()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    y_raw = df[target_col].copy()

    metadata = {
        'original_column': target_col,
        'transform': transform,
        'original_min': float(y_raw.min()),
        'original_max': float(y_raw.max()),
        'original_mean': float(y_raw.mean())
    }

    if transform == 'log':
        # Simple log transform
        y = np.log(y_raw)
        logger.info(f"Applied log transform to target. Range: {y.min():.4f} to {y.max():.4f}")

    elif transform == 'delta_log':
        # Log difference (growth rate)
        y = np.log(y_raw).diff()
        logger.info(f"Applied delta-log transform to target. Range: {y.min():.4f} to {y.max():.4f}")

    else:
        # No transform
        y = y_raw
        logger.info(f"No transform applied to target")

    metadata['transformed_min'] = float(y.min()) if not pd.isna(y.min()) else None
    metadata['transformed_max'] = float(y.max()) if not pd.isna(y.max()) else None

    return y, metadata


def create_direct_horizon_targets(
    y: pd.Series,
    horizons: List[int]
) -> Dict[int, pd.Series]:
    """
    Build direct (non-recursive) horizon-specific targets from a single
    origin-aligned target series.

    STRUCTURAL FIX (spec 3.1 / section 12): the feature matrix X is built so
    that row t contains only information available at forecast origin t
    (e.g. CO2e_lag1 at row t = CO2e_{t-1}). For a *direct* h-step-ahead
    forecast, the correct supervised pair is (X_t, y_{t+h}) - NOT (X_{t+h},
    y_{t+h}), which is what a naive "shift the train/test split boundary by
    h rows and reuse row-aligned X" walk-forward scheme implicitly does. The
    latter lets features like CO2e_lag1 at row t+h reference CO2e_{t+h-1},
    which is not observable at the true origin t for h > 1 - a leakage bug.

    This function returns, for each horizon h, the series y_h where
    y_h.loc[t] == y.loc[t + h] (aligned back onto the origin index t), so
    that direct per-horizon models are trained as (X_t, y_h.loc[t]) pairs.
    Rows near the end of the series where t+h falls outside the index become
    NaN and must be dropped by the caller together with the corresponding
    X rows.

    Args:
        y: Target series (original or transformed scale) with a
            DatetimeIndex at quarterly frequency, in temporal order.
        horizons: Horizons to build (e.g. [1, 2, 4]).

    Returns:
        Dict mapping horizon -> shifted target series aligned to the origin
        index (same index as `y`).
    """
    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("create_direct_horizon_targets requires y to have a DatetimeIndex")

    targets = {}
    for h in horizons:
        if h < 1:
            raise ValueError(f"Horizon must be >= 1, got {h}")
        targets[h] = y.shift(-h)
        targets[h].name = f"{y.name or 'target'}_h{h}"
    return targets


def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
    prefix: str = 'lag'
) -> pd.DataFrame:
    """
    Create lagged features for specified columns.

    Args:
        df: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag orders (e.g., [1, 2, 3, 4])
        prefix: Prefix for lag column names

    Returns:
        DataFrame with lag features
    """
    logger = get_logger()
    lag_features = {}

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping lag creation")
            continue

        for lag in lags:
            lag_col_name = f"{col}_{prefix}{lag}"
            lag_features[lag_col_name] = df[col].shift(lag)

    lag_df = pd.DataFrame(lag_features, index=df.index)
    logger.info(f"Created {len(lag_features)} lag features")

    return lag_df


def create_seasonality_features(
    df: pd.DataFrame,
    method: str = 'dummies'
) -> pd.DataFrame:
    """
    Create seasonality features.

    Args:
        df: DataFrame with datetime index
        method: 'dummies' for quarter dummies, 'sincos' for sine/cosine encoding

    Returns:
        DataFrame with seasonality features
    """
    logger = get_logger()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for seasonality features")

    quarter = df.index.quarter

    if method == 'dummies':
        # Quarter dummy variables (Q1 as reference)
        season_df = pd.DataFrame(index=df.index)
        for q in [2, 3, 4]:
            season_df[f'Q{q}'] = (quarter == q).astype(int)
        logger.info("Created quarter dummy features (Q2, Q3, Q4)")

    elif method == 'sincos':
        # Sine/cosine encoding
        # Quarter 1-4 maps to angle 0 to 2*pi*(3/4)
        angle = 2 * np.pi * (quarter - 1) / 4
        season_df = pd.DataFrame({
            'season_sin': np.sin(angle),
            'season_cos': np.cos(angle)
        }, index=df.index)
        logger.info("Created sine/cosine seasonality features")

    else:
        raise ValueError(f"Unknown seasonality method: {method}")

    return season_df


def create_shock_features(
    df: pd.DataFrame,
    config: FeatureConfig
) -> pd.DataFrame:
    """
    Create shock indicator features (COVID, energy crisis, etc.).

    Args:
        df: DataFrame with datetime index
        config: Feature configuration

    Returns:
        DataFrame with shock features
    """
    logger = get_logger()
    shock_df = pd.DataFrame(index=df.index)

    if config.include_covid_dummy:
        # COVID dummy
        from ..core.utils import quarter_to_date
        covid_start = quarter_to_date(config.covid_start)
        covid_end = quarter_to_date(config.covid_end)

        shock_df['COVID'] = ((df.index >= covid_start) & (df.index <= covid_end)).astype(int)
        logger.info(f"Created COVID dummy ({config.covid_start} to {config.covid_end})")

    if config.include_energy_crisis:
        # Energy crisis dummy
        from ..core.utils import quarter_to_date
        crisis_start = quarter_to_date(config.energy_crisis_start)
        crisis_end = quarter_to_date(config.energy_crisis_end)

        shock_df['EnergyCrisis'] = ((df.index >= crisis_start) & (df.index <= crisis_end)).astype(int)
        logger.info(f"Created Energy Crisis dummy ({config.energy_crisis_start} to {config.energy_crisis_end})")

    return shock_df


def create_rate_of_change_features(
    df: pd.DataFrame,
    columns: List[str],
    log_transform: bool = True
) -> pd.DataFrame:
    """
    Create rate-of-change (delta/growth) features.

    Args:
        df: Input DataFrame
        columns: Columns to create rate-of-change features for
        log_transform: If True, use log difference (growth rate); otherwise simple difference

    Returns:
        DataFrame with rate-of-change features
    """
    logger = get_logger()
    roc_features = {}

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping rate-of-change creation")
            continue

        if log_transform:
            # Log difference = growth rate (Δlog)
            # Handle zeros/negatives by adding small constant if needed
            series = df[col].copy()
            if (series <= 0).any():
                series = series + series[series > 0].min() * 0.01
            roc_features[f'{col}_dlog'] = np.log(series).diff()
        else:
            # Simple first difference
            roc_features[f'{col}_diff'] = df[col].diff()

    roc_df = pd.DataFrame(roc_features, index=df.index)
    logger.info(f"Created {len(roc_features)} rate-of-change features")

    return roc_df


def create_intensity_features(
    df: pd.DataFrame,
    target_col: str = 'CO2e',
    denominator_cols: Optional[List[str]] = None,
    min_lag: int = 1
) -> pd.DataFrame:
    """
    Create intensity/per-capita features (e.g. CO2e_per_Population).

    BUG FIX (spec 3.1): these features must never divide a contemporaneous,
    untransformed target value by a contemporaneous denominator, since that
    directly leaks CO2e_t into a predictor used to forecast CO2e_{t+h}. Both
    the numerator and denominator are therefore lagged by `min_lag` quarters
    before the ratio is formed, i.e.:

        CO2e_per_Population_t = CO2e_{t-min_lag} / Population_{t-min_lag}

    This is a historical, forecast-available ratio, not a same-quarter
    intensity measure. `min_lag` must be >= 1; callers requiring extra
    horizon-specific safety margin (e.g. H2/H4) can pass a larger value.

    Args:
        df: Input DataFrame
        target_col: Target column (numerator, will be lagged)
        denominator_cols: Columns to use as denominators (e.g., Population)
        min_lag: Minimum number of quarters to lag both numerator and
            denominator (must be >= 1; default 1).

    Returns:
        DataFrame with intensity features
    """
    logger = get_logger()
    intensity_features = {}

    if min_lag < 1:
        raise ValueError(
            f"create_intensity_features: min_lag must be >= 1 to avoid "
            f"target leakage (CO2e_t must never appear in a predictor used "
            f"to forecast CO2e_t or later), got min_lag={min_lag}"
        )

    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found")
        return pd.DataFrame(index=df.index)

    # Default denominator columns
    if denominator_cols is None:
        denominator_cols = ['Population']

    numerator_lagged = df[target_col].shift(min_lag)

    for denom_col in denominator_cols:
        if denom_col not in df.columns:
            logger.warning(f"Denominator column '{denom_col}' not found, skipping")
            continue

        # Lag denominator by the same amount, avoid division by zero
        denom_lagged = df[denom_col].shift(min_lag).replace(0, np.nan)
        intensity_features[f'{target_col}_per_{denom_col}'] = numerator_lagged / denom_lagged

    intensity_df = pd.DataFrame(intensity_features, index=df.index)
    logger.info(
        f"Created {len(intensity_features)} intensity features "
        f"(numerator and denominator both lagged by {min_lag} quarter(s))"
    )

    return intensity_df


def create_weather_features(
    df: pd.DataFrame,
    temp_col: str = 'Air_Temp',
    base_temp: float = 18.0
) -> pd.DataFrame:
    """
    Create weather-derived features like Heating Degree Days (HDD).

    Args:
        df: Input DataFrame
        temp_col: Temperature column name
        base_temp: Base temperature for HDD calculation (default 18°C)

    Returns:
        DataFrame with weather features
    """
    logger = get_logger()
    weather_features = {}

    if temp_col not in df.columns:
        logger.warning(f"Temperature column '{temp_col}' not found, skipping weather features")
        return pd.DataFrame(index=df.index)

    # Heating Degree Days proxy: max(0, base_temp - temp)
    weather_features['HDD_proxy'] = np.maximum(0, base_temp - df[temp_col])

    # Cooling Degree Days proxy: max(0, temp - base_temp)
    weather_features['CDD_proxy'] = np.maximum(0, df[temp_col] - base_temp)

    weather_df = pd.DataFrame(weather_features, index=df.index)
    logger.info(f"Created {len(weather_features)} weather features")

    return weather_df


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [4, 8],
    functions: List[str] = ['mean', 'std']
) -> pd.DataFrame:
    """
    Create rolling window features.

    Args:
        df: Input DataFrame
        columns: Columns to create rolling features for
        windows: Window sizes
        functions: Aggregation functions

    Returns:
        DataFrame with rolling features
    """
    logger = get_logger()
    rolling_features = {}

    for col in columns:
        if col not in df.columns:
            continue

        for window in windows:
            for func in functions:
                feat_name = f"{col}_roll{window}_{func}"
                if func == 'mean':
                    rolling_features[feat_name] = df[col].rolling(window=window, min_periods=1).mean()
                elif func == 'std':
                    rolling_features[feat_name] = df[col].rolling(window=window, min_periods=1).std()
                elif func == 'min':
                    rolling_features[feat_name] = df[col].rolling(window=window, min_periods=1).min()
                elif func == 'max':
                    rolling_features[feat_name] = df[col].rolling(window=window, min_periods=1).max()

    rolling_df = pd.DataFrame(rolling_features, index=df.index)
    logger.info(f"Created {len(rolling_features)} rolling features")

    return rolling_df


def engineer_features(
    df: pd.DataFrame,
    config: Config,
    target_col: str = None
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Main feature engineering function.

    Args:
        df: Input DataFrame
        config: Configuration object
        target_col: Target column name (overrides config)

    Returns:
        Tuple of (feature DataFrame, target Series, metadata)
    """
    logger = get_logger()
    logger.info("Starting feature engineering...")

    if target_col is None:
        target_col = config.data.target_column

    metadata = {
        'feature_columns': [],
        'lag_features': [],
        'rolling_features': [],
        'seasonality_features': [],
        'shock_features': [],
        'n_original_features': 0
    }

    # Create target
    y, target_meta = create_target_variable(
        df, target_col, config.data.target_transform
    )
    metadata['target_transform'] = target_meta

    # Get original features (excluding target)
    feature_cols = [c for c in df.columns if c != target_col]
    metadata['n_original_features'] = len(feature_cols)

    # Start with original features
    X = df[feature_cols].copy()

    # Hard removal (spec section 2): TEC, CEI, and anything derived from them
    # must never enter any candidate pool, not even behind an opt-in flag.
    # The raw source file may still contain these columns - drop them here,
    # structurally, before any downstream feature construction (lags,
    # rate-of-change, intensity ratios) can reference them, rather than
    # relying on the registry enforcement below to merely reject them.
    REMOVED_RAW_COLUMNS = ('TEC', 'CEI')
    removed_present = [c for c in REMOVED_RAW_COLUMNS if c in X.columns]
    if removed_present:
        X = X.drop(columns=removed_present)
        metadata['removed_columns'] = removed_present
        logger.info(f"Dropped removed raw columns per spec section 2: {removed_present}")

    # Create lag features
    lag_cols = config.features.lag_features
    if lag_cols:
        # If target is in lag_cols, use the raw target column
        if target_col in lag_cols or 'CO2e' in lag_cols:
            lag_target_df = create_lag_features(
                df[[target_col]],
                [target_col],
                config.features.lag_orders
            )
            X = pd.concat([X, lag_target_df], axis=1)
            metadata['lag_features'].extend(lag_target_df.columns.tolist())

        # Lag other features
        other_lag_cols = [c for c in lag_cols if c in X.columns]
        if other_lag_cols:
            lag_df = create_lag_features(
                X[other_lag_cols],
                other_lag_cols,
                config.features.lag_orders
            )
            X = pd.concat([X, lag_df], axis=1)
            metadata['lag_features'].extend(lag_df.columns.tolist())

    # Create rolling features (if enabled)
    if getattr(config.features, 'include_rolling_features', False):
        rolling_cols = getattr(config.features, 'rolling_columns', ['CO2e'])
        rolling_windows = getattr(config.features, 'rolling_windows', [4, 8])
        rolling_funcs = getattr(config.features, 'rolling_functions', ['mean', 'std'])

        # Use target column for rolling features
        roll_df_target = df[[target_col]].copy()
        roll_df_target.columns = ['CO2e']  # Normalize name

        rolling_df = create_rolling_features(
            roll_df_target,
            ['CO2e'],
            windows=rolling_windows,
            functions=rolling_funcs
        )
        # Shift rolling features by 1 to avoid data leakage
        rolling_df = rolling_df.shift(1)
        X = pd.concat([X, rolling_df], axis=1)
        metadata['rolling_features'] = rolling_df.columns.tolist()
        logger.info(f"Created {len(rolling_df.columns)} rolling features (shifted by 1 to avoid leakage)")

    # Create rate-of-change features (if enabled)
    if getattr(config.features, 'include_roc_features', False):
        roc_cols = getattr(config.features, 'roc_columns', ['GDP'])
        if 'TEC' in roc_cols:
            raise ValueError(
                "roc_columns includes 'TEC', which is removed per spec section 2 "
                "(no TEC-derived feature may be created, not even opt-in)."
            )
        # Also add target column rate-of-change
        roc_all_cols = roc_cols + [target_col]
        roc_df = create_rate_of_change_features(
            df[list(set(roc_all_cols) & set(df.columns))],
            list(set(roc_all_cols) & set(df.columns)),
            log_transform=True
        )
        # Shift by 1 to avoid leakage for target-related features
        for col in roc_df.columns:
            if target_col in col:
                roc_df[col] = roc_df[col].shift(1)
        X = pd.concat([X, roc_df], axis=1)
        metadata['roc_features'] = roc_df.columns.tolist()
        logger.info(f"Created {len(roc_df.columns)} rate-of-change features")

    # Create intensity features (if enabled) - opt-in, default off (spec 3.1)
    if getattr(config.features, 'include_intensity_features', False):
        intensity_denominators = getattr(config.features, 'intensity_denominators', ['Population'])
        if 'TEC' in intensity_denominators:
            raise ValueError(
                "intensity_denominators includes 'TEC', which is removed per spec "
                "section 2 (no TEC-derived feature may be created, not even opt-in)."
            )
        intensity_min_lag = getattr(config.features, 'intensity_min_lag', 1)
        intensity_df = create_intensity_features(
            df,
            target_col=target_col,
            denominator_cols=intensity_denominators,
            min_lag=intensity_min_lag
        )
        X = pd.concat([X, intensity_df], axis=1)
        metadata['intensity_features'] = intensity_df.columns.tolist()
        logger.info(f"Created {len(intensity_df.columns)} intensity features")

    # NOTE: CEI is removed per spec section 2 (no CEI-derived feature may be
    # created, not even opt-in) - the raw column is already dropped above, and
    # no lagged-CEI feature is constructed here anymore (previously
    # create_lagged_cei_feature / CEI_lag1, spec 3.2's now-superseded
    # governance approach).

    # Create weather features (if enabled)
    if getattr(config.features, 'include_weather_features', False):
        temp_col = getattr(config.features, 'temperature_column', 'Air_Temp')
        base_temp = getattr(config.features, 'hdd_base_temp', 18.0)
        weather_df = create_weather_features(df, temp_col=temp_col, base_temp=base_temp)
        X = pd.concat([X, weather_df], axis=1)
        metadata['weather_features'] = weather_df.columns.tolist()
        logger.info(f"Created {len(weather_df.columns)} weather features")

    # Create seasonality features
    season_df = create_seasonality_features(df, config.features.seasonality_type)
    X = pd.concat([X, season_df], axis=1)
    metadata['seasonality_features'] = season_df.columns.tolist()

    # Create shock features
    shock_df = create_shock_features(df, config.features)
    X = pd.concat([X, shock_df], axis=1)
    metadata['shock_features'] = shock_df.columns.tolist()

    # Predictor-governance enforcement (spec section 5 / bug-audit Finding
    # A-1): config/feature_registry.yaml is meant to be the single source of
    # truth for which columns are allowed into any model matrix, but until
    # now nothing actually consulted it here - FeatureRegistry.enforce()
    # existed but was unused, so a feature that drifted out of the registry
    # (e.g. a rename, or a new column added to `df` upstream) could silently
    # enter X. When enabled, this raises on any X column absent from the
    # registry, and on any column present but not `retained_after_audit`.
    # No opt-in override set exists any more: the one prior use case
    # (raw contemporaneous CEI) is gone now that CEI is fully removed
    # per spec section 2, not merely excluded-by-default.
    if getattr(config.features, 'enforce_availability_registry', True):
        from .registry import FeatureRegistry

        registry = FeatureRegistry.load(getattr(config, 'feature_registry_path', 'config/feature_registry.yaml'))

        explicit_overrides = set()

        unknown = [c for c in X.columns if c not in registry.entries]
        if unknown:
            raise ValueError(
                f"enforce_availability_registry=True but the following X "
                f"columns are not present in {registry.source_path}: "
                f"{unknown}. Add them to the feature registry with an "
                f"explicit leakage/availability classification before they "
                f"can enter a model matrix."
            )

        not_retained = [c for c in X.columns if not registry.is_retained(c)]
        blocked = [c for c in not_retained if c not in explicit_overrides]
        if blocked:
            raise ValueError(
                f"enforce_availability_registry=True but the following X "
                f"columns are marked retained_after_audit: false in "
                f"{registry.source_path} and were not reached via an "
                f"explicit sensitivity opt-in: {blocked}. Either exclude "
                f"them upstream or add the corresponding opt-in flag."
            )

        metadata['registry_governance'] = {
            'enforced': True,
            'registry_path': str(registry.source_path),
            'n_registry_entries': len(registry),
            'explicit_overrides_used': sorted(explicit_overrides),
        }
        logger.info(
            f"Feature registry governance enforced: all {len(X.columns)} X "
            f"columns are known and retained_after_audit "
            f"(overrides: {sorted(explicit_overrides) or 'none'})"
        )
    else:
        metadata['registry_governance'] = {'enforced': False}

    metadata['feature_columns'] = X.columns.tolist()
    metadata['n_total_features'] = len(X.columns)

    logger.info(f"Feature engineering complete. Total features: {metadata['n_total_features']}")

    return X, y, metadata


def create_feature_dictionary(
    X: pd.DataFrame,
    metadata: Dict[str, Any],
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create a feature dictionary describing all features.

    Args:
        X: Feature DataFrame
        metadata: Feature metadata
        output_path: Path to save dictionary (optional)

    Returns:
        Feature dictionary DataFrame
    """
    feature_dict = []

    for col in X.columns:
        entry = {
            'feature': col,
            'dtype': str(X[col].dtype),
            'n_unique': X[col].nunique(),
            'min': X[col].min() if X[col].dtype in [np.float64, np.int64] else None,
            'max': X[col].max() if X[col].dtype in [np.float64, np.int64] else None,
            'mean': X[col].mean() if X[col].dtype in [np.float64, np.int64] else None,
            'missing_pct': (X[col].isnull().sum() / len(X)) * 100
        }

        # Categorize feature type
        if col in metadata.get('lag_features', []):
            entry['category'] = 'lag'
        elif col in metadata.get('rolling_features', []):
            entry['category'] = 'rolling'
        elif col in metadata.get('seasonality_features', []):
            entry['category'] = 'seasonality'
        elif col in metadata.get('shock_features', []):
            entry['category'] = 'shock'
        else:
            entry['category'] = 'original'

        feature_dict.append(entry)

    df_dict = pd.DataFrame(feature_dict)

    if output_path:
        df_dict.to_csv(output_path, index=False)

    return df_dict
