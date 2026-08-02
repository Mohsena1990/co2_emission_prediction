"""
General utilities for CO2 forecasting framework.
"""
import re
import numpy as np
import pandas as pd
import json
import pickle
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(data: Any, path: Union[str, Path], indent: int = 2):
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """Load data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(obj: Any, path: Union[str, Path]):
    """Save object to pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load object from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def hash_dict(d: Dict) -> str:
    """Create a hash of a dictionary for caching/identification."""
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _make_quarter_timestamp(year: int, quarter: int) -> pd.Timestamp:
    """Build the first-of-quarter Timestamp for a given (year, quarter)."""
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"Quarter must be in 1-4, got {quarter} (year={year})")
    month = (quarter - 1) * 3 + 1
    return pd.Timestamp(year=year, month=month, day=1)


_QUARTER_STRING_RE = re.compile(r'^(?P<year>\d{4})[\s\-_/]?Q(?P<q>[1-4])$', re.IGNORECASE)


def quarter_to_date(value: Any) -> pd.Timestamp:
    """
    Convert a quarter identifier to the first-of-quarter Timestamp.

    This is the single, canonical quarter parser for the framework - do not
    duplicate this logic elsewhere. Supported inputs:

    - Strings: '1999Q1', '1999-Q1', '1999_Q1', '1999/Q1', '1999 Q1' (case-insensitive)
    - Decimal year.quarter encodings as produced by the source spreadsheet,
      e.g. 1999.1 -> 1999Q1, 1999.4 -> 1999Q4. The fractional digit IS the
      quarter number (not a calendar fraction of the year); only .1-.4 are
      valid and anything else raises ValueError rather than being guessed.
    - pandas.Timestamp / datetime.date / datetime.datetime (mapped to the
      first day of the quarter containing that date).
    - Plain 4-digit year integers are rejected (ambiguous - no quarter given).

    Raises:
        ValueError: if the value cannot be unambiguously mapped to a quarter.
    """
    # Timestamp / datetime passthrough
    if isinstance(value, pd.Timestamp):
        return _make_quarter_timestamp(value.year, value.quarter)
    if isinstance(value, (datetime, date)):
        ts = pd.Timestamp(value)
        return _make_quarter_timestamp(ts.year, ts.quarter)

    # String formats: 'YYYYQn', 'YYYY-Qn', 'YYYY_Qn', 'YYYY/Qn', 'YYYY Qn'
    if isinstance(value, str):
        s = value.strip()
        m = _QUARTER_STRING_RE.match(s)
        if m:
            return _make_quarter_timestamp(int(m.group('year')), int(m.group('q')))

        # Fall back to a bare decimal string, e.g. "1999.1"
        try:
            return quarter_to_date(float(s))
        except (ValueError, TypeError):
            pass

        raise ValueError(
            f"Cannot parse quarter string: {value!r}. Expected formats like "
            f"'1999Q1', '1999-Q1', or decimal '1999.1'."
        )

    # Numeric decimal encodings: YYYY.n where n in {1,2,3,4} is the quarter digit
    if isinstance(value, (int, float, np.integer, np.floating)):
        year = int(np.floor(value))
        frac = (float(value) - year) * 10
        q = int(round(frac))
        if q < 1 or q > 4 or abs(frac - q) > 1e-6:
            raise ValueError(
                f"Cannot interpret decimal quarter encoding {value!r}: fractional "
                f"part does not map to a valid quarter 1-4 (got fractional digit "
                f"{frac:.4f}). Refusing to silently guess."
            )
        return _make_quarter_timestamp(year, q)

    raise ValueError(f"Cannot parse quarter value of type {type(value)}: {value!r}")


def year_quarter_to_date(year: Any, quarter: Any) -> pd.Timestamp:
    """
    Convert separate integer year and quarter columns/values to a Timestamp.
    Use this when the source data provides year and quarter as two distinct
    integer columns rather than a single combined identifier.
    """
    return _make_quarter_timestamp(int(year), int(quarter))


def date_to_quarter(date: pd.Timestamp) -> str:
    """Convert timestamp to quarter string."""
    return f"{date.year}Q{date.quarter}"


def ensure_quarterly_index(df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
    """
    Ensure DataFrame has a quarterly datetime index.
    """
    df = df.copy()

    if date_col is not None and date_col in df.columns:
        df[date_col] = df[date_col].apply(quarter_to_date)
        df = df.set_index(date_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert index
        df.index = df.index.map(quarter_to_date)

    df.index = pd.DatetimeIndex(df.index, freq='QS')
    df.index.name = 'date'
    return df


def calculate_weighted_mae(
    errors_by_horizon: Dict[int, float],
    weights: Dict[int, float]
) -> float:
    """
    Calculate weighted MAE across horizons.

    Args:
        errors_by_horizon: MAE for each horizon {1: mae1, 2: mae2, 4: mae4}
        weights: Weights for each horizon {1: 0.5, 2: 0.3, 4: 0.2}

    Returns:
        Weighted MAE
    """
    total = 0.0
    for h, mae in errors_by_horizon.items():
        w = weights.get(h, 0.0)
        total += w * mae
    return total


def inverse_log_transform(y_log: np.ndarray) -> np.ndarray:
    """Inverse of a plain log transform (target_transform='log')."""
    return np.exp(y_log)


def to_original_scale(values: np.ndarray, transform: str) -> np.ndarray:
    """
    Invert config.data.target_transform so every reported metric/prediction
    is on original-scale CO2e (spec section 2/17: "All evaluation metrics,
    plots, annual totals, and safeguards must use original-scale values" /
    "Calculate all metrics on original-scale CO2e"). Models may still be
    TRAINED on the transformed target (a standard, valid choice) - nothing
    downstream of this function should ever see transformed values.

    Bug history: this was missing from src/pipeline/experiment.py's
    original-scale reporting for the entire time that module existed (found
    only by noticing a reported weighted_mae under 0.1 was physically
    implausible for CO2e measured in hundreds of thousands of tonnes), and
    was ALSO missing from the pre-existing scripts/04_evaluate_and_safeguards.py's
    quarterly-level metrics (only the separate annual-consistency check
    there ever called inverse_log_transform). Both now share this one
    function rather than risk a third divergent implementation.

    Args:
        values: Array of target-scale values (e.g. y_test.values, y_pred).
        transform: config.data.target_transform ('log', 'none'/None, or
            other).

    Returns:
        Original-scale values.

    Raises:
        ValueError: for any transform this function does not know how to
            invert (spec 3.3: raise a clear error rather than silently
            reporting transformed-scale numbers as if they were original-
            scale). Only 'log' and 'none' are currently supported -
            'delta_log' requires a per-fold reference level from the
            untransformed series that callers do not currently thread
            through.
    """
    if transform in (None, 'none'):
        return np.asarray(values)
    if transform == 'log':
        return inverse_log_transform(np.asarray(values))
    raise ValueError(
        f"to_original_scale: original-scale inversion for "
        f"target_transform='{transform}' is not implemented - only 'log' "
        f"(the default) and 'none' are supported. Reporting transformed-"
        f"scale metrics as if they were original-scale CO2e is exactly what "
        f"spec section 2/17 forbids; this raises instead of doing that "
        f"silently (spec 3.3's inversion-safety principle applied here too)."
    )


def invert_delta_log(
    log_diffs: Union[np.ndarray, List[float]],
    reference_log_level: float
) -> np.ndarray:
    """
    Reconstruct original-scale target levels from predicted log-differences
    (target_transform='delta_log').

    Given the log level of the target at the forecast origin (the last truly
    observed value before the forecast window - never a predicted or test-set
    value), reconstruct the forecasted levels as:

        y_hat[t] = exp(reference_log_level + cumsum(log_diffs)[t])

    Args:
        log_diffs: Sequence of predicted log-differences (Delta log y), in
            forecast-order starting immediately after the reference period.
        reference_log_level: log(y) at the forecast origin (t=0), i.e. the
            last available true value before the first predicted diff.

    Returns:
        Array of reconstructed original-scale levels, same length as log_diffs.

    Raises:
        ValueError: if reference_log_level is missing/non-finite or log_diffs
            contains non-finite values - a delta_log forecast cannot be safely
            reconstructed without a valid anchor, and returning a bogus level
            silently would be worse than failing loudly.
    """
    log_diffs = np.asarray(log_diffs, dtype=float)

    if reference_log_level is None or not np.isfinite(reference_log_level):
        raise ValueError(
            "invert_delta_log requires a finite reference_log_level (the log "
            "of the last observed target value at the forecast origin). "
            "delta_log forecasts cannot be reconstructed without it."
        )
    if log_diffs.size == 0:
        return log_diffs
    if not np.all(np.isfinite(log_diffs)):
        raise ValueError(
            "invert_delta_log received non-finite predicted log-differences; "
            "refusing to reconstruct levels from them."
        )

    cumulative = np.cumsum(log_diffs)
    return np.exp(reference_log_level + cumulative)


def assert_prediction_alignment(y_test: Any, y_pred: Any, context: str = "") -> None:
    """
    Enforce the invariant `len(y_pred) == len(y_test)` explicitly (bug 3.4).

    Every model-evaluation loop in the pipeline must call this immediately
    after `model.predict(...)` and before zipping/aligning predictions with
    actuals. Silent truncation via `zip()` or silent broadcasting via numpy
    must never be allowed to hide a shape mismatch - this raises a clear
    exception instead.

    Args:
        y_test: Test-fold actual values (anything with __len__).
        y_pred: Model predictions (anything with __len__).
        context: Optional extra context (e.g. "model=lstm, fold=3, h=2") to
            include in the error message.

    Raises:
        ValueError: if len(y_pred) != len(y_test).
    """
    n_test = len(y_test)
    n_pred = len(y_pred)
    if n_test != n_pred:
        ctx = f" ({context})" if context else ""
        raise ValueError(
            f"Prediction length mismatch{ctx}: len(y_pred)={n_pred} != "
            f"len(y_test)={n_test}. Refusing to silently truncate via zip() "
            f"or broadcast - fix the model's predict() contract instead."
        )


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    return a / b if b != 0 else default


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_year_from_quarter(date: pd.Timestamp) -> int:
    """Get year from quarterly timestamp."""
    return date.year


def aggregate_to_annual(
    quarterly_values: pd.Series,
    agg_func: str = 'sum'
) -> pd.Series:
    """Aggregate quarterly values to annual."""
    annual = quarterly_values.groupby(quarterly_values.index.year)
    if agg_func == 'sum':
        return annual.sum()
    elif agg_func == 'mean':
        return annual.mean()
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Period):
            return str(obj)
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        return super().default(obj)


def save_json_numpy(data: Any, path: Union[str, Path], indent: int = 2):
    """Save data to JSON file with numpy support."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)
