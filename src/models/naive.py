"""
Naive baselines (spec section 18): required reference candidates alongside
the 5 tuned forecasting models, NOT counted toward Stream A's 20 or Stream
B's 100 candidates (they carry no configuration/FS identity - run once,
attached to both streams' comparison tables as reference rows).

    naive_lag1 (persistence): y_hat_{origin+h} = y_origin
        The last actual value known at the forecast origin, repeated for
        every horizon h. Always leakage-safe (origin is always known).

    naive_lag4 (seasonal naive): y_hat_{origin+h} = y_{origin+h-4}
        For h in {1, 2, 4}, origin+h-4 <= origin, so this always resolves
        to a value at or before the forecast origin - never the target
        quarter or later. Falls back to the persistence value if the
        target-minus-4-quarters date isn't in the training history (can
        happen for h=4 in the earliest fold, where origin+h-4 == origin
        exactly - which IS in history - so this fallback is defensive only).

Both fit() on a pandas Series with a quarterly DatetimeIndex (matches the
rest of the pipeline's X_train/y_train contract) and predict() from the
test DataFrame's DatetimeIndex, ignoring the feature columns entirely -
consistent with these being feature-free reference forecasts.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from .base import BaseForecaster, ModelRegistry

_QUARTER_OFFSET = pd.DateOffset(months=3)


@ModelRegistry.register('naive_lag1')
class NaiveLag1Forecaster(BaseForecaster):
    """Persistence baseline: y_hat_{origin+h} = y_origin for every horizon."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__('naive_lag1', params)
        self._last_value: Optional[float] = None

    def fit(self, X, y: pd.Series) -> 'NaiveLag1Forecaster':
        self._last_value = float(y.iloc[-1])
        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        n = len(X) if hasattr(X, '__len__') else 1
        return np.full(n, self._last_value, dtype=float)


@ModelRegistry.register('naive_lag4')
class NaiveLag4Forecaster(BaseForecaster):
    """Seasonal-naive baseline: y_hat_{origin+h} = y_{origin+h-4}."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__('naive_lag4', params)
        self._y_train: Optional[pd.Series] = None

    def fit(self, X, y: pd.Series) -> 'NaiveLag4Forecaster':
        self._y_train = y.copy()
        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(X, 'index'):
            raise ValueError(
                "NaiveLag4Forecaster.predict requires a DatetimeIndex-bearing "
                "X (test features) to resolve each target date's t-4 lookup."
            )
        fallback = float(self._y_train.iloc[-1])
        preds = []
        for target_date in X.index:
            lookup_date = target_date - 4 * _QUARTER_OFFSET
            if lookup_date in self._y_train.index:
                preds.append(float(self._y_train.loc[lookup_date]))
            else:
                preds.append(fallback)
        return np.array(preds, dtype=float)
