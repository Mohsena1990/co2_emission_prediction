"""
Tests for LSTM sequence-alignment fixes (bug 3.4): predict() must always
return exactly len(y_test) predictions, with no zero/repeat-padding
fallback, and must raise explicitly on any contract violation.

Skips gracefully if torch is not installed in this environment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from src.models.lstm import LSTMModel
from src.core.utils import assert_prediction_alignment


def _make_series(n=60, n_features=3, seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.date_range('2000-01-01', periods=n, freq='QS')
    X = pd.DataFrame(rng.randn(n, n_features), index=dates,
                      columns=[f'f{i}' for i in range(n_features)])
    y = pd.Series(100 + rng.randn(n).cumsum(), index=dates, name='y')
    return X, y


def _tiny_lstm(lookback=4):
    return LSTMModel(params={
        'lookback': lookback, 'hidden_size': 4, 'num_layers': 1,
        'dropout': 0.0, 'learning_rate': 0.01, 'batch_size': 8,
        'max_epochs': 3, 'patience': 3, 'seed': 42,
    })


class TestLSTMPredictionLength:
    @pytest.mark.parametrize("lookback,n_test", [
        (4, 1), (4, 4), (4, 8), (8, 1), (8, 4), (12, 1), (2, 1),
    ])
    def test_len_y_pred_equals_len_y_test(self, lookback, n_test):
        X, y = _make_series(n=50)
        model = _tiny_lstm(lookback=lookback)
        train_size = 40
        X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
        X_test = X.iloc[train_size:train_size + n_test]
        y_test = y.iloc[train_size:train_size + n_test]

        model.fit(X_train, y_train)
        predict_input = model.build_predict_input(X_train, X_test)
        y_pred = model.predict(predict_input, n_targets=len(X_test))

        assert len(y_pred) == len(y_test)
        # Must not raise
        assert_prediction_alignment(y_test, y_pred, context="test")

    def test_lookback_shorter_than_training_window(self):
        X, y = _make_series(n=50)
        model = _tiny_lstm(lookback=4)
        X_train, y_train = X.iloc[:40], y.iloc[:40]
        X_test, y_test = X.iloc[40:44], y.iloc[40:44]
        model.fit(X_train, y_train)
        predict_input = model.build_predict_input(X_train, X_test)
        y_pred = model.predict(predict_input, n_targets=len(X_test))
        assert len(y_pred) == 4

    def test_lookback_greater_than_outer_test_length(self):
        # test fold has 1 row, lookback is 12 -> must still work via
        # build_predict_input pulling history from training data.
        X, y = _make_series(n=50)
        model = _tiny_lstm(lookback=12)
        X_train, y_train = X.iloc[:40], y.iloc[:40]
        X_test, y_test = X.iloc[40:41], y.iloc[40:41]
        model.fit(X_train, y_train)
        predict_input = model.build_predict_input(X_train, X_test)
        assert len(predict_input) == 13  # 12 history + 1 test row
        y_pred = model.predict(predict_input, n_targets=len(X_test))
        assert len(y_pred) == 1

    def test_one_row_and_multi_row_test_periods(self):
        X, y = _make_series(n=50)
        model = _tiny_lstm(lookback=4)
        X_train, y_train = X.iloc[:40], y.iloc[:40]
        model.fit(X_train, y_train)

        for n_test in (1, 2, 4, 6):
            X_test = X.iloc[40:40 + n_test]
            predict_input = model.build_predict_input(X_train, X_test)
            y_pred = model.predict(predict_input, n_targets=n_test)
            assert len(y_pred) == n_test

    def test_predict_without_history_raises_instead_of_padding(self):
        # Regression guard: the old behaviour silently padded with repeated
        # rows and returned a wrong-length result. Now it must raise.
        X, y = _make_series(n=50)
        model = _tiny_lstm(lookback=8)
        model.fit(X.iloc[:40], y.iloc[:40])

        X_test_only = X.iloc[40:42]  # only 2 rows, no history prepended
        with pytest.raises(ValueError):
            model.predict(X_test_only)

    def test_inconsistent_n_targets_raises(self):
        X, y = _make_series(n=50)
        model = _tiny_lstm(lookback=4)
        X_train, y_train = X.iloc[:40], y.iloc[:40]
        X_test = X.iloc[40:44]
        model.fit(X_train, y_train)
        predict_input = model.build_predict_input(X_train, X_test)
        with pytest.raises(ValueError):
            model.predict(predict_input, n_targets=999)

    def test_fit_raises_clearly_when_too_few_rows_for_lookback(self):
        # Regression guard: previously this crashed deep inside PyTorch's
        # DataLoader/RandomSampler with a confusing "num_samples=0" error.
        X, y = _make_series(n=20)
        model = _tiny_lstm(lookback=12)
        X_train, y_train = X.iloc[:8], y.iloc[:8]  # only 8 rows < lookback
        with pytest.raises(ValueError, match="lookback"):
            model.fit(X_train, y_train)

    def test_build_predict_input_raises_on_insufficient_history(self):
        X, y = _make_series(n=30)
        model = _tiny_lstm(lookback=12)
        X_train, y_train = X.iloc[:16], y.iloc[:16]  # 4 sequences, fits fine
        model.fit(X_train, y_train)
        # build_predict_input needs the tail lookback=12 rows as history;
        # deliberately pass a too-short slice.
        short_history = X_train.iloc[:8]
        X_test = X.iloc[16:18]
        with pytest.raises(ValueError):
            model.build_predict_input(short_history, X_test)


class TestAssertPredictionAlignment:
    def test_matching_lengths_pass(self):
        assert_prediction_alignment([1, 2, 3], [1.0, 2.0, 3.0])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            assert_prediction_alignment([1, 2, 3], [1.0, 2.0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
