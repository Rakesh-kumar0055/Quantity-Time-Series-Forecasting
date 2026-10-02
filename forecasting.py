from typing import Any

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    from statsmodels.tsa.arima_model import ARIMA

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def train_test_split(ts):
    """Chronological 80/20 train-test split."""

    split_index = int(
        len(ts) * 0.80
    )

    train = ts.iloc[:split_index].copy()
    test = ts.iloc[split_index:].copy()

    return train, test


def naive_forecast(train, test):

    forecast = pd.Series(
        train.iloc[-1],
        index=test.index,
        name="Naive Forecast"
    )

    return forecast


def train_arima(train, test):

    model: Any = ARIMA(
        train,
        order=(1, 1, 1)
    )

    fitted_model = model.fit()

    forecast = fitted_model.forecast(
        steps=len(test)
    )

    forecast = pd.Series(
        forecast,
        index=test.index
    )

    return fitted_model, forecast


def train_sarima(train, test):

    model: Any = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(
        disp=False
    )

    forecast = fitted_model.forecast(
        steps=len(test)
    )

    forecast = pd.Series(
        forecast,
        index=test.index
    )

    return fitted_model, forecast


def calculate_metrics(test, forecast):

    mae = mean_absolute_error(
        test,
        forecast
    )

    rmse = np.sqrt(
        mean_squared_error(
            test,
            forecast
        )
    )

    return mae, rmse