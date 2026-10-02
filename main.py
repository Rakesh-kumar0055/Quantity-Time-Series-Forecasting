from pathlib import Path
import sys
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

# Add both project root and src directory to the import path so imports work
# whether the project is opened from the workspace root or executed from elsewhere.
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# `src` is a directory in this project, not a package installed in site-packages.
# Import directly from the module names inside that folder so static analyzers and
# runtime execution agree on the import path.
from data_loader import load_data  # type: ignore[import-not-found]
from preprocessing import prepare_time_series  # type: ignore[import-not-found]
from analysis import exploratory_analysis  # type: ignore[import-not-found]
from forecasting import (  # type: ignore[import-not-found]
    train_test_split,
    naive_forecast,
    train_arima,
    train_sarima,
    calculate_metrics,
)
from evaluation import compare_models  # type: ignore[import-not-found]


DATA_PATH = Path(
    "data/Raw Data_Predictive Analysis.xlsx"
)

FORECAST_DAYS = 30


def main():

    print("=" * 70)
    print("QUANTITY TIME SERIES FORECASTING")
    print("=" * 70)

    # ------------------------------------------------
    # Load data
    # ------------------------------------------------

    df = load_data(
        DATA_PATH
    )

    print("\nFirst five rows:")
    print(df.head())

    # ------------------------------------------------
    # Preprocessing
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)

    df, ts = prepare_time_series(
        df
    )

    # ------------------------------------------------
    # EDA
    # ------------------------------------------------

    exploratory_analysis(
        df,
        ts
    )

    # ------------------------------------------------
    # Train test split
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAIN-TEST SPLIT")
    print("=" * 70)

    train, test = train_test_split(
        ts
    )

    print(
        f"Training observations: {len(train)}"
    )

    print(
        f"Testing observations : {len(test)}"
    )

    print(
        f"Training period: "
        f"{train.index.min()} to {train.index.max()}"
    )

    print(
        f"Testing period: "
        f"{test.index.min()} to {test.index.max()}"
    )

    # ------------------------------------------------
    # Naive baseline
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("NAIVE BASELINE")
    print("=" * 70)

    naive = naive_forecast(
        train,
        test
    )

    naive_mae, naive_rmse = calculate_metrics(
        test,
        naive
    )

    print(
        f"Naive MAE : {naive_mae:.4f}"
    )

    print(
        f"Naive RMSE: {naive_rmse:.4f}"
    )

    # ------------------------------------------------
    # ARIMA
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("ARIMA(1,1,1)")
    print("=" * 70)

    arima_model, arima = train_arima(
        train,
        test
    )

    arima_mae, arima_rmse = calculate_metrics(
        test,
        arima
    )

    print(
        f"ARIMA MAE : {arima_mae:.4f}"
    )

    print(
        f"ARIMA RMSE: {arima_rmse:.4f}"
    )

    # ------------------------------------------------
    # SARIMA
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("SARIMA(1,1,1)x(1,0,1,7)")
    print("=" * 70)

    sarima_model, sarima = train_sarima(
        train,
        test
    )

    sarima_mae, sarima_rmse = calculate_metrics(
        test,
        sarima
    )

    print(
        f"SARIMA MAE : {sarima_mae:.4f}"
    )

    print(
        f"SARIMA RMSE: {sarima_rmse:.4f}"
    )

    # ------------------------------------------------
    # Model comparison
    # ------------------------------------------------

    results = compare_models(
        test,
        naive,
        arima,
        sarima,
        calculate_metrics
    )

    # ------------------------------------------------
    # Select lowest RMSE
    # ------------------------------------------------

    model_objects = {
        "Naive Baseline": {
            "rmse": naive_rmse
        },
        "ARIMA(1,1,1)": {
            "rmse": arima_rmse
        },
        "SARIMA(1,1,1)x(1,0,1,7)": {
            "rmse": sarima_rmse
        }
    }

    selected_model_name = min(
        model_objects,
        key=lambda name:
        model_objects[name]["rmse"]
    )

    print("\n" + "=" * 70)
    print("SELECTED MODEL")
    print("=" * 70)

    print(
        f"Selected model: {selected_model_name}"
    )

    print(
        f"Test RMSE: "
        f"{model_objects[selected_model_name]['rmse']:.4f}"
    )

    # ------------------------------------------------
    # Train final model on complete data
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("FUTURE FORECAST")
    print("=" * 70)

    future_index = pd.date_range(
        start=ts.index[-1] + pd.Timedelta(days=1),
        periods=FORECAST_DAYS,
        freq="D"
    )

    if selected_model_name == "ARIMA(1,1,1)":

        arima_result = train_arima(
            ts,
            pd.Series(
                index=future_index,
                dtype=float
            )
        )
        final_model = cast(Any, arima_result[0])

        forecast_values = cast(Any, final_model).forecast(
            steps=FORECAST_DAYS
        )
        future_forecast = pd.Series(
            forecast_values,
            index=future_index
        )

    elif selected_model_name == "SARIMA(1,1,1)x(1,0,1,7)":

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        final_model = SARIMAX(
            ts,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(
            disp=False
        )

        forecast_values = cast(Any, final_model).forecast(
            steps=FORECAST_DAYS
        )
        future_forecast = pd.Series(
            forecast_values,
            index=future_index
        )

    else:

        future_forecast = pd.Series(
            [ts.iloc[-1]] * FORECAST_DAYS,
            index=future_index
        )

    future_forecast = pd.Series(
        future_forecast,
        index=future_index
    )

    # Quantity cannot be negative
    future_forecast = future_forecast.clip(
        lower=0
    )

    print(
        f"Forecast period: "
        f"{future_forecast.index.min().date()} "
        f"to "
        f"{future_forecast.index.max().date()}"
    )

    print(
        f"Average forecast quantity: "
        f"{future_forecast.mean():.2f}"
    )

    print(
        f"Minimum forecast quantity: "
        f"{future_forecast.min():.2f}"
    )

    print(
        f"Maximum forecast quantity: "
        f"{future_forecast.max():.2f}"
    )

    print(
        f"Total forecast quantity: "
        f"{future_forecast.sum():.2f}"
    )

    # ------------------------------------------------
    # Save forecast
    # ------------------------------------------------

    output_dir = Path(
        "outputs"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    forecast_df = future_forecast.to_frame(
        "Forecast Quantity"
    )

    forecast_df.to_csv(
        output_dir / "future_forecast.csv"
    )

    # ------------------------------------------------
    # Forecast graph
    # ------------------------------------------------

    history_window = min(
        90,
        len(ts)
    )

    plt.figure(
        figsize=(15, 7)
    )

    plt.plot(
        ts.index[-history_window:],
        ts.iloc[-history_window:].to_numpy(dtype=float, copy=False),
        label="Historical Quantity"
    )

    plt.plot(
        future_forecast.index,
        future_forecast.to_numpy(dtype=float, copy=False),
        label="Future Forecast",
        linewidth=2
    )

    plt.axvline(
        ts.index[-1],
        linestyle="--",
        alpha=0.7,
        label="Forecast Start"
    )

    plt.title(
        f"Next {FORECAST_DAYS} Days Quantity Forecast",
        fontsize=16
    )

    plt.xlabel("Order Date")
    plt.ylabel("Predicted Quantity")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "figures" /
        "future_forecast.png",
        dpi=300
    )

    plt.show()

    print("\n" + "=" * 70)
    print("PROJECT COMPLETED")
    print("=" * 70)

    print("\nGenerated outputs:")
    print("outputs/future_forecast.csv")
    print("outputs/figures/")


if __name__ == "__main__":
    main()