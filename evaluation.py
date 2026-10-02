from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("outputs/figures")


def compare_models(
    test,
    naive_forecast,
    arima_forecast,
    sarima_forecast,
    calculate_metrics
):

    naive_mae, naive_rmse = calculate_metrics(
        test,
        naive_forecast
    )

    arima_mae, arima_rmse = calculate_metrics(
        test,
        arima_forecast
    )

    sarima_mae, sarima_rmse = calculate_metrics(
        test,
        sarima_forecast
    )

    results = pd.DataFrame({
        "Model": [
            "Naive Baseline",
            "ARIMA(1,1,1)",
            "SARIMA(1,1,1)x(1,0,1,7)"
        ],
        "MAE": [
            naive_mae,
            arima_mae,
            sarima_mae
        ],
        "RMSE": [
            naive_rmse,
            arima_rmse,
            sarima_rmse
        ]
    })

    results = (
        results
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # Model comparison chart

    plt.figure(figsize=(11, 6))

    x = range(len(results))
    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        results["MAE"],
        width,
        label="MAE"
    )

    plt.bar(
        [i + width / 2 for i in x],
        results["RMSE"],
        width,
        label="RMSE"
    )

    plt.xticks(
        list(x),
        results["Model"].tolist(),
        rotation=15
    )

    plt.ylabel("Error")
    plt.title("Forecasting Model Comparison")
    plt.legend()
    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_DIR / "model_comparison.png",
        dpi=300
    )

    plt.show()

    # Actual vs forecast

    plt.figure(figsize=(15, 7))

    plt.plot(
        test.index,
        test.values,
        label="Actual Test Data",
        linewidth=2
    )

    plt.plot(
        arima_forecast.index,
        arima_forecast.values,
        label="ARIMA Forecast",
        linewidth=2
    )

    plt.plot(
        sarima_forecast.index,
        sarima_forecast.values,
        label="SARIMA Forecast",
        linewidth=2
    )

    plt.title(
        "Actual Quantity vs Forecasts",
        fontsize=16
    )

    plt.xlabel("Order Date")
    plt.ylabel("Total Quantity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "actual_vs_forecast.png",
        dpi=300
    )

    plt.show()

    return results