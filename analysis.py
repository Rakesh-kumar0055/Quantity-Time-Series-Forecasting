from pathlib import Path
from importlib import import_module

import matplotlib.pyplot as plt
import seaborn as sns

try:
    seasonal_decompose = import_module(
        "statsmodels.tsa.seasonal"
    ).seasonal_decompose
except ModuleNotFoundError:
    seasonal_decompose = None


OUTPUT_DIR = Path("outputs/figures")


def create_output_directory():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def exploratory_analysis(df, ts):
    """Perform time-series exploratory analysis."""

    create_output_directory()

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nDaily quantity statistics:")
    print(ts.describe())

    # ------------------------------------------------
    # Daily quantity
    # ------------------------------------------------

    plt.figure(figsize=(15, 6))

    plt.plot(
        ts.index,
        ts.values,
        linewidth=1
    )

    plt.title(
        "Daily Quantity Sold Over Time",
        fontsize=16
    )

    plt.xlabel("Order Date")
    plt.ylabel("Total Quantity")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "daily_quantity_trend.png",
        dpi=300
    )

    plt.show()

    # ------------------------------------------------
    # Distribution
    # ------------------------------------------------

    plt.figure(figsize=(12, 5))

    sns.histplot(
        ts,
        bins=40,
        kde=True
    )

    plt.title(
        "Distribution of Daily Quantity",
        fontsize=16
    )

    plt.xlabel("Daily Quantity")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "daily_quantity_distribution.png",
        dpi=300
    )

    plt.show()

    # ------------------------------------------------
    # Weekly / Monthly
    # ------------------------------------------------

    weekly_quantity = ts.resample("W").sum()
    monthly_quantity = ts.resample("ME").sum()

    plt.figure(figsize=(15, 6))

    plt.plot(
        weekly_quantity.index,
        weekly_quantity.values
    )

    plt.title("Weekly Quantity Sold")
    plt.xlabel("Week")
    plt.ylabel("Quantity")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "weekly_quantity.png",
        dpi=300
    )

    plt.show()

    plt.figure(figsize=(15, 6))

    plt.plot(
        monthly_quantity.index,
        monthly_quantity.values
    )

    plt.title("Monthly Quantity Sold")
    plt.xlabel("Month")
    plt.ylabel("Quantity")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "monthly_quantity.png",
        dpi=300
    )

    plt.show()

    # ------------------------------------------------
    # Moving averages
    # ------------------------------------------------

    rolling_7 = ts.rolling(window=7).mean()
    rolling_30 = ts.rolling(window=30).mean()

    plt.figure(figsize=(15, 7))

    plt.plot(
        ts.index,
        ts.values,
        alpha=0.35,
        label="Daily Quantity"
    )

    plt.plot(
        rolling_7.index,
        rolling_7.values,
        linewidth=2,
        label="7-Day Moving Average"
    )

    plt.plot(
        rolling_30.index,
        rolling_30.values,
        linewidth=2,
        label="30-Day Moving Average"
    )

    plt.title(
        "Daily Quantity with Moving-Average Smoothing",
        fontsize=16
    )

    plt.xlabel("Order Date")
    plt.ylabel("Quantity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "moving_averages.png",
        dpi=300
    )

    plt.show()

    # ------------------------------------------------
    # Day of week
    # ------------------------------------------------

    dow_quantity = (
        df.assign(
            DayOfWeek=df["OrderDate"].dt.day_name()
        )
        .groupby("DayOfWeek")["total_qty_sales"]
        .sum()
    )

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    dow_quantity = dow_quantity.reindex(day_order)

    plt.figure(figsize=(11, 5))

    sns.barplot(
        x=dow_quantity.index,
        y=dow_quantity.values
    )

    plt.title(
        "Quantity Sold by Day of Week",
        fontsize=16
    )

    plt.xlabel("Day of Week")
    plt.ylabel("Total Quantity")
    plt.xticks(rotation=25)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "day_of_week_quantity.png",
        dpi=300
    )

    plt.show()

    # ------------------------------------------------
    # Seasonal decomposition
    # ------------------------------------------------

    if len(ts) >= 14 and seasonal_decompose is not None:

        decomposition = seasonal_decompose(
            ts,
            model="additive",
            period=7,
            extrapolate_trend="freq"
        )

        fig = decomposition.plot()

        fig.set_size_inches(
            15,
            10
        )

        fig.suptitle(
            "Weekly Seasonal Decomposition",
            y=1.02,
            fontsize=16
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "seasonal_decomposition.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
    elif len(ts) >= 14:
        print(
            "Skipping seasonal decomposition: install statsmodels to enable it."
        )

    return {
        "weekly": weekly_quantity,
        "monthly": monthly_quantity,
        "day_of_week": dow_quantity
    }