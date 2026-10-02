import pandas as pd


def prepare_time_series(df):
    """Clean the dataset and create a continuous daily quantity series."""

    required_columns = ["OrderDate", "total_qty_sales"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

    # Convert columns
    df["OrderDate"] = pd.to_datetime(
        df["OrderDate"],
        errors="coerce"
    )

    df["total_qty_sales"] = pd.to_numeric(
        df["total_qty_sales"],
        errors="coerce"
    )

    # Count invalid values
    invalid_dates = df["OrderDate"].isna().sum()
    invalid_quantity = df["total_qty_sales"].isna().sum()

    # Remove invalid records
    df = df.dropna(
        subset=required_columns
    ).copy()

    # Aggregate transactions by date
    daily_quantity = (
        df.groupby("OrderDate")["total_qty_sales"]
        .sum()
        .sort_index()
    )

    # Create continuous daily series
    ts = daily_quantity.asfreq("D")

    missing_dates = ts.isna().sum()

    # Following notebook assumption:
    # missing calendar dates are treated as zero sales
    ts = ts.fillna(0)

    print("\nData preparation completed.")
    print(f"Invalid dates removed    : {invalid_dates}")
    print(f"Invalid quantities removed: {invalid_quantity}")
    print(f"Missing calendar dates   : {missing_dates}")
    print(
        f"Date range               : "
        f"{ts.index.min()} to {ts.index.max()}"
    )

    return df, ts