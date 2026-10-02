import pandas as pd
from pathlib import Path


def load_data(file_path):
    """Load the Excel forecasting dataset."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {file_path}\n"
            "Place 'Raw Data_Predictive Analysis.xlsx' inside the data folder."
        )

    df = pd.read_excel(file_path)

    print("\nDataset loaded successfully.")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    return df