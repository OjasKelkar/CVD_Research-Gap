# missing_data.py\n\n
import numpy as np
import pandas as pd

# Missingness levels used in our experiment
MISSINGNESS_LEVELS = [0.0, 0.10, 0.20, 0.30]


def introduce_missing_values(X, missing_rate, random_state=42):
    """
    Introduce MCAR (Missing Completely At Random)
    into predictor features only.

    Parameters:
        X: Input feature DataFrame
        missing_rate: Percentage of values to mask
        random_state: Random seed

    Returns:
        DataFrame containing simulated missing values
    """

    if not 0 <= missing_rate < 1:
        raise ValueError("Missing rate must be between 0 and 1.")

    X_missing = X.copy()

    # Use a fixed random generator
    rng = np.random.default_rng(random_state)

    # Select only existing non-missing cells
    valid_positions = np.argwhere(X_missing.notna().to_numpy())

    # Calculate number of cells to mask
    n_missing = int(len(valid_positions) * missing_rate)

    if n_missing > 0:
        selected_indices = rng.choice(
            len(valid_positions),
            size=n_missing,
            replace=False
        )

        selected_positions = valid_positions[selected_indices]

        for row, col in selected_positions:
            X_missing.iat[row, col] = np.nan

    print(f"\nMissingness level: {missing_rate * 100:.0f}%")
    print("Additional missing values:", n_missing)
    print(
        "Total missing values:",
        int(X_missing.isnull().sum().sum())
    )

    return X_missing


def generate_missingness_datasets(X, random_state=42):
    """
    Generate datasets for all missingness conditions.
    """

    datasets = {}

    for rate in MISSINGNESS_LEVELS:

        datasets[rate] = introduce_missing_values(
            X,
            missing_rate=rate,
            random_state=random_state
        )

    return datasets


if __name__ == "__main__":

    from src.data_loader import get_data

    X, y = get_data()

    datasets = generate_missingness_datasets(X)

    print("\n========== MISSING DATA SUMMARY ==========")

    for rate, data in datasets.items():

        print(f"\nCondition: {rate * 100:.0f}%")

        print("Dataset shape:", data.shape)

        print(
            "Missing values:",
            data.isnull().sum().sum()
        )
