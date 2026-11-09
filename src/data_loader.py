# data_loader.py\n\n
import pandas as pd
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset location
DATASET_PATH = PROJECT_ROOT / "data" / "processed.cleveland.data"

# Dataset column names
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num"
]

TARGET_COLUMN = "num"


def load_dataset():
    """
    Load the Cleveland Heart Disease dataset.
    Missing values represented by '?' are converted to NaN.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}\n"
            "Please place processed.cleveland.data inside the data folder."
        )

    df = pd.read_csv(
        DATASET_PATH,
        header=None,
        names=COLUMNS,
        na_values=["?"],
        skipinitialspace=True
    )

    print("Dataset loaded successfully!")

    return df


def clean_dataset(df):
    """
    Convert dataset columns into numeric format
    and remove records with missing target values.
    """

    df = df.copy()

    for column in COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove records where target is missing
    df = df.dropna(subset=[TARGET_COLUMN])

    print("Dataset cleaning completed!")

    return df


def split_features_target(df):
    """
    Separate independent variables and target variable.

    Original target:
    0 = No heart disease
    1,2,3,4 = Heart disease

    Convert into binary classification:
    0 = No disease
    1 = Disease
    """

    X = df.drop(columns=[TARGET_COLUMN])

    y = (df[TARGET_COLUMN] > 0).astype(int)

    print("Features and target separated!")

    return X, y


def get_data():

    df = load_dataset()

    df = clean_dataset(df)

    X, y = split_features_target(df)

    return X, y


# Execute this section when running the file directly
if __name__ == "__main__":

    X, y = get_data()

    print("\n========== DATASET INFORMATION ==========")

    print("Number of records:", X.shape[0])

    print("Number of features:", X.shape[1])

    print("\nTarget Class Distribution:")

    print(y.value_counts().sort_index())

    print("\nMissing Values in Each Feature:")

    print(X.isnull().sum())

    print("\nFirst Five Records:")

    print(X.head())
