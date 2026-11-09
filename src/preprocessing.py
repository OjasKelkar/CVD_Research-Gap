# preprocessing.py\n\n
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import (
    SimpleImputer,
    KNNImputer,
    IterativeImputer
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Numerical features
NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

# Categorical features
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]


def get_preprocessor(strategy="median"):
    """
    Create a preprocessing pipeline using the selected
    numerical imputation strategy.

    Available strategies:
        median
        knn
        iterative

    Categorical features use most-frequent imputation.
    One-hot encoding is applied to categorical features.
    """

    strategy = strategy.lower()

    # Select numerical imputation method
    if strategy == "median":

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

    elif strategy == "knn":

        numeric_pipeline = Pipeline([
            ("scaler_before_imputation", StandardScaler()),
            ("imputer", KNNImputer(n_neighbors=5))
        ])

    elif strategy == "iterative":

        numeric_pipeline = Pipeline([
            (
                "imputer",
                IterativeImputer(
                    max_iter=10,
                    random_state=42
                )
            ),
            ("scaler", StandardScaler())
        ])

    else:
        raise ValueError(
            "Invalid strategy. Choose median, knn, or iterative."
        )

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    # Combine numerical and categorical pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor


if __name__ == "__main__":

    from src.data_loader import get_data
    from src.missing_data import introduce_missing_values

    X, y = get_data()

    # Introduce 20% additional missingness
    X_missing = introduce_missing_values(
        X,
        missing_rate=0.20,
        random_state=42
    )

    strategies = ["median", "knn", "iterative"]

    print("\n========== IMPUTATION TEST ==========")

    for strategy in strategies:

        preprocessor = get_preprocessor(strategy)

        # Fit and transform for this basic module test
        X_processed = preprocessor.fit_transform(X_missing)

        print(f"\nStrategy: {strategy.upper()}")

        print("Processed data shape:", X_processed.shape)

        print(
            "Remaining missing values:",
            np.isnan(X_processed).sum()
        )
