# models.py\n\n
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def get_models(random_state=42):
    """
    Create five machine-learning classification models.

    All models support probability prediction,
    which is required for ROC-AUC, PR-AUC and calibration.
    """

    models = {

        # 1. Logistic Regression
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=random_state
        ),

        # 2. Support Vector Machine with RBF kernel
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=random_state
        ),

        # 3. Random Forest
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        ),

        # 4. Extreme Gradient Boosting
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        ),

        # 5. Categorical Boosting
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=4,
            learning_rate=0.03,
            loss_function="Logloss",
            verbose=0,
            random_seed=random_state,
            allow_writing_files=False,
            thread_count=-1
        )
    }

    return models


if __name__ == "__main__":

    models = get_models()

    print("\n========== MODEL CONFIGURATION ==========")

    for name, model in models.items():

        print(f"\nModel: {name}")

        print("Algorithm:", type(model).__name__)

        print("Probability prediction: Available")

    print("\nTotal models configured:", len(models))
