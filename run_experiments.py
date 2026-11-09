
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data_loader import get_data
from src.missing_data import introduce_missing_values
from src.preprocessing import get_preprocessor
from src.models import get_models
from src.evaluation import evaluate_predictions


# =====================================================
# 1. EXPERIMENT SETTINGS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = TABLES_DIR / "experiment_results.csv"
SUMMARY_FILE = TABLES_DIR / "experiment_summary.csv"
PREDICTIONS_FILE = TABLES_DIR / "oof_predictions.csv"

# Keep True for the first small test.
# Change to False to run the complete experiment.
QUICK_TEST = False

if QUICK_TEST:
    MISSINGNESS_LEVELS = [0.20]
    IMPUTATION_STRATEGIES = ["median"]
    RANDOM_SEEDS = [42]
    print("\nQUICK TEST MODE: 5 model configurations.")
else:
    MISSINGNESS_LEVELS = [0.0, 0.10, 0.20, 0.30]
    IMPUTATION_STRATEGIES = ["median", "knn", "iterative"]
    RANDOM_SEEDS = [42, 2026]
    print("\nFULL EXPERIMENT MODE: 600 model fits.")


# =====================================================
# 2. LOAD DATA
# =====================================================

print("\nLoading dataset...")

X, y = get_data()

print("Dataset shape:", X.shape)
print("Target distribution:")
print(y.value_counts().sort_index())


# =====================================================
# 3. PREPARE RESULT FILES
# =====================================================

for file_path in [RESULTS_FILE, SUMMARY_FILE, PREDICTIONS_FILE]:
    if file_path.exists():
        file_path.unlink()


# =====================================================
# 4. RUN EXPERIMENTS
# =====================================================

all_results = []

total_configs = (
    len(RANDOM_SEEDS)
    * len(MISSINGNESS_LEVELS)
    * len(IMPUTATION_STRATEGIES)
    * len(get_models())
)

completed = 0

for seed in RANDOM_SEEDS:

    print("\n" + "=" * 55)
    print("RANDOM SEED:", seed)
    print("=" * 55)

    for missing_rate in MISSINGNESS_LEVELS:

        # Generate one fixed masked dataset for this seed
        X_missing = introduce_missing_values(
            X,
            missing_rate=missing_rate,
            random_state=seed
        )

        for strategy in IMPUTATION_STRATEGIES:

            for model_name, model in get_models(
                random_state=seed
            ).items():

                completed += 1

                print("\n" + "-" * 55)
                print(
                    f"Experiment {completed}/{total_configs}"
                )
                print("Missingness:", missing_rate)
                print("Imputation:", strategy)
                print("Model:", model_name)

                # Preprocessing is fitted only on training folds
                pipeline = Pipeline([
                    ("preprocessor", get_preprocessor(strategy)),
                    ("classifier", model)
                ])

                cv = StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=seed
                )

                # Obtain out-of-fold predicted probabilities
                y_prob = cross_val_predict(
                    pipeline,
                    X_missing,
                    y,
                    cv=cv,
                    method="predict_proba",
                    n_jobs=1
                )[:, 1]

                # Calculate evaluation metrics
                metrics = evaluate_predictions(y, y_prob)

                result = {
                    "Seed": seed,
                    "Missingness": missing_rate,
                    "Imputation": strategy,
                    "Model": model_name,
                    **metrics
                }

                all_results.append(result)

                # Save results after every configuration
                pd.DataFrame(all_results).to_csv(
                    RESULTS_FILE,
                    index=False
                )

                # Save out-of-fold probabilities for later figures
                predictions = pd.DataFrame({
                    "Sample_Index": range(len(y)),
                    "Actual": y.to_numpy(),
                    "Predicted_Probability": y_prob,
                    "Seed": seed,
                    "Missingness": missing_rate,
                    "Imputation": strategy,
                    "Model": model_name
                })

                predictions.to_csv(
                    PREDICTIONS_FILE,
                    mode="a",
                    header=not PREDICTIONS_FILE.exists(),
                    index=False
                )

                print("ROC-AUC:", round(metrics["ROC-AUC"], 4))
                print("F1 Score:", round(metrics["F1 Score"], 4))
                print("Brier Score:", round(metrics["Brier Score"], 4))
                print("Configuration completed!")


# =====================================================
# 5. CREATE SUMMARY TABLE
# =====================================================

results_df = pd.DataFrame(all_results)

metric_columns = [
    "Accuracy", "Precision", "Recall", "F1 Score",
    "ROC-AUC", "PR-AUC", "Brier Score"
]

summary_df = results_df.groupby(
    ["Missingness", "Imputation", "Model"]
)[metric_columns].agg(["mean", "std"])

summary_df.columns = [
    f"{metric}_{statistic}"
    for metric, statistic in summary_df.columns
]

summary_df = summary_df.reset_index()

summary_df.to_csv(SUMMARY_FILE, index=False)


# =====================================================
# 6. FINAL OUTPUT
# =====================================================

print("\n" + "=" * 55)
print("EXPERIMENTS COMPLETED!")
print("=" * 55)

print("\nTotal configurations completed:", len(results_df))

print("\nResults saved at:")
print(RESULTS_FILE)

print("\nSummary saved at:")
print(SUMMARY_FILE)

print("\nOut-of-fold predictions saved at:")
print(PREDICTIONS_FILE)

print("\nExperiment summary:")
print(summary_df.to_string(index=False))
# run_experiments.py\n\n