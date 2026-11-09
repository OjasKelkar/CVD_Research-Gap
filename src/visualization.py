
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


# ==========================================================
# PATHS AND SETTINGS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

RESULTS_FILE = TABLES_DIR / "experiment_results.csv"
OOF_FILE = TABLES_DIR / "oof_predictions.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 120
})


# ==========================================================
# LOAD EXPERIMENTAL RESULTS
# ==========================================================

def load_results():

    results = pd.read_csv(RESULTS_FILE)
    oof = pd.read_csv(OOF_FILE)

    print("Experimental results loaded:", results.shape)
    print("OOF predictions loaded:", oof.shape)

    return results, oof


# ==========================================================
# SAVE FIGURE
# ==========================================================

def save_figure(filename):

    path = FIGURES_DIR / filename

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Figure saved:", path.name)


# ==========================================================
# FIGURE 1: ROC-AUC COMPARISON
# ==========================================================

def plot_roc_auc(results):

    grouped = results.groupby(
        ["Missingness", "Model"]
    )["ROC-AUC"].mean().reset_index()

    plt.figure()

    for model in grouped["Model"].unique():

        data = grouped[grouped["Model"] == model]

        plt.plot(
            data["Missingness"] * 100,
            data["ROC-AUC"],
            marker="o",
            linewidth=2,
            label=model
        )

    plt.xlabel("Additional Missingness (%)")
    plt.ylabel("ROC-AUC")
    plt.title("ROC-AUC Comparison Across Missingness Levels")
    plt.xticks([0, 10, 20, 30])
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure("figure1_roc_auc_comparison.png")


# ==========================================================
# FIGURE 2: F1-SCORE COMPARISON
# ==========================================================

def plot_f1_score(results):

    grouped = results.groupby(
        ["Missingness", "Model"]
    )["F1 Score"].mean().reset_index()

    plt.figure()

    for model in grouped["Model"].unique():

        data = grouped[grouped["Model"] == model]

        plt.plot(
            data["Missingness"] * 100,
            data["F1 Score"],
            marker="o",
            linewidth=2,
            label=model
        )

    plt.xlabel("Additional Missingness (%)")
    plt.ylabel("F1 Score")
    plt.title("F1-Score Comparison Across Missingness Levels")
    plt.xticks([0, 10, 20, 30])
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure("figure2_f1_score_comparison.png")


# ==========================================================
# FIGURE 3: BRIER SCORE COMPARISON
# ==========================================================

def plot_brier_score(results):

    grouped = results.groupby(
        ["Missingness", "Model"]
    )["Brier Score"].mean().reset_index()

    plt.figure()

    for model in grouped["Model"].unique():

        data = grouped[grouped["Model"] == model]

        plt.plot(
            data["Missingness"] * 100,
            data["Brier Score"],
            marker="o",
            linewidth=2,
            label=model
        )

    plt.xlabel("Additional Missingness (%)")
    plt.ylabel("Brier Score")
    plt.title("Brier Score Across Missingness Levels")
    plt.xticks([0, 10, 20, 30])
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure("figure3_brier_score_comparison.png")


# ==========================================================
# FIGURE 4: IMPUTATION STRATEGY COMPARISON
# ==========================================================

def plot_imputation_comparison(results):

    grouped = results.groupby(
        ["Missingness", "Imputation"]
    )["ROC-AUC"].mean().reset_index()

    plt.figure()

    for strategy in grouped["Imputation"].unique():

        data = grouped[grouped["Imputation"] == strategy]

        plt.plot(
            data["Missingness"] * 100,
            data["ROC-AUC"],
            marker="o",
            linewidth=2,
            label=strategy.capitalize()
        )

    plt.xlabel("Additional Missingness (%)")
    plt.ylabel("Mean ROC-AUC")
    plt.title("Comparison of Imputation Strategies")
    plt.xticks([0, 10, 20, 30])
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure("figure4_imputation_comparison.png")


# ==========================================================
# SELECT BEST CONFIGURATION USING BRIER SCORE
# ==========================================================

def get_best_configuration(results):

    grouped = results.groupby(
        ["Missingness", "Imputation", "Model"]
    )["Brier Score"].mean().reset_index()

    best = grouped.loc[grouped["Brier Score"].idxmin()]

    print("\nBest configuration based on mean Brier Score:")
    print(best.to_string())

    return best


# ==========================================================
# FIGURE 5: CALIBRATION CURVE
# ==========================================================

def plot_calibration(oof, best):

    selected = oof[
        (oof["Missingness"] == best["Missingness"]) &
        (oof["Imputation"] == best["Imputation"]) &
        (oof["Model"] == best["Model"])
    ].copy()

    if selected.empty:
        print("Calibration plot skipped: no matching predictions.")
        return

    actual = selected["Actual"]
    probabilities = selected["Predicted_Probability"]

    fraction_positive, mean_predicted = calibration_curve(
        actual,
        probabilities,
        n_bins=10,
        strategy="uniform"
    )

    plt.figure()

    plt.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        linewidth=2,
        label="Model calibration"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positive Cases")
    plt.title("Calibration Curve")
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure("figure5_calibration_curve.png")


# ==========================================================
# FIGURE 6: ROC CURVE
# ==========================================================

def plot_roc_curve(oof, best):

    selected = oof[
        (oof["Missingness"] == best["Missingness"]) &
        (oof["Imputation"] == best["Imputation"]) &
        (oof["Model"] == best["Model"])
    ].copy()

    if selected.empty:
        print("ROC curve skipped: no matching predictions.")
        return

    actual = selected["Actual"]
    probabilities = selected["Predicted_Probability"]

    fpr, tpr, thresholds = roc_curve(actual, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Selected Configuration")
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure("figure6_roc_curve.png")


# ==========================================================
# FIGURE 7: CONFUSION MATRIX
# ==========================================================

def plot_confusion_matrix(oof, best):

    selected = oof[
        (oof["Missingness"] == best["Missingness"]) &
        (oof["Imputation"] == best["Imputation"]) &
        (oof["Model"] == best["Model"])
    ].copy()

    if selected.empty:
        print("Confusion matrix skipped: no matching predictions.")
        return

    predicted = (
        selected["Predicted_Probability"] >= 0.5
    ).astype(int)

    cm = confusion_matrix(
        selected["Actual"],
        predicted,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Disease", "Disease"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", values_format="d")

    ax.set_title("Confusion Matrix at Threshold 0.5")

    save_figure("figure7_confusion_matrix.png")


# ==========================================================
# MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":

    print("\n========== GENERATING RESEARCH FIGURES ==========\n")

    results, oof = load_results()

    plot_roc_auc(results)
    plot_f1_score(results)
    plot_brier_score(results)
    plot_imputation_comparison(results)

    best = get_best_configuration(results)

    plot_calibration(oof, best)
    plot_roc_curve(oof, best)
    plot_confusion_matrix(oof, best)

    print("\n==============================================")
    print("FIGURE GENERATION COMPLETED!")
    print("Figures saved in:", FIGURES_DIR)
    print("==============================================")
