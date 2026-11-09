# evaluation.py\n\n
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

from sklearn.calibration import calibration_curve


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    """
    Evaluate binary classification using predicted probabilities.

    Parameters:
        y_true: Actual class labels
        y_prob: Predicted probabilities for class 1
        threshold: Classification decision threshold

    Returns:
        Dictionary containing evaluation metrics
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError("Actual and predicted values must have equal length.")

    if not np.all(np.isfinite(y_prob)):
        raise ValueError("Predicted probabilities contain invalid values.")

    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    # Convert probabilities into class predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    # Probability-based metrics
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_prob)

        pr_auc = average_precision_score(y_true, y_prob)

    else:
        roc_auc = np.nan
        pr_auc = np.nan

    brier = brier_score_loss(y_true, y_prob)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Brier Score": brier,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


def get_calibration_data(y_true, y_prob, n_bins=10):
    """
    Calculate calibration curve data.

    Returns:
        mean_predicted_probability
        fraction_of_positive_samples
    """

    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="uniform"
    )

    return mean_predicted, fraction_positive


def print_evaluation_results(results):
    """
    Display evaluation metrics in readable format.
    """

    print("\n========== MODEL EVALUATION ==========")

    for metric, value in results.items():

        if isinstance(value, (int, np.integer)):
            print(f"{metric}: {value}")

        else:
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":

    print("Evaluation module loaded successfully!")

    print("\nMetrics configured:")

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC",
        "Brier Score",
        "Confusion Matrix",
        "Calibration Curve"
    ]

    for metric in metrics:
        print("-", metric)
