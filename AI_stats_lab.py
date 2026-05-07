"""
AI_stats_lab.py

Lab: Training and Evaluating Classification Models

Topics:
- Confusion matrix
- Recall
- Fallout
- Precision
- Accuracy
- Thresholding prediction scores
- Effect of changing threshold
- Training two classifiers
- Comparing model performance

Instructions:
- Implement all functions.
- Do NOT change function names.
- Do NOT print inside functions.
- Return exactly the required formats.
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Question 1: Confusion Matrix, Metrics, and Threshold Effects
# ============================================================

def confusion_matrix_counts(y_true, y_pred):
    """
    Compute confusion matrix counts for binary classification.

    Returns:
        (TP, FP, FN, TN)
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    return (TP, FP, FN, TN)


def classification_metrics(y_true, y_pred):
    """
    Compute classification metrics.

    Returns:
        Dictionary containing:
            recall
            fallout
            precision
            accuracy
    """

    TP, FP, FN, TN = confusion_matrix_counts(y_true, y_pred)

    # Recall
    recall_den = TP + FN
    recall = TP / recall_den if recall_den != 0 else 0.0

    # Fallout
    fallout_den = FP + TN
    fallout = FP / fallout_den if fallout_den != 0 else 0.0

    # Precision
    precision_den = TP + FP
    precision = TP / precision_den if precision_den != 0 else 0.0

    # Accuracy
    accuracy_den = TP + FP + FN + TN
    accuracy = (TP + TN) / accuracy_den if accuracy_den != 0 else 0.0

    return {
        "recall": recall,
        "fallout": fallout,
        "precision": precision,
        "accuracy": accuracy
    }


def apply_threshold(scores, threshold):
    """
    Convert prediction scores into binary predictions.
    """

    scores = np.array(scores)

    predictions = np.where(scores >= threshold, 1, 0)

    return predictions


def threshold_metrics_analysis(y_true, scores, thresholds):
    """
    Analyze how changing threshold affects metrics.
    """

    results = []

    for threshold in thresholds:

        predictions = apply_threshold(scores, threshold)

        metrics = classification_metrics(y_true, predictions)

        result = {
            "threshold": threshold,
            "recall": metrics["recall"],
            "fallout": metrics["fallout"],
            "precision": metrics["precision"],
            "accuracy": metrics["accuracy"]
        }

        results.append(result)

    return results


# ============================================================
# Question 2: Train Two Classifiers and Evaluate Them
# ============================================================

def train_two_classifiers(X_train, y_train):
    """
    Train Logistic Regression and Decision Tree classifiers.
    """

    logistic_model = LogisticRegression(max_iter=1000)

    decision_tree_model = DecisionTreeClassifier(random_state=0)

    logistic_model.fit(X_train, y_train)

    decision_tree_model.fit(X_train, y_train)

    return {
        "logistic_regression": logistic_model,
        "decision_tree": decision_tree_model
    }


def evaluate_classifier(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a trained classifier.
    """

    # Positive class probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Convert probabilities to predictions
    predictions = apply_threshold(probabilities, threshold)

    # Confusion matrix counts
    TP, FP, FN, TN = confusion_matrix_counts(y_test, predictions)

    # Metrics
    metrics = classification_metrics(y_test, predictions)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "recall": metrics["recall"],
        "fallout": metrics["fallout"],
        "precision": metrics["precision"],
        "accuracy": metrics["accuracy"]
    }


def compare_classifiers(X_train, y_train, X_test, y_test, threshold=0.5):
    """
    Train and evaluate two classifiers.
    """

    models = train_two_classifiers(X_train, y_train)

    logistic_results = evaluate_classifier(
        models["logistic_regression"],
        X_test,
        y_test,
        threshold
    )

    decision_tree_results = evaluate_classifier(
        models["decision_tree"],
        X_test,
        y_test,
        threshold
    )

    return {
        "logistic_regression": logistic_results,
        "decision_tree": decision_tree_results
    }


if __name__ == "__main__":
    print("Implement all required functions.")
