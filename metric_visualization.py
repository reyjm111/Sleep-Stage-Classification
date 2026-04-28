import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def compute_fold_metrics(y_true, y_pred, y_proba, class_labels):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "cm": confusion_matrix(y_true, y_pred, labels=class_labels),
    }

    if y_proba is not None:
        try:
            results["roc_auc_ovr_weighted"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted"
            )
        except ValueError:
            results["roc_auc_ovr_weighted"] = np.nan
    else:
        results["roc_auc_ovr_weighted"] = np.nan

    return results


def summarize_cv_results(all_results):
    metric_keys = [k for k in all_results[0].keys() if k != "cm"]
    summary = {
        metric: np.nanmean([fold[metric] for fold in all_results])
        for metric in metric_keys
    }
    return summary


def print_cv_summary(summary):
    print("\nMean CV results:")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")


def print_class_distribution(y):
    class_counts = pd.Series(y).value_counts().sort_index()
    class_props = pd.Series(y).value_counts(normalize=True).sort_index()

    print("\nClass counts:")
    print(class_counts)

    print("\nClass proportions:")
    print(class_props.round(3))


def print_full_classification_report(all_y_true, all_y_pred, class_labels):
    print("\nClassification report:")
    print(
        classification_report(
            all_y_true,
            all_y_pred,
            labels=class_labels,
            digits=3,
            zero_division=0
        )
    )


def plot_normalized_confusion_matrix(cms, class_labels, title="Normalized Confusion Matrix Across All CV Folds"):
    cm_sum = np.sum(cms, axis=0)
    cm_norm = cm_sum.astype(float) / cm_sum.sum(axis=1, keepdims=True)

    print("\nRow-normalized confusion matrix:")
    print(cm_norm)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_labels)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_training_history(histories):
    """
    Plot average loss and val_loss across folds
    """
    plt.figure(figsize=(8, 5))

    max_epochs = max(len(h["loss"]) for h in histories)

    # Pad histories so they align
    loss_matrix = []
    val_loss_matrix = []

    for h in histories:
        loss = h["loss"]
        val_loss = h["val_loss"]

        loss = loss + [np.nan] * (max_epochs - len(loss))
        val_loss = val_loss + [np.nan] * (max_epochs - len(val_loss))

        loss_matrix.append(loss)
        val_loss_matrix.append(val_loss)

    loss_mean = np.nanmean(loss_matrix, axis=0)
    val_loss_mean = np.nanmean(val_loss_matrix, axis=0)

    plt.plot(loss_mean, label="Train Loss")
    plt.plot(val_loss_mean, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Average Training Curve Across Folds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()