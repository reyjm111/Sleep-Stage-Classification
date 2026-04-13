import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from metric_visualization import (
    compute_fold_metrics,
    summarize_cv_results,
    print_cv_summary,
    print_class_distribution,
    print_full_classification_report,
    get_feature_importance_df,
)


def train(X, y, groups, model, n_splits=5, task='5-class', random_state=42):
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    if task == '3-class':

        y = np.where(y == 1, 1, np.where(y == 2, 2, 3))
    
    class_labels = np.sort(np.unique(y))

    all_results = []
    all_y_true = []
    all_y_pred = []
    all_cms = []

    print_class_distribution(y)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        print(f"\nFold {fold}/{n_splits}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        y_pred = fold_model.predict(X_test)

        if hasattr(fold_model, "predict_proba"):
            y_proba = fold_model.predict_proba(X_test)
        else:
            y_proba = None

        fold_metrics = compute_fold_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            class_labels=class_labels
        )

        all_results.append(fold_metrics)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_cms.append(fold_metrics["cm"])

        print(
            f"accuracy={fold_metrics['accuracy']:.4f}, "
            f"balanced_accuracy={fold_metrics['balanced_accuracy']:.4f}, "
            f"f1_macro={fold_metrics['f1_macro']:.4f}"
        )

    summary = summarize_cv_results(all_results)
    print_cv_summary(summary)
    print_full_classification_report(all_y_true, all_y_pred, class_labels)

    return {
        "fold_results": all_results,
        "summary": summary,
        "all_y_true": np.array(all_y_true),
        "all_y_pred": np.array(all_y_pred),
        "cms": all_cms,
        "class_labels": class_labels
    }