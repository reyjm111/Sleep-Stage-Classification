import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from metric_visualization import (
    compute_fold_metrics,
    summarize_cv_results,
    print_cv_summary,
    print_class_distribution,
    print_full_classification_report,
)
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from cnn_model import build_cnn, build_hybrid_cnn_lstm_attention
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.preprocessing import StandardScaler

def make_context_windows(
    X_signal,
    y,
    groups,
    context=3,
    X_features=None
):
    """
    General context window builder.

    Args:
        X_signal:   (N, ch, samples)
        X_features: (N, n_features) or None
        y:          (N,)
        groups:     (N,)
        context:    odd int (1,3,5,...)

    Returns:
        Xs_ctx: (N_new, context, ch, samples)
        Xf_ctx: (N_new, n_features) or None
        y_ctx:  (N_new,)
        g_ctx:  (N_new,)
    """

    assert context % 2 == 1, "context must be odd"

    if context == 1:
        Xs = X_signal[:, None, :, :]
        if X_features is not None:
            return Xs, X_features, y, groups
        return Xs, y, groups

    half = context // 2

    Xs_ctx, Xf_ctx, y_ctx, g_ctx = [], [], [], []

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]

        Xs_g = X_signal[idx]
        y_g = y[idx]

        if X_features is not None:
            Xf_g = X_features[idx]

        if len(Xs_g) < context:
            continue

        for i in range(half, len(Xs_g) - half):
            # signal window
            window = Xs_g[i - half:i + half + 1]
            Xs_ctx.append(window)

            # features (center only)
            if X_features is not None:
                Xf_ctx.append(Xf_g[i])

            y_ctx.append(y_g[i])
            g_ctx.append(g)

    Xs_ctx = np.stack(Xs_ctx)
    y_ctx = np.array(y_ctx)
    g_ctx = np.array(g_ctx)

    if X_features is not None:
        Xf_ctx = np.stack(Xf_ctx)
        return Xs_ctx, Xf_ctx, y_ctx, g_ctx

    return Xs_ctx, y_ctx, g_ctx

def smooth_proba_grouped(y_proba, groups, window=3):
    """
    Smooth probabilities within each group (subject).

    Args:
        y_proba: (N, n_classes)
        groups: (N,)
        window: odd integer (e.g., 3, 5, 7)

    Returns:
        smoothed probabilities (N, n_classes)
    """
    assert window % 2 == 1, "window must be odd"

    y_proba = np.asarray(y_proba)
    groups = np.asarray(groups)

    smoothed = np.zeros_like(y_proba)
    half = window // 2

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        proba_g = y_proba[idx]

        for i in range(len(proba_g)):
            start = max(0, i - half)
            end = min(len(proba_g), i + half + 1)

            smoothed[idx[i]] = proba_g[start:end].mean(axis=0)

    return smoothed

def train(X, y, groups, model, n_splits=5, task='5-class', smoothing=True, random_state=42):
    
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

        if fold_model.__class__.__name__ == "XGBClassifier":

            y_train_xgb = y_train - 1

            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            fold_model.fit(
                X_train_scaled,
                y_train_xgb,
                sample_weight=sample_weights,
                eval_set=[(X_test_scaled, y_test - 1)],
                verbose=False
            )

            y_proba = fold_model.predict_proba(X_test_scaled)

        else:
            fold_model.fit(X_train, y_train)

            if hasattr(fold_model, "predict_proba"):
                y_proba = fold_model.predict_proba(X_test)
            else:
                y_pred = fold_model.predict(X_test)
                y_proba = None

        if y_proba is not None:
            if smoothing:
                y_proba = smooth_proba_grouped(
                    y_proba,
                    groups[test_idx],
                    window=5
                )

            y_proba = y_proba / (y_proba.sum(axis=1, keepdims=True) + 1e-8)
            y_pred = np.argmax(y_proba, axis=1) + 1

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

def train_cnn(X, y, groups, n_splits=5, n_epochs=20, batch_size=32, task='5-class', context=3, random_state=42):

    # 3-class mapping
    if task == '3-class':
        y = np.where(y == 1, 1, np.where(y == 2, 2, 3))

    # build context windows before CV within subject boundaries
    X, y, groups = make_context_windows(X, y, groups, context=context)

    class_labels = np.sort(np.unique(y))
    n_classes = len(class_labels)

    all_results = []
    all_y_true = []
    all_y_pred = []
    all_cms = []

    print_class_distribution(y)

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        print(f"\nFold {fold}/{n_splits}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # reshape for CNN (assuming X is 2D: samples x features)
        X_train = np.transpose(X_train, (0, 1, 3, 2))  # (N, context, samples, ch)
        X_test  = np.transpose(X_test,  (0, 1, 3, 2))

        # merge context into channels
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], -1)
        X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[2],  -1)

        # one-hot encode
        y_train_oh = to_categorical(y_train - 1, num_classes=n_classes)
        y_test_oh = to_categorical(y_test - 1, num_classes=n_classes)

        # setting class weights to address imbalance
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )

        # map actual label to weight, then convert to weight for Keras
        label_to_weight = dict(zip(np.unique(y_train), class_weights))
        class_weights = {label - 1: weight for label, weight in label_to_weight.items()}

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]

        # Build model per fold
        model = build_cnn(input_shape=X_train.shape[1:], n_classes=n_classes)

        model.fit(
            X_train,
            y_train_oh,
            validation_data=(X_test, y_test_oh),
            epochs=n_epochs,
            batch_size=batch_size,
            class_weight=class_weights, 
            callbacks=callbacks,  
            verbose=1
        )

        # Predictions
        y_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_proba, axis=1) + 1

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

def train_hybrid_cnn_lstm_attention_smoothed(
    X_signal,
    X_features,
    y,
    groups,
    n_splits=5,
    n_epochs=20,
    batch_size=32,
    task="5-class",
    context=3,
    smoothing_window=5,
    random_state=42
):

    if task == "3-class":
        y = np.where(y == 1, 1, np.where(y == 2, 2, 3))

    # context windows
    X_signal, X_features, y, groups = make_context_windows(
        X_signal, y, groups, context=context, X_features=X_features
    )

    class_labels = np.sort(np.unique(y))
    n_classes = len(class_labels)

    all_results = []
    all_y_true = []
    all_y_pred = []
    all_cms = []

    print_class_distribution(y)

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X_signal, y, groups=groups), start=1
    ):
        print(f"\nFold {fold}/{n_splits}")

        Xs_train, Xs_test = X_signal[train_idx], X_signal[test_idx]
        Xf_train, Xf_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        groups_test = groups[test_idx]

        # reshape signal
        Xs_train = np.transpose(Xs_train, (0, 1, 3, 2))
        Xs_test  = np.transpose(Xs_test,  (0, 1, 3, 2))

        # normalize features
        feat_mean = Xf_train.mean(axis=0, keepdims=True)
        feat_std = Xf_train.std(axis=0, keepdims=True) + 1e-6
        Xf_train = (Xf_train - feat_mean) / feat_std
        Xf_test  = (Xf_test  - feat_mean) / feat_std

        y_train_oh = to_categorical(y_train - 1, num_classes=n_classes)
        y_test_oh  = to_categorical(y_test  - 1, num_classes=n_classes)

        # class weights
        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = {
            label - 1: weight for label, weight in zip(np.unique(y_train), cw)
        }

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                min_delta=0.002,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_delta=0.002,
                min_lr=1e-6,
                verbose=1
            )
        ]

        model = build_hybrid_cnn_lstm_attention(
            signal_input_shape=Xs_train.shape[1:],
            feature_input_shape=Xf_train.shape[1],
            n_classes=n_classes
        )

        model.fit(
            {"signal_input": Xs_train, "feature_input": Xf_train},
            y_train_oh,
            validation_data=(
                {"signal_input": Xs_test, "feature_input": Xf_test},
                y_test_oh
            ),
            epochs=n_epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        # Probabilistic predictions
        y_proba = model.predict(
            {"signal_input": Xs_test, "feature_input": Xf_test},
            verbose=0
        )

        # Probability smoothing
        y_proba_smooth = smooth_proba_grouped(
            y_proba,
            groups_test,
            window=smoothing_window
        )

        # Normalize probabilities after smoothing
        y_proba_smooth = y_proba_smooth / (
            y_proba_smooth.sum(axis=1, keepdims=True) + 1e-8
        )

        y_pred = np.argmax(y_proba_smooth, axis=1) + 1

        # Metrics
        fold_metrics = compute_fold_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba_smooth,
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