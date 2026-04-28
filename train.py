import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from metric_visualization import (
    compute_fold_metrics,
    summarize_cv_results,
    print_cv_summary,
    print_class_distribution,
    print_full_classification_report, 
    plot_training_history
)
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from cnn_model import build_hybrid_cnn_flexible, sleep_cnn_model
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.preprocessing import StandardScaler
import gc
from model import build_rf_model

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

def train_rf_fold(X_train, y_train, X_test):
    rf = build_rf_model()
    rf.fit(X_train, y_train)
    return rf.predict_proba(X_test)


def combine_probas(cnn_proba, rf_proba, use_ensemble, weight_rf=0.6):
    if not use_ensemble or rf_proba is None:
        return cnn_proba
    return weight_rf * rf_proba + (1 - weight_rf) * cnn_proba

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

            fold_model.fit(
            X_train,
            y_train_xgb,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test - 1)],
            verbose=False
        )

            y_proba = fold_model.predict_proba(X_test)

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
                    window=3
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

def train_hybrid_flexible(
    X_signal,
    X_features,
    y,
    groups,

    use_feature_branch=True,
    use_lstm_attention=True,
    use_rf_ensemble=False,

    rf_weight=0.6,
    n_splits=5,
    n_epochs=40,
    batch_size=8,
    task="3-class",
    context=5,
    smoothing_window=5,
    random_state=42
):

    print(f"\nModel config → feature={use_feature_branch}, lstm={use_lstm_attention}, rf={use_rf_ensemble}")

    if task == "3-class":
        y = y.copy()
        y[y >= 3] = 3

    print_class_distribution(y)

    # Context windows
    X_signal, X_features, y, groups = make_context_windows(
        X_signal, y, groups, context=context, X_features=X_features
    )

    X_signal = np.asarray(X_signal, dtype=np.float32)
    X_features = np.asarray(X_features, dtype=np.float32)

    class_labels = np.sort(np.unique(y))
    n_classes = len(class_labels)

    all_results, all_y_true, all_y_pred, all_cms = [], [], [], []

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X_signal, y, groups=groups), start=1
    ):
        print(f"\n================ Fold {fold}/{n_splits} ================")

        Xs_train = X_signal[train_idx]
        Xs_test  = X_signal[test_idx]
        Xf_train = X_features[train_idx]
        Xf_test  = X_features[test_idx]
        y_train  = y[train_idx]
        y_test   = y[test_idx]

        groups_test = groups[test_idx]

        Xs_train = np.transpose(Xs_train, (0, 1, 3, 2))
        Xs_test  = np.transpose(Xs_test,  (0, 1, 3, 2))

        # Feature normalization
        mean = Xf_train.mean(axis=0, keepdims=True)
        std  = Xf_train.std(axis=0, keepdims=True) + 1e-6
        Xf_train = (Xf_train - mean) / std
        Xf_test  = (Xf_test  - mean) / std

        y_train_oh = tf.keras.utils.to_categorical(y_train - 1, n_classes)
        y_test_oh  = tf.keras.utils.to_categorical(y_test  - 1, n_classes)

        # Class weights
        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = {
            label - 1: weight for label, weight in zip(np.unique(y_train), cw)
        }

        # CNN Model
        feature_dim = Xf_train.shape[1] if use_feature_branch else None

        model = build_hybrid_cnn_flexible(
            Xs_train.shape[1:],
            feature_dim,
            n_classes,
            use_lstm_attention=use_lstm_attention,
            use_feature_branch=use_feature_branch
        )

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

        train_inputs = {"signal_input": Xs_train}
        val_inputs   = {"signal_input": Xs_test}

        if use_feature_branch:
            train_inputs = {
                "signal_input": Xs_train,
                "feature_input": Xf_train
            }
            val_inputs = {
                "signal_input": Xs_test,
                "feature_input": Xf_test
            }
        else:
            train_inputs = Xs_train
            val_inputs   = Xs_test

        # Training
        history = model.fit(
            train_inputs,
            y_train_oh,
            validation_data=(val_inputs, y_test_oh),
            epochs=n_epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=2
        )

        print(f"Final val acc: {history.history['val_accuracy'][-1]:.4f}")

        # CNN Predictions
        test_inputs = {"signal_input": Xs_test}
        if use_feature_branch:
            test_inputs = {
                "signal_input": Xs_test,
                "feature_input": Xf_test
            }
        else:
            test_inputs = Xs_test

        cnn_proba = model.predict(test_inputs, batch_size=batch_size, verbose=0)

        # RF Ensemble
        if use_rf_ensemble:
            rf_proba = train_rf_fold(Xf_train, y_train, Xf_test)
            final_proba = rf_weight * rf_proba + (1 - rf_weight) * cnn_proba
        else:
            final_proba = cnn_proba

        # Smoothing
        final_proba = smooth_proba_grouped(
            final_proba,
            groups_test,
            window=smoothing_window
        )

        final_proba /= final_proba.sum(axis=1, keepdims=True)

        y_pred = np.argmax(final_proba, axis=1) + 1

        # Metrics
        fold_metrics = compute_fold_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=final_proba,
            class_labels=class_labels
        )

        print(
            f"accuracy={fold_metrics['accuracy']:.4f}, "
            f"balanced_accuracy={fold_metrics['balanced_accuracy']:.4f}, "
            f"f1_macro={fold_metrics['f1_macro']:.4f}"
        )

        all_results.append(fold_metrics)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_cms.append(fold_metrics["cm"])

        # Memory cleaning
        tf.keras.backend.clear_session()
        del model, cnn_proba
        gc.collect()

    summary = summarize_cv_results(all_results)

    print("\nFINAL SUMMARY:")
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

def sleep_model_train(
    X_signal,
    X_features,
    y,
    groups,
    rf_weight=0.35,
    n_splits=5,
    n_epochs=40,
    batch_size=8,
    context=5,
    smoothing_window=7,
    random_state=42
):
    """
    FINAL BEST MODEL RUN
    """

    print("\n RUNNING FINAL BEST MODEL")

    # 3-class mapping
    y = y.copy()
    y[y >= 3] = 3

    print_class_distribution(y)

    # Context windows
    X_signal, X_features, y, groups = make_context_windows(
        X_signal, y, groups, context=context, X_features=X_features
    )

    X_signal = np.asarray(X_signal, dtype=np.float32)
    X_features = np.asarray(X_features, dtype=np.float32)

    class_labels = np.sort(np.unique(y))
    n_classes = len(class_labels)

    all_results, all_y_true, all_y_pred, all_cms = [], [], [], []
    histories = []

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X_signal, y, groups=groups), start=1
    ):
        print(f"\n================ Fold {fold}/{n_splits} ================")

        Xs_train = X_signal[train_idx]
        Xs_test  = X_signal[test_idx]
        Xf_train = X_features[train_idx]
        Xf_test  = X_features[test_idx]
        y_train  = y[train_idx]
        y_test   = y[test_idx]
        groups_test = groups[test_idx]

        Xs_train = np.transpose(Xs_train, (0, 1, 3, 2))
        Xs_test  = np.transpose(Xs_test,  (0, 1, 3, 2))

        # Normalize features
        mean = Xf_train.mean(axis=0, keepdims=True)
        std  = Xf_train.std(axis=0, keepdims=True) + 1e-6
        Xf_train = (Xf_train - mean) / std
        Xf_test  = (Xf_test  - mean) / std

        # Labels
        y_train_oh = tf.keras.utils.to_categorical(y_train - 1, n_classes)
        y_test_oh  = tf.keras.utils.to_categorical(y_test  - 1, n_classes)

        # Class weights (slightly boosted for minorities)
        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = {
            label - 1: weight for label, weight in zip(np.unique(y_train), cw)
        }

        # Model
        model = sleep_cnn_model(
            signal_input_shape=Xs_train.shape[1:],
            feature_input_shape=Xf_train.shape[1],
            n_classes=n_classes
        )

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Training
        history = model.fit(
            {
                "signal_input": Xs_train,
                "feature_input": Xf_train
            },
            y_train_oh,
            validation_data=(
                {
                    "signal_input": Xs_test,
                    "feature_input": Xf_test
                },
                y_test_oh
            ),
            epochs=n_epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=2
        )

        histories.append(history.history)

        # CNN predictions
        cnn_proba = model.predict(
            {
                "signal_input": Xs_test,
                "feature_input": Xf_test
            },
            batch_size=batch_size,
            verbose=0
        )

        # RF
        rf_proba = train_rf_fold(Xf_train, y_train, Xf_test)

        # Ensemble
        final_proba = rf_weight * rf_proba + (1 - rf_weight) * cnn_proba

        # Smoothing
        final_proba = smooth_proba_grouped(
            final_proba,
            groups_test,
            window=smoothing_window
        )

        final_proba /= final_proba.sum(axis=1, keepdims=True)

        y_pred = np.argmax(final_proba, axis=1) + 1

        # Metrics
        fold_metrics = compute_fold_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=final_proba,
            class_labels=class_labels
        )

        print(
            f"accuracy={fold_metrics['accuracy']:.4f}, "
            f"balanced_accuracy={fold_metrics['balanced_accuracy']:.4f}, "
            f"f1_macro={fold_metrics['f1_macro']:.4f}"
        )

        all_results.append(fold_metrics)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_cms.append(fold_metrics["cm"])

        tf.keras.backend.clear_session()
        del model, cnn_proba, rf_proba, final_proba
        gc.collect()

    # Summary
    summary = summarize_cv_results(all_results)

    print("\nFINAL SUMMARY:")
    print_cv_summary(summary)
    print_full_classification_report(all_y_true, all_y_pred, class_labels)

    # Plot training curves
    plot_training_history(histories)

    return {
        "fold_results": all_results,
        "summary": summary,
        "all_y_true": np.array(all_y_true),
        "all_y_pred": np.array(all_y_pred),
        "cms": all_cms,
        "class_labels": class_labels,
        "histories": histories
    }

def sleep_model_train_weight_smoothing_sweep(
    X_signal,
    X_features,
    y,
    groups,
    rf_weights=[0.3, 0.5, 0.6, 0.7, 0.8],
    smoothing_windows=[1, 3, 5, 7],
    n_splits=4,
    n_epochs=40,
    batch_size=8,
    context=1,
    random_state=42
):

    print("\nRunning WEIGHT + SMOOTHING SWEEP")

    y = y.copy()
    y[y >= 3] = 3

    print_class_distribution(y)

    # Context
    X_signal, X_features, y, groups = make_context_windows(
        X_signal, y, groups, context=context, X_features=X_features
    )

    X_signal = np.asarray(X_signal, dtype=np.float32)
    X_features = np.asarray(X_features, dtype=np.float32)

    class_labels = np.sort(np.unique(y))
    n_classes = len(class_labels)

    # 🔥 store results per (weight, smoothing)
    results = {
        (w, s): []
        for w in rf_weights
        for s in smoothing_windows
    }

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X_signal, y, groups=groups), start=1
    ):
        print(f"\n================ Fold {fold}/{n_splits} ================")

        # Split
        Xs_train = X_signal[train_idx]
        Xs_test  = X_signal[test_idx]
        Xf_train = X_features[train_idx]
        Xf_test  = X_features[test_idx]
        y_train  = y[train_idx]
        y_test   = y[test_idx]
        groups_test = groups[test_idx]

        # reshape
        Xs_train = np.transpose(Xs_train, (0, 1, 3, 2))
        Xs_test  = np.transpose(Xs_test,  (0, 1, 3, 2))

        # Normalize features
        mean = Xf_train.mean(axis=0, keepdims=True)
        std  = Xf_train.std(axis=0, keepdims=True) + 1e-6
        Xf_train = (Xf_train - mean) / std
        Xf_test  = (Xf_test  - mean) / std

        # Labels
        y_train_oh = tf.keras.utils.to_categorical(y_train - 1, n_classes)
        y_test_oh  = tf.keras.utils.to_categorical(y_test  - 1, n_classes)

        # Class weights
        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = {
            label - 1: weight for label, weight in zip(np.unique(y_train), cw)
        }

        # --- Build model ---
        model = sleep_cnn_model(
            signal_input_shape=Xs_train.shape[1:],
            feature_input_shape=Xf_train.shape[1],
            n_classes=n_classes
        )

        # --- Callbacks (RESTORED) ---
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # --- Train CNN ONCE ---
        history = model.fit(
            {
                "signal_input": Xs_train,
                "feature_input": Xf_train
            },
            y_train_oh,
            validation_data=(
                {
                    "signal_input": Xs_test,
                    "feature_input": Xf_test
                },
                y_test_oh
            ),
            epochs=n_epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=2
        )

        print(f"Final val acc: {history.history['val_accuracy'][-1]:.4f}")

        # --- Predictions ONCE ---
        cnn_proba = model.predict(
            {
                "signal_input": Xs_test,
                "feature_input": Xf_test
            },
            batch_size=batch_size,
            verbose=0
        )

        rf_proba = train_rf_fold(Xf_train, y_train, Xf_test)

        # 🔥 Sweep BOTH weight and smoothing
        for w in rf_weights:

            base_proba = w * rf_proba + (1 - w) * cnn_proba

            for s in smoothing_windows:

                final_proba = smooth_proba_grouped(
                    base_proba,
                    groups_test,
                    window=s
                )

                final_proba /= final_proba.sum(axis=1, keepdims=True)

                y_pred = np.argmax(final_proba, axis=1) + 1

                fold_metrics = compute_fold_metrics(
                    y_true=y_test,
                    y_pred=y_pred,
                    y_proba=final_proba,
                    class_labels=class_labels
                )

                results[(w, s)].append(fold_metrics)

        # --- Memory cleanup ---
        tf.keras.backend.clear_session()
        del model, cnn_proba, rf_proba
        gc.collect()

    # --- Summarize ---
    summary_table = []

    for (w, s), folds in results.items():
        summary = summarize_cv_results(folds)

        summary_table.append({
            "rf_weight": w,
            "cnn_weight": 1 - w,
            "smoothing": s,
            "accuracy": summary["accuracy"],
            "balanced_accuracy": summary["balanced_accuracy"],
            "f1_macro": summary["f1_macro"]
        })

    df = pd.DataFrame(summary_table)

    df["score"] = 0.5 * df["balanced_accuracy"] + 0.5 * df["f1_macro"]
    df = df.sort_values(by="score", ascending=False)

    print("\nFINAL RESULTS:\n")
    print(df)

    return df