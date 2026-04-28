import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

def build_hybrid_cnn_flexible(
    signal_input_shape,
    feature_input_shape,
    n_classes,
    use_lstm_attention=True,
    use_feature_branch=True
):
    # create lightweight cnn model
    def build_epoch_cnn(input_shape):
        inp = tf.keras.Input(shape=input_shape)

        x = layers.Conv1D(16, 7, strides=2, padding="same")(inp) 
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv1D(32, 5, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv1D(64, 5, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv1D(64, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.GlobalAveragePooling1D()(x)

        return tf.keras.Model(inp, x)

    # Raw Signal Branch
    signal_input = tf.keras.Input(shape=signal_input_shape, name="signal_input")

    epoch_cnn = build_epoch_cnn(signal_input_shape[1:])
    x = layers.TimeDistributed(epoch_cnn)(signal_input)

    # LSTM & Attention
    if use_lstm_attention:
        x = layers.Bidirectional(
            layers.LSTM(
                32,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2
            )
        )(x)

        # Atention layer
        attn = layers.Dense(32, activation="tanh")(x)
        attn = layers.Dense(1)(attn)
        attn = layers.Softmax(axis=1)(attn)

        x = layers.Multiply()([x, attn])
        x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)
        x = layers.Dropout(0.3)(x)

    else:
        x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Feature Branch
    if use_feature_branch:
        feature_input = tf.keras.Input(
            shape=(feature_input_shape,), name="feature_input"
        )

        f = layers.BatchNormalization()(feature_input)

        f = layers.Dense(128, activation="relu")(f)
        f = layers.Dropout(0.3)(f)
        f = layers.Dense(64, activation="relu")(f)
        f = layers.Dense(32, activation="relu")(f)

        combined = layers.Concatenate()([x, f])
        inputs = [signal_input, feature_input]

    else:
        combined = x
        inputs = signal_input

    # Final layer
    z = layers.Dense(32, activation="relu")(combined)
    z = layers.Dropout(0.3)(z)

    outputs = layers.Dense(n_classes, activation="softmax")(z)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def sleep_cnn_model(
    signal_input_shape,
    feature_input_shape,
    n_classes
):
    """
    Best model:
    CNN (TimeDistributed) + Feature Branch
    NO LSTM, NO attention
    """

    # CNN
    def build_epoch_cnn(input_shape):
        inp = tf.keras.Input(shape=input_shape)

        x = layers.Conv1D(16, 7, strides=2, padding="same")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv1D(32, 5, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv1D(64, 5, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv1D(64, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.GlobalAveragePooling1D()(x)

        return tf.keras.Model(inp, x)

    # Signal Branch
    signal_input = tf.keras.Input(shape=signal_input_shape, name="signal_input")

    epoch_cnn = build_epoch_cnn(signal_input_shape[1:])
    x = layers.TimeDistributed(epoch_cnn)(signal_input)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Feature Branch
    feature_input = tf.keras.Input(
        shape=(feature_input_shape,), name="feature_input"
    )

    f = layers.BatchNormalization()(feature_input)

    f = layers.Dense(128, activation="relu")(f)
    f = layers.Dropout(0.3)(f)
    f = layers.Dense(64, activation="relu")(f)
    f = layers.Dense(32, activation="relu")(f)

    # Combine raw signal and feature heads
    combined = layers.Concatenate()([x, f])

    z = layers.Dense(32, activation="relu")(combined)
    z = layers.Dropout(0.3)(z)

    outputs = layers.Dense(n_classes, activation="softmax")(z)

    model = tf.keras.Model(
        inputs=[signal_input, feature_input],
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model