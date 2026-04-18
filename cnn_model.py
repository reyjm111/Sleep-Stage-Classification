import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape, n_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # temporal filtering
    x = tf.keras.layers.Conv1D(32, kernel_size=7, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # deeper temporal abstraction
    x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model