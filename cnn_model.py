import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

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

def build_hybrid_cnn_lstm_attention(signal_input_shape, feature_input_shape, n_classes):

    def build_epoch_cnn(input_shape):
        inp = tf.keras.Input(shape=input_shape)

        x = layers.Conv1D(32, 7, padding="same", use_bias=False,
                          kernel_regularizer=regularizers.l2(5e-5))(inp)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)

        x = layers.Conv1D(64, 5, padding="same", use_bias=False,
                          kernel_regularizer=regularizers.l2(5e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)

        x = layers.Conv1D(128, 3, padding="same", use_bias=False,
                          kernel_regularizer=regularizers.l2(5e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.GlobalAveragePooling1D()(x)

        return tf.keras.Model(inp, x, name="epoch_cnn")

    # Raw signal branch
    signal_input = tf.keras.Input(shape=signal_input_shape, name="signal_input")

    epoch_cnn = build_epoch_cnn(signal_input_shape[1:])
    x = layers.TimeDistributed(epoch_cnn)(signal_input)

    # LSTM with sequencing 
    x = layers.Bidirectional(layers.LSTM(
        64,
        return_sequences=True, 
        dropout=0.3,
        recurrent_dropout=0.2
    ))(x)

    # Attention
    attention = layers.Attention()([x, x]) 
    x = layers.GlobalAveragePooling1D()(attention)

    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(5e-5))(x)
    x = layers.Dropout(0.3)(x)

    # Feature Branch
    feature_input = tf.keras.Input(shape=(feature_input_shape,), name="feature_input")

    f = layers.BatchNormalization()(feature_input)
    f = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(5e-5))(f)
    f = layers.Dropout(0.3)(f)
    f = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(5e-5))(f)

    # Fuse Signal Branch and Feature Branch
    combined = layers.Concatenate()([x, f])

    z = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(5e-5))(combined)
    z = layers.Dropout(0.3)(z)

    outputs = layers.Dense(n_classes, activation="softmax")(z)

    model = models.Model(
        inputs=[signal_input, feature_input],
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

