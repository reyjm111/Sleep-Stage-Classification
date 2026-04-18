from tensorflow.keras import layers, models

def build_cnn(input_shape, n_classes=5):
    model = models.Sequential([
        
        # Input: (T, 2)
        layers.Input(shape=input_shape),

        # Conv Block 1
        layers.Conv1D(32, kernel_size=7, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),

        # Conv Block 2
        layers.Conv1D(64, kernel_size=5, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),

        # Conv Block 3
        layers.Conv1D(128, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),

        # Global pooling
        layers.GlobalAveragePooling1D(),

        # Dense head
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(n_classes, activation='softmax')
    ])

    return model