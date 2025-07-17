import tensorflow as tf
from tensorflow.keras import layers, Model

# Define a basic residual block
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    # First conv layer
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    # Second conv layer
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)

    # Adjust shortcut if needed
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add shortcut and conv output
    y = layers.add([shortcut, y])
    y = layers.Activation('relu')(y)

    return y

# Build ResNet-18 model
def build_resnet18(input_shape=(224, 224, 3), num_classes=1000):
    input_tensor = layers.Input(shape=input_shape)

    # Initial conv + max pool
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Residual blocks
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)

    x = residual_block(x, filters=128, stride=2)
    x = residual_block(x, filters=128)

    x = residual_block(x, filters=256, stride=2)
    x = residual_block(x, filters=256)

    x = residual_block(x, filters=512, stride=2)
    x = residual_block(x, filters=512)

    # Global average pooling & output
    x = layers.GlobalAveragePooling2D()(x)
    output_tensor = layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

# Instantiate the model
resnet18 = build_resnet18()

# Display the model summary
resnet18.summary()
