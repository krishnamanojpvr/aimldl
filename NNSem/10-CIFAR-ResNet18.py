import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.utils import to_categorical

# Load and preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# Residual block
def residual_block(x, filters, stride=1):
    shortcut = x

    # First conv
    y = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    # Second conv
    y = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(y)
    y = layers.BatchNormalization()(y)

    # Adjust shortcut if needed
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add skip connection
    out = layers.add([y, shortcut])
    out = layers.Activation('relu')(out)
    return out

# Build ResNet-18
def build_resnet18(input_shape=(32,32,3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # conv2_x
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # conv3_x
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)

    # conv4_x
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)

    # conv5_x
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Instantiate and compile the model
model = build_resnet18()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')
