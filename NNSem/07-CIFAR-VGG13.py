import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np

# 1. Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values to [0,1]
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# Convert labels from shape (num_samples,1) → (num_samples,)
y_train = y_train.squeeze()
y_test  = y_test.squeeze()

# Print dataset shapes
print('Train:', x_train.shape, y_train.shape)
print('Test :', x_test.shape, y_test.shape)

# 2. Build VGG13-like model using Sequential
model = models.Sequential([
    # Conv Block 1
    layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(32,32,3)),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D((2,2), strides=2, padding='same'),

    # Conv Block 2
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D((2,2), strides=2, padding='same'),

    # Conv Block 3
    layers.Conv2D(256, (3,3), padding='same', activation='relu'),
    layers.Conv2D(256, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D((2,2), strides=2, padding='same'),

    # Conv Block 4
    layers.Conv2D(512, (3,3), padding='same', activation='relu'),
    layers.Conv2D(512, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D((2,2), strides=2, padding='same'),

    # Conv Block 5
    layers.Conv2D(512, (3,3), padding='same', activation='relu'),
    layers.Conv2D(512, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D((2,2), strides=2, padding='same'),

    # Flatten + Fully Connected layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes
])

# Show model summary
model.summary()

# 3. Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # use sparse labels
    metrics=['accuracy']
)

# 4. Train the model
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,  # use part of train data for validation
    verbose=2
)

# 5. Evaluate on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
