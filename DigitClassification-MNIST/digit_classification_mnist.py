import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

# Load and preprocess dataset
(x_train, y_train), (x_val, y_val) = datasets.mnist.load_data()

# Normalize to [-1, 1]
x_train = 2 * tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.0 - 1
x_val = 2 * tf.convert_to_tensor(x_val, dtype=tf.float32) / 255.0 - 1

# Flatten input images: [batch, 28, 28] -> [batch, 784]
x_train_flat = tf.reshape(x_train, (-1, 28 * 28))
x_val_flat = tf.reshape(x_val, (-1, 28 * 28))

# One-hot encode labels
y_train = tf.one_hot(y_train, depth=10)
y_val = tf.one_hot(y_val, depth=10)

# Define model
model = keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)   # logits output
])

# Define optimizer
optimizer = optimizers.SGD(learning_rate=0.1)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['accuracy']   # optional: reports how often highest logit matches label
)

# Train the model
model.fit(
    x_train_flat, y_train,
    epochs=50,
    batch_size=32,             # default is 32; you can change
    validation_data=(x_val_flat, y_val)
)

# Evaluate the model (optional; usually done automatically at the end of training)
loss, acc = model.evaluate(x_val_flat, y_val)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}")
