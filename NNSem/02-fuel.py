import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

# Download dataset
dataset_path = keras.utils.get_file(
    "auto-mpg.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)

# Load dataset
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(
    dataset_path, names=column_names,
    na_values="?", comment='\t', sep=" ", skipinitialspace=True
)
dataset = raw_dataset.copy()
dataset = dataset.dropna()

# One-hot encode 'Origin'
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0

# Split into train/test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Separate labels
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Normalize
train_stats = train_dataset.describe().transpose()
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Build model using keras.Sequential or subclassing
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[9]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(0.001),
    loss='mse',                  # Mean Squared Error
    metrics=['mae', 'mse']       # Mean Absolute Error and MSE
)

# Train the model
history = model.fit(
    normed_train_data, train_labels,
    epochs=200,
    batch_size=32,
    validation_split=0.2,        # use part of training data for validation
    verbose=1                    # print progress
)

# Evaluate the model on test set
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print(f"\nTest MAE: {mae:.4f}, Test MSE: {mse:.4f}")
