import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Set parameters
vocab_size = 10000           # Use top 10,000 words in the dataset
max_sequence_length = 250    # Pad / truncate reviews to this length
embedding_dim = 128
units = 64
batch_size = 64
epochs = 3

# Load the IMDb dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences so they are all the same length
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_sequence_length)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_sequence_length)

# Build the model using Sequential
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    LSTM(units),
    # GRU(units),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
