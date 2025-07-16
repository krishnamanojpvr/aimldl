import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define dataset parameters
batchsz = 128
total_words = 10000
max_review_len = 80
embedding_len = 100

# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)

print("Training Data Shape:", x_train.shape)
print("Length of First Training Sample:", len(x_train[0]))
print("Training Labels Shape:", y_train.shape)
print("Test Data Shape:", x_test.shape)
print("Length of First Test Sample:", len(x_test[0]))
print("Test Labels Shape:", y_test.shape)

# Get the word-to-index mapping
word_index = keras.datasets.imdb.get_word_index()
print("Total words in dataset:", len(word_index))
for k, v in list(word_index.items())[:10]:
    print(f"Word: {k}, Index: {v}")

# Modify word_index for special tokens
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Create reverse index mapping
reverse_word_index = dict((value, key) for (key, value) in word_index.items())

# Decode function
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decoded_review = decode_review(x_train[0])
print(decoded_review)

# Pad and truncate sequences
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

# Wrap into Dataset objects
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)

print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)

# Define model using Sequential
def build_model(units):
    model = keras.Sequential([
        layers.Embedding(total_words, embedding_len, input_length=max_review_len),
        layers.SimpleRNN(units, dropout=0.5, return_sequences=True),
        layers.SimpleRNN(units, dropout=0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Train & test
def main():
    units = 64
    epochs = 20
    model = build_model(units)
    model.compile(
        optimizer=tf.optimizers.Adam(0.001),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    model.evaluate(db_test)

# Run the training
main()


