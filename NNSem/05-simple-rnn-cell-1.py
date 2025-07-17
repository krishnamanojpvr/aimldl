# code ex - 1
import tensorflow as tf

# Define the input feature length
n = 4

# Create a SimpleRNNCell with a cell state vector length of 3
cell = tf.keras.layers.SimpleRNNCell(3)

# Build the cell with the input shape (batch_size=None, feature_dim=n)
cell.build(input_shape=(None, n))

# Get the trainable variables: kernel (Wxh), recurrent_kernel (Whh), and bias (b)
trainable_variables = cell.trainable_variables

# Print the trainable variables
for var in trainable_variables:
    print(f"{var.name}: shape={var.shape}")
