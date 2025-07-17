import tensorflow as tf

x = tf.random.normal([4, 80, 100])
cell0 = tf.keras.layers.SimpleRNNCell(64)
cell1 = tf.keras.layers.SimpleRNNCell(64)
h0 = [tf.zeros([4, 64])]
h1 = [tf.zeros([4, 64])]
output_sequence = []

for xt in tf.unstack(x, axis=1):
    out0, h0 = cell0(xt, h0)
    out1, h1 = cell1(out0, h1)
    output_sequence.append(out1)

final_output = tf.stack(output_sequence, axis=1)
print("Final Output Shape:", final_output.shape)
