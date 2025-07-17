import tensorflow as tf

x = tf.random.normal([4, 80, 100])
cell0 = tf.keras.layers.SimpleRNNCell(64)
cell1 = tf.keras.layers.SimpleRNNCell(64)
h0 = [tf.zeros([4, 64])]
h1 = [tf.zeros([4, 64])]
output_sequence_cell0 = []
output_sequence_cell1 = []

for xt in tf.unstack(x, axis=1):
    out0, h0 = cell0(xt, h0)
    output_sequence_cell0.append(out0)

for xt in output_sequence_cell0:
    out1, h1 = cell1(xt, h1)
    output_sequence_cell1.append(out1)

final_output_cell0 = tf.stack(output_sequence_cell0, axis=1)
final_output_cell1 = tf.stack(output_sequence_cell1, axis=1)
print("Final Output for cell0 Shape:", final_output_cell0.shape)
print("Final Output for cell1 Shape:", final_output_cell1.shape)
