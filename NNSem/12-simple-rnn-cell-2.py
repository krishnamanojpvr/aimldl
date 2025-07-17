import tensorflow as tf

h0 = [tf.zeros([4, 64])]
x = tf.random.normal([4, 80, 100])
cell = tf.keras.layers.SimpleRNNCell(64)
h = h0
state_history = []

for xt in tf.unstack(x, axis=1):
    out, h = cell(xt, h)
    state_history.append(out)

final_output = out
print(final_output.shape)
