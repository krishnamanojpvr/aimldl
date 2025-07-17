import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 2  # input + hidden + output

        # Initialize weights and biases
        self.weights = [np.random.rand(input_size, hidden_sizes[0])]
        self.biases = [np.zeros((1, hidden_sizes[0]))]

        for i in range(len(hidden_sizes) - 1):
            self.weights.append(np.random.rand(hidden_sizes[i], hidden_sizes[i+1]))
            self.biases.append(np.zeros((1, hidden_sizes[i+1])))

        self.weights.append(np.random.rand(hidden_sizes[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, x):
        self.layer_outputs = []
        input_layer = x.reshape(1, -1)  # ensure input is 2D

        for i in range(self.num_layers - 1):
            weighted_sum = np.dot(input_layer, self.weights[i]) + self.biases[i]
            layer_output = sigmoid(weighted_sum)
            self.layer_outputs.append(layer_output)
            input_layer = layer_output

        return input_layer

    def backward(self, x, y, output, learning_rate):
        deltas = [None] * self.num_layers
        error = y - output
        deltas[-1] = error * sigmoid_derivative(output)

        # Backpropagate through hidden layers
        for i in range(self.num_layers - 2, 0, -1):
            error = np.dot(deltas[i+1], self.weights[i].T)
            deltas[i] = error * sigmoid_derivative(self.layer_outputs[i-1])

        # Update weights and biases
        input_layer = x.reshape(1, -1)
        self.weights[0] += learning_rate * np.dot(input_layer.T, deltas[1])
        self.biases[0] += learning_rate * deltas[1]

        for i in range(1, self.num_layers - 1):
            prev_output = self.layer_outputs[i-1]
            self.weights[i] += learning_rate * np.dot(prev_output.T, deltas[i+1])
            self.biases[i] += learning_rate * deltas[i+1]

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(len(X)):
                x = X[i]
                target = y[i].reshape(1, -1)
                output = self.forward(x)
                self.backward(x, target, output, learning_rate)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch} completed.")

    def predict(self, x):
        return self.forward(x)

if __name__ == "__main__":
    # Dataset: XOR with one-hot encoded labels
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])

    # Network configuration
    input_size = 2
    hidden_sizes = [20, 50, 25]
    output_size = 2
    learning_rate = 0.1
    epochs = 10000

    # Initialize and train the network
    nn = NeuralNetwork(input_size, hidden_sizes, output_size)
    nn.train(X, y, epochs, learning_rate)

    # Test predictions
    print("\n=== Predictions ===")
    for i in range(len(X)):
        prediction = nn.predict(X[i])
        print(f"Input: {X[i]}, Actual: {y[i]}, Predicted: {prediction}")
