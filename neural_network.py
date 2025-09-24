import numpy as np

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training data (Weight, Height)
X = np.array([
    [-2, -1],   # Alice
    [25, 6],    # Bob
    [17, 4],    # Charlie
    [-15, -6]   # Diana
], dtype=float)

# Labels (Gender: 1 = female, 0 = male)
y = np.array([[1], [0], [0], [1]], dtype=float)

# Seed for reproducibility
np.random.seed(42)

# Neural network architecture
input_neurons = 2    # Weight, Height
hidden_neurons = 3   # can adjust
output_neurons = 1   # Gender

# Weights and biases initialization
weights_input_hidden = np.random.randn(input_neurons, hidden_neurons)
weights_hidden_output = np.random.randn(hidden_neurons, output_neurons)
bias_hidden = np.zeros((1, hidden_neurons))
bias_output = np.zeros((1, output_neurons))

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    # ---- Forward pass ----
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # ---- Compute error ----
    error = y - final_output

    # ---- Backpropagation ----
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # ---- Update weights and biases ----
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # ---- Print loss occasionally ----
    if epoch % 2000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ---- Final predictions ----
print("\nFinal predictions:")
for i, sample in enumerate(X):
    pred = final_output[i][0]
    label = "Female" if pred > 0.5 else "Male"
    print(f"Input: {sample}, Prediction: {pred:.4f} → {label}, Actual: {'Female' if y[i][0]==1 else 'Male'}")
