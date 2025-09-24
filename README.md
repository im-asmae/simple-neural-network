# Simple Neural Network for Gender Prediction
This is a small neural network implemented in Python using NumPy. The network predicts gender based on a simplified dataset containing "weight" and "height" features.

## Project Overview

This project demonstrates a basic implementation of a feedforward neural network with:
- 1 hidden layer
- Sigmoid activation function
- Backpropagation for training
- Manual weight and bias updates (no frameworks like TensorFlow or PyTorch)

-> It is intended for learning purposes and understanding the fundamentals of neural networks.

## Original Dataset
| Name    | Weight (lb) | Height (in) | Gender |
| ------- | ----------- | ----------- | ------ |
| Alice   | 133         | 65          | F      |
| Bob     | 160         | 72          | M      |
| Charlie | 152         | 70          | M      |
| Diana   | 120         | 60          | F      |

### Simplified Dataset

To make the neural network easier to understand and train manually, we simplified the dataset by subtracting 135 from the weight and 66 from the height, and converting gender to numeric labels (1 = Female, 0 = Male):
| Name    | Weight (minus 135) | Height (minus 66) | Gender |
| ------- | ------------------ | ----------------- | ------ |
| Alice   | -2                 | -1                | 1      |
| Bob     | 25                 | 6                 | 0      |
| Charlie | 17                 | 4                 | 0      |
| Diana   | -15                | -6                | 1      |

**Reason for Simplification:**
- Reduces large numbers, making computations easier and more intuitive for learning.
- Converts categorical labels into numeric values required for training.
- Allows you to clearly observe how the neural network learns and updates weights during training.

## How It Works

1. Forward Pass: Computes the hidden layer outputs and the final output using the sigmoid activation function.
2. Loss Calculation: Uses Mean Squared Error (MSE) between predictions and actual labels.
3. Backpropagation: Calculates gradients and updates weights and biases.
4. Training: Repeats for a set number of epochs to minimize the loss.

## Neural Network Architecture

- Input layer: 2 neurons (weight, height)
- Hidden layer: 3 neurons
- Output layer: 1 neuron (gender prediction)
- Activation function: Sigmoid

## How to Run:
**1. Clone this repository:**
```bash
  git clone https://github.com/im-asmae/simple-neural-network.git
  cd simple-neural-network
```
**2. Install dependencies:**
```bash
  pip install numpy
```
**3. Run the script:**
```bash
  python neural_network.py
``` 
- You will see the training progress and final predictions in the console.

## Example Output
```text
Epoch 0, Loss: 0.3478
Epoch 2000, Loss: 0.0011
Epoch 4000, Loss: 0.0005
Epoch 6000, Loss: 0.0003
Epoch 8000, Loss: 0.0002

Final predictions:
Input: [-2. -1.], Prediction: 0.9830 → Female, Actual: Female
Input: [25.  6.], Prediction: 0.0129 → Male, Actual: Male
Input: [17.  4.], Prediction: 0.0129 → Male, Actual: Male
Input: [-15.  -6.], Prediction: 0.9879 → Female, Actual: Female
```

## Learnings
- Implemented a neural network from scratch
- Understood **forward and backward propagation**
- Learned how to update **weights and biases manually**

## License
This project is for educational purposes only and open for modification. No formal license is applied.

## Future Improvements
- Add more data points for better accuracy.
- Experiment with different activation functions.
- Implement multiple hidden layers.
   
