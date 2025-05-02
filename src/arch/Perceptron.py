import numpy as np

class Perceptron:
    """
        Stores Perceptron
        
        Attributes:
        weights: weights for input
        bias: bias for weighted sum
    """

    def __init__(self, num_input):
        self.num_input = num_input
        self.bias = 0
        limit = np.sqrt(6 / num_input)
        self.weights = np.random.uniform(-limit, limit, num_input)

    def setOutput(self, out):
        self.output = out

    def calculate(self, input):
        self.input = input
        self.z = np.dot(input, self.weights) + self.bias
        return self.z

    def backprop(self, pred, f):
        learningRate = 0.4
        error = pred - self.z
        grad = f(self.z) * error
        self.weights -= learningRate * self.input.T @ grad
        self.bias -= learningRate * np.sum(grad)
        return error