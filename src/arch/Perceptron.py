import numpy as np


# ? NONLINEAR FUNCTIONS
# * 1. Sigmoid - center: (0,0.5)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# * 2. Hyperbolic Tangent
def tanh(x):
    return 2 * sigmoid(2 * x) - 1

# * 3. Rectified Linear Unit 
def ReLU(x, r=0):
    return max(r * x, x)

# * 4. Exponential Linear Unit
def ELU(x, alpha=1.67326, lam=1.0507):
    return lam * max(alpha * (np.exp(x) - 1), x)


# * Linear - no change
def linear(z):
    return z

# * 2. Softplus
def softplus(x):
    if x > 20:
        return x
    return np.log(1 + np.exp(x))


class Perceptron:
    """
        Stores Perceptron
        
        Attributes:
        weights: weights for input
        bias: bias for weighted sum
        activation: string for function
        func: activation function
    """

    def __init__(self, weights, act='sigmoid'):
        funcy = {
            'sigmoid' : sigmoid,
            'tanh' : tanh,
            'ReLU' : ReLU,
            'ELU' : ELU,
            'softmax' : linear,
            'softplus' : softplus
        }
        self.weights = weights
        self.bias = 0
        self.activation = act
        self.func = funcy[act]

    def calculate(self, input):
        self.input = input
        self.out = np.dot(self.weights, input) + self.bias
        return float(self.func(self.out))

    def backprop(self, pred):
        learningRate = 0.4
        error = 1 - pred
        self.weights -= learningRate * error * dsigmoid(self.out) * np.array(self.input)
        self.bias -= learningRate * error * dsigmoid(self.out)