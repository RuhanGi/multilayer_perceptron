import numpy as np


# ? NONLINEAR FUNCTIONS
# * 1. Sigmoid - center: (0,0.5)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# * 2. Hyperbolic Tangent
def tanh(x):
    return 2 * sigmoid(2 * x) - 1

# * 3. Rectified Linear Unit 
def ReLU(x, r=0):
    return max(r * x, x)

# * 4. Exponential Linear Unit
def ELU(x, alpha=1.67326, lam=1.0507):
    return lam * max(alpha * (np.exp(x) - 1), x)


# ? Exponential Linear Units
# * 1. Softmax for Multi-Class
def softmax(z):
    exp = np.exp(z)
    return exp / np.sum(exp)

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
            'softplus': softplus
        }
        self.weights = weights
        self.bias = 0
        self.activation = act
        self.func = funcy[act]

    def calculate(self, input):
        return float(self.func(np.dot(self.weights, input) + self.bias))
