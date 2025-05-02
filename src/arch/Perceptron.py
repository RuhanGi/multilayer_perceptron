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

def dtanh(x):
    return 4 * dsigmoid(2 * x)

# * 3. Rectified Linear Unit 
def ReLU(x, r=0):
    return max(r * x, x)

def dReLU(x, r=0):
    return 1 if x > 0 else r

# * 4. Exponential Linear Unit
def ELU(x, alpha=1.67326, lam=1.0507):
    return lam * max(alpha * (np.exp(x) - 1), x)

def dELU(x, alpha=1.67326, lam=1.0507):
    return lam if x > 0 else lam * alpha * np.exp(x) * (np.exp(x) - 1)

# * Linear - no change
def linear(z):
    return z

def dlinear(z):
    return np.ones_like(z)

# * 2. Softplus
def softplus(x):
    return x if x > 20 else np.log(1 + np.exp(x))

def dsoftplus(x):
    return 1 if x > 20 else 1 - sigmoid(-x)


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
        dfuncy = {
            'sigmoid' : dsigmoid,
            'tanh' : dtanh,
            'ReLU' : dReLU,
            'ELU' : dELU,
            'softmax' : dlinear,
            'softplus' : dsoftplus
        }
        self.weights = weights
        self.bias = 0
        self.activation = act
        self.func = funcy[act]
        self.dfunc = dfuncy[act]

    def setOutput(self, out):
        self.output = out

    def calculate(self, input):
        self.input = input
        self.preFunc = np.dot(self.input, self.weights) + self.bias
        self.output = self.func(self.preFunc)
        return self.output

    def backprop(self, pred):
        learningRate = 0.4
        error = pred - self.output
        grad = self.dfunc(self.preFunc) * error
        self.weights -= learningRate * self.input.T @ grad
        self.bias -= learningRate * np.sum(grad)
        return error