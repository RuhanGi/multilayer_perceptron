from dataclasses import dataclass, field
import numpy as np

def sigmoid(x):
    out = np.empty_like(x)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1 + exp_x)
    return out

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return 2 * sigmoid(2 * x) - 1

def dtanh(x):
    return 4 * dsigmoid(2 * x)

def ReLU(x, r=0):
    return np.maximum(r * x, x)

def dReLU(x, r=0):
    return np.where(x > 0, 1, r)

def ELU(x, alpha=1.67326, lam=1.0507):
    return lam * np.maximum(alpha * (np.exp(x) - 1), x)

def dELU(x, alpha=1.67326, lam=1.0507):
    return lam * np.where(x > 0, 1, alpha * np.exp(x) * (np.exp(x) - 1))

def linear(z):
    return z

def dlinear(z):
    return np.ones_like(z)

def softmax(z):
    exp = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp / np.sum(exp, axis=0, keepdims=True)

def softplus(x):
    return x if x > 20 else np.log(1 + np.exp(x))

def dsoftplus(x):
    return 1 if x > 20 else 1 - sigmoid(-x)

@dataclass
class DenseLayer:
    """
        Stores a single Layer of nodes
        
        Attributes:
        num_inputs: number of inputs
        num_nodes: number of nodes layer
        act: activation function
        perceps: stores every Perceptron in that layer
    """
    num_input: int
    num_nodes: int
    act: str

    def __post_init__(self):
        funcy = {
            'sigmoid' : (sigmoid, dsigmoid),
            'tanh' : (tanh, dtanh),
            'ReLU' : (ReLU, dReLU),
            'ELU' : (ELU, dELU),
            'softmax' : (softmax, dlinear),
            'softplus' : (softplus, dsoftplus)
        }
        self.func = funcy[self.act][0]
        self.dfunc = funcy[self.act][1]
        limit = np.sqrt(6 / self.num_input)
        self.weights = np.random.uniform(-limit, limit, (self.num_input+1, self.num_nodes))
        self.velocity = np.zeros_like(self.weights)
        self.momentum = np.zeros_like(self.weights)

    def calculate(self, input):
        self.input = np.hstack([input, np.ones((input.shape[0], 1))])
        self.z = np.dot(self.input, self.weights)
        self.output = self.func(self.z)
        return self.output

    def backprop(self, error, learningRate):
        assert error.shape[1] == self.num_nodes, "shape mismatch"
        grad = error * self.dfunc(self.z)
        self.weights -= learningRate * self.input.T @ grad
        return grad @ self.weights[:-1].T
    
    def adambackprop(self, error, learningRate):
        assert error.shape[1] == self.num_nodes, "shape mismatch"
        grad = error * self.dfunc(self.z)
        delt = self.input.T @ grad

        decay1, decay2, epsilon = 0.9, 0.99, 10**-8
        self.momentum = decay1 * self.momentum + (1-decay1) * delt
        self.velocity = decay2 * self.velocity + (1-decay2) * delt**2

        self.weights -= learningRate * self.momentum / (np.sqrt(self.velocity) + epsilon)
        return grad @ self.weights[:-1].T
    
    def rmsbackprop(self, error, learningRate):
        assert error.shape[1] == self.num_nodes, "shape mismatch"
        grad = error * self.dfunc(self.z)
        delt = self.input.T @ grad

        decay, epsilon = 0.95, 10**-8
        self.velocity = decay * self.velocity + (1-decay) * delt**2

        self.weights -= learningRate * delt / (np.sqrt(self.velocity) + epsilon)
        return grad @ self.weights[:-1].T
