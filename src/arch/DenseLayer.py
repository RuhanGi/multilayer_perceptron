from dataclasses import dataclass, field
import numpy as np
from .Perceptron import Perceptron


# ? NONLINEAR FUNCTIONS
# * 1. Sigmoid - center: (0,0.5)
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

# * 2. Hyperbolic Tangent
def tanh(x):
    return 2 * sigmoid(2 * x) - 1

def dtanh(x):
    return 4 * dsigmoid(2 * x)

# * 3. Rectified Linear Unit 
def ReLU(x, r=0):
    return np.max(r * x, x)

def dReLU(x, r=0):
    return 1 if x > 0 else r

# * 4. Exponential Linear Unit
def ELU(x, alpha=1.67326, lam=1.0507):
    return lam * np.max(alpha * (np.exp(x) - 1), x)

def dELU(x, alpha=1.67326, lam=1.0507):
    return lam if x > 0 else lam * alpha * np.exp(x) * (np.exp(x) - 1)

# * Linear - no change
def linear(z):
    return z

def dlinear(z):
    return np.ones_like(z)

def softmax(z):
    exp = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp / np.sum(exp, axis=0, keepdims=True)

# * 2. Softplus
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
            'sigmoid' : sigmoid,
            'tanh' : tanh,
            'ReLU' : ReLU,
            'ELU' : ELU,
            'softmax' : softmax,
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
        self.func = funcy[self.act]
        self.dfunc = dfuncy[self.act]
        self.perceps = np.array(
            [Perceptron(self.num_input) for _ in range(self.num_nodes)],
            dtype=object
        )

    def calculate(self, input):
        self.output = self.func(np.array([p.calculate(input) for p in self.perceps]))
        return self.output.T

    def backprop(self, error):
        """
        error: array storing errors of each node
        """
        assert error.shape[1] == self.num_nodes, "shape mismatch"

        newerr = []
        for i,p in enumerate(self.perceps):
            newerr.append(p.backprop(error[:,i], self.dfunc))
        newerr = np.sum(np.array(newerr), axis=0)
        return newerr
        