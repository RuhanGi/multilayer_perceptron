from dataclasses import dataclass, field
import numpy as np
from .Perceptron import Perceptron


def softmax(z):
    exp = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp / np.sum(exp, axis=0, keepdims=True)


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
    act: str = 'sigmoid'
    perceps: list = field(default_factory=list)

    def __post_init__(self):
        limit = np.sqrt(6 / self.num_input)
        for i in range(self.num_nodes):
            self.perceps.append(Perceptron(
                np.random.uniform(-limit, limit, self.num_input),
                act=self.act
            ))

    def calculate(self, input):
        if self.act == 'softmax':
            soft = softmax([p.calculate(input) for p in self.perceps])
            for p,s in zip(self.perceps, soft):
                p.setOutput(s)
            return soft
        else:
            return [p.calculate(input) for p in self.perceps]

    def backprop(self, pred):
        error = pred
        for p in self.perceps:
            error = p.backprop(pred)
        return error
        