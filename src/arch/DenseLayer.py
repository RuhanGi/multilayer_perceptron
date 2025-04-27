from dataclasses import dataclass, field
import numpy as np
from .Perceptron import Perceptron


@dataclass
class DenseLayer:
    """
        Stores a single Layer of nodes
        
        Attributes:
        num_inputs: number of inputs
        num_nodes: number of nodes layer

        activation: activation function
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
        return [p.calculate(input) for p in self.perceps]
