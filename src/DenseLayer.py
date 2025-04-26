from dataclasses import dataclass, field
import numpy as np


@dataclass
class DenseLayer:
    """
        Stores a single Layer of nodes
        
        Attributes:
        num_inputs: number of inputs
        num_nodes: number of nodes layer

        activation: activation function
        weights: weight matrix
    """
    num_input: int
    num_nodes: int
    activation: str = 'sigmoid'
    weights: str = 'identity'
    biases: np.ndarray = field(init=False) 
    wmatrix: np.ndarray = field(init=False)

    def __post_init__(self):
        self.biases = np.zeros(self.num_nodes)
        
        if self.weights == 'identity':
            self.wmatrix = np.identity(self.num_nodes)
        elif self.weights == 'heUniform':
            self.wmatrix = np.random.uniform(
                low = -np.sqrt(6/self.num_nodes),
                high = np.sqrt(6/self.num_nodes),
                size=(self.num_input, self.num_nodes)
            )
            
