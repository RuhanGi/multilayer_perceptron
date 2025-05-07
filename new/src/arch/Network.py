# from .DenseLayer import DenseLayer
import sys

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
CYAN = "\033[96m"
GRAY = "\033[97m"
BLACK = "\033[98m"
RESET = "\033[0m"

class Network:
    """
        Whole Neural Network with Parameters and Layers
    """

    def __init__(self, train, train_out):
        limit = np.sqrt(6 / num_input)
        self.weights = np.random.uniform(-limit, limit, (num_input+1, num_nodes))
        self.func, self.dfunc = af.getFunc(act)

