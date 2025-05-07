from .DenseLayer import DenseLayer
import numpy as np
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
        self.means = np.average(train, axis=0)
        self.stds = np.std(train, axis=0)
        self.train = self.normalize(train)

        num_inputs = train.shape[0]
        self.layers = [DenseLayer(num_inputs, num_inputs, 'sigmoid')]
        # ! EXPERIMENTING WITH ADDING INPUT LAYER
        # self.layers = []

    def normalize(self, inputs):
        assert len(self.means) == inputs.shape[1], "shape mismatch"
        assert len(self.stds) == inputs.shape[1], "shape mismatch"
        return (inputs - self.means) / self.stds

    def addLayer(self, num_nodes, act='sigmoid'):
        layer = DenseLayer(self.layers[-1].weights.shape[1], num_nodes, act)
        self.layers.append(layer)

    def calculate(self, passer):
        for layer in self.layers:
            passer = layer.calculate(passer)
        return passer
