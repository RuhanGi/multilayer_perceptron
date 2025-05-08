from .DenseLayer import DenseLayer
from .lossfunc import getLoss
import matplotlib.pyplot as plt
import numpy as np
import sys

class Network:
    """
        Whole Neural Network with Parameters and Layers
    """

    def __init__(self, train, train_out):
        assert len(train) == len(train_out), "shape mismatch"
        self.means = np.average(train, axis=0)
        self.stds = np.std(train, axis=0)
        self.train = self.normalize(train)
        self.mapper = {lab: i for i, lab in enumerate(dict.fromkeys(train_out))}
        self.train_hot = self.makeOnehot(train_out)

        # ! EXPERIMENTING WITH ADDING INPUT LAYER
        # num_inputs = train.shape[1]
        # self.layers = [DenseLayer(num_inputs, num_inputs, 'sigmoid')]
        self.layers = []

    def normalize(self, inputs):
        assert len(self.means) == inputs.shape[1], "shape mismatch"
        assert len(self.stds) == inputs.shape[1], "shape mismatch"
        return (inputs - self.means) / self.stds

    def addLayer(self, num_nodes, act='sigmoid'):
        if len(self.layers) > 0:
            num_inputs = self.layers[-1].weights.shape[1]
        else:
            num_inputs = self.train.shape[1]
        layer = DenseLayer(num_inputs, num_nodes, act)
        self.layers.append(layer)

    def forward(self, passer, normalize=False):
        if normalize:
            passer = self.normalize(passer)
        for layer in self.layers:
            passer = layer.forward(passer)
        return passer

    def backprop(self, passer, learningRate, normalize=False, optimizer='default'):
        if normalize:
            passer = self.normalize(passer)
        for layer in reversed(self.layers):
            passer = layer.backprop(passer, learningRate)
        return passer

    def makeOnehot(self, y_out):
        indices = np.array([self.mapper[label] for label in y_out])
        return np.eye(len(self.mapper))[indices]

    def plotMetric(self, y, c, what):
        plt.plot(y, color=c, label=what)
        plt.ylim(0, int(np.max(y)+1))
        plt.xlim(0, len(y) - 1)
        plt.tight_layout()
        plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'escape' else None)
        plt.show()

    def fit(self, val, val_out, plot=False, epochs=10, loss='crossEntropy',
             batch_size=32,learningRate=0.2, optimizer='minibatch'):
        assert self.train.shape[1] == val.shape[1], "feature mismatch"
        assert set(val_out).issubset(set(self.mapper.keys())), "unidentified label found"

        self.addLayer(len(self.mapper), act='softmax')

        val = self.normalize(val)
        one_hot = self.makeOnehot(val_out)
        lFunc, dlFunc = getLoss(loss)
    
        # TODO make metrics array
        valloss = [lFunc(self.forward(val), one_hot)]

        for e in range(epochs):
            for i in range(0, len(self.train), batch_size):
                probs = self.forward(self.train[i:i+batch_size])
                # loss.append(lFunc(probs, self.train_hot[i:i+batch_size]))
                dLdp = dlFunc(probs, self.train_hot[i:i+batch_size])
                self.backprop(dLdp, learningRate)
            valloss.append(lFunc(self.forward(val), one_hot))

        self.plotMetric(valloss, 'red', 'Validation Loss')

        # TODO early stopping
        # TODO plot metrics

        print(valloss)
