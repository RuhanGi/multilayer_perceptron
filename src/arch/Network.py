from .DenseLayer import DenseLayer
from .lossfunc import getLoss
import matplotlib.pyplot as plt
import numpy as np
import copy

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
CYAN = "\033[96m"
GRAY = "\033[97m"
BLACK = "\033[98m"
RESET = "\033[0m"

def plotMetric(y, c, what):
        plt.plot(y, color=c, label=what)
        plt.ylim(0, int(np.max(y)+1))
        plt.tight_layout()
        plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'escape' else None)
        plt.show()

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
        self.layers = []

    def normalize(self, inputs):
        assert len(self.means) == inputs.shape[1], "shape mismatch"
        assert len(self.stds) == inputs.shape[1], "shape mismatch"
        return (inputs - self.means) / self.stds

    def makeOnehot(self, y_out):
        indices = np.array([self.mapper[label] for label in y_out])
        one_hot = np.eye(len(self.mapper))[indices]
        return one_hot

    def addLayer(self, num_nodes, act='sigmoid'):
        if len(self.layers) > 0:
            num_inputs = self.layers[-1].weights.shape[1]
        else:
            num_inputs = self.train.shape[1]
        layer = DenseLayer(num_inputs, num_nodes, act)
        self.layers.append(layer)

    def forward(self, passer, normalize=False):
        assert passer.shape[1] == self.train.shape[1], "shape mismatch"
        if normalize:
            passer = self.normalize(passer)
        for layer in self.layers:
            passer = layer.forward(passer)
        return passer

    def backprop(self, passer, learningRate, normalize=False, optimizer='default'):
        assert passer.shape[1] == len(self.mapper), "shape mismatch"
        for layer in reversed(self.layers):
            passer = layer.backprop(passer, learningRate)
        return passer

    def metricize(self, val, one_hot, metrics, lFunc):
        tprob = self.forward(self.train)
        vprob = self.forward(val)
        ttrue = np.argmax(self.train_hot, axis=1)
        vtrue = np.argmax(one_hot, axis=1)
        tpred = np.argmax(tprob, axis=1)
        vpred = np.argmax(vprob, axis=1)
        metrics['Train']['Acc'].append(np.mean(tpred == ttrue))
        metrics['Val']['Acc'].append(np.mean(vpred == vtrue))

        metrics['Train']['Loss'].append(lFunc(tprob, self.train_hot))
        metrics['Val']['Loss'].append(lFunc(vprob, one_hot))

        # tp = np.sum((pred == 1) & (val_out == 1))
        # fp = np.sum((pred == 1) & (val_out != 1))
        # fn = np.sum((pred != 1) & (val_out == 1))

        # metrics['Recall'].append(tp / (tp + fn + 1e-8))
        # metrics['Prec'].append(tp / (tp + fp + 1e-8))
        # metrics['F1'].append(tp / (tp + (fp + fn) / 2 + 1e-8))

        # TODO more metrics

    def error(self, probs, one_hot):
        assert probs.shape == one_hot.shape, "shape mismatch"
        return probs - one_hot

    def fit(self, val, val_out, plot=False, epochs=400, loss='crossEntropy',
             batch_size=8,learningRate=0.01, optimizer='minibatch'):

        assert len(val) == len(val_out), "sample mismatch"
        assert self.train.shape[1] == val.shape[1], "feature mismatch"
        assert set(val_out).issubset(set(self.mapper.keys())), "unidentified label found"

        self.addLayer(len(self.mapper), act='softmax')
        val = self.normalize(val)
        one_hot = self.makeOnehot(val_out)
        lFunc = getLoss(loss)

        metrics = {
            "Train": {'Loss': [], 'Acc': [], 'Recall': [], 'Prec': [], 'F1': []},
            "Val": {'Loss': [], 'Acc': [], 'Recall': [], 'Prec': [], 'F1': []}
        }
        self.metricize(val, one_hot, metrics, lFunc)

        minloss, best_lay = 100, None
        patience, wait = 10, 0

        for e in range(epochs):
            for i in range(0, len(self.train), batch_size):
                probs = self.forward(self.train[i:i+batch_size])
                dLdp = self.error(probs, self.train_hot[i:i+batch_size])
                self.backprop(dLdp, learningRate)
            self.metricize(val, one_hot, metrics, lFunc)
            print(f"{YELLOW}\rEpoch {e}/{epochs} - " +
                f"TrainLoss: {PURPLE}{metrics['Train']['Loss'][-1]:.6f}{YELLOW} " +
                f"ValLoss: {PURPLE}{metrics['Val']['Loss'][-1]:.6f}{YELLOW}", end="")
            if metrics['Val']['Loss'][-1] < minloss:
                minloss = metrics['Val']['Loss'][-1]
                best_lay = copy.deepcopy(self.layers)
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        plotMetric(metrics['Val']['Loss'], 'red', 'Validation Loss')
        plotMetric(metrics['Val']['Acc'], 'blue', 'ValAcc')
