from .DenseLayer import DenseLayer
import matplotlib.pyplot as plt
import numpy as np
import copy
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

def binaryCross(true, pred):
    assert len(true) == len(pred), "length mismatch"
    return -np.mean(true * np.log(pred) + (1-true) * np.log(1-pred))

def plotMetrics(tmetrics, vmetrics):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        axs[0].plot(tmetrics['Acc'], 'g:', label='Train Accuracy')
        axs[0].plot(vmetrics['Acc'], 'g-', label='Validation Accuracy')
        axs[0].plot(tmetrics['F1'], 'b:', label='Train F1')
        axs[0].plot(vmetrics['F1'], 'b-', label='Validation F1')
        axs[0].axhline(y=1, color='k', linestyle=':')
        axs[0].set_title('Accuracy')
        axs[0].legend()

        axs[1].plot(tmetrics['Loss'], 'r:', label='Train Loss')
        axs[1].plot(vmetrics['Loss'], 'r-', label='Validation Loss')
        axs[1].set_title('Loss')
        axs[1].legend()

        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'escape' else None)
        plt.show()

class Network:
    """
        Whole Neural Network with Parameters and Layers
    """

    def __init__(self, train, train_out, val, val_out, seed=8716):
        assert len(train) == len(train_out), "sample mismatch"
        assert len(val) == len(val_out), "sample mismatch"
        assert train.shape[1] == val.shape[1], "feature mismatch"
        assert np.all(np.isin(val_out, train_out)), "unidentified label found"

        self.means = np.average(train, axis=0)
        self.stds = np.std(train, axis=0)
        train = (train - self.means) / self.stds
        val = (val - self.means) / self.stds

        unique = np.unique(train_out)
        self.mapper = {label: i for i, label in enumerate(unique)}
        train_out = np.array([self.mapper[v] for v in train_out])
        val_out = np.array([self.mapper[v] for v in val_out])

        self.train = train
        self.train_out = train_out
        self.val = val
        self.val_out = val_out

        self.layers = []
        np.random.seed(seed)

    def predict(self, inputs, val_out=None, needsNormal=False):
        if needsNormal:
            inputs = (inputs - self.means) / self.stds

        for layer in self.layers:
            inputs = layer.calculate(inputs)

        if val_out is not None:
            val_out = np.array([self.mapper[v] for v in val_out])
            return binaryCross(val_out, inputs[:,1])
        else:
            return inputs

    def addLayer(self, num_nodes, activation='sigmoid'):
        num_input = self.train.shape[1] if len(self.layers) == 0 else self.layers[-1].num_nodes
        self.layers.append(DenseLayer(num_input, num_nodes, activation))

    def metricize(self, metrics, val, val_out):
        val = self.predict(val)
        pred = np.argmax(val, axis=1)

        metrics['Loss'].append(binaryCross(val_out, val[:,1]))
        metrics['Acc'].append(np.mean(pred == val_out))

        tp = np.sum((pred == 1) & (val_out == 1))
        fp = np.sum((pred == 1) & (val_out != 1))
        fn = np.sum((pred != 1) & (val_out == 1))

        metrics['Recall'].append(tp / (tp + fn + 1e-8))
        metrics['Prec'].append(tp / (tp + fp + 1e-8))
        metrics['F1'].append(tp / (tp + (fp + fn) / 2 + 1e-8))

    def fit(self, learningRate=0.01, batch_size=8, epochs=400, optimizer='minibatch'):
        try:
            self.addLayer(len(self.mapper), 'softmax')
        
            tmetrics = {'Loss': [], 'Acc': [], 'Recall': [], 'Prec': [], 'F1': []}
            vmetrics = {'Loss': [], 'Acc': [], 'Recall': [], 'Prec': [], 'F1': []}
            self.metricize(tmetrics, self.train, self.train_out)
            self.metricize(vmetrics, self.val, self.val_out)

            best_acc, best_lay = 0, None
            patience, wait = 10, 0

            for e in range(epochs):
                for i in range(0, len(self.train), batch_size):
                    output = self.predict(self.train[i:i+batch_size])
            
                    error = output - np.eye(len(self.mapper))[self.train_out[i:i+batch_size]]
                    for layer in reversed(self.layers):
                        if optimizer == 'adam':
                            error = layer.adambackprop(error, learningRate)
                        elif optimizer == 'rmsprop':
                            error = layer.rmsbackprop(error, learningRate)
                        else:
                            error = layer.backprop(error, learningRate)
        
                self.metricize(tmetrics, self.train, self.train_out)
                self.metricize(vmetrics, self.val, self.val_out)
                if vmetrics['Acc'][-1] > best_acc:
                    best_acc = vmetrics['Acc'][-1]
                    best_lay = copy.deepcopy(self.layers)
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
            self.layers = best_lay
            # plotMetrics(tmetrics, vmetrics)
            return best_acc, vmetrics
        except Exception as e:
            print(RED + "Error: " + str(e) + RESET)
            sys.exit(1)
