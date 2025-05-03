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

def categoricalCrossentropy(true, pred):
    logpred = np.log(pred)
    return sum(t*lp for t,lp in zip(true, logpred))

def onehot(arr, classes):
    return np.eye(classes)[arr]

class Network:
    """
        Whole Neural Network with Parameters and Layers
    """

    def __init__(self, num_input):
        self.num_input = num_input
        self.layers = []

    def addLayer(self, num_nodes, activation='sigmoid'):
        inputs = self.layers[-1].num_nodes if len(self.layers) else self.num_input
        self.layers.append(DenseLayer(inputs, num_nodes, activation))

    def fit(self, train, train_out, val, val_out, loss='categoricalCrossentropy',
                learningRate=0.01, batch_size=8, epochs=1):
        try:
            assert len(train) == len(train_out), "sample mismatch"
            assert len(val) == len(val_out), "sample mismatch"
            assert train.shape[1] == val.shape[1], "feature mismatch"
            assert np.all(np.isin(val_out, train_out)), "unidentified label found"
        
            means = np.average(train, axis=0)
            stds = np.std(train, axis=0)
            train = (train - means) / stds
            val = (val - means) / stds

            unique = np.unique(train_out)
            mapper = {label: i for i, label in enumerate(unique)}
            train_out = np.array([mapper[v] for v in train_out])
            val_out = np.array([mapper[v] for v in val_out])


            for _ in range(epochs): # ! EPOCHS SET TO 1 ONLY
                for i in range(0, len(train), batch_size):
                    output = train[i:i+batch_size]
                    for layer in self.layers:
                        output = layer.calculate(output)

                    error = output - np.eye(len(unique))[train_out[i:i+batch_size]]
                    for layer in reversed(self.layers):
                        error = layer.backprop(error)
        
            # for layer in self.layers:
            #     predictions = layer.calculate(predictions)
            # predictions = np.argmax(predictions, axis=0)

            # count = 0
            # for i in range(len(predictions)):
            #     if predictions[i] == val.iloc[i, 1]:
            #         count += 1
            # print("Accuracy: ", count/len(predictions)*100, "%")

        except Exception as e:
            print(RED + "Error: " + str(e) + RESET)
            sys.exit(1)
