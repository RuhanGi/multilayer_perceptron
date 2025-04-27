from Network import Network
import pandas as pd
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


def loadData(fil):
    try:
        return pd.read_csv(fil, header=None)
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)

def epoch(ndata, y, th):
    # TODO backpropagation and gradient descent
    # TODO Nesterov momentum, RMSprop, adam,
    return th - learningRate * grad / m

def trainModel(data, y, headers, n):
    try:
        classes = np.unique(y)

        # TODO create the weight matrix

        # * normalize

        maxiterations = 100
        for i in range(maxiterations): # TODO add tqdm()
            th = epoch(ndata, y, th)
            print(f"\rEpoch [{i}/{maxiterations}]",end="")

            # TODO add train_loss, trainF1, val_loss, valF1, accuracy

            # * add early stopping when overfitting valF1 score decreases

        # TODO learning curve graphs: Loss + Accuracy
        # TODO multiple curves on same graph (models)
        # TODO history of metrics obtained
    
        print(GREEN + "\rModel Trained!" + (" " * 30) + RESET)

        # * denormalize

        return th
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print(GREEN + " Usage:  " + YELLOW + "python3 train.py {traindata}.csv {valdata}.csv" + RESET)
        sys.exit(0)

    train = loadData(sys.argv[1])
    val = loadData(sys.argv[2])

    features = train.columns[2:]
    train[features] = (train[features] - train[features].mean()) / train[features].std()
    train[1] = train[1].map({'M': 1, 'B': 0})

    n = Network(train.shape[1]-2)
    n.addLayer(24)
    n.addLayer(24)
    n.addLayer(2)
    print(n.fit(train, val))


if __name__ == "__main__":
    main()
