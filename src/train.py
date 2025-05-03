from arch import Network
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

def loadData(fil):
    try:
        df = pd.read_csv(fil, header=None)
        train = np.array(df.iloc[:, 2:])
        train_out = np.array(df.iloc[:,1])
        return train, train_out
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print(GREEN + " Usage:  " + YELLOW + "python3 train.py {traindata}.csv {valdata}.csv" + RESET)
        sys.exit(0)

    np.set_printoptions(linewidth=200, suppress=True)

    train, train_out = loadData(sys.argv[1])
    val, val_out = loadData(sys.argv[2])

    n = Network(train.shape[1])
    n.addLayer(24)
    n.addLayer(15)
    n.addLayer(2, activation='softmax')
    n.fit(train, train_out, val, val_out)


if __name__ == "__main__":
    main()
