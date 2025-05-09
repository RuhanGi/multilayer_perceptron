from arch import Network
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
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
        df = pd.read_csv(fil, header=None)
        df.dropna(inplace=True)
        features = np.array(df.iloc[:, 2:])
        labels = np.array(df.iloc[:,1])
        return features, labels
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)

def plotMetrics(metrics):
    # * Plot Loss function of different optimizers
    # * Plot 'Acc', 'F1'
    plt.plot(metrics['Val']['Loss'], color='red')

    plt.tight_layout()
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'escape' else None)
    plt.show()

def main():
    if len(sys.argv) != 3:
        print(GREEN + " Usage:  " + YELLOW + "python3 train.py {traindata}.csv {valdata}.csv" + RESET)
        sys.exit(0)

    np.set_printoptions(suppress=True)

    train, train_out = loadData(sys.argv[1])
    val, val_out = loadData(sys.argv[2])

    n = Network(train, train_out)
    n.addLayer(24)
    n.addLayer(15)
    metrics = n.fit(val, val_out)

    plotMetrics(metrics)

    with open('net.pkl', 'wb') as f:
        pickle.dump(n, f)

if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except Exception as e:
    #     print(RED + "Error: " + str(e) + RESET)
    #     sys.exit(1)
