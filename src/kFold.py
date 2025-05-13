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


def loadData(fil):
    try:
        df = pd.read_csv(fil, header=None)
        df.dropna(inplace=True)
        features = np.array(df.iloc[:, 2:])
        labels = np.array(df.iloc[:, 1])
        return features, labels
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)


def splitData(features, labels, ratio=0.8):
    shuffle = np.random.permutation(len(features))
    size = int(ratio * len(features))
    train = features[shuffle[:size]]
    train_out = labels[shuffle[:size]]
    val = features[shuffle[size:]]
    val_out = labels[shuffle[size:]]
    return train, train_out, val, val_out


def runOneFold(features, labels):
    train, train_out, val, val_out = splitData(features, labels)

    n = Network(train, train_out)
    n.addLayer(24)
    n.addLayer(15)
    n.fit(val, val_out, opt="")

    return n.calcLoss(val, val_out)


def main():
    if len(sys.argv) != 2:
        print(GREEN + " Usage:  " + YELLOW + "python3 kFold.py {data}.csv" + RESET)
        sys.exit(0)

    features, labels = loadData(sys.argv[1])

    k = 20
    losses = []
    for _ in range(k):
        losses = np.append(losses, runOneFold(features, labels))
    minL = np.min(losses)
    meanL = np.mean(losses)
    maxL = np.max(losses)
    goodL = np.sum(losses < 0.08)

    print(BLUE + f"Distribution of {k}-Fold Cross Validation:")
    print(YELLOW + "- Minimum Loss:", GREEN if minL < 0.08 else RED, minL)
    print(YELLOW + "- Average Loss:", GREEN if meanL < 0.08 else RED, meanL)
    print(YELLOW + "- Maximum Loss:", GREEN if maxL < 0.08 else RED, maxL)
    print(YELLOW + "- %Loss < 0.08:", GREEN if goodL >= k / 3 else RED, f"{goodL/k*100:.2f}%")


if __name__ == "__main__":
    main()
