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
        labels = np.array(df.iloc[:,1])
        return features, labels
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)

def make_onehot(labels, mapper=None):
    if mapper is None:
        unique = np.unique(labels)
        mapper = {label: i for i, label in enumerate(unique)}
    return np.array([self.mapper[v] for v in train_out])

# ! check if needed
def undo_onehot(onehot, mapper):
    return np.eye(len(mapper))[onehot]

def main():
    if len(sys.argv) != 3:
        print(GREEN + " Usage:  " + YELLOW + "python3 train.py {traindata}.csv {valdata}.csv" + RESET)
        sys.exit(0)

    train, train_out = loadData(sys.argv[1])
    val, val_out = loadData(sys.argv[2])

    n = Network(train, train_out)
    
    print(n.calculate())

if __name__ == "__main__":
    main()
