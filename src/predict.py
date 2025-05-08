from arch import Network
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

def loadNetwork(fil):
    try:
        with open(fil, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print(GREEN + " Usage:  " + YELLOW + "python3 train.py {traindata}.csv {valdata}.csv" + RESET)
        sys.exit(0)

    inputs, val_out = loadData(sys.argv[1])
    net = loadNetwork(sys.argv[2])

    output = net.predict(inputs, val_out=val_out, needsNormal=True)
    print(f"Binary Cross Entropy Loss is: ", output)

if __name__ == "__main__":
    main()
