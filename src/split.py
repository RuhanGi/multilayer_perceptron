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
        df.drop_duplicates(inplace=True)
        return df
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)

def splitData(df, ratio=0.8):
    shuffle = np.random.permutation(df.index)
    size = int(ratio * len(df))
    return df.iloc[shuffle[:size]], df.iloc[shuffle[size:]]

def main():
    if len(sys.argv) != 2:
        print(RED + "Pass Data to Split!" + RESET)
        sys.exit(1)

    df = loadData(sys.argv[1])
    ratio = 0.8

    train, val = splitData(df, ratio=ratio)

    train.to_csv("data/train.csv", header=False, index=False)
    val.to_csv("data/val.csv", header=False, index=False)
    print(GREEN + f"{ratio*100:.0f}% Train/Test Split Successfull!" + RESET)

if __name__ == "__main__":
    main()
