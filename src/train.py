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

    # TODO investigate trainF1 score not improving over time
    # TODO prediction program
    # TODO Nesterov momentum, RMSprop, adam / compare different models in same graph

    single = True
    if single:
        n = Network(train, train_out, val, val_out)
        n.addLayer(24)
        n.addLayer(15)
        acc = n.fit()
        print(BLUE + f"Accuracy: {PURPLE}{acc*100:.4f}%" + RESET)
    else:
        accs = []
        k = 20
        bs, ba, bn = 0,0, None
        offset = np.random.randint(10**5)
        for i in range(offset, offset+k):
            n = Network(train, train_out, val, val_out, seed=i)
            n.addLayer(24)
            n.addLayer(15)
            acc = n.fit()
            print(GRAY + f"\rModel: {i-offset}/{k}", end="")
            accs.append(acc)
            if acc > ba:
                bs, ba, bn = i, acc, n

        print(BLUE + f"\rBest Accuracy: {PURPLE}{ba*100:.4f}%{BLUE} by Seed {i}" + RESET)
        print(f"{GREEN}{k}-Fold Acc: {PURPLE}{np.average(accs)*100:.4f}%{RESET}")


if __name__ == "__main__":
    main()
