import matplotlib.pyplot as plt
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
        train = np.array(df.iloc[:, 2:])
        train_out = np.array(df.iloc[:,1])
        return train, train_out
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)

def run(train, train_out, val, val_out, seed=8716, opt='minibatch'):
    n = Network(train, train_out, val, val_out, seed)
    n.addLayer(24)
    n.addLayer(15)
    acc, vmetrics = n.fit(optimizer=opt)
    return acc, n, vmetrics

def plotModel(axs, metrics, label, color):
        axs[0].plot(metrics['Acc'], color, label=label + ' Accuracy')
        axs[1].plot(metrics['Loss'], color, label=label + ' Loss')

def main():
    if len(sys.argv) != 3:
        print(GREEN + " Usage:  " + YELLOW + "python3 train.py {traindata}.csv {valdata}.csv" + RESET)
        sys.exit(0)

    train, train_out = loadData(sys.argv[1])
    val, val_out = loadData(sys.argv[2])

    single = True
    if single:
        acc, n, batchmet = run(train, train_out, val, val_out)
        print(BLUE + f"Minibatch Accuracy: {PURPLE}{acc*100:.4f}%" + RESET)
        acc, adamn, adamet = run(train, train_out, val, val_out, opt='adam')
        print(BLUE + f"Adam Accuracy: {PURPLE}{acc*100:.4f}%" + RESET)
        acc, n, rmsmet = run(train, train_out, val, val_out, opt='rmsprop')
        print(BLUE + f"RMSProp Accuracy: {PURPLE}{acc*100:.4f}%" + RESET)

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        plotModel(axs, batchmet, 'Batch', 'g')
        plotModel(axs, adamet, 'Adam', 'b')
        plotModel(axs, rmsmet, 'RMSProp', 'r')

        axs[0].axhline(y=1, color='k', linestyle=':')
        axs[0].set_title('Accuracy')
        axs[0].legend()
        axs[1].set_title('Loss')
        axs[1].legend()

        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'escape' else None)
        plt.show()
        with open('adamn.pkl', 'wb') as f:
            pickle.dump(adamn, f)
    else:
        accs = []
        k = 400
        bs, ba, bn = 0,0, None
        offset = np.random.randint(10**5)
        for i in range(offset, offset+k):
            acc, n, met = run(train, train_out, val, val_out, seed=i, opt='adam')
            print(GRAY + f"\rModel: {i-offset}/{k}", end="")
            accs.append(acc)
            if acc > ba:
                bs, ba, bn = i, acc, n

        print(BLUE + f"\rBest Accuracy: {PURPLE}{ba*100:.4f}%{BLUE} by Seed {i}" + RESET)
        print(f"{GREEN}{k}-Fold Acc: {PURPLE}{np.average(accs)*100:.4f}%{RESET}")

if __name__ == "__main__":
    main()
