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

def plotMetrics(*metrics):
    metric_names = ['Loss', 'Acc', 'F1', 'Prec']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()

    for i, name in enumerate(metric_names):
        for idx, metric in enumerate(metrics):
            color = colors[idx % len(colors)]
            ax[i].plot(
                range(len(metric['Val'][name])),
                metric['Val'][name], color=color,
                label=metric['Opt']+' Val'
            )
            ax[i].plot(
                range(len(metric['Train'][name])),
                metric['Train'][name],
                color=color,
                linestyle='dotted',
                label=metric['Opt']+' Train'
            )
        ax[i].set_title(name)
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel(name)
        ax[i].legend()
        ax[i].grid(True)
    ax[0].set_ylim(0, 0.3)

    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'escape' else None)
    plt.show()

def run(train, train_out, val, val_out, opt='adam'):
    n = Network(train, train_out)
    n.addLayer(24)
    n.addLayer(15)
    metrics = n.fit(val, val_out, opt=opt)
    return n, metrics

def main():
    if len(sys.argv) != 3:
        print(GREEN + " Usage:  " + YELLOW + "python3 train.py {traindata}.csv {valdata}.csv" + RESET)
        sys.exit(0)

    parser = argparse.ArgumentParser(description='Training script for your model.')

    # parser.add_argument('--layer', nargs='+', type=int, required=True,
    #                     help='Number of units in each layer, e.g., 24 24 24')
    # parser.add_argument('--epochs', type=int, required=True,
    #                     help='Number of training epochs')
    # parser.add_argument('--loss', type=str, required=True,
    #                     help='Loss function to use, e.g., categoricalCrossentropy')
    # parser.add_argument('--batch_size', type=int, required=True,
    #                     help='Batch size for training')
    # parser.add_argument('--learning_rate', type=float, required=True,
    #                     help='Learning rate for optimizer')
    # args = parser.parse_args()

    np.set_printoptions(suppress=True)

    train, train_out = loadData(sys.argv[1])
    val, val_out = loadData(sys.argv[2])

    n, metrics = run(train, train_out, val, val_out, opt='')
    n2, metrics2 = run(train, train_out, val, val_out, opt='rmsprop')
    n3, metrics3 = run(train, train_out, val, val_out, opt='adam')

    plotMetrics(metrics, metrics2, metrics3)

    with open('net.pkl', 'wb') as f:
        pickle.dump(n, f)

# TODO
# * Check BINARYYYY Cross Entropy is below 0.08
# * Optimizers: Nesterov momentum, RMSprop, Adam

# * 2. Pass personalization of networks as parameters
# * 5. Evaluate the learning phase with multiple metrics.

if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except Exception as e:
    #     print(RED + "Error: " + str(e) + RESET)
    #     sys.exit(1)
