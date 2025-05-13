from arch import Network
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
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

def run(layers, train, train_out, val, val_out, opt='adam'):
    n = Network(train, train_out)
    for l in layers:
        n.addLayer(l)
    metrics = n.fit(val, val_out, opt=opt)
    return n, metrics

def main():
    parser = argparse.ArgumentParser(description='Training script for your model.')
    parser.add_argument('train_csv', type=str, help='Path to training CSV file')
    parser.add_argument('val_csv', type=str, help='Path to validation CSV file')
    parser.add_argument('--layers', nargs='+', type=int, default=[24, 15],
                        help='Number of nodes in each layer, e.g., --layers 24 15')
    args = parser.parse_args()

    train, train_out = loadData(args.train_csv)
    val, val_out = loadData(args.val_csv)
    layers = args.layers

    n1, metrics1 = run(layers, train, train_out, val, val_out, opt='')
    n2, metrics2 = run(layers, train, train_out, val, val_out, opt='rmsprop')
    n3, metrics3 = run(layers, train, train_out, val, val_out, opt='adam')

    plotMetrics(metrics1, metrics2, metrics3)
    l1, l2, l3 = metrics1['Val']['Loss'][-1], metrics2['Val']['Loss'][-1], metrics3['Val']['Loss'][-1]
    if l1 < l2 and l1 < l3:
        n = n1
    else:
        n = n2 if l2 < l3 else n3
    with open('net.pkl', 'wb') as f:
        pickle.dump(n, f)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)
