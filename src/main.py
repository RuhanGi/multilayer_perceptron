import matplotlib.pyplot as plt
import seaborn as sns
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
        return pd.read_csv(fil, header=None)
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)


def cleanData(df):
    try:
        float_cols = df.select_dtypes(include=['float64'])
        df.dropna(inplace=True, subset=float_cols.columns)
        df.drop_duplicates(inplace=True)
        return df
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)

def plotData(df):
    df.iloc[:, 2:32].boxplot(figsize=(15, 12), rot=90)
    plt.title("Boxplot of Features")
    plt.yscale("log")
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'escape' else None)
    plt.show()

def plot_correlation_matrix(df):
    correlation_matrix = df.iloc[:, 2:].corr()

    mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, mask=mask)
    plt.title("Correlation Matrix of Columns 2 to End")
    plt.tight_layout()
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'escape' else None)
    plt.show()

def main():
    if len(sys.argv) != 2:
        print(RED + "Pass Data to Train!" + RESET)
        sys.exit(1)

    df = loadData(sys.argv[1])
    df = cleanData(df)
    
    # print(CYAN, df.describe())
    
    # plotData(df)
    
    plot_correlation_matrix(df)


if __name__ == "__main__":
    main()
