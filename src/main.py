import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys

from arch import Network


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

def plotData(df):
    diagnosis_col = df.columns[1]
    score_cols = df.select_dtypes(include="float64").columns
    target_corr = df.iloc[:, 1:].corr()[1].drop(1)
    sorted_features = target_corr.abs().sort_values(ascending=False).index

    df_long = df.melt(id_vars=diagnosis_col, value_vars=score_cols,
        var_name="Feature", value_name="Value")

    df_long["Feature"] = pd.Categorical(df_long["Feature"], categories=sorted_features, ordered=True)

    plt.figure(figsize=(20, 12))
    sns.boxplot(
        data=df_long,
        x="Feature",
        y="Value",
        hue=diagnosis_col
        # ,log_scale=True
    )
    plt.ylim(-4, 6)
    plt.xticks(rotation=90)
    plt.title("Boxplots of Features by Diagnosis (M or B)")
    plt.tight_layout()
    plt.gcf().canvas.mpl_connect(
        'key_press_event', 
        lambda event: plt.close() if event.key == 'escape' else None
    )
    plt.show()

def plot_correlation_matrix(df):
    correlation_matrix = df.iloc[:, 1:].corr()

    mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, mask=mask)
    plt.title("Correlation Matrix of Columns 2 to End")
    plt.tight_layout()
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'escape' else None)
    plt.show()

# TODO Feature Selection Techniques:
# *Here are some ways to automatically select relevant features:
# *Univariate selection (e.g., using statistical tests like Chi-square, ANOVA, etc.)
# *Feature importance (e.g., using models like Decision Trees or Random Forest, which rank features by importance)
# *L1 regularization (Lasso) in linear models, which penalizes less important features and forces their coefficients to zero.
# *PCA (Principal Component Analysis) for dimensionality reduction (which can also help avoid collinearity).

def plotFunc(df):
    plt.figure(figsize=(12, 8))

    top = [29,24,9,22,4]
    # top = [29,24,9,22,4,25,2,5,8,28,7,27,12,14,15]
    plt.scatter(df[top].sum(axis=1) / len(top), df[1])
    for i, col in enumerate(top):
        plt.scatter(df[col], df[1] + 0.05*(i+1))

    # plt.xlim(-3, 5)
    plt.ylim(-0.5, 2)
    plt.tight_layout()
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'escape' else None)
    plt.show()

def main():
    if len(sys.argv) != 2:
        print(RED + "Pass Data to Train!" + RESET)
        sys.exit(1)

    df = loadData(sys.argv[1])

    features = df.columns[2:]
    df[features] = (df[features] - df[features].mean()) / df[features].std()
    df[1] = df[1].map({'M': 1, 'B': 0})
    # plotData(df)
    # plot_correlation_matrix(df)
    plotFunc(df)


if __name__ == "__main__":
    main()
    
