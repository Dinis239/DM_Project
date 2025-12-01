import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import pandas as pd


def visualize_dimensionality_reduction(transformation: np.array,
                                       targets: list) -> None:
    '''
    Function to plot a scatter plot of the t-SNE or UMAP output.

    Arguments:
        ----------
         - transformation(np.array): Array of dimensionality reduction output
         - targets(list): Series containing the assigned
         cluster of all observations

    Returns:
        ----------
         - None, although a plot is produced.
    '''
    # create a scatter plot of the dimensionality reduction output
    ax = sns.scatterplot(x=transformation[:, 0],
                         y=transformation[:, 1],
                         hue=targets,
                         legend='full',
                         edgecolor=None,
                         palette='tab10')
    sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


def boxplot_grid(data: pd.DataFrame, 
                 variables: list,
                 cluster_var: str,
                 color: str = None) -> None:
    """
    Plot a boxploty grid based on the data.

    Parameters:
        ----------
         - data (pd.DataFrame): The DataFrame containing the data.
         - variables (list): The column names of the variables to be plotted.
         - color (str, optional): Color for the bars. Defaults to None.
         - edgecolor (str, optional): Color for the bars edges.
         Defaults to 'black'.

    Returns:
        ----------
         None, but a plot is produced
    """
    a = math.ceil(len(variables)/3)
    b = 3
    fig, axes = plt.subplots(a, b, figsize=(25, 75))
    axes = axes.flatten()
    for i, column in enumerate(variables):
        sns.boxplot(x=data[cluster_var], y=column,
                    data=data, ax=axes[i], color=color)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        axes[i].set_title(column)
    axes_to_turn_off = (a * b) - len(variables)
    for i in range(1, axes_to_turn_off + 1):
        axes[-i].axis('off')
    plt.tight_layout()
    plt.show()
