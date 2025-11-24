import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
