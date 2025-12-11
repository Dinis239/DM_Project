import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def visualize_dimensionality_reduction(transformation: np.array,
                                       targets: list) -> None:
    '''
    Function to plot a scatter plot of the output of a 2D dimnsionality
    reduction method, colored by a specified list of targets.

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
    Plot a grid of boxplots, split by the cluster variable,
    based on the data.

    Parameters:
        ----------
         - data (pd.DataFrame): The DataFrame containing the data.
         - variables (list): The column names of the variables to be plotted.
         - cluster_var (str): The name of the variable containing the cluster
         assigments for which each boxplot will be made.
         - color (str, optional): Color for the bars. Defaults to None.

    Returns:
        ----------
         None, but a plot is produced
    """
    a = math.ceil(len(variables)/3)
    b = 3
    _, axes = plt.subplots(a, b, figsize=(25, 25))
    axes = axes.flatten()
    for i, column in enumerate(variables):
        sns.boxplot(x=cluster_var, y=column,
                    data=data, ax=axes[i], color=color,
                    palette='rainbow', hue=cluster_var)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        axes[i].set_title(column)
    axes_to_turn_off = (a * b) - len(variables)
    for i in range(1, axes_to_turn_off + 1):
        axes[-i].axis('off')
    plt.tight_layout()
    plt.show()


def plot_comparing_avr_clusters(clusters_centroids_data: pd.DataFrame,
                                colum_to_keep: str) -> None:
    '''
    Create a plot with the mean distribution per cluster group
    and the overall mean

    Arguments:
        ----------
         - clusters_centroids_data(pd.DataFrame): dataframe
         grouped by cluster.
         - colum_to_keep(str): name of the column containing
         the label of each cluster

    Returns:
        ----------
         None, but a scatterplot is produced.
    '''
    mean_row = pd.DataFrame([['Overal Mean'] +
                            list(clusters_centroids_data.iloc[:, 1:].mean())],
                            columns=clusters_centroids_data.columns)

    cluster_centroids_analysis = pd.concat([clusters_centroids_data,
                                            mean_row])

    # Normalize the values
    cluster_centroids_analysis.iloc[:, 1:] = list(MinMaxScaler().fit_transform(
        cluster_centroids_analysis.iloc[:, 1:]))

    # Transform to long format
    melt_data = pd.melt(cluster_centroids_analysis,
                        id_vars=colum_to_keep,
                        var_name='variable',
                        value_name='value')

    sns.set_style("whitegrid")
    sns.scatterplot(melt_data, x='value', y='variable',
                    hue=colum_to_keep, s=100)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
