import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

def histogram_grid(data: pd.DataFrame, variables: list,
                   color: str = None, edgecolor: str = 'black') -> None:
    """
    Plot a histogram grid based on the data.

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
    a = math.ceil(math.sqrt(len(variables)))
    b = math.ceil(len(variables) / a)
    fig, axes = plt.subplots(a, b, figsize=(25, 25))
    axes = axes.flatten()
    for i, column in enumerate(variables):
        sns.histplot(x=column, data=data, ax=axes[i], color=color,
                     edgecolor=edgecolor)
        axes[i].set_title(column)
    axes_to_turn_off = (a * b) - len(variables)
    for i in range(1, axes_to_turn_off + 1):
        axes[-i].axis('off')
    plt.tight_layout()
    plt.show()
