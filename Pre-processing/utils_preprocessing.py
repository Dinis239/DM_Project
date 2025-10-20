import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math


def histogram_grid(data: pd.DataFrame,
                   variables: list,
                   color: str = None,
                   edgecolor: str = 'black') -> None:
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
    _, axes = plt.subplots(a, b, figsize=(25, 25))
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


def outlier_filter_IQR(data: pd.DataFrame, variables: list,
                       outlier_type: str = 'normal',
                       return_dataframe: bool = False) -> None:
    """
    Evaluate the outliers of a dataset

    Prints the percentage of the dataset that is retained
    after excluding outliers and can optionally return a
    filtered DataFrame without outliers.

    Parameters:
        ----------
         - data (pd.DataFrame): The DataFrame containing the data.
         - variables (list): The column names of the variables to be evaluated.
         - type (string): The type of outliers to evaluate.
         Defaults to 'normal'.
         - return_dataframe(boolean): Whether to return the filtered dataset
         without outliers.

    Returns:
        ----------
        None, optionally a filtered Dataframe
    """
    dic_type = {'normal': 1.5, 'extreme': 3}
    data_no_out = data.copy()
    for variable in variables:
        P_25 = np.nanpercentile(data_no_out[variable], 25)
        P_75 = np.nanpercentile(data_no_out[variable], 75)
        IQR = P_75 - P_25
        data_no_out = data_no_out[(data_no_out[variable] <= P_75 +
                                   dic_type[outlier_type] * IQR)
                                  & (data_no_out[variable] >= P_25 -
                                     dic_type[outlier_type] * IQR)]
    print(f"Excluding all {outlier_type} outliers, "
          f"we are left with {round((len(data_no_out)/len(data))*100, 2)}% "
          "of our dataset")
    if return_dataframe:
        return data_no_out


def outlier_count_IQR(data: pd.DataFrame, variables: list,
                      outlier_type: str = 'normal') -> pd.DataFrame:
    """
    Evaluate the outliers of a dataset

    Returns a dataframe including the variable names and the
    correspoding number of outliers, according to the outlier_type
    parameter.

    Parameters:
        ----------
         - data (pd.DataFrame): The DataFrame containing the data.
         - variables (list): The column names of the variables to be evaluated.
         - type (string): The type of outliers to evaluate.
         Defaults to 'normal'.
    Returns:
        ----------
         - outlier_count_df (pd.DataFrame): Dataframe containing outlier counts
         by variable.
    """
    outlier_count_df = pd.DataFrame(columns=['Variable', 'N Outliers'])
    dic_type = {'normal': 1.5, 'extreme': 3}
    data_no_out_var = data.copy()
    for variable in variables:
        P_25 = np.nanpercentile(data[variable], 25)
        P_75 = np.nanpercentile(data[variable], 75)
        IQR = P_75 - P_25
        data_no_out_var = data[(data[variable] >= P_75 +
                                dic_type[outlier_type] * IQR) |
                               (data[variable] <= P_25 -
                                dic_type[outlier_type] * IQR)]
        outlier_count_df = pd.concat([outlier_count_df,
                                      pd.DataFrame([[variable,
                                                     len(data_no_out_var)]],
                                                   columns=['Variable',
                                                            'N Outliers'])])
    return outlier_count_df.set_index('Variable')
