# Data Mining 1 Project - Employee Absence Segmentation

In this project, we will tackle the task of seggregating and profiling absences of  employees of a brazilian company based on a dataset which include anonymized data about absences including absence-specific information and information about the employee committing each absence.

For clarity, this project has been split into multiple notebooks which work in a chain-like way where data is exported from one notebook to the next. For this reason, in theory, it is better to run all notebooks in order.

For more practical ease of use all of the datasets that are exported and required in future notebooks have been exported to the datasets folder allowing for the option to run one notebook without having to run all of the previous ones.

Below is the structure of the delivery:
 - Datasets Folder:
  - absenteesim_data - Raw dataset
  - data_for_modeling - Preprocessed data to input into models
  - data_for_analysis - Preprocessed data to analyze clusters and compare clustering solutions
  - data_for_dashboard - Used for the streamlit dashboard. Similar to data_for_analysis, but without dummy encoding.
  - 