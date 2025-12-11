# Data Mining 1 Project - Employee Absence Segmentation
## Group 15
**Members:**
- Beatriz Boura - 20250272
- Dinis Gaspar - 20221869
- Leonor Cardoso - 20250546
- Margarida Cruz - 20221929

In this project, we will tackle the task of seggregating and profiling absences of  employees of a brazilian company based on a dataset which include anonymized data about absences including absence-specific information and information about the employee committing each absence.

For clarity, this project has been split into multiple notebooks which work in a chain-like way where data is exported from one notebook to the next. For this reason, in theory, it is better to run all notebooks in order.

For more practical ease of use all of the datasets that are exported and required in future notebooks have been exported to the datasets folder allowing for the option to run one notebook without having to run all of the previous ones.

If you wish to run the project locally, we recommend creating a virtual environment with Python version 3.12.4, ipython and pip. Then using pip install all required packages from the requirements.txt file. With anaconda this process would be as follows. In the Anaconda prompt:
- 1 - Run the following command "create --name DM1_Group15_2025env python=3.12.4 ipython pip" - You may change the name
- 2 - Run the following command  "conda activate DM1_Group15_2025env"
- 3 - Within the anaconda prompt navigate to where the delivery is, entering the delivery folder. Copy the path of the delivery folder and put "cd" followed by the path in the prompt
- 4 - Run the following command "pip install -r requirements.txt" 

Below is the structure of the delivery:
- requirements.txt - Contains the list of required packages and versions to run the project.
- README.md - This file
- 0_Home.py - Python script contain the technical implementation of the landing page of the streamlit dashboard
- Datasets folder:
  + absenteesim_data.csv - Raw dataset
  + data_for_modeling.csv - Preprocessed dataset to input into models
  + data_for_analysis.csv - Preprocessed dataset to analyze clusters and compare clustering solutions
  + data_for_dashboard.csv - Used for the streamlit dashboard. Similar to data_for_analysis, but without dummy 
  encoding.
  + SOM_8_clusters_labels.csv - Cluster labels for SOM (needed to analyze results in line with the rest of the clustering notebook)
  + classified_data.csv - data_for_analysis but with final cluster labels
  + final_cluster_centroids.csv - Dataset with mean values for all variables per cluster
- Pre-processing folder:
  + EDA_Preprocessing.ipynb - First Notebook - Jupyter notebook containing the technical implementation of the Exploratory Data Analysis and Preprocessing stages of the project.
  + utils_preprocessing.py - Python script with auxiliary functions used in the EDA_Preprocessing notebook
- Clustering folder:
  + Clustering.ipynb - Second Notebook - Jupyter notebook containing the technical implementation of the Cluster Modeling stage of the project.
  + Cluster_analysis.ipynb - Third Notebook - Jupyter notebook containing the analysis of the final clusters
  + utils_clustering.py - Python script with auxiliary functions used in the notebooks of the folder
- Pages folder:
  + 1_Exploratory_Analysis.py - Python script contain the technical implementation of the "Exploratory Analysis" page of the streamlit dashboard
  + 2_Cluster_Analysis.py - Python script contain the technical implementation of the "Cluster Analysis" page of the streamlit dashboard