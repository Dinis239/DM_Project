import os
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Features used for the radar chart and rounding in the centroid table.
INTERPRETABLE_FEATURES = ['transportation_expense', 'body_mass_index',
                          'absenteeism_time_in_hours', 'age',
                          'commute_cost_per_km', 'number_of_children',
                          'number_of_pets', 'disciplinary_failure',
                          'higher_education', 'risk_behavior']

# Creating a function that maps column names to clearer
# and interpretable representations.


def create_display_map(df_columns):
    # Builds readable names for raw column names
    """
    Creates a user-friendly mapping for column display names.
    """

    # Direct mappings for standard variables
    mapping = {'transportation_expense': 'Transportation Expense',
               'body_mass_index': 'Body Mass Index (BMI)',
               'absenteeism_time_in_hours': 'Absenteeism Time (Hours)',
               'age': 'Age',
               'commute_cost_per_km': 'Commute Cost (per km)',
               'number_of_children': 'Number of Children',
               'number_of_pets': 'Number of Pets',
               'disciplinary_failure': 'Disciplinary Failure',
               'higher_education': 'Higher Education',
               'risk_behavior': 'Risk Behavior',
               'Final_Cluster': 'Cluster ID',
               'Cluster Description': 'Cluster Description'}

    # Creates mappings for dynamic dummy-coded columns
    # Replace underscores with spaces, capitalizes where
    # needed
    for col_name in df_columns:
        if col_name not in mapping:
            if col_name.startswith('month_'):
                mapping[col_name] = col_name.replace('month_', '')
            elif col_name.startswith('is_'):
                new_col = col_name.replace('is_', '').replace('_', ' ')
                mapping[col_name] = new_col.title()
            elif col_name.startswith('reason_'):
                suffix = col_name.replace('reason_', '').replace('_', ' ')
                mapping[col_name] = f'Reason: {suffix.title()}'
            else:
                mapping[col_name] = col_name.replace('_', ' ').title()
    return mapping


# File paths for the datasets
CLASSIFIED_FILE = 'Datasets/classified_data.csv'
CENTROIDS_FILE = 'Datasets/final_cluster_centroids.csv'

# Function to load the data


def load_cluster_data(classified_path, centroids_path):
    # Loads cluster data from CSV files
    """
    Loads and preprocesses data, standardizing column names.
    """

    # Validate file existence
    # Not theroetically needed, but good to have
    # just in case
    if (
        not os.path.exists(classified_path)
        or not os.path.exists(centroids_path)
    ):
        st.error(f"Error: Data files not found. Expected: {classified_path},"
                 f" {centroids_path}")
        st.stop()

    # Reading the data
    classified_df = pd.read_csv(classified_path, index_col=0)
    centroids_df = pd.read_csv(centroids_path, index_col=0)

    # Standardize description column naming
    centroids_df.rename(columns={'Description': 'Cluster Description'},
                        inplace=True)

    return classified_df, centroids_df


# Load datasets

df_classified, df_centroids = load_cluster_data(CLASSIFIED_FILE,
                                                CENTROIDS_FILE)

# Adding a fallback solution in case, for some unexpected reason,
# the loaded files turn out to be empty
if df_classified.empty or df_centroids.empty:
    st.markdown("An error has occurred loading the data")
    st.stop()

# Create readable column maps (getting the display column name map,
# and reverse map)
CENTROID_COLUMN_MAP = create_display_map(df_centroids.columns)
CENTROID_DISPLAY_TO_VAR = {v: k for k, v in CENTROID_COLUMN_MAP.items()}

# Title and intro description
st.title("⭐ Cluster Analysis")
st.markdown("This page provides insights into "
            "the segments found in your data.")

# Initializing toggle state for description visibility
if 'show_descriptions' not in st.session_state:
    st.session_state.show_descriptions = False


# Function that toggles description visibility
def toggle_descriptions():
    """Toggles the cluster descriptions visibility to
    whatever the opposite of the current session state is."""
    st.session_state.show_descriptions = (
        not st.session_state.show_descriptions)


# Initializing toggle state for centroid table visibility
if 'show_centroids' not in st.session_state:
    st.session_state.show_centroids = False


# Function that toggles centroid table visibility
def toggle_centroids():
    """Toggles the cluster centroids table visibility
    to whatever the opposite of the current session state is."""
    st.session_state.show_centroids = (
        not st.session_state.show_centroids)


# Initializing toggle state for scatter plot visibility
if 'show_scatter' not in st.session_state:
    st.session_state.show_scatter = False


# Function that toggles scatter plot visibility
def toggle_scatter():
    """Toggles the visibility of the scatter plot sections to
    whatever the opposite of the current session state is."""
    st.session_state.show_scatter = (
        not st.session_state.show_scatter)


# Section for cluster filter selection
st.header("Filter Clusters")
st.markdown("Use the controls below to select the clusters "
            "you want to review in detail.")

# Prepare cluster options for selection
all_cluster_ids = df_centroids['Final_Cluster'].unique()

# Creating the selection box for cluster filtering
# with all clusters by default
selected_clusters = st.multiselect("Filter Clusters to Display:",
                                   options=all_cluster_ids,
                                   default=all_cluster_ids)

# Ensure user selected clusters
# If not stop until clusters are selected
if not selected_clusters:
    st.info("Please select at least one cluster to view its profile.")
    st.stop()

# Filter centroids based on selection
df_centroids_filtered = df_centroids[df_centroids['Final_Cluster']
                                     .isin(selected_clusters)].copy()

# SECTION 1: Cluster count visualization
st.header("Cluster Count Distribution")

# Getting a dataframe with the counts for each cluster
df_classified_filtered = df_classified[df_classified['Final_Cluster']
                                       .isin(selected_clusters)].copy()

cluster_counts = (df_classified_filtered['Final_Cluster'].value_counts()
                  .reset_index())
cluster_counts.columns = ['Final_Cluster', 'Count']

# Plotting the counts of each cluster in a bar chart
fig_bar = px.bar(cluster_counts,
                 x='Final_Cluster',
                 y='Count',
                 title='Number of Observations per Cluster',
                 color='Final_Cluster',
                 labels={'Final_Cluster': 'Cluster ID'})

# Disable the legend to avoid double information
fig_bar.update_layout(showlegend=False)

st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# SECTION 2: Table of centroid values

# Getting a copy of the centroids to display on a table
df_centroids_display = df_centroids_filtered.copy()

# Rename columns of the display centroids
# dataframe using readable map
df_centroids_display.rename(columns=CENTROID_COLUMN_MAP, inplace=True)

# Reordering columns so that the ID and description come first.
feature_cols = [col for col in df_centroids_display.columns
                if col not in ['Cluster ID', 'Cluster Description']]
df_centroids_display = df_centroids_display[
    ['Cluster ID', 'Cluster Description'] + feature_cols]

# Formatting all columns to appear with 2 decimal places.
df_centroids_display.iloc[:, 2:] = df_centroids_display.iloc[:, 2:].round(2)\
    .map(lambda x: f"{x:.2f}")

# Creating a button to toggle the cluster centroid table on or off
st.button(('Hide' if st.session_state.show_centroids else 'Show')
          + " Centroids",
          on_click=toggle_centroids)

if st.session_state.show_centroids:
    st.header("Detailed Centroids (Mean Values)")

    # Drop the description column, as it doesn't make
    # sense to display it in a table
    df_centroid_values = df_centroids_display.drop(
        columns=['Cluster Description'])

    # Show centroid table
    st.dataframe(df_centroid_values, hide_index=True, use_container_width=True)

st.markdown("---")

# SECTION 3: Cluster Descriptions and Suggestions

# Creating a button to toggle the cluster Description and Suggestions on or off
st.button(('Hide' if st.session_state.show_descriptions else 'Show')
          + " Descriptions",
          on_click=toggle_descriptions)

if st.session_state.show_descriptions:
    st.subheader("Cluster Descriptions and Recommendations")
    st.markdown("---")

    # Displaying the cluster and its associated description
    # and suggestions
    for i, row in df_centroids_display.iterrows():
        st.markdown(f"## ⭐ {row['Cluster ID']}")
        st.markdown(row['Cluster Description'])
        st.markdown("---")

# Ensuring that there is always a section separator even
# if we are not displaying the table
else:
    st.markdown("---")

# SECTION 4: Scatter Plot

# Creating a button to toggle the scatter plot on or off
st.button(('Hide' if st.session_state.show_scatter else 'Show')
          + " Scatter Plot",
          on_click=toggle_scatter)

if st.session_state.show_scatter:
    st.header("Cluster Scatter Plot")

    # Getting the list of available columns for the plot
    numerical_for_plot = [CENTROID_COLUMN_MAP[col]
                          for col in df_classified_filtered.columns
                          if col in INTERPRETABLE_FEATURES]

    # Creating the side-by-side layout
    # for the 2 drop-down menus
    col_x, col_y = st.columns(2)

    with col_x:
        # Creating the drop-down menu for the
        # variable on the x axis
        scatter_x_display = st.selectbox("Select **X-axis** Variable:",
                                         options=numerical_for_plot,
                                         # Sets the default option to BMI
                                         index=4)

    with col_y:
        # Creating the drop-down menu for the
        # variable on the y axis
        scatter_y_display = st.selectbox("Select **Y-axis** Variable:",
                                         options=numerical_for_plot,
                                         # Sets the default option to
                                         # be Absenteeism time
                                         index=5)

    # Obtain the "actual" variable name from the display name
    # for the 2 variables to plot
    scatter_x = CENTROID_DISPLAY_TO_VAR.get(scatter_x_display)
    scatter_y = CENTROID_DISPLAY_TO_VAR.get(scatter_y_display)

    # Plotly scatter plot with a title and axis labels
    # and points colored based on cluster
    if scatter_x != scatter_y:
        fig_scatter = px.scatter(df_classified_filtered,
                                 x=scatter_x,
                                 y=scatter_y,
                                 color='Final_Cluster',
                                 title=(f"Clustering by {scatter_x_display} "
                                        f"vs {scatter_y_display}"),
                                 labels={scatter_x: scatter_x_display,
                                         scatter_y: scatter_y_display,
                                         'Final_Cluster': 'Cluster'})

        st.plotly_chart(fig_scatter, use_container_width=True)
    # Just in case the user selects the same variable twice
    else:
        st.info("Please select 2 different variables.")


st.markdown("---")


# SECTION 5: Radar Chart
st.header("Visual Cluster Comparison (Radar Chart)")
st.markdown("Compare the **normalized** profiles of selected clusters "
            "across multiple dimensions. We recommend having at least 5"
            "features selected.")


# Creating the side-by-side layout
# for the 2 drop-down menus
col_select, col_features = st.columns(2)

with col_select:
    # Populating the left column menu with the
    # drop-down menu for cluster selection. By
    # default including all clusters.
    radar_selected_clusters = st.multiselect(
        "Select Clusters to Compare on Radar:",
        options=all_cluster_ids,
        default=all_cluster_ids)

with col_features:
    # Populating the right column menu with the
    # drop-down menu for column selection, with the display names,
    # with only the "interpretable" features we defined
    # at the start. By default selecting: 'transportation_expense',
    # 'body_mass_index','absenteeism_time_in_hours', 'age' and
    # 'commute_cost_per_km'.
    interpretable_display_names = [CENTROID_COLUMN_MAP[col]
                                   for col in INTERPRETABLE_FEATURES
                                   if col in CENTROID_COLUMN_MAP]

    radar_selected_features_display = st.multiselect(
        "Select Features for Radar Axes:",
        options=interpretable_display_names,
        default=interpretable_display_names[:5])

    # Mapping the selected features from display column name back
    # to the original dataset column name.
    radar_selected_features = [CENTROID_DISPLAY_TO_VAR[name]
                               for name in radar_selected_features_display]

if radar_selected_clusters and radar_selected_features:
    df_plot = df_centroids[df_centroids['Final_Cluster']
                           .isin(radar_selected_clusters)].copy()

    # Creating a dataframe with the cluster and the selected variables
    df_plot = df_plot[['Final_Cluster'] + radar_selected_features]

    # Creating a copy to to normalize the data, which is required for
    # the plotting of a radar plot. For a radar plot, the data will
    # be normalized using MinMax scaler, as we want a 0 to 1 scaling, from
    # the minimum to the maximum value. Obviously we can't scale the cluster
    # (and we don't want to), but we do join it back to the normalzed data.
    scaler = MinMaxScaler()
    df_normalized = df_plot[['Final_Cluster']].join(
        pd.DataFrame(scaler.fit_transform(df_plot[radar_selected_features]),
                     columns=radar_selected_features,
                     index=df_plot.index))

    # Melting to get a row per feature per cluster, so we can get the
    # radar plot.
    df_melted = df_normalized.melt(id_vars=['Final_Cluster'],
                                   value_vars=radar_selected_features,
                                   var_name='Feature',
                                   value_name='Normalized_Value')

    # Mapping to the display names
    df_melted['Feature'] = df_melted['Feature'].map(CENTROID_COLUMN_MAP)

    # Plotting the radar plot.
    # A line_polar plot represents each line of a dataframe
    # as a vertex with polar coordinates where features are axes.
    # When connected and colored by cluster it produces the full radar plot.
    fig_radar = px.line_polar(df_melted,
                              r='Normalized_Value',
                              theta='Feature',
                              color='Final_Cluster',
                              line_close=True,
                              title='Normalized Cluster Profiles Comparison',
                              height=550)

    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar, use_container_width=True)

# In case no clusters or no columns are selected.
else:
    st.info("Please select clusters and features for the radar chart.")
