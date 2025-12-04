# Code with consistent simple comments added throughout
import os
import pandas as pd
import plotly.express as px
import streamlit as st
import pandas.api.types as types

# Features used for the radar chart and rounding in the centroid table.
INTERPRETABLE_FEATURES = [
    'transportation_expense', 'body_mass_index',
    'absenteeism_time_in_hours', 'age', 'commute_cost_per_km',
    'number_of_children', 'number_of_pets',
    'disciplinary_failure', 'higher_education',
    'risk_behavior',
]

# Creating a function that maps column names to clearer
# and interpretable representations.


def create_display_map(df_columns):
    # Builds readable names for raw column names
    """
    Creates a user-friendly mapping for column names.
    """

    # Direct mappings for standard variables
    mapping = {
        'transportation_expense': 'Transportation Expense',
        'body_mass_index': 'Body Mass Index (BMI)',
        'absenteeism_time_in_hours': 'Absenteeism Time (Hours)',
        'age': 'Age',
        'commute_cost_per_km': 'Commute Cost (per km)',
        'number_of_children': 'Num. Children',
        'number_of_pets': 'Num. Pets',
        'disciplinary_failure': 'Disciplinary Failure',
        'higher_education': 'Higher Education',
        'risk_behavior': 'Risk Behavior',
        'Final_Cluster': 'Cluster ID',
        'Cluster Description': 'Cluster Description',
    }

    # Creates mappings for dynamic dummy-coded columns
    for col in df_columns:
        if col not in mapping:
            if col.startswith('month_'):
                mapping[col] = col.replace('month_', '')
            elif col.startswith('is_'):
                new_col = col.replace('is_', '').replace('_', ' ')
                mapping[col] = new_col.title()
            elif col.startswith('reason_'):
                suffix = col.replace('reason_', '').replace('_', ' ')
                mapping[col] = f'Reason: {suffix.title()}'
            else:
                mapping[col] = col.replace('_', ' ').title()
    return mapping


# File paths for the datasets
CLASSIFIED_FILE = 'Datasets/classified_data.csv'
CENTROIDS_FILE = 'Datasets/final_cluster_centroids.csv'

# Function to load the data


@st.cache_data
def load_cluster_data(classified_path, centroids_path):
    # Loads cluster data from CSV files
    """
    Loads and preprocesses data, standardizing column names.
    """

    # Validate file existence
    if (
        not os.path.exists(classified_path)
        or not os.path.exists(centroids_path)
    ):
        st.error(
            f"Error: Data files not found. Expected: {classified_path}, "
            f"{centroids_path}"
        )
        st.stop()

    # Reading the data
    df_classified = pd.read_csv(classified_path, index_col=0)
    df_centroids = pd.read_csv(centroids_path, index_col=0)

    # Ensures consistent naming for Final_Cluster
    for df_data in [df_classified, df_centroids]:
        cluster_col = next(
            (col for col in df_data.columns if col.lower() == 'final_cluster'),
            None,
        )
        if cluster_col:
            df_data.rename(columns={cluster_col: 'Final_Cluster'},
                           inplace=True)
        else:
            st.error("Error: Could not find 'final_cluster' column.")
            st.stop()

    # Convert cluster to string for consistency
    df_classified['Final_Cluster'] = df_classified['Final_Cluster'].astype(str)
    df_centroids['Final_Cluster'] = df_centroids['Final_Cluster'].astype(str)

    # Standardize description column naming
    if 'Description' in df_centroids.columns:
        df_centroids.rename(
            columns={'Description': 'Cluster Description'},
            inplace=True,
        )

    return df_classified, df_centroids


# Load datasets

df_classified, df_centroids = load_cluster_data(
    CLASSIFIED_FILE,
    CENTROIDS_FILE,
)

# Create readable column maps
CENTROID_COLUMN_MAP = create_display_map(df_centroids.columns)
CENTROID_DISPLAY_TO_VAR = {v: k for k, v in CENTROID_COLUMN_MAP.items()}

# Title and intro description
st.title("⭐ Cluster Analysis")
st.markdown("This page provides insights into "
            "the segments found in your data.")

# Initializing toggle state for description visibility
if 'show_descriptions' not in st.session_state:
    st.session_state.show_descriptions = True


# Function that toggles description visibility
def toggle_descriptions():
    st.session_state.show_descriptions = (
        not st.session_state.show_descriptions
    )


# Section for cluster filter selection
st.header("Filter Clusters")
st.markdown(
    "Use the controls below to select the clusters you want to "
    "review in detail."
)

# Prepare cluster options for selection
all_cluster_ids = df_centroids['Final_Cluster'].unique().tolist()
selected_clusters = st.multiselect(
    "Filter Clusters to Display:",
    options=all_cluster_ids,
    default=all_cluster_ids,
)

# Ensure user selected clusters
if not selected_clusters:
    st.info("Please select at least one cluster to view its profile.")
    st.stop()

# Filter centroids based on selection
df_centroids_filtered = df_centroids[
    df_centroids['Final_Cluster'].isin(selected_clusters)
].copy()
df_centroids_display = df_centroids_filtered.copy()

# Rename columns using readable map
description_col_name = 'Cluster Description'
df_centroids_display.rename(columns=CENTROID_COLUMN_MAP, inplace=True)

# Reordering columns
is_desc_present = description_col_name in df_centroids_display.columns
ordered_cols = (
    ['Cluster ID', description_col_name]
    if is_desc_present
    else ['Cluster ID']
)

remaining_cols = [
    col for col in df_centroids_display.columns if col not in ordered_cols
]
df_centroids_display = df_centroids_display[ordered_cols + remaining_cols]

# Formatting numerical columns
for col in df_centroids_display.columns:
    original_col = CENTROID_DISPLAY_TO_VAR.get(col)

    is_feature = original_col in INTERPRETABLE_FEATURES
    is_reason_or_time = (
        'reason:' in col.lower()
        or 'month' in col.lower()
        or 'week' in col.lower()
    )
    is_numeric = types.is_numeric_dtype(df_centroids_display[col])

    if (is_feature or is_reason_or_time) and is_numeric:
        df_centroids_display[col] = (
            df_centroids_display[col]
            .round(2)
            .map(lambda x: f"{x:.2f}")
        )

# SECTION 1: Cluster count visualization
if not df_classified.empty:
    st.header("Cluster Count Distribution")

    df_classified_filtered = df_classified[
        df_classified['Final_Cluster'].isin(selected_clusters)
    ].copy()

    cluster_counts = (
        df_classified_filtered['Final_Cluster']
        .value_counts()
        .reset_index()
    )
    cluster_counts.columns = ['Final_Cluster', 'Count']

    fig_bar = px.bar(
        cluster_counts,
        x='Final_Cluster',
        y='Count',
        title='Number of Observations per Cluster',
        color='Final_Cluster',
        labels={'Final_Cluster': 'Cluster ID'},
    )
    fig_bar.update_layout(showlegend=False)

    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# SECTION 2: Table of centroid values
st.header("Detailed Centroids (Mean Values)")

if description_col_name in df_centroids_display.columns:
    df_centroid_values = df_centroids_display.drop(
        columns=[description_col_name]
    )
else:
    df_centroid_values = df_centroids_display.copy()

# Show centroid table
st.dataframe(df_centroid_values, hide_index=True, use_container_width=True)

st.markdown("---")

# SECTION 3: Cluster Descriptions
st.button(
    (
        'Hide'
        if st.session_state.show_descriptions
        else 'Show'
    )
    + " Descriptions",
    on_click=toggle_descriptions,
)

if st.session_state.show_descriptions:
    st.subheader("Cluster Descriptions")
    st.markdown("---")

    if description_col_name in df_centroids_display.columns:

        for i, row in df_centroids_display.iterrows():
            cluster_id = row['Cluster ID']
            raw_description = row[description_col_name]

            lines = raw_description.strip().split('\n')
            cleaned_markdown_list = []
            cluster_name = None

            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith(('-', '*', '+')):
                    item_text = stripped_line[1:].lstrip().capitalize()
                    cleaned_markdown_list.append(f"* {item_text}")
                elif stripped_line and not cluster_name:
                    cluster_name = stripped_line.capitalize()
                elif stripped_line:
                    cleaned_markdown_list.append(
                        stripped_line.capitalize()
                    )

            name_for_title = (
                cluster_name
                if cluster_name
                else f"Cluster {cluster_id}"
            )
            description_content = "\n".join(cleaned_markdown_list)

            st.markdown(f"## ⭐ {name_for_title}")
            st.markdown(description_content)

            if i < len(df_centroids_display) - 1:
                st.markdown("---")
    else:
        st.warning("The 'Cluster Description' column "
                   "was not found in the data.")

st.markdown("---")

# --- SECTION 4: Radar Chart ---
st.header("Visual Cluster Comparison (Radar Chart)")
st.markdown(
    "Compare the **normalized** profiles of selected clusters across "
    "multiple dimensions."
)

col_select, col_features = st.columns(2)

with col_select:
    radar_selected_clusters = st.multiselect(
        "Select Clusters to Compare on Radar:",
        options=all_cluster_ids,
        default=all_cluster_ids,
    )

with col_features:
    interpretable_display_names = [
        CENTROID_COLUMN_MAP[col]
        for col in INTERPRETABLE_FEATURES
        if col in CENTROID_COLUMN_MAP
    ]

    radar_selected_features_display = st.multiselect(
        "Select Features for Radar Axes:",
        options=interpretable_display_names,
        default=interpretable_display_names[:5],
    )

    radar_selected_features = [
        CENTROID_DISPLAY_TO_VAR[name]
        for name in radar_selected_features_display
    ]

if radar_selected_clusters and radar_selected_features:
    df_plot = df_centroids[
        df_centroids['Final_Cluster'].isin(radar_selected_clusters)
    ].copy()

    df_plot = df_plot[['Final_Cluster'] + radar_selected_features]

    df_normalized = df_plot.copy()

    for feature in radar_selected_features:
        min_val = df_normalized[feature].min()
        max_val = df_normalized[feature].max()

        if max_val != min_val:
            df_normalized[feature] = (
                (df_normalized[feature] - min_val)
                / (max_val - min_val)
            )
        else:
            df_normalized[feature] = 0.5

    df_melted = df_normalized.melt(
        id_vars=['Final_Cluster'],
        value_vars=radar_selected_features,
        var_name='Feature',
        value_name='Normalized_Value',
    )

    df_melted['Feature'] = df_melted['Feature'].map(
        CENTROID_COLUMN_MAP
    )

    fig_radar = px.line_polar(
        df_melted,
        r='Normalized_Value',
        theta='Feature',
        color='Final_Cluster',
        line_close=True,
        title='Normalized Cluster Profiles Comparison',
        height=550,
    )

    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info(
        "Please select clusters and features for the radar chart "
        "comparison."
    )

st.markdown("---")

# --- SECTION 5: Scatter Plot ---
if not df_classified.empty:
    st.header("Cluster Scatter Plot")

    numerical_for_plot = [
        col for col in df_classified.columns
        if col in INTERPRETABLE_FEATURES
    ]

    if len(numerical_for_plot) >= 2:
        col_x, col_y = st.columns(2)

        default_x_idx = (
            numerical_for_plot.index('absenteeism_time_in_hours')
            if 'absenteeism_time_in_hours' in numerical_for_plot
            else 0
        )
        default_y_idx = (
            numerical_for_plot.index('age')
            if 'age' in numerical_for_plot
            else 1
        )

        if default_x_idx == default_y_idx and len(numerical_for_plot) > 1:
            default_y_idx = 1
        elif (
            default_x_idx == default_y_idx
            and len(numerical_for_plot) == 1
        ):
            default_y_idx = 0

        with col_x:
            scatter_x = st.selectbox(
                "Select X-axis Variable:",
                options=numerical_for_plot,
                index=default_x_idx,
            )

        with col_y:
            scatter_y = st.selectbox(
                "Select Y-axis Variable:",
                options=numerical_for_plot,
                index=default_y_idx,
            )

        df_scatter_filtered = df_classified[
            df_classified['Final_Cluster'].isin(selected_clusters)
        ].copy()

        fig_scatter = px.scatter(
            df_scatter_filtered,
            x=scatter_x,
            y=scatter_y,
            color='Final_Cluster',
            title=(
                f"Clustering by {CENTROID_COLUMN_MAP.get(scatter_x)} "
                f"vs {CENTROID_COLUMN_MAP.get(scatter_y)}"
            ),
            labels={
                scatter_x: CENTROID_COLUMN_MAP.get(scatter_x),
                scatter_y: CENTROID_COLUMN_MAP.get(scatter_y),
                'Final_Cluster': 'Cluster ID',
            },
        )

        fig_scatter.update_layout(
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.05,
            )
        )

        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning(
            "Not enough numerical columns available in classified data "
            "for scatter plot."
        )
else:
    st.error("No classified data loaded for visualization.")
