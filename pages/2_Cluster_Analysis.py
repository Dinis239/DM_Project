import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pandas.api.types as types # Import types for explicit check

# --- Configuration & Utility Functions ---

# Features used for the radar chart visualization and rounding in the centroid table.
INTERPRETABLE_FEATURES = [
    'transportation_expense', 'body_mass_index', 'absenteeism_time_in_hours', 
    'age', 'commute_cost_per_km', 'service_time', 'years_until_retirement',
    'number_of_children', 'number_of_pets', 'disciplinary_failure', 
    'higher_education', 'risk_behavior'
]

def create_display_map(df_columns):
    """Creates a user-friendly mapping for column names, ensuring uniqueness for dummies."""
    mapping = {
        'transportation_expense': 'Transportation Expense', 'service_time': 'Service Time (Years)', 
        'years_until_retirement': 'Years Until Retirement', 'body_mass_index': 'Body Mass Index (BMI)', 
        'absenteeism_time_in_hours': 'Absenteeism Time (Hours)', 'age': 'Age', 
        'commute_cost_per_km': 'Commute Cost (per km)', 'number_of_children': 'Num. Children', 
        'number_of_pets': 'Num. Pets', 'disciplinary_failure': 'Disciplinary Failure', 
        'higher_education': 'Higher Education', 'risk_behavior': 'Risk Behavior', 
        'Final_Cluster': 'Cluster ID', 'Cluster Description': 'Cluster Description'
    }
    
    # Custom logic to handle unique dummy variable names
    for col in df_columns:
        if col not in mapping:
            if col.startswith('month_'):
                mapping[col] = col.replace('month_', '')
            elif col.startswith('is_'):
                mapping[col] = col.replace('is_', '').replace('_', ' ').title()
            elif col.startswith('reason_'):
                suffix = col.replace('reason_', '').replace('_', ' ').title()
                mapping[col] = f'Reason: {suffix}'
            else:
                mapping[col] = col.replace('_', ' ').title()
    return mapping

# --- Data Loading ---

CLASSIFIED_FILE = 'Datasets/classified_data.csv'
CENTROIDS_FILE = 'Datasets/final_cluster_centroids.csv'

@st.cache_data
def load_cluster_data(classified_path, centroids_path):
    """Loads and preprocesses the classified data and centroids, standardizing column names."""
    try:
        df_classified = pd.read_csv(classified_path, index_col=0)
        df_centroids = pd.read_csv(centroids_path, index_col=0)
        
        # Standardize 'Final_Cluster' name (case-insensitive check)
        for df, path in [(df_classified, classified_path), (df_centroids, centroids_path)]:
            cluster_col = next((col for col in df.columns if col.lower() == 'final_cluster'), None)
            if cluster_col:
                df.rename(columns={cluster_col: 'Final_Cluster'}, inplace=True)
            else:
                st.error(f"Error: Could not find a 'final_cluster' column in {path}.")
                st.stop()
        
        # Ensure 'Final_Cluster' is treated as string/categorical
        df_classified['Final_Cluster'] = df_classified['Final_Cluster'].astype(str)
        df_centroids['Final_Cluster'] = df_centroids['Final_Cluster'].astype(str)
        
        # Rename the Description column
        if 'Description' in df_centroids.columns:
            df_centroids.rename(columns={'Description': 'Cluster Description'}, inplace=True)
            
        return df_classified, df_centroids

    except FileNotFoundError as e:
        st.error(f"Error loading cluster files: {e}. Please ensure files are in the 'Datasets' folder.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during data processing: {e}")
        st.stop()

# Load data and create mappings
df_classified, df_centroids = load_cluster_data(CLASSIFIED_FILE, CENTROIDS_FILE)
CENTROID_COLUMN_MAP = create_display_map(df_centroids.columns)
CENTROID_DISPLAY_TO_VAR = {v: k for k, v in CENTROID_COLUMN_MAP.items()}


# --- Page Content ---
st.title("⭐ Cluster Analysis")
st.markdown("This page provides insights into the segments found in your data.")

# --- Cluster Profiles and Centroids ---
st.header("Cluster Profiles and Centroids")
st.markdown("Use the controls below to select the clusters you want to review in detail.")

# 1a. Filter Control
all_cluster_ids = df_centroids['Final_Cluster'].unique().tolist()
selected_clusters = st.multiselect(
    "Filter Clusters to Display:",
    options=all_cluster_ids,
    default=all_cluster_ids
)

# Apply Filter
if not selected_clusters:
    st.warning("Please select at least one cluster to view its profile.")
    st.markdown("---")
    st.stop()

df_centroids_filtered = df_centroids[df_centroids['Final_Cluster'].isin(selected_clusters)].copy()
df_centroids_display = df_centroids_filtered.copy()

# 1b. Prepare DataFrame for Display
description_col_name = 'Cluster Description'
df_centroids_display.rename(columns=CENTROID_COLUMN_MAP, inplace=True)

# Reorder columns
ordered_cols = ['Cluster ID', description_col_name] if description_col_name in df_centroids_display.columns else ['Cluster ID']
remaining_cols = [col for col in df_centroids_display.columns if col not in ordered_cols]
df_centroids_display = df_centroids_display[ordered_cols + remaining_cols]


# --- ENFORCE TWO DECIMAL PLACES AND STRING FORMATTING ---
for col in df_centroids_display.columns:
    original_col = CENTROID_DISPLAY_TO_VAR.get(col)
    
    if original_col in INTERPRETABLE_FEATURES or 'reason:' in col.lower() or 'month' in col.lower() or 'week' in col.lower():
        
        if types.is_numeric_dtype(df_centroids_display[col]):
            
            # Apply rounding and then enforce the string formatting (.2f)
            df_centroids_display[col] = (
                df_centroids_display[col]
                .round(2)
                .map(lambda x: f"{x:.2f}")
            )


## Cluster Descriptions using Vertical List
st.subheader("Cluster Descriptions")
st.markdown("---")

if description_col_name in df_centroids_display.columns:
    
    for i, row in df_centroids_display.iterrows():
        cluster_id = row['Cluster ID']
        raw_description = row[description_col_name]
        
        # List Parsing and Cleaning
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
                cleaned_markdown_list.append(stripped_line.capitalize())
        
        # Title and Content finalization
        name_for_title = cluster_name if cluster_name else f"Cluster {cluster_id}"
        description_content = "\n".join(cleaned_markdown_list)
        
        # Render Description
        st.markdown(f"## ⭐ {cluster_id}")
        st.markdown(description_content)

        if i < len(df_centroids_display) - 1:
            st.markdown("---")
else:
    st.warning("The 'Cluster Description' column was not found in the data.")


## Detailed Centroids Table
st.subheader("Detailed Centroids (Mean Values)")
# Drop description column before display
if description_col_name in df_centroids_display.columns:
    df_centroid_values = df_centroids_display.drop(columns=[description_col_name])
else:
    df_centroid_values = df_centroids_display.copy()

st.dataframe(df_centroid_values, hide_index=True, use_container_width=True)
st.markdown("---")

# --- Visual Cluster Comparison (Radar Chart) ---
st.header("Visual Cluster Comparison (Radar Chart)")
st.markdown("Compare the normalized profiles of selected clusters across multiple dimensions.")

# Controls for Radar Chart
col_select, col_features = st.columns(2)
with col_select:
    radar_selected_clusters = st.multiselect(
        "Select Clusters to Compare on Radar:",
        options=df_centroids['Final_Cluster'].unique().tolist(),
        default=df_centroids['Final_Cluster'].unique().tolist()
    )
with col_features:
    interpretable_display_names = [CENTROID_COLUMN_MAP[col] for col in INTERPRETABLE_FEATURES if col in CENTROID_COLUMN_MAP]
    radar_selected_features_display = st.multiselect(
        "Select Features for Radar Axes:",
        options=interpretable_display_names,
        default=interpretable_display_names[:5]
    )
    radar_selected_features = [CENTROID_DISPLAY_TO_VAR[name] for name in radar_selected_features_display]


if radar_selected_clusters and radar_selected_features:
    df_plot = df_centroids[df_centroids['Final_Cluster'].isin(radar_selected_clusters)].copy()
    df_plot = df_plot[['Final_Cluster'] + radar_selected_features]

    # Normalize data for the radar chart (Min-Max Scaling)
    df_normalized = df_plot.copy()
    for feature in radar_selected_features:
        min_val = df_normalized[feature].min()
        max_val = df_normalized[feature].max()
        
        if max_val != min_val:
            df_normalized[feature] = (df_normalized[feature] - min_val) / (max_val - min_val)
        else:
            df_normalized[feature] = 0.5
            
    # Melt the normalized DataFrame
    df_melted = df_normalized.melt(
        id_vars=['Final_Cluster'], value_vars=radar_selected_features, var_name='Feature', value_name='Normalized_Value'
    )
    df_melted['Feature'] = df_melted['Feature'].map(CENTROID_COLUMN_MAP)

    # Generate Radar Chart 
    fig_radar = px.line_polar(
        df_melted, r='Normalized_Value', theta='Feature', color='Final_Cluster', line_close=True,
        title='Normalized Cluster Profiles Comparison', height=550
    )
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("Please select clusters and features for the radar chart comparison.")

st.markdown("---")


# --- Cluster Distribution and Scatter Plot ---
if not df_classified.empty:
    
    # Cluster Distribution (Bar Chart)
    st.subheader("Cluster Count Distribution")
    df_classified_filtered = df_classified[df_classified['Final_Cluster'].isin(selected_clusters)].copy()
    cluster_counts = df_classified_filtered['Final_Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Final_Cluster', 'Count']
    fig_bar = px.bar(
        cluster_counts, x='Final_Cluster', y='Count', color='Final_Cluster', 
        title='Number of Observations per Cluster', text='Count'
    )
    st.plotly_chart(fig_bar, use_container_width=True)


    # Interactive Scatter Plot
    st.subheader("Cluster Scatter Plot")
    numerical_for_plot = [col for col in df_classified.columns if col in INTERPRETABLE_FEATURES]

    if len(numerical_for_plot) >= 2:
        col_x, col_y = st.columns(2)

        with col_x:
            scatter_x = st.selectbox(
                "Select X-axis Variable:", options=numerical_for_plot,
                index=numerical_for_plot.index('absenteeism_time_in_hours') if 'absenteeism_time_in_hours' in numerical_for_plot else 0
            )
        with col_y:
            scatter_y = st.selectbox(
                "Select Y-axis Variable:", options=numerical_for_plot,
                index=numerical_for_plot.index('age') if 'age' in numerical_for_plot else 0
            )

        # Ensure scatter plot data is also filtered by the user's selection
        df_scatter_filtered = df_classified[df_classified['Final_Cluster'].isin(selected_clusters)].copy()

        fig_scatter = px.scatter(
            df_scatter_filtered, x=scatter_x, y=scatter_y, color='Final_Cluster',
            title=f'Clustering by {CENTROID_COLUMN_MAP.get(scatter_x, scatter_x)} vs {CENTROID_COLUMN_MAP.get(scatter_y, scatter_y)}',
            hover_data=['Final_Cluster', 'number_of_children', 'body_mass_index']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Not enough numerical columns available in classified data for scatter plot.")
else:
    st.error("No classified data loaded for visualization.")