import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. Global Column Definitions and Mapping ---

# Define the core variable groups based on the final structure
CORE_NUMERICAL = ['transportation_expense', 'service_time', 'years_until_retirement', 'body_mass_index', 'absenteeism_time_in_hours', 'commute_cost_per_km']
CORE_COUNT = ['number_of_children', 'number_of_pets']
CORE_CATEGORICAL = ['month_of_absence', 'reason_for_absence', 'weekday_type']
CORE_FLAG = ['disciplinary_failure', 'higher_education', 'risk_behavior']
ALL_CLEAN_COLS = CORE_CATEGORICAL + CORE_FLAG

# 🚨 USER-FRIENDLY NAME MAPPING 🚨
COLUMN_MAP = {
    'transportation_expense': 'Transportation Expense',
    'service_time': 'Service Time (Years)',
    'years_until_retirement': 'Years Until Retirement',
    'body_mass_index': 'Body Mass Index (BMI)',
    'absenteeism_time_in_hours': 'Absenteeism Time (Hours)',
    'commute_cost_per_km': 'Commute Cost (per km)',
    'number_of_children': 'Number of Children',
    'number_of_pets': 'Number of Pets',
    'month_of_absence': 'Month of Absence',
    'reason_for_absence': 'Reason for Absence',
    'weekday_type': 'Day Type',
    'disciplinary_failure': 'Disciplinary Failure',
    'higher_education': 'Higher Education',
    'risk_behavior': 'Risk Behavior'
}

# Create reverse map for easy lookup: Display Name -> Variable Name
DISPLAY_TO_VAR = {v: k for k, v in COLUMN_MAP.items()}


def clean_categorical_values(df, columns_to_clean):
    """Standardizes categorical string values in the DataFrame."""
    for col in columns_to_clean:
        if col in df.columns and pd.api.types.is_string_dtype(df[col].dtype):
            df[col] = (df[col]
                        .str.strip()
                        .str.replace('_', ' ', regex=False)
                        .str.title()
                      )
    return df

def map_flags_to_yes_no(df, flag_columns):
    """Maps 1/0 integer values in binary flag columns to 'Yes'/'No' strings."""
    mapping = {1: 'Yes', 0: 'No'}
    for col in flag_columns:
        if col in df.columns:
            df[col] = df[col].map(mapping).astype('category')
    return df

# --- 2. Data Loading ---

@st.cache_data
def load_data(file_path):
    """Loads the actual data from the specified file path, sets index, and cleans values."""
    try:
        df = pd.read_csv(file_path, index_col=0) 
        
        # 1. Ensure flag and count columns are treated as integers (temporarily for safe mapping)
        for col_group in [CORE_FLAG, CORE_COUNT]:
            for col in col_group:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # 2. Map 1/0 to 'Yes'/'No' for flag columns
        df = map_flags_to_yes_no(df, CORE_FLAG)
        
        # 3. Apply value cleaning for string-based categorical columns
        df = clean_categorical_values(df, CORE_CATEGORICAL) 
        
        # 4. Convert categorical columns to the 'category' dtype
        for col in CORE_CATEGORICAL:
             if col in df.columns:
                 df[col] = df[col].astype('category')
                 
        return df

    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please verify the path relative to your dashboard.py file.")
        return pd.DataFrame() 

# --- Global Data Loading ---
FILE_PATH = 'Datasets/data_for_dashboard.csv'
data_for_analysis = load_data(FILE_PATH)

# If the file couldn't be loaded, display an empty page
if data_for_analysis.empty:
    st.stop()
    
# Get the final list of columns available in the loaded data
all_columns = data_for_analysis.columns.tolist()
all_numerical_cols = [col for col in all_columns if col in CORE_NUMERICAL + CORE_COUNT]

# Final display options list
DISPLAY_OPTIONS = [COLUMN_MAP[col] for col in all_columns if col in COLUMN_MAP]

# --------------------------------------------------------------------------------------
## 📊 Dashboard Main Content

# --- 3. Sidebar for Data Filtering ---

st.sidebar.header("🔍 Data Filtering")

def apply_filters(df):
    """Applies all sidebar filters to the DataFrame."""
    df_filtered = df.copy()

    # --- Categorical/Flag Filters (Multiselect) ---
    st.sidebar.subheader("Categorical & Flag Filters")
    
    cat_and_flag_cols = CORE_CATEGORICAL + CORE_FLAG
    
    for col_name in cat_and_flag_cols:
        if col_name in df.columns:
            display_name = COLUMN_MAP[col_name]
            unique_values = df[col_name].unique().tolist()
            
            selected_values = st.sidebar.multiselect(
                display_name,
                options=unique_values,
                default=unique_values
            )
            df_filtered = df_filtered[df_filtered[col_name].isin(selected_values)]

    # --- Numerical/Count Filters (Sliders) ---
    st.sidebar.subheader("Numerical & Count Filters")
    
    # Filter for Absenteeism Time (Hours)
    if 'absenteeism_time_in_hours' in df.columns:
        col_name = 'absenteeism_time_in_hours'
        min_abs, max_abs = float(df[col_name].min()), float(df[col_name].max())
        abs_range = st.sidebar.slider(
            COLUMN_MAP[col_name],
            min_value=min_abs,
            max_value=max_abs,
            value=(min_abs, max_abs),
            step=1.0,
            format="%.1f"
        )
        df_filtered = df_filtered[
            (df_filtered[col_name] >= abs_range[0]) & 
            (df_filtered[col_name] <= abs_range[1])
        ]

    # Filter for Body Mass Index (BMI)
    if 'body_mass_index' in df.columns:
        col_name = 'body_mass_index'
        min_bmi, max_bmi = float(df[col_name].min()), float(df[col_name].max())
        bmi_range = st.sidebar.slider(
            COLUMN_MAP[col_name],
            min_value=min_bmi,
            max_value=max_bmi,
            value=(min_bmi, max_bmi),
            step=0.1,
            format="%.1f"
        )
        df_filtered = df_filtered[
            (df_filtered[col_name] >= bmi_range[0]) & 
            (df_filtered[col_name] <= bmi_range[1])
        ]
    
    # Filter for Number of Children (Count variable)
    if 'number_of_children' in df.columns:
        col_name = 'number_of_children'
        max_children = int(df[col_name].max())
        children_val = st.sidebar.slider(
            COLUMN_MAP[col_name],
            min_value=0,
            max_value=max_children,
            value=max_children
        )
        df_filtered = df_filtered[df_filtered[col_name] <= children_val]
    
    return df_filtered

# Apply all filters to get the final data used for visualization
df_filtered = apply_filters(data_for_analysis)

# --------------------------------------------------------------------------------------

# --- 4. Dashboard Title and Introduction / Metrics ---
st.title("📊 Exploratory Data Analysis")
st.markdown(f"**Data Loaded:** **{data_for_analysis.shape[0]}** total observations. Filters applied below.")

st.header("Quick Filtered Data Summary")

def get_metric(df, col, func):
    if df.empty or col not in df.columns:
        return "N/A"
    if func == 'mean':
        return f"{df[col].mean():.2f}"
    if func == 'std':
        return f"{df[col].std():.2f}"
    if func == 'median':
        return f"{df[col].median():.2f}"
    return str(df.shape[0])


# Metrics Row 1
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Observations (Filtered)", value=get_metric(df_filtered, 'absenteeism_time_in_hours', 'count'))
with col2:
    st.metric(label="Avg. Absenteeism (Hours)", value=get_metric(df_filtered, 'absenteeism_time_in_hours', 'mean'))
with col3:
    st.metric(label="Median Absenteeism (Hours)", value=get_metric(df_filtered, 'absenteeism_time_in_hours', 'median'))
with col4:
    st.metric(label="Std. Dev. Absenteeism (Hours)", value=get_metric(df_filtered, 'absenteeism_time_in_hours', 'std'))


st.divider()

# --- 5. Feature Distribution Plot ---

if df_filtered.empty:
    st.error("No observations match the current filter criteria. Please adjust your filters.")
    st.stop()

st.header("Feature Distribution Plot")

# Use DISPLAY NAMES for the selectbox options
feature_display_name = st.selectbox(
    "Select a feature to visualize its distribution on the FILTERED data:",
    options=DISPLAY_OPTIONS
)

# Convert selected display name back to the actual variable name for plotting
feature_to_plot = DISPLAY_TO_VAR[feature_display_name] 

if feature_to_plot in all_numerical_cols:
    # Histogram for numerical/count data
    fig = px.histogram(
        df_filtered, 
        x=feature_to_plot, 
        title=f'Histogram of {feature_display_name} (Filtered Data)',
        marginal="box",
        color_discrete_sequence=['#FF7F0E']
    )
    fig.update_layout(xaxis_title=feature_display_name, yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    
elif feature_to_plot in CORE_CATEGORICAL or feature_to_plot in CORE_FLAG:
    # Bar chart for categorical and binary data
    fig = px.bar(
        df_filtered[feature_to_plot].value_counts().reset_index(name='Count'),
        x=feature_to_plot,
        y='Count',
        title=f'Count of {feature_display_name} (Filtered Data)',
        color=feature_to_plot,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_xaxes(title=feature_display_name, tickangle=45) 
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- 6. Scatter Plot for Variable Relationships ---
st.header("Variable Relationship Explorer (Scatter Plot)")

numerical_display_options = [COLUMN_MAP[col] for col in all_numerical_cols if col in COLUMN_MAP]

if numerical_display_options:
    default_x_display = COLUMN_MAP.get('absenteeism_time_in_hours', numerical_display_options[0])
    default_y_display = COLUMN_MAP.get('body_mass_index', numerical_display_options[0])

    col_x, col_y = st.columns(2)

    with col_x:
        scatter_x_display = st.selectbox(
            "Select **X-axis** Variable:",
            options=numerical_display_options,
            index=numerical_display_options.index(default_x_display) if default_x_display in numerical_display_options else 0
        )

    with col_y:
        scatter_y_display = st.selectbox(
            "Select **Y-axis** Variable:",
            options=numerical_display_options,
            index=numerical_display_options.index(default_y_display) if default_y_display in numerical_display_options else 0
        )

    # Convert selected display names back to actual variable names
    scatter_x = DISPLAY_TO_VAR.get(scatter_x_display)
    scatter_y = DISPLAY_TO_VAR.get(scatter_y_display)

    if scatter_x in df_filtered.columns and scatter_y in df_filtered.columns:
        if pd.api.types.is_numeric_dtype(df_filtered[scatter_x]) and pd.api.types.is_numeric_dtype(df_filtered[scatter_y]):
            scatter_fig = px.scatter(
                df_filtered,
                x=scatter_x,
                y=scatter_y,
                title=f'{scatter_x_display} vs {scatter_y_display} (Filtered Data)',
                trendline="ols",
                opacity=0.6
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        else:
            st.warning(f"Columns '{scatter_x_display}' and/or '{scatter_y_display}' are not strictly numeric in the filtered data. Cannot create scatter plot.")
    else:
        st.warning("Selected columns were not found in the loaded dataset. Please check your data file.")

else:
    st.warning("No numerical columns found to create a scatter plot.")