import streamlit as st
import pandas as pd
import plotly.express as px
import pandas.api.types as types

# 1. Global Column Definitions and Mapping

CORE_NUMERICAL = ['transportation_expense', 'service_time',
                  'years_until_retirement', 'body_mass_index',
                  'absenteeism_time_in_hours', 'commute_cost_per_km']
CORE_COUNT = ['number_of_children', 'number_of_pets']
CORE_CATEGORICAL = ['month_of_absence', 'reason_for_absence', 'weekday_type']
CORE_FLAG = ['disciplinary_failure', 'higher_education', 'risk_behavior']
ALL_NUMERICAL_VARS = CORE_NUMERICAL + CORE_COUNT

# Map of original column names to user-friendly display names.
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
    'risk_behavior': 'Risk Behavior',
    'age': 'Age'
}
DISPLAY_TO_VAR = {col_mapped: og_col
                  for og_col, col_mapped in COLUMN_MAP.items()}


def map_flags_to_yes_no(dataframe, flag_columns):
    """Maps 1/0 integer values in binary flag columns to 'Yes'/'No' strings."""
    mapping = {1: 'Yes', 0: 'No'}
    for col in flag_columns:
        if col in dataframe.columns:
            # Safely convert to integer before mapping
            dataframe[col] = pd.to_numeric(dataframe[col],
                                           errors='coerce').fillna(0).astype(int)
            dataframe[col] = dataframe[col].map(mapping).astype('category')
    return dataframe


# 2. Data Loading

@st.cache_data
def load_data(file_path):
    """Loads the data, sets index, and cleans flag values."""
    try:
        data = pd.read_csv(file_path, index_col=0)

        # Ensure count columns are integers
        for col in CORE_COUNT:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col],
                                          errors='coerce').fillna(0).astype(int)

        # Map 1/0 to 'Yes'/'No' for flag columns
        data = map_flags_to_yes_no(data, CORE_FLAG)

        return data

    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. "
                 "Please verify the path relative to your dashboard.py file.")
        st.stop()
        return pd.DataFrame()


# Global Data Loading
FILE_PATH = 'Datasets/data_for_dashboard.csv'
data_for_analysis = load_data(FILE_PATH)

if data_for_analysis.empty:
    st.stop()

# Get the final list of columns available in the loaded data
all_cols = data_for_analysis.columns.tolist()
all_numerical_cols = [col for col in all_cols
                      if col in ALL_NUMERICAL_VARS]
DISPLAY_OPTIONS = [COLUMN_MAP[col] for col in all_cols
                   if col in COLUMN_MAP]

# -----------------------------------------------------------------------------
# 3. Sidebar for Data Filtering

st.sidebar.header("🔍 Data Filtering")


def apply_filters(dataframe):
    """Applies all sidebar filters to the DataFrame."""
    df_filtered = dataframe.copy()

    # Categorical & Flag Filters (Multiselect)
    st.sidebar.subheader("Categorical & Flag Filters")
    cat_and_flag_cols = CORE_CATEGORICAL + CORE_FLAG

    for col_name in cat_and_flag_cols:
        if col_name in dataframe.columns:
            display_name = COLUMN_MAP[col_name]
            unique_values = dataframe[col_name].unique().tolist()

            selected_values = st.sidebar.multiselect(
                display_name,
                options=unique_values,
                default=unique_values
            )
            df_filtered = df_filtered[
                df_filtered[col_name].isin(selected_values)]

    # Numerical & Count Filters (Sliders)
    st.sidebar.subheader("Numerical & Count Filters")

    # Filters to apply: Hours, BMI, Children, and Pets
    sidebar_num_filters = ['absenteeism_time_in_hours',
                           'body_mass_index', 'number_of_children',
                           'number_of_pets']

    for col_name in sidebar_num_filters:
        if col_name in dataframe.columns:
            display_name = COLUMN_MAP[col_name]
            # Ensure min/max values are based on the column's data type
            min_val, max_val = float(dataframe[col_name].min()), \
                               float(dataframe[col_name].max())

            # Determine step size and format based on variable type
            step = 1.0
            format_str = "%.1f"
            is_count = col_name in CORE_COUNT

            if is_count:
                step = 1
                format_str = "%d"
                # CRITICAL FIX: Cast min/max to int when step is int
                min_val, max_val = int(min_val), int(max_val)


            value_tup = (min_val, max_val)
            # Children and Pets filter is just a maximum value, not a range slider
            if is_count:
                value_tup = max_val

            val_range = st.sidebar.slider(
                display_name,
                min_value=min_val,
                max_value=max_val,
                value=value_tup,
                step=step,
                format=format_str
            )

            if is_count:
                # Filter: <= selected max value
                df_filtered = df_filtered[df_filtered[col_name] <= val_range]
            else:
                # Filter: between min and max range (expects tuple/list)
                df_filtered = df_filtered[
                    (df_filtered[col_name] >= val_range[0]) &
                    (df_filtered[col_name] <= val_range[1])
                ]

    return df_filtered


# Apply all filters to get the final data used for visualization
df_filtered = apply_filters(data_for_analysis)

# -----------------------------------------------------------------------------

# 4. Dashboard Title and Introduction / Metrics
st.title("📊 Exploratory Data Analysis")
st.markdown(f"**Data Loaded:** **{data_for_analysis.shape[0]}** total "
            f"observations. **{df_filtered.shape[0]}** observations meet the "
            "current filter criteria.")

st.header("Quick Filtered Data Summary")


def get_metric(dataframe, col, func):
    """Calculates and formats a key metric."""
    if dataframe.empty or col not in dataframe.columns:
        return "N/A"

    if func == 'count':
        return str(dataframe.shape[0])

    if types.is_numeric_dtype(dataframe[col]):
        if func == 'mean':
            return f"{dataframe[col].mean():.2f}"
        if func == 'std':
            return f"{dataframe[col].std():.2f}"
        if func == 'median':
            return f"{dataframe[col].median():.2f}"
    return "N/A"


# Metrics Row 1
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Obs. (Filtered)",
              value=get_metric(df_filtered, 'absenteeism_time_in_hours',
                               'count'))
with col2:
    st.metric(label="Avg. Absenteeism (Hours)",
              value=get_metric(df_filtered, 'absenteeism_time_in_hours',
                               'mean'))
with col3:
    st.metric(label="Median Absenteeism (Hours)",
              value=get_metric(df_filtered, 'absenteeism_time_in_hours',
                               'median'))
with col4:
    st.metric(label="Std. Dev. Absenteeism (Hours)",
              value=get_metric(df_filtered, 'absenteeism_time_in_hours',
                               'std'))


st.divider()

# 5. Feature Distribution Plot

if df_filtered.empty:
    st.error("No observations match the current filter criteria. "
             "Please adjust your filters.")
    st.stop()

st.header("Feature Distribution Plot")

feature_display_name = st.selectbox(
    "Select a feature to visualize its distribution on the FILTERED data:",
    options=DISPLAY_OPTIONS
)
feature_to_plot = DISPLAY_TO_VAR[feature_display_name]

if feature_to_plot in ALL_NUMERICAL_VARS:
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
    df_counts = df_filtered[feature_to_plot].value_counts().reset_index(
        name='Count')
    df_counts.columns = [feature_to_plot, 'Count']

    fig = px.bar(
        df_counts,
        x=feature_to_plot,
        y='Count',
        title=f'Count of {feature_display_name} (Filtered Data)')
    fig.update_xaxes(title=feature_display_name, tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# 6. Scatter Plot for Variable Relationships
st.header("Variable Relationship Explorer (Scatter Plot)")

numerical_display_options = [COLUMN_MAP[col] for col in all_numerical_cols
                             if col in COLUMN_MAP]

if numerical_display_options:
    default_x_display = COLUMN_MAP.get('absenteeism_time_in_hours',
                                       numerical_display_options[0])
    default_y_display = COLUMN_MAP.get('body_mass_index',
                                       numerical_display_options[0])

    col_x, col_y = st.columns(2)

    with col_x:
        scatter_x_display = st.selectbox(
            "Select **X-axis** Variable:",
            options=numerical_display_options,
            index=(numerical_display_options.index(default_x_display)
                   if default_x_display in numerical_display_options else 0)
        )

    with col_y:
        scatter_y_display = st.selectbox(
            "Select **Y-axis** Variable:",
            options=numerical_display_options,
            index=(numerical_display_options.index(default_y_display)
                   if default_y_display in numerical_display_options else 0)
        )

    scatter_x = DISPLAY_TO_VAR.get(scatter_x_display)
    scatter_y = DISPLAY_TO_VAR.get(scatter_y_display)

    if scatter_x in df_filtered.columns and scatter_y in df_filtered.columns:
        if (types.is_numeric_dtype(df_filtered[scatter_x]) and
                types.is_numeric_dtype(df_filtered[scatter_y])):
            scatter_fig = px.scatter(
                df_filtered,
                x=scatter_x,
                y=scatter_y,
                title=(f'{scatter_x_display} vs {scatter_y_display} '
                       '(Filtered Data)'),
                trendline="ols",
                opacity=0.6
            )
            scatter_fig = px.scatter(
                df_filtered,
                x=scatter_x,
                y=scatter_y,
                title=(f'{scatter_x_display} vs {scatter_y_display} '
                       '(Filtered Data)'),
                trendline="ols",
                opacity=0.6
            )

            scatter_fig.update_traces(
                selector={'mode': 'lines'},
                line={'color': 'red', 'width': 3}
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        else:
            st.warning(f"Columns '{scatter_x_display}' and/or "
                       f"'{scatter_y_display}' are not strictly numeric in the "
                       "filtered data. Cannot create scatter plot.")
    else:
        st.warning("Selected columns were not found in the loaded dataset. "
                   "Please check your data file.")

else:
    st.warning("No numerical columns found to create a scatter plot.")