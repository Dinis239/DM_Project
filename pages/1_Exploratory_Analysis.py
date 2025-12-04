import streamlit as st
import pandas as pd
import plotly.express as px
import pandas.api.types as types
import os

# 1. Global Column Definitions and Mapping
# Used for filtering and plotting appropriate
# visualization types

# CORE_NUMERICAL: Numeric features used for quantitative analysis
# CORE_COUNT: Discrete count-based variables
# CORE_CATEGORICAL: Categorical fields representing labels or classes
# CORE_FLAG: Binary indicator variables
# ALL_NUMERICAL_VARS: Combined numeric and count variables

CORE_NUMERICAL = ['transportation_expense', 'body_mass_index',
                  'absenteeism_time_in_hours', 'commute_cost_per_km', 'age']
CORE_COUNT = ['number_of_children', 'number_of_pets']
CORE_CATEGORICAL = ['month_of_absence', 'reason_for_absence', 'weekday_type']
CORE_FLAG = ['disciplinary_failure', 'higher_education', 'risk_behavior']
ALL_NUMERICAL_VARS = CORE_NUMERICAL + CORE_COUNT

# Creating a month order constant to use for plotting in the correct order
MONTH_ORDER = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# Map of original column names to user-friendly display names.
# Used for more clarity and interpretability in the dashboard.
# A reverse map is also created.

COLUMN_MAP = {
    'transportation_expense': 'Transportation Expense',
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
            dataframe[col] = pd.to_numeric(
                dataframe[col]).astype(int)
            dataframe[col] = dataframe[col].map(mapping).astype('category')
    return dataframe


# 2. Data Loading

@st.cache_data
def load_data(file_path):
    """Loads the data, sets index, and cleans flag values."""
    try:
        # Load data
        data = pd.read_csv(file_path, index_col=0)

        # Ensure count columns are integers
        for col in CORE_COUNT:
            if col in data.columns:
                data[col] = pd.to_numeric(
                    data[col]).astype(int)

        # Map 1/0 to 'Yes'/'No' for flag columns
        data = map_flags_to_yes_no(data, CORE_FLAG)

        return data

    # The code below should never be needed, but it's there
    # in case there's an issue and the file is missing.
    except FileNotFoundError:
        st.error(
            f"Error: The file '{file_path}' was not found. "
            f"Please verify the path relative to your dashboard.py file. "
            f"(Checked path: {os.getcwd()}/{file_path})"
        )
        st.stop()


# Global Data Loading
FILE_PATH = 'Datasets/data_for_dashboard.csv'
data_for_analysis = load_data(FILE_PATH)

# Stop if the loaded data is empty
if data_for_analysis.empty:
    st.stop()

# Get the final list of columns available in the loaded data
all_cols = data_for_analysis.columns.tolist()
all_numerical_cols = [col for col in all_cols
                      if col in ALL_NUMERICAL_VARS]
DISPLAY_OPTIONS = [COLUMN_MAP[col] for col in all_cols
                   if col in COLUMN_MAP]

# 3. Sidebar for Data Filtering

st.sidebar.header("🔍 Data Filtering")

# This function applies the filters defined in the sidebar
# of the dashboard.


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
            min_val = float(dataframe[col_name].min())
            max_val = float(dataframe[col_name].max())

            # Determine step size and format based on variable type
            step = 1.0
            format_str = "%.1f"
            is_count = col_name in CORE_COUNT

            if is_count:
                step = 1
                format_str = "%d"
                # Cast min/max to int when step is int
                min_val = int(min_val)
                max_val = int(max_val)

            value_tup = (min_val, max_val)
            # Children and Pets filter is just a maximum value
            if is_count:
                # For count variables, use max_val as the default max selection
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
                df_filtered = df_filtered[
                    df_filtered[col_name] <= val_range
                ]
            else:
                # Filter: between min and max range
                df_filtered = df_filtered[
                    (df_filtered[col_name] >= val_range[0]) &
                    (df_filtered[col_name] <= val_range[1])
                ]

    return df_filtered


# Apply all filters to get the final data used for visualization
df_filtered = apply_filters(data_for_analysis)

# 4. Dashboard Title and Introduction / Metrics
st.title("📊 Exploratory Data Analysis")
st.markdown(f"Data Source: **{data_for_analysis.shape[0]}** total absences")
st.header("Quick Filtered Data Summary")


def get_metric(dataframe, col):
    """Calculates the mean and formats a key metric."""
    if dataframe.empty or col not in dataframe.columns:
        return "N/A"

    if types.is_numeric_dtype(dataframe[col]):
        return f"{dataframe[col].mean():.1f}"
    return "N/A"


# Displaying relevant metrics.
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(label="Filtered Absences",
              value=df_filtered.shape[0])
with col2:
    st.metric(
        label="Avg. Absenteeism (Hours)",
        value=get_metric(df_filtered, 'absenteeism_time_in_hours')
    )
with col3:
    st.metric(label="Avg. BMI",
              value=get_metric(df_filtered, 'body_mass_index'))
with col4:
    st.metric(
        label="Avg. level of Higher Education (%)",
        value=get_metric(df_filtered, 'higher_education')
    )
with col5:
    st.metric(
        label="Avg. Age",
        value=get_metric(df_filtered, 'age')
    )


st.divider()

# 5. Feature Distribution Plot (Plotly)

if df_filtered.empty:
    st.error(
        "No observations match the current filter criteria. "
        "Please adjust your filters."
    )
    st.stop()

st.header("Feature Distribution Plot")

feature_display_name = st.selectbox(
    "Select a feature to visualize its distribution on the FILTERED data:",
    options=DISPLAY_OPTIONS
)
feature_to_plot = DISPLAY_TO_VAR[feature_display_name]

# Placeholder for the Plotly figure
fig = None

if feature_to_plot in ALL_NUMERICAL_VARS:
    # Option to select plot type
    plot_type = st.radio(
        "Select Plot Type:",
        ('Histogram', 'Boxplot'),
        key='num_plot_type'
    )

    if plot_type == 'Histogram':
        fig = px.histogram(
            df_filtered,
            x=feature_to_plot,
            nbins=30,  # Default number of bins
            title=(f'Histogram of {feature_display_name} '
                   f'(Filtered Data)'),
            labels={feature_to_plot: feature_display_name,
                    'count': 'Count'},
            color_discrete_sequence=['#FF7F0E']
        )
        fig.update_layout(bargap=0.05)  # Add slight gap between bars

    elif plot_type == 'Boxplot':
        fig = px.box(
            df_filtered,
            x=feature_to_plot,
            title=(f'Boxplot of {feature_display_name} '
                   f'(Filtered Data)'),
            labels={feature_to_plot: feature_display_name},
            color_discrete_sequence=['#FF7F0E']
        )
        # Remove Y-axis title for horizontal boxplot
        fig.update_yaxes(title_text="")

elif feature_to_plot in CORE_CATEGORICAL or feature_to_plot in CORE_FLAG:
    # Count plot (Bar chart) for categorical and binary data

    # Define plot order for categorical variables
    if feature_to_plot == 'month_of_absence':
        category_order = MONTH_ORDER
    elif pd.api.types.is_categorical_dtype(
            data_for_analysis[feature_to_plot]
    ):
        category_order = (
            data_for_analysis[feature_to_plot].cat.categories.tolist()
        )
    else:
        category_order = df_filtered[feature_to_plot].unique().tolist()

    # Calculate counts manually for plotting
    df_counts = df_filtered[feature_to_plot].value_counts().reset_index()
    df_counts.columns = [feature_to_plot, 'Count']

    # Ensure correct order if specified
    category_orders_dict = {feature_to_plot: category_order}

    fig = px.bar(
        df_counts,
        x=feature_to_plot,
        y='Count',
        category_orders=category_orders_dict,
        title=(f'Count of {feature_display_name} '
               f'(Filtered Data)'),
        labels={feature_to_plot: feature_display_name,
                'Count': 'Count'},
        color_discrete_sequence=['#FF7F0E']
    )
    # Rotate x-labels for readability
    fig.update_xaxes(tickangle=45)


# Display the Plotly figure
if fig:
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# 6. Variable Relationship Explorer (Plotly Scatter)
st.header("Variable Relationship Explorer")

numerical_display_options = [COLUMN_MAP[col] for col in all_numerical_cols
                             if col in COLUMN_MAP]

if numerical_display_options:
    default_x_display = COLUMN_MAP.get(
        'absenteeism_time_in_hours',
        numerical_display_options[0]
    )
    default_y_display = COLUMN_MAP.get(
        'body_mass_index',
        numerical_display_options[0]
    )

    col_x, col_y = st.columns(2)

    with col_x:
        # Set default index safely
        try:
            default_x_index = numerical_display_options.index(
                default_x_display
            )
        except ValueError:
            default_x_index = 0

        scatter_x_display = st.selectbox(
            "Select **X-axis** Variable:",
            options=numerical_display_options,
            index=default_x_index,
            key='scatter_x_select'
        )

    with col_y:
        # Set default index safely
        try:
            default_y_index = numerical_display_options.index(
                default_y_display
            )
        except ValueError:
            default_y_index = 0

        scatter_y_display = st.selectbox(
            "Select **Y-axis** Variable:",
            options=numerical_display_options,
            index=default_y_index,
            key='scatter_y_select'
        )

    scatter_x = DISPLAY_TO_VAR.get(scatter_x_display)
    scatter_y = DISPLAY_TO_VAR.get(scatter_y_display)

    if scatter_x in df_filtered.columns and scatter_y in df_filtered.columns:
        if (types.is_numeric_dtype(df_filtered[scatter_x]) and
                types.is_numeric_dtype(df_filtered[scatter_y])):

            # Plotly scatter plot with OLS trendline
            fig_scatter = px.scatter(
                df_filtered,
                x=scatter_x,
                y=scatter_y,
                trendline="ols",
                title=(f'{scatter_x_display} vs {scatter_y_display} '
                       f'(Filtered Data)'),
                labels={scatter_x: scatter_x_display,
                        scatter_y: scatter_y_display},
                opacity=0.6,
                color_discrete_sequence=["#3700FF"],
                trendline_color_override='red'
            )

            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning(
                f"Columns '{scatter_x_display}' and/or "
                f"'{scatter_y_display}' are not strictly numeric in the"
                f" filtered data. Cannot create scatter plot."
            )
    else:
        st.warning(
            "Selected columns were not found in the loaded dataset. "
            "Please check your data file."
        )

else:
    st.warning("No numerical columns found to create a scatter plot.")
