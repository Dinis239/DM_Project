import os
import streamlit as st
import pandas as pd
import plotly.express as px

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
MONTH_ORDER = ['January', 'February', 'March', 'April',
               'May', 'June', 'July', 'August', 'September',
               'October', 'November', 'December']

# Map of original column names to user-friendly display names.
# Used for more clarity and interpretability in the dashboard.
# A reverse map is also created, to be used for mapping back
# from selection options to dataset variable names.
COLUMN_MAP = {'transportation_expense': 'Transportation Expense',
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
              'age': 'Age'}

DISPLAY_TO_VAR = {col_mapped: og_col
                  for og_col, col_mapped in COLUMN_MAP.items()}


def map_flags(dataframe, flag_columns):
    """Maps 1/0 integer values in binary flag columns to 'Yes'/'No' strings."""
    mapping = {1: 'Yes', 0: 'No'}
    for col in flag_columns:
        if col in dataframe.columns:
            # Covert the column to category datatype
            dataframe[col] = dataframe[col].map(mapping).astype('category')
    return dataframe


# 2. Data Loading
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
        data = map_flags(data, CORE_FLAG)

        return data

    # The code below should never be needed, but it's there
    # in case there's an issue and the file is missing.
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. "
                 f"Please verify the path relative to your dashboard.py file. "
                 f"(Checked path: {os.getcwd()}/{file_path})")
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
            unique_values = dataframe[col_name].unique()

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

            # Creating a tuple with the minimum and
            # maximum values of the feature
            value_tup = (min_val, max_val)

            # Creating the slidebar for filtering in the dashboard
            val_range = st.sidebar.slider(display_name,
                                          min_value=min_val,
                                          max_value=max_val,
                                          value=value_tup,
                                          step=1.0,
                                          format="%d")

            # Filter: between min and max range
            df_filtered = df_filtered[(df_filtered[col_name] >= val_range[0]) &
                                      (df_filtered[col_name] <= val_range[1])]

    return df_filtered


# Apply all filters to get the final data used for visualization
filtered_df = apply_filters(data_for_analysis)

# 4. Dashboard Title and Introduction / Metrics
st.title("📊 Exploratory Data Analysis")
st.markdown(f"Data Source: **{data_for_analysis.shape[0]}** total absences")
st.header("Quick Filtered Data Summary")


# Displaying relevant metrics in a side-by-side column layout
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(label="Filtered Absences",
              value=filtered_df.shape[0])
with col2:
    st.metric(label="Avg. Absenteeism (Hours)",
              value=round(filtered_df['absenteeism_time_in_hours'].mean(), 1))
with col3:
    st.metric(label="Avg. BMI",
              value=round(filtered_df['body_mass_index'].mean(), 1))
with col4:
    st.metric(label="Avg. Transportation Expense (€)",
              value=round(filtered_df['transportation_expense'].mean(), 1))
with col5:
    st.metric(label="Avg. Age",
              value=round(filtered_df['age'].mean(), 1))


st.divider()

# 5. Feature Distribution Plot (Plotly)

if filtered_df.empty:
    st.error("No observations match the current filter criteria. "
             "Please adjust your filters.")
    st.stop()

st.header("Feature Distribution Plot")

# Creating the drop-down to use for choosing variables
# using the variable display names
feature_display_name = st.selectbox(
    "Select a feature to visualize its distribution on the FILTERED data:",
    options=DISPLAY_OPTIONS)
feature_to_plot = DISPLAY_TO_VAR[feature_display_name]


if feature_to_plot in ALL_NUMERICAL_VARS:
    # Option to select plot type
    plot_type = st.radio(
        "Select Plot Type:",
        ('Histogram', 'Boxplot'),
        key='num_plot_type'
    )

    if plot_type == 'Histogram':
        # Plotting the histogram, while adding axis labels,
        # title and defining the colors
        fig = px.histogram(filtered_df,
                           x=feature_to_plot,
                           nbins=30,
                           title=(f'Histogram of {feature_display_name} '
                                  f'(Filtered Data)'),
                           labels={feature_to_plot: feature_display_name,
                                   'count': 'Count'},
                           color_discrete_sequence=["#009B0D"])
        fig.update_layout(bargap=0.05)  # Add slight gap between bars

    elif plot_type == 'Boxplot':
        # Plotting the boxplot, while adding axis labels,
        # title and defining the colors
        fig = px.box(filtered_df,
                     x=feature_to_plot,
                     title=(f'Boxplot of {feature_display_name} '
                            f'(Filtered Data)'),
                     labels={feature_to_plot: feature_display_name},
                     color_discrete_sequence=['#009B0D'])

elif feature_to_plot in CORE_CATEGORICAL or feature_to_plot in CORE_FLAG:
    # Count plot (Bar chart) for categorical and binary data

    # Define plot order for categorical variables
    if feature_to_plot == 'month_of_absence':
        # Ensuring month order
        category_order = MONTH_ORDER
    else:
        category_order = filtered_df[feature_to_plot].unique().tolist()

    # Calculate counts manually for plotting
    df_counts = filtered_df[feature_to_plot].value_counts().reset_index()
    df_counts.columns = [feature_to_plot, 'Count']

    # Ensure correct order if specified
    category_orders_dict = {feature_to_plot: category_order}

    fig = px.bar(df_counts,
                 x=feature_to_plot,
                 y='Count',
                 category_orders=category_orders_dict,
                 title=(f'Count of {feature_display_name} '
                        f'(Filtered Data)'),
                 labels={feature_to_plot: feature_display_name,
                         'Count': 'Count'},
                 color_discrete_sequence=['#009B0D'])
    # Rotate x-labels for readability
    fig.update_xaxes(tickangle=45)


# Display the Plotly figure
st.plotly_chart(fig, use_container_width=True)

st.divider()

# 6. Variable Relationship Explorer (Plotly Scatter)
st.header("Variable Relationship Explorer")

# For all the numerical columns, get the column display name
numerical_display_options = [COLUMN_MAP[col] for col in all_numerical_cols
                             if col in COLUMN_MAP]

# The if and elses at the end are simply in case the data is not
# in the right structure, which for us, in theory it should always be

# Creating the side-by-side layout
# for the 2 drop-down menus
col_x, col_y = st.columns(2)

with col_x:
    # Creating the drop-down menu for the
    # variable on the x axis
    scatter_x_display = st.selectbox("Select **X-axis** Variable:",
                                     options=numerical_display_options,
                                     # Sets the default option to be BMI
                                     index=3)

with col_y:
    # Creating the drop-down menu for the
    # variable on the y axis
    scatter_y_display = st.selectbox("Select **Y-axis** Variable:",
                                     options=numerical_display_options,
                                     # Sets the default option to
                                     # be Absenteeism time
                                     index=4)

# Obtain the "actual" variable name from the display name
# for the 2 variables to plot
scatter_x = DISPLAY_TO_VAR.get(scatter_x_display)
scatter_y = DISPLAY_TO_VAR.get(scatter_y_display)


# Plotly scatter plot with OLS trendline
# with a title and axis labels
fig_scatter = px.scatter(filtered_df,
                         x=scatter_x,
                         y=scatter_y,
                         trendline="ols",
                         title=(f'{scatter_x_display} vs '
                                f'{scatter_y_display} '
                                f'(Filtered Data)'),
                         labels={scatter_x: scatter_x_display,
                                 scatter_y: scatter_y_display},
                         opacity=0.6,
                         color_discrete_sequence=["#009B0D"],
                         trendline_color_override='red')

st.plotly_chart(fig_scatter, use_container_width=True)
