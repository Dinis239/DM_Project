import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Absence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Customization ---
# --- Main Page Content (Home/Landing Page) ---

st.title("📊 Employee Absence Analysis Dashboard")
st.markdown("""
This dashboard provides a comprehensive analysis of employee absenteeism
data, split into two pages: **Exploratory Data Analysis** and
**Cluster Analysis**. Use the sidebar to navigate between them.
""")

st.markdown("---")

# 📈 Exploratory Analysis

st.subheader("Page 1: Exploratory Analysis")
st.markdown("""
This section focuses on understanding the raw data and main demographic
trends in employee absence.
Here you can:
* **Filter** the data by demographic factors (e.g., Age, BMI, Education).
* Visualize **univariate distributions** (histograms).
* Explore **bivariate relationships** (scatter plots) between variables like
**Age vs. Absenteeism Hours**.
* Identify **raw patterns** before any modeling or clustering is applied.
""")

st.markdown("---")

# ⭐ Cluster Analysis

st.subheader("Page 2: Cluster Analysis")
st.markdown("""
This section dives into segmentation, revealing natural groupings
(clusters) of absences based on their characteristics and the characteristics
of the employees performing the absence.
This is crucial for targeted interventions. Here you can:
* Review **detailed profiles** for each cluster, including a description and
mean values (centroids).
* Use the **Radar Chart** to visually compare cluster profiles across key
features.
* Filter the view to focus only on clusters of interest.
""")
