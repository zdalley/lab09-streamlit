# Here is some starter code to get the data:
import pandas as pd
import numpy as np
import plotly.express as px

import streamlit as st
from sklearn.datasets import fetch_openml

# Cache the data
@st.cache_data
def load_data():
    dat = fetch_openml(data_id=40945, parser = 'auto')
    return dat.frame

# Load the data
titanic_df = load_data()

# Add a Title
st.title("Titanic Data Analysis App")

# User Inputs
num_rows = st.slider("Number of Rows", 10, len(titanic_df), 20)
filter_name = st.text_input("Filter by Name", "")
show_histogram = st.checkbox("Show Histogram of Ages")
x_axis_feature = st.selectbox("X-Axis Feature", titanic_df.columns)

# Filtered Data
filtered_df = titanic_df[titanic_df['name'].str.contains(filter_name, case=False)]

# Create sidebar for summary statistics
summary_sidebar = st.sidebar.subheader("Summary")

# Mean Age of Deceased Females by Class
st.sidebar.write("Mean Age of Deceased Females by Class:")
deceased_females = filtered_df[
    (filtered_df["sex"] == "female") & (filtered_df["survived"] == "0")
]
mean_age_by_class = deceased_females.groupby("pclass")["age"].mean().reset_index()
st.sidebar.dataframe(mean_age_by_class)

# Survival Counts
st.sidebar.write("Survival Counts:")
survival_counts = pd.DataFrame({
    "Status": ["Did Not Survive", "Survived"],
    "Count": filtered_df["survived"].value_counts().sort_index()
}).set_index("Status")
st.sidebar.dataframe(survival_counts)

# Create tabs
tab1, tab2 = st.tabs(["Data", "Visualization"])

# Data Tab
with tab1:
    st.dataframe(filtered_df.head(num_rows))

# Visualization Tab
with tab2:
    if show_histogram:
        st.subheader("Age Distribution")
        hist_fig = px.histogram(filtered_df, x="age", nbins=20,
                                title="Age Distribution",
                                labels={"age": "Age", "count": "Count"})
        st.plotly_chart(hist_fig)

    fig = px.scatter(filtered_df, x=x_axis_feature, y="age", color="survived", hover_data=["name"])
    st.plotly_chart(fig)

