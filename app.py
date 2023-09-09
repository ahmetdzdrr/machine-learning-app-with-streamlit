import os
import pandas as pd
import streamlit as st
import utils as ut
import warnings

warnings.simplefilter("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

image_path = "ml.png"
st.sidebar.image(image_path)
st.sidebar.write("<h1 style='text-align: center;'>Machine Learning App</h1>", unsafe_allow_html=True)

option = st.sidebar.radio("", ["Upload", "Exploratory Data Analysis", "Machine Learning"])

if option == 'Upload':
    uploaded_file = st.file_uploader("Choose a file (CSV)", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file, index_col=None)
        data.to_csv("data.csv", index=None)
        st.write("First 5 observations of data")
        st.dataframe(data.head())
        st.write("Last 5 observations of data")
        st.dataframe(data.tail())
        st.success("You can pass Exploratory Data Analysis section", icon="✅")

if os.path.exists("data.csv"):
    df = pd.read_csv("data.csv", index_col=None)

    if option == 'Exploratory Data Analysis':
        if os.path.exists("data.csv"):
            st.title("Exploratory Data Analysis")
            st.markdown("---")
            st.write("### About Dataset")
            ut.data_analysis(df)
            st.write("### Data Visualization")
            ut.pie_visualization(df)
            ut.barplot_visualization(df)
            st.markdown("---")
            st.write("### Feature Comparison")
            ut.feature_comparison(df)
            st.markdown("---")
            st.write("### Correlation Matrix")
            ut.correlation_matrix(df)

    if option == 'Machine Learning':
        if os.path.exists("data.csv"):
            st.title("Machine Learning")
            st.markdown("---")
            ut.data_preprocessing(df)
            st.markdown("---")
            ut.processing_null_values(df)
            st.markdown("---")
            st.write("### Encoding Processes")
            ut.encoding_process(df)
            st.markdown("---")
            st.write("### Scaling Processes")
            ut.scaling_process(df)
            st.markdown("---")
            st.write("### Model and Evaluate")
            ut.select_model(df)

else:
    st.warning("You must upload a CSV file!", icon="⚠️")
