from util import load_and_clean_data
from eda import run_eda
import models
import streamlit as st
import plotly

st.title("Analyzing effects of climate change factor on local weather")
st.sidebar.subheader("Choose a model")
classifier = st.sidebar.selectbox("Models", ("LSTM (Average Temperature)", "Random Forest (CO2)", "Prophet (Average Temperature)"))

def display_graphs(graphs):
    for key, value in graphs.items():
        if isinstance(value, plotly.graph_objs._figure.Figure):
            st.header(key)
            st.plotly_chart(value)
        else:
            st.header(key)
            st.pyplot(value)

def display_eda_graphs(eda_graphs):
    st.title("Your data at a glance")
    display_graphs(eda_graphs)

def display_model_graphs(model_graphs):
    st.title("Predictions for the selected model")
    display_graphs(model_graphs)

st.write("Hint: Downloading the NOAA weather database for your city at https://www.ncdc.noaa.gov/cdo-web/search ensure it conforms to that format.")

uploaded_file = st.file_uploader("Upload a dataset in .csv format")
my_bar = st.progress(0)

if classifier == "Prophet (Average Temperature)":
    st.info('Selected model is - Prophet (Average Temperature)')
    number_pred = st.text_input('Number of days to predict')
elif classifier == "LSTM (Average Temperature)":
    st.info('Selected model is - LSTM (Average Temperature)')
else:
    st.info('Selected model is - Random Forest (CO2)')    

if st.button('Submit'):
    weather = load_and_clean_data(uploaded_file)
    my_bar.progress(33)
    eda_graphs = run_eda(weather)
    my_bar.progress(66)
    if classifier == "LSTM (Average Temperature)":
        model_graphs = models.run_lstm(weather, 60, 7)
    elif classifier == "Random Forest (CO2)":
        model_graphs = models.run_random_forest(weather)
    else:
        model_graphs = models.run_fb_prophet(weather, number_pred)
    my_bar.progress(100)
    display_model_graphs(model_graphs)
    display_eda_graphs(eda_graphs)


