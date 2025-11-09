import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
import plotly.express as px

#import the .csv file
df = pd.read_csv("carbon_emissions_china.csv")

#convert to data frame
df_carbonEmission = pd.DataFrame(df)

st.title("The Entire Dataset")

st.write(df_carbonEmission)
st.header("First Few Lines")
st.write(df_carbonEmission.head())


st.subheader("Basic Exploratory Data")
st.write(df_carbonEmission.describe())

#identify how much categories there are
st.subheader("Number of Categories and How Much in each Categories")
df_categories = df_carbonEmission["Sector"].unique()
st.write(df_categories)
df_categories = df_carbonEmission["Sector"].value_counts()
st.write(df_categories)

# STATES

st.title("State Analysis")

# --- Average Emmission Per State ---
st.subheader("Average MtCO2 Emmission Per State:")
df_averageEmissionPerState = pd.DataFrame(df_carbonEmission.groupby('State')['MtCO2 per day'].mean())
st.write(df_averageEmissionPerState)

st.bar_chart(df_averageEmissionPerState)

# --- States With Average MtCO2 per day Higher Than 0.3 ---
st.text("These are the states with average MtCO2 higher than 0.3.")
st.write(df_averageEmissionPerState[df_averageEmissionPerState['MtCO2 per day'] > 0.3])
st.bar_chart(df_averageEmissionPerState[df_averageEmissionPerState['MtCO2 per day'] > 0.3])

df_selectedStates = ['Hebei', 'Inner Mongolia', 'Jiangsu', 'Shandong']
df_topStates = df_carbonEmission[df_carbonEmission['State'].isin(df_selectedStates)]

x = df_topStates[(df_topStates['State'] == 'Hebei') & (df_topStates['Sector'] == 'Industry')]
st.write(x)

x['Date'] = pd.to_datetime(x['Date'], dayfirst = True)

st.line_chart(x, x = 'Date', y = 'MtCO2 per day')

# SECTOR

st.title("Sector Analysis")

# --- Average Emmission Per Sector ---

st.subheader("Average MtCO2 Emmission Per Sector")
df_averageEmissionPerSector = df_carbonEmission.groupby('Sector')['MtCO2 per day'].mean()
st.write(df_averageEmissionPerSector)

st.bar_chart(df_averageEmissionPerSector)

