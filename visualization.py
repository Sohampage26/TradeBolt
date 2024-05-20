import math
import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as mean_squared_error
import random
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from keras.models import Sequential
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import requests
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping

def show_visualization_page():
    try:
        uploaded_file = st.file_uploader("Upload a CSV file: ")
        data_dir = uploaded_file
        df = pd.read_csv(data_dir,  na_values=['null'], index_col='Date', parse_dates=True, infer_datetime_format=True)
        st.markdown('<p style="color: blue; font-size: 18px;">Top 5 records of the Dataset:</p>', unsafe_allow_html=True)
        st.write(df.head())
        st.markdown('<p style="color: green; font-size: 18px;">Bottom 5 records of the Dataset:</p>', unsafe_allow_html=True)
        st.write(df.tail())
        st.markdown('<p style="color: red; font-size: 18px;">Sample records of the Dataset:</p>', unsafe_allow_html=True)
        st.write(df.sample(25))
        st.markdown('<p style="color: #F875AA; font-size: 18px;">Size of the Dataset:</p>', unsafe_allow_html=True)
        st.write('Row Size:',df.shape[0])
        st.write('Column Size:',df.shape[1])
        st.markdown('<p style="color: #F9B572; font-size: 18px;">Columns are:</p>', unsafe_allow_html=True)
        st.write(df.columns)
        st.markdown('<p style="color: #190482; font-size: 18px;">Description related to Dataset are:</p>', unsafe_allow_html=True)
        st.write(df.describe())
        st.markdown('<h3 style="color: #940B92; text-align: center;">Data Preprocessing</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #3A4D39; font-size: 18px;">Null Values in the Dataset:</p>', unsafe_allow_html=True)
        st.write(df.isnull().sum())
        st.markdown('<p style="color: #706233; font-size: 18px;">Duplicate Records   in the Dataset:</p>', unsafe_allow_html=True)
        st.write(df.duplicated().sum())
        st.markdown('<p style="color: #9A4444; font-size: 18px;">Unique Values in the Dataset:</p>', unsafe_allow_html=True)
        st.write(df.nunique())
        st.markdown('<h3 style="color: #9D76C1; text-align: center;">Exploratory Data Analysis</h3>', unsafe_allow_html=True)
        corr_matrix = df.corr()
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        st.markdown('<p style="color: #739072; font-size: 18px;">Correlation Heatmap:</p>', unsafe_allow_html=True)
        st.pyplot(fig)
        fig = plt.figure(figsize=(15, 6))
        df['High'].plot()
        df['Low'].plot()
        plt.ylabel(None)
        plt.xlabel(None)
        st.markdown('<p style="color: #0174BE; font-size: 18px;">High & Low Price:</p>', unsafe_allow_html=True)
        plt.legend(['High Price', 'Low Price'])
        plt.tight_layout()
        st.pyplot(fig)
        fig = plt.figure(figsize=(15, 6))
        df['Open'].plot()
        df['Close'].plot()
        plt.ylabel(None)
        plt.xlabel(None)
        st.markdown('<p style="color: #CE5A67; font-size: 18px;">Opening & Closing Price:</p>', unsafe_allow_html=True)
        plt.legend(['Open Price', 'Close Price'])
        plt.tight_layout()
        st.pyplot(fig)
        fig = plt.figure(figsize=(15, 6))
        df['Volume'].plot()
        plt.ylabel('Volume')
        plt.xlabel(None)
        st.markdown('<p style="color: #F9B572; font-size: 18px;">Sales Volume of :</p>', unsafe_allow_html=True)
        plt.tight_layout()
        st.pyplot(fig)
        fig = plt.figure(figsize=(15, 6))
        df['Adj Close'].pct_change().hist(bins=50)
        plt.ylabel('Daily Return')
        st.markdown('<p style="color: #940B92; font-size: 18px;"> Daily Return:</p>', unsafe_allow_html=True)
        plt.tight_layout()
        st.pyplot(fig)
        output_var = pd.DataFrame(df['Adj Close'])
        features = ['Open', 'High', 'Low', 'Volume']
        pairplot = sns.pairplot(df[features])
        st.markdown('<p style="color: #363062; font-size: 18px;">Features Visualization:</p>', unsafe_allow_html=True)
        st.pyplot(pairplot.fig)
    except Exception as e:
        st.error("Please Upload CSV.")