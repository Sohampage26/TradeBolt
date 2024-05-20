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
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def show_home_page():
    square_style = "background-color: #f0d799; color: black; padding: 20px; border-radius: 10px; margin: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);"
    
    st.markdown("<h1 style='font-size:60px; color:#33ccff; text-align:center;'>TradeBolt</h1>", unsafe_allow_html=True)

    # Centered subheading
    st.markdown("<h1 style='font-size:60px;color:#33ccff; text-align:center;'>Powering Your Trading with ML</h1>", unsafe_allow_html=True)
        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='" + square_style + "'>"
                    "<h2 style='color: #4CAF50;'>Welcome to Stock Price Prediction</h2>"
                    "<p style='font-size: 16px;'>Explore stock data, visualize trends, and predict prices with ease. Whether you're a beginner or pro, our tools and algorithms empower data-driven investing for maximum returns.</p>"
                    "<p style='font-size: 16px;font-style:italic'><u>Upload historical data, use advanced models like ARIMA and LSTM, and uncover valuable insights for confident decision-making.</u></p>"
                    "</div>", unsafe_allow_html=True)

    # Second Square - Understanding Blue Chip vs Non-Blue Chip Stocks
    with col2:
        lottie_url = "https://lottie.host/7d252b8c-2362-4e43-a207-b60324bd8da1/ECxymRfbRe.json"
        lottie_json = load_lottieurl(lottie_url)
        st_lottie(lottie_json,width=600,height=300)

    col3, col4 = st.columns(2)
    with col3:
        lottie_url = "https://lottie.host/6b0cff74-2783-4cb3-b919-511b6ce8e065/TqZJ6SDBkH.json"
        lottie_json = load_lottieurl(lottie_url)
        st_lottie(lottie_json,width=500,height=400)

    with col4:
        st.markdown("<div style='" + square_style + "'>"
                    "<h2 style='color: #4CAF50;'>Unlock Insights with Dynamic Visualizations</h2>"
                    "<p style='font-size: 16px;'>Dive into a world of visual data exploration! Our platform offers an array of captivating visualizations, from dynamic line plots to insightful correlation heatmaps. Decode stock data intricacies, spot trends, and anticipate market shifts with our intuitive tools.</p>"
                    "<p style='font-size: 16px;'>Navigate the market landscape effortlessly, whether you're seeking short-term gains or long-term strategies. Let our visually rich platform empower your decision-making journey for smarter, more informed investments.</p>"
                    "</div>", unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("<div style='" + square_style + "'>"
                    "<h2 style='color: #4CAF50;'>Precision in Every Prediction</h2>"
                    "<p style='font-size: 16px;'>Embark on a journey with advanced algorithms like ARIMA and LSTM. Tailored for accuracy, ARIMA decodes short-term trends, while LSTM excels in unraveling long-term dependencies and market dynamics.</p>"
                    "<p style='font-size: 16px;'>Marrying these powerful algorithms, our platform delivers precise forecasts, empowering your investment strategy with data-driven confidence.</p>"
                    "</div>", unsafe_allow_html=True)
        
        with col6:
            lottie_url = "https://lottie.host/94ca9db4-16b7-4370-9fe4-ca9b7126cba1/kMbrz3Z5ar.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json,width=500,height=400)

    col7, col8 = st.columns(2)
    with col8:
        st.markdown("<div style='" + square_style + "'>"
                "<h2 style='color: #4CAF50;'>Navigate Future Markets with Confidence</h2>"
                "<p style='font-size: 16px;'>Explore stock price forecasts for diverse horizons: <b>week</b>, <b>fortnight</b>, and <b>month</b> ahead. Backed by historical data and advanced ML models, our predictions unveil insights into potential price shifts and trading prospects.</p>"
                "<p style='font-size: 16px;'>Tailor your strategy with precision. Short-term forecasts aid traders in seizing immediate opportunities, while long-term projections empower investors for strategic portfolio planning and decision-making.</p>"
                "<p style='font-size: 16px;'>Blend forecasted trends with thorough analysis to make informed decisions, optimizing returns and managing risk effectively in your investment journey.</p>"
                "</div>", unsafe_allow_html=True)
        
    with col7:
            lottie_url = "https://lottie.host/65dd98e0-454f-4100-b930-69e14039229e/Llntar7xEd.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json,width=500,height=400)

    col71, col81 = st.columns(2)
    with col71:
        st.markdown("<div style='" + square_style + "'>" +
                "<h2 style='color: #4CAF50;'> <span style='color: blue;'>Blue Chip</span> vs Non-Blue Chip Stocks</h2>" +
                "<ul style='font-size: 18px;'>" +
                "<li><b><span style='color: blue;'>Blue Chip Stocks:</span></b> Shares of large, reputable companies known for stable earnings, strong finances, and regular dividends. Industry leaders with a proven track record.</li>" +
                "<li><b><span style='color: green;'>Non-Blue Chip Stocks:</span></b> Typically smaller, higher-growth potential companies with more volatility. Offers chances for significant capital gains but with increased risk.</li>" +
                "</ul>" +
                "<p style='font-size: 18px;'>Knowing the difference helps align your portfolio with your financial goals and risk preference.</p>" +
                "</div>", unsafe_allow_html=True)
        
    with col81:
            lottie_url = "https://lottie.host/272fd877-c66f-439c-8573-b86f47166f4b/ZDyefI70sr.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json,width=500,height=400)

    
    uploaded_file = st.file_uploader("Upload a CSV file: ")
    try:
        data_dir = uploaded_file
        df = pd.read_csv(data_dir,  na_values=['null'], index_col='Date', parse_dates=True, infer_datetime_format=True)
        if uploaded_file:
            col9, col10 = st.columns(2)
            with col9:
                lottie_url = "https://lottie.host/c65c0bf7-7e88-47f9-a988-2a5f70a06aca/fZvqGW9tEi.json"
                lottie_json = load_lottieurl(lottie_url)
                st_lottie(lottie_json,width=400,height=200)

            with col10:
                lottie_url = "https://lottie.host/f972bd19-053a-4132-8060-82bb4f23a5e4/UJ5UiaDEtQ.json"
                lottie_json = load_lottieurl(lottie_url)
                st_lottie(lottie_json,width=500,height=400)
    except:
        lottie_url = "https://lottie.host/f972bd19-053a-4132-8060-82bb4f23a5e4/UJ5UiaDEtQ.json"
        lottie_json = load_lottieurl(lottie_url)
        st_lottie(lottie_json,width=1000,height=400)
