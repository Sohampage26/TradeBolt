import streamlit as st
from streamlit_option_menu import option_menu
from home import show_home_page
from visualization import show_visualization_page
from models import show_models_page
from forecasting import show_forecasting_page
from admin import admin_panel
from hft import hft
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
st.set_page_config(page_title='TradeBolt', layout='wide', page_icon=":mag_right:")

with st.sidebar:
    selected = option_menu("DashBoard", ["Home","Admin", 'Visualization', 'Models', 'Forecasting','HFT'],
                           icons=['house', 'shield','graph-down', 'box', 'diagram-2','graph-up'], menu_icon="cast", default_index=0,
                           styles={"nav-link-selected": {"background-color": "green"}})

if selected == 'Home':
    show_home_page()
elif selected == 'Admin':
    admin_panel()
elif selected == 'Visualization':
    show_visualization_page()
elif selected == 'Models':
    show_models_page()
elif selected == 'Forecasting':
    show_forecasting_page()
elif selected == 'HFT':
    hft()


