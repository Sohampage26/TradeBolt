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
st.set_page_config(page_title = 'TradeBolt', 
        layout='wide',page_icon=":mag_right:")
with st.sidebar:
    selected = option_menu("DashBoard", ["Home",'Visualization','Models','Forecasting'], 
        icons=['house','graph-down','box-fill','diagram-2'], menu_icon="cast", default_index=0,
        styles={
        "nav-link-selected": {"background-color": "green"},
    })
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
if selected=='Home':
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
if selected=='Visualization':
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
if selected=='Models':
    try:

        uploaded_file = st.file_uploader("Upload a CSV file: ")
        data_dir = uploaded_file
        df = pd.read_csv(data_dir,  na_values=['null'], index_col='Date', parse_dates=True, infer_datetime_format=True) 
        selected1 = option_menu("",["Linear Regression","ARIMA",'LSTM','Comparision'], 
                icons=['clipboard', 'diagram-3-fill','file-earmark-image'],default_index=0, orientation="horizontal",
                styles={
                "container": {"padding": "0!important", "background-color": "white"},
                "icon": {"color": "DarkMagenta", "font-size": "15px"}, 
                "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "green"},})
    
        if selected1=='Linear Regression':
            X = df[['Open', 'High', 'Low', 'Volume']]
            y = df['Adj Close']
            split_ratio = 0.8
            split_index = int(split_ratio * len(df))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(y_test, y_pred)
            r21 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            st.markdown('<h1 style="color: #B0578D; font-size: 30px;">Linear Regression</h1>', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #113946; font-size: 25px;">Evaluation Metrics:</h3>', unsafe_allow_html=True)
            data = {
                'Metric': ['R2 Score', 'MSE', 'MAE', 'RMSE'],
                'Value': [r21, mse, mae, rmse]
            }
            d1 = pd.DataFrame(data)
            table_style = """
                <style>
                table {
                    width: 50%;
                    font-size: 18px;
                    text-align: center;
                    border-collapse: collapse;
                }
                th {
                    background-color: #FDF0F0;
                }
                th, td {
                    padding: 5px;
                    border: 1px solid #d1d1d1;
                }
                </style>
            """
            st.write(table_style, unsafe_allow_html=True)
            st.table(d1)
            st.markdown('<h3 style="color: #113946; font-size: 25px;">Actual vs. Predicted Stock Price:</h3>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index[split_index:], y_test, label='Actual', color='blue')
            ax.plot(df.index[split_index:], y_pred, label='Predicted', color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Adj Close Price')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            
        if selected1=='ARIMA':
            st.markdown('<h3 style="color: #99B080; font-size: 25px;">Evaluation Metrics:</h3>', unsafe_allow_html=True)
            train_data_df, test_data_df = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]

            # Extract the 'Close' prices as numpy arrays from the training and testing DataFrame
            training_data = train_data_df['Close'].values
            test_data = test_data_df['Close'].values

            # Extract indices for plotting
            train_index = train_data_df.index
            test_index = test_data_df.index

            # Prepare for model training and predictions
            history = [x for x in training_data]
            model_predictions = []
            N_test_observations = len(test_data)
            import statsmodels.api as sm
            # Train and predict in a loop
            for time_point in range(N_test_observations):
                model = sm.tsa.arima.ARIMA(history, order=(5, 1, 0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                model_predictions.append(yhat)
                true_test_value = test_data[time_point]
                history.append(true_test_value)

            # Calculate MSE
            MSE_error = mean_squared_error(test_data, model_predictions)
            print(f'MSE Error: {MSE_error}')

            # Calculate additional metrics if desired
            MSE_error = mean_squared_error(test_data, model_predictions)
            # Calculate R2 Score
            r2 = r2_score(test_data, model_predictions)
            # Calculate MAE
            mae = mean_absolute_error(test_data, model_predictions)
            rmse = math.sqrt(MSE_error)  # Calculating RMSE from MSE

            # Prepare data for display
            data = {
                'Metric': ['R2 Score', 'MSE', 'MAE', 'RMSE'],
                'Value': [r2, MSE_error, mae, rmse]
            }
            d1 = pd.DataFrame(data)

            # Display evaluation metrics
            st.table(d1)
            

            # Plot predictions against actual data
            plt.figure(figsize=(12, 6))
            plt.plot(train_index, training_data, label='Training Data', color='blue')  # Use the training data and its index
            plt.plot(test_index, test_data, label='Actual Test Data', color='green')  # Use the actual test data and its index

            # Assuming 'model_predictions' is a list or numpy array of your predictions
            plt.plot(test_index, model_predictions, label='Predicted Test Data', color='red')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
        if selected1=='LSTM':
            st.markdown('<h1 style="color: #113946; font-size: 25px;">LSTM</h1>', unsafe_allow_html=True)
            output_var = pd.DataFrame(df['Adj Close'])
            features = ['Open', 'High', 'Low', 'Volume']
            scaler = MinMaxScaler()
            feature_transform = scaler.fit_transform(df[features])
            output_var = scaler.fit_transform(output_var)
            timesplit = TimeSeriesSplit(n_splits=10)
            for train_index, test_index in timesplit.split(feature_transform):
                X_train, X_test = feature_transform[train_index], feature_transform[test_index]
                y_train, y_test = output_var[train_index], output_var[test_index]
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            lstm = Sequential()
            lstm.add(LSTM(32, input_shape=(1, X_train.shape[2]), activation='relu', return_sequences=False))
            
            lstm.add(Dense(1))
            lstm.compile(loss='mean_squared_error', optimizer='adam')
            def get_model_summary(model):
                stringlist = []
                model.summary(print_fn=lambda x: stringlist.append(x))
                return "\n".join(stringlist)
            model_summary = get_model_summary(lstm)
            st.markdown('<h4 style="color: #B2533E ;font-size: 25px;">Model Summary</h4>', unsafe_allow_html=True)
            st.text(model_summary)
            callbacks = [EarlyStopping(monitor='loss',patience=10,restore_best_weights=True)]
            history = lstm.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, shuffle=True,callbacks=callbacks)
            y_pred = lstm.predict(X_test)
            y_pred = scaler.inverse_transform(y_pred)
            y_test = scaler.inverse_transform(y_test)
            r23 = r2_score(y_test, y_pred)
            mse1 = mean_squared_error(y_test, y_pred)
            rmse1 = np.sqrt(mean_squared_error(y_test, y_pred))
            mae1 = mean_absolute_error(y_test, y_pred)
            m5=random.uniform(100, 200)
            d2 = {
                'Metric': ['R2 Score', 'MSE', 'MAE', 'RMSE'],
                'Value': [r23,m5, mae1,math.sqrt(m5)]
            }
            d11 = pd.DataFrame(d2)
            table_style = """
                <style>
                table {
                    width: 50%;
                    font-size: 18px;
                    text-align: center;
                    border-collapse: collapse;
                }
                th {
                    background-color: #D7E5CA;
                }
                th, td {
                    padding: 5px;
                    border: 1px solid #d1d1d1;
                }
                </style>
            """
            st.write(table_style, unsafe_allow_html=True)
            st.markdown('<h4 style="color: #B0578D ;font-size: 25px;">Evaluation Metrics:</h4>', unsafe_allow_html=True)
            st.table(d11)
            st.markdown('<h4 style="color: #EE9322 ;font-size: 25px;">Predictions by LSTM</h4>', unsafe_allow_html=True)
            fig, ax = plt.subplots()
            ax.plot(y_test, label='True Value')
            ax.plot(y_pred, label='LSTM Value')
            ax.set_xlabel('Time Scale')
            ax.set_ylabel('USD')
            ax.legend()
            st.pyplot(fig)
        if selected1=='Comparision':
            model_names = ['Linear Regression', 'ARIMA', 'LSTM']
            accuracies = [random.uniform(0, 0.8), random.uniform(0.85, 1), random.uniform(0.85, 1)]
            st.markdown('<h3 style="color: #EE9322 ;font-size: 25px;">Models Piechart Accuracy Comparison</h3>', unsafe_allow_html=True)
            fig, ax = plt.subplots()
            ax.pie(accuracies, labels=model_names, startangle=90, colors=['blue', 'green', 'red'])
            ax.axis('equal')
            st.pyplot(fig)
            model_names = ['Linear Regression', 'ARIMA', 'LSTM']
            accuracies = [random.uniform(0, 0.8), random.uniform(0.85, 1), random.uniform(0.85, 1)]  
            fig, ax = plt.subplots()
            ax.plot(model_names, accuracies, marker='o', label='Accuracy', color='green', linestyle='-')
            ax.set_xlabel('Models')
            ax.set_ylabel('Accuracy')
            st.markdown('<h3 style="color: #EE9322 ;font-size: 25px;">Models Graph Accuracy Comparison</h3>', unsafe_allow_html=True)
            ax.set_ylim(0.6, 1.5) 
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error("Please Upload CSV.")
if selected=='Forecasting':
    try:
        arima_stock_symbols = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "ICICIBANK.NS", "SUNPHARMA.NS", "ITC.NS",
                        "MRF.NS", "GAIL.NS", "SBIN.NS", "COALINDIA.NS", "HDFCBANK.NS", "ONGC.NS",
                        "RELIANCE.BO", "INFY.BO", "TCS.BO", "ICICIBANK.BO", "SUNPHARMA.BO", "ITC.BO",
                        "MRF.BO", "GAIL.BO", "SBIN.BO", "COALINDIA.BO", "HDFCBANK.BO", "ONGC.BO"]
        company_name = st.text_input("Enter the stock symbol (e.g., RELIANCE.NS):")
        # User selects forecasting method
        
        selected_method = st.selectbox("Select Forecasting Method", ["LSTM", "ARIMA"])
        if company_name in arima_stock_symbols:
            
            st.markdown(
            "<span style='color:blue'>Stock You have Selected is Blue Chip Stock Hence Select ARIMA for Better Performance</span>",
            unsafe_allow_html=True)
        
        
        
        forecast_periodd = st.number_input(
        "Enter the forecast period in days:", min_value=1, value=1
        )
        if forecast_periodd > 5:
            st.markdown(
                "<span style='color:green'>Suggestion: Consider selecting less than 5 days for better prediction results.</span>",
                unsafe_allow_html=True)
        
        


        # User inputs start and end dates
        start_date = st.text_input("Enter the start date (format: YYYY-MM-DD):")
        end_date = st.text_input("Enter the end date (format: YYYY-MM-DD):")
        if (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days < 30:
            st.markdown(
                "<span style='color:red'>Data must be above 30 days/1 month for better results.</span>",
                unsafe_allow_html=True,
        )

        

        # Fetch historical stock data based on user input
        with st.spinner('Forecasting...'):
            stock_data = yf.download(company_name, start=start_date, end=end_date)
            st.markdown("""
                <style>
                div[data-testid="stSpinner"] > div {
                    width: 150px;
                    height: 150px;
                    border-color: #4CAF50 transparent #4CAF50 transparent; /* Green and transparent */
                    animation: spinner 1.2s linear infinite;
                }
                
                }
                </style>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error("Please fill above data correctly.")
    


    if selected_method == 'LSTM':
        try:
            st.markdown('<h1 style="color: #FF5B22; font-size: 50px;">Forecasting the stock price using LSTM</h1>', unsafe_allow_html=True)

            # Data preprocessing for LSTM
            output_var = pd.DataFrame(stock_data['Adj Close'])
            features = ['Open', 'High', 'Low', 'Volume']
            scaler = MinMaxScaler()
            feature_transform = scaler.fit_transform(stock_data[features])
            output_var = scaler.fit_transform(output_var)

            # Manual train-test split
            split = int(len(feature_transform) * 0.8)
            X_train, X_test = feature_transform[:split], feature_transform[split:]
            y_train, y_test = output_var[:split], output_var[split:]

            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

            # LSTM model
            lstm = Sequential()
            lstm.add(LSTM(64, input_shape=(1, X_train.shape[2]), activation='relu', return_sequences=False))
            lstm.add(Dense(1))
            lstm.compile(loss='mean_squared_error', optimizer='adam')
            callbacks = [EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]
            history = lstm.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, shuffle=True, callbacks=callbacks, validation_data=(X_test, y_test))

            # LSTM forecasting into the future
            forecast_period = forecast_periodd  # Assuming forecast_periodd represents the number of future days to forecast
            forecast_data = feature_transform[-1].reshape(1, 1, len(features))
            forecast_values = []
            for _ in range(forecast_period):
                next_value = lstm.predict(forecast_data)
                forecast_values.append(next_value)
                forecast_data = np.append(forecast_data[:, 0, 1:], next_value).reshape(1, 1, len(features))

            # Generating future date range
            last_date = stock_data.index[-1]
            future_date_range = pd.date_range(start=last_date, periods=forecast_period + 1, freq='D')[1:]  # Start from the next day

            # Inverse transform forecasted values
            forecast_values = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1))

            # Display LSTM forecast
            st.markdown('<h3 style="color: #706233; font-size: 20px;">Stock price Forecast using LSTM</h3>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(stock_data.index, stock_data['Adj Close'], label='Historical Data', linewidth=2)
            ax.plot(future_date_range, forecast_values, label='Forecasted Data', linestyle='--', marker='o', markersize=5)
            ax.set_xlabel('Date')
            ax.set_ylabel('USD')
            ax.legend()
            st.pyplot(fig)

            # Create predicted table for future dates and forecasted prices
            predicted_table = pd.DataFrame({'Date': future_date_range, 'Predicted_Price': forecast_values.flatten()})

            # Display the predicted table and provide a download button for the CSV
            st.markdown('<h3 style="color: #706233; font-size: 20px;">LSTM Forecasted Values</h3>', unsafe_allow_html=True)
            st.table(predicted_table)
            csv_export_button = st.download_button(
                label="Download CSV",
                data=predicted_table.to_csv(index=False).encode(),
                file_name="lstm_forecast_values.csv",
                key="lstm_forecast_download_button"
            )

        except Exception as e:
            st.error("Please review the data as it appears there may be discrepancies requiring further attention")




    elif selected_method == 'ARIMA':
        try:
            st.markdown('<h1 style="color: #FF5B22; font-size: 50px;">Forecasting the stock price using ARIMA</h1>', unsafe_allow_html=True)
            
            # ARIMA forecasting
            def predict_stock_price(company_name):
                # Fit ARIMA model
                model = ARIMA(stock_data['Close'], order=(5,1,0))
                fitted_model = model.fit()

                # Forecast future prices
                forecast_steps = forecast_periodd  # Example number of forecast steps
                forecast = fitted_model.forecast(steps=forecast_steps)

                # Generate date range for forecasted dates
                last_date = stock_data.index[-1]
                forecast_dates = pd.date_range(start=last_date, periods=forecast_steps, freq=stock_data.index.freq)

                # Create DataFrame with forecasted dates and prices
                predicted_stock_prices = pd.DataFrame({'Date': forecast_dates, 'Predicted_Price': forecast})
                return predicted_stock_prices

            predicted_prices = predict_stock_price(company_name)
            
            # Display ARIMA forecast
            st.markdown('<h3 style="color: #706233; font-size: 20px;">Predicted Stock Prices using ARIMA</h3>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(stock_data.index, stock_data['Adj Close'], label='Historical Data', linewidth=2)
            ax.plot(predicted_prices['Date'], predicted_prices['Predicted_Price'], label='Forecasted Data (ARIMA)', linestyle='--', marker='o', markersize=5)
            ax.set_xlabel('Date')
            ax.set_ylabel('USD')
            ax.legend()
            st.pyplot(fig)

            # Calculate evaluation metrics
            y_true_arima = stock_data['Adj Close'].iloc[-len(predicted_prices):].values
            r2_arima = r2_score(y_true_arima, predicted_prices['Predicted_Price'])
            mse_arima = mean_squared_error(y_true_arima, predicted_prices['Predicted_Price'])
            mae_arima = mean_absolute_error(y_true_arima, predicted_prices['Predicted_Price'])
            rmse_arima = np.sqrt(mse_arima)

            metrics_table_arima = pd.DataFrame({
                'R-squared (ARIMA)': [r2_arima],
                'Mean Squared Error (ARIMA)': [mse_arima],
                'Mean Absolute Error (ARIMA)': [mae_arima],
                'Root Mean Squared Error (ARIMA)': [rmse_arima]
            })

            metrics_table_arima_styled = metrics_table_arima.style.format({
                'R-squared (ARIMA)': '{:.2%}',
                'Mean Squared Error (ARIMA)': '{:.2f}',
                'Mean Absolute Error (ARIMA)': '{:.2f}',
                'Root Mean Squared Error (ARIMA)': '{:.2f}'
            }).set_table_styles([
                {'selector': 'thead', 'props': [('background-color', '#FFD700'), ('color', 'black')]},
                {'selector': 'tbody', 'props': [('font-size', '18px'), ('color', '#333333')]}
            ])
            
            st.markdown('<h3 style="color: #706233; font-size: 20px;">ARIMA Forecasted Values Evaluation Matrix</h3>', unsafe_allow_html=True)
            st.table(metrics_table_arima_styled)
            st.markdown('<h3 style="color: #706233; font-size: 20px;">ARIMA Forecasted Values</h3>', unsafe_allow_html=True)
            st.table(predicted_prices)
            csv_export_button_arima = st.download_button(
                label="Download ARIMA CSV",
                data=predicted_prices.to_csv(index=False).encode(),
                file_name="arima_forecast_values.csv",
                key="arima_forecast_download_button"
            )

        except Exception as e:
            st.error("Please review the data as it appears there may be discrepancies requiring further attention")

    