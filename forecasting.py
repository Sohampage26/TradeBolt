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
import json

def show_forecasting_page():
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
            future_date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periodd, freq='D')  # Start from the next day

            # Inverse transform forecasted values
            forecast_values = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1))

            # Display LSTM forecast
            st.markdown('<h3 style="color: #706233; font-size: 20px;">Stock price Forecast using LSTM</h3>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(stock_data.index, stock_data['Adj Close'], label='Historical Data', linewidth=2)
            ax.plot(future_date_range, forecast_values, label='Forecasted Data', linestyle='--', marker='o', markersize=5)
            ax.set_xlabel('Date')
            ax.set_ylabel('INR')
            ax.legend()
            st.pyplot(fig)
            # Calculate evaluation metrics for LSTM forecast
            y_true_lstm = stock_data['Adj Close'].iloc[-forecast_period - 1:-1].values
            r2_lstm = r2_score(y_true_lstm, forecast_values.flatten())
            mse_lstm = mean_squared_error(y_true_lstm, forecast_values.flatten())
            mae_lstm = mean_absolute_error(y_true_lstm, forecast_values.flatten())
            rmse_lstm = np.sqrt(mse_lstm)

            # Create a DataFrame to display the evaluation metrics
            metrics_table_lstm = pd.DataFrame({
                'R-squared (LSTM)': [r2_lstm],
                'Mean Squared Error (LSTM)': [mse_lstm],
                'Mean Absolute Error (LSTM)': [mae_lstm],
                'Root Mean Squared Error (LSTM)': [rmse_lstm]
            })

            # Format the table
            metrics_table_lstm_styled = metrics_table_lstm.style.format({
                'R-squared (LSTM)': '{:.2%}',
                'Mean Squared Error (LSTM)': '{:.2f}',
                'Mean Absolute Error (LSTM)': '{:.2f}',
                'Root Mean Squared Error (LSTM)': '{:.2f}'
            }).set_table_styles([
                {'selector': 'thead', 'props': [('background-color', '#FFD700'), ('color', 'black')]},
                {'selector': 'tbody', 'props': [('font-size', '18px'), ('color', '#333333')]}
            ])

            # Display the evaluation metrics table
            st.markdown('<h3 style="color: #706233; font-size: 20px;">Evaluation Metrics for LSTM Forecast</h3>', unsafe_allow_html=True)
            st.table(metrics_table_lstm_styled)



            # Create predicted table for future dates and forecasted prices
            predicted_table = pd.DataFrame({'Date': future_date_range+pd.Timedelta(days=1), 'Predicted_Price': forecast_values.flatten()})

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
            def load_arima_parameters():
                try:
                    with open('arima_config.json', 'r') as config_file:
                        params = json.load(config_file)
                    return params
                except FileNotFoundError:
                    print("Configuration file not found. Using default parameters.")
                    return {'p': 5, 'd': 1, 'q': 0}
            params = load_arima_parameters()
            p, d, q = params['p'], params['d'], params['q']
            print(p,d,q)

            
            
            # ARIMA forecasting
            def predict_stock_price(company_name):
                # Fit ARIMA model
                model = ARIMA(stock_data['Close'], order=(p,d,q))
                fitted_model = model.fit()

                # Forecast future prices
                forecast_steps = forecast_periodd  # Example number of forecast steps
                forecast = fitted_model.forecast(steps=forecast_steps)

                # Generate date range for forecasted dates
                last_date = stock_data.index[-1]
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq=stock_data.index.freq)

                # Create DataFrame with forecasted dates and prices
                predicted_stock_prices = pd.DataFrame({'Date': forecast_dates+pd.Timedelta(days=1), 'Predicted_Price': forecast})
                return predicted_stock_prices

            predicted_prices = predict_stock_price(company_name)
            
            # Display ARIMA forecast
            st.markdown('<h3 style="color: #706233; font-size: 20px;">Predicted Stock Prices using ARIMA</h3>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(stock_data.index, stock_data['Adj Close'], label='Historical Data', linewidth=2)
            ax.plot(predicted_prices['Date'], predicted_prices['Predicted_Price'], label='Forecasted Data (ARIMA)', linestyle='--', marker='o', markersize=5)
            ax.set_xlabel('Date')
            ax.set_ylabel('INR')
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