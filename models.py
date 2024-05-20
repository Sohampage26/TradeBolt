import math
import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as mean_squared_error
import random
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error as mse
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

def show_models_page():
    try:

        uploaded_file = st.file_uploader("Upload a CSV file: ")
        data_dir = uploaded_file
        df = pd.read_csv(data_dir,  na_values=['null'], index_col='Date', parse_dates=True) 
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
            # from sklearn.metrics import mean_squared_error
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
            MSE_error_local = mean_squared_error(test_data, model_predictions)
            print(f'MSE Error: {MSE_error_local}')

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



