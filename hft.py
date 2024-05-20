import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Function to fetch real-time data from Yahoo Finance
def get_real_time_data(ticker, interval='1m', period='1d'):
    data = yf.download(ticker, interval=interval, period=period)
    return data['Close']

# Calculate MACD
def calculate_macd(data, fast_window=12, slow_window=26, signal_window=9):
    fast_mavg = data.ewm(span=fast_window, adjust=False).mean()
    slow_mavg = data.ewm(span=slow_window, adjust=False).mean()
    macd = fast_mavg - slow_mavg
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# Moving average crossover trading strategy with RSI and MACD
def moving_average_crossover_strategy_with_rsi_macd(data, short_window=50, long_window=200, rsi_window=14, fast_window=12, slow_window=26, signal_window=9):
    # Calculate short-term and long-term moving averages
    short_mavg = data.rolling(window=short_window, min_periods=1).mean()
    long_mavg = data.rolling(window=long_window, min_periods=1).mean()

    # Calculate RSI
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_window, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate MACD
    macd, signal = calculate_macd(data, fast_window, slow_window, signal_window)

    # Generate signals based on crossover, RSI, and MACD
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 'HOLD'
    signals.loc[(short_mavg > long_mavg) & (rsi > 70) & (macd > signal), 'Signal'] = 'SELL'  # Sell signal
    signals.loc[(short_mavg < long_mavg) & (rsi < 30) & (macd < signal), 'Signal'] = 'BUY'  # Buy signal
    return signals

# Main function for the HFT program with Streamlit UI
def hft():
    st.title('High-Frequency Trading ')

    # Input options for ticker symbol and interval
    ticker = st.text_input('Enter Ticker Symbol', 'AAPL')
    interval = st.selectbox('Select Interval', ('1m', '5m', '15m', '30m', '1h'))

    # Fetch real-time data
    data = get_real_time_data(ticker, interval)

    # Moving average crossover strategy with RSI and MACD
    signals = moving_average_crossover_strategy_with_rsi_macd(data)
    current_signal = signals['Signal'].iloc[-1]

    # Display signals with different colors
    if current_signal == 'BUY':
        st.write('Current Signal: ', f'<span style="color:green; font-size:20px;">BUY</span>', unsafe_allow_html=True)
    elif current_signal == 'SELL':
        st.write('Current Signal: ', f'<span style="color:orange; font-size:20px;">SELL</span>', unsafe_allow_html=True)
    else:
        st.write('Current Signal: ', f'<span style="color:grey; font-size:20px;">HOLD</span>', unsafe_allow_html=True)

    # Display real-time data chart
    st.subheader('Real-Time Data Chart')
    st.line_chart(data, use_container_width=True)

