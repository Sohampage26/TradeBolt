import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, time
import altair as alt

# Function to check if the market is open
def is_market_open():
    market_open_time = time(9, 15)
    market_close_time = time(15, 30)
    current_time = datetime.now().time()
    return market_open_time <= current_time <= market_close_time

# Function to get real-time data without caching
def get_real_time_data(ticker, interval, period):
    data = yf.download(tickers=ticker, period=period, interval=interval)
    return data

def macd_rsi_divergence_strategy(data, fast_window=12, slow_window=26, signal_window=9, rsi_window=14):
    # Calculate MACD
    exp1 = data['Close'].ewm(span=fast_window, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow_window, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    # Calculate RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_window, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Generate signals based on MACD and RSI divergence
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 'HOLD'

    # MACD Signal
    signals.loc[macd > signal, 'Signal'] = 'BUY'
    signals.loc[macd < signal, 'Signal'] = 'SELL'

    # RSI Signal
    signals.loc[rsi < 30, 'Signal'] = 'BUY'
    signals.loc[rsi > 70, 'Signal'] = 'SELL'

    # MACD and RSI Divergence Signal
    bullish_divergence = (macd.diff() > 0) & (rsi.diff() < 0) & (macd > 0) & (rsi < 70)
    bearish_divergence = (macd.diff() < 0) & (rsi.diff() > 0) & (macd < 0) & (rsi > 30)
    signals.loc[bullish_divergence, 'Signal'] = 'BUY'
    signals.loc[bearish_divergence, 'Signal'] = 'SELL'

    return signals

def hft():
    st.title('High-Frequency Trading')

    ticker = st.text_input('Enter Ticker Symbol', 'RELIANCE.NS')
    interval = st.selectbox('Select Interval', ('1m', '5m', '15m', '30m', '1h'))
    period = st.selectbox('Select Period', ('1d', '5d', '1mo'))

    data = get_real_time_data(ticker, interval, period)

    # Debug: Show the current time and market status
    current_time = datetime.now()
    market_status = "open" if is_market_open() else "closed"
    st.write(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}, Market status: {market_status}")

    if not is_market_open():
        st.write("The market is currently closed. Displaying data up to the last market close.")

    if not data.empty and 'Close' in data.columns:
        signals = macd_rsi_divergence_strategy(data)
        current_signal = signals['Signal'].iloc[-1]

        if current_signal == 'BUY':
            st.write('Current Signal: ', f'<span style="color:green; font-size:20px;">BUY</span>', unsafe_allow_html=True)
        elif current_signal == 'SELL':
            st.write('Current Signal: ', f'<span style="color:red; font-size:20px;">SELL</span>', unsafe_allow_html=True)
        else:
            st.write('Current Signal: ', f'<span style="color:grey; font-size:20px;">HOLD</span>', unsafe_allow_html=True)

        # Add a column to indicate price change
        data['Change'] = data['Close'].diff()
        data['Color'] = np.where(data['Change'] >= 0, 'green', 'red')

        # Create an Altair line chart with conditional color
        line_chart = alt.Chart(data.reset_index()).mark_line().encode(
            x='Datetime:T',
            y='Close:Q',
            color=alt.condition(
                alt.datum.Change >= 0,
                alt.value('green'),  # The positive color
                alt.value('red')  # The negative color
            )
        ).properties(
            width='container'
        )

        # Display the chart in Streamlit
        st.altair_chart(line_chart, use_container_width=True)
        
        if not is_market_open():
            st.write("Note: The above data is up to the last market close. Data will be updated when the market reopens.")
    else:
        st.write("Error: No data retrieved for the given ticker symbol and interval.")
