import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Function to fetch real-time data from Yahoo Finance
def get_real_time_data(ticker, interval='1h', period='1d'):
    # Using Yahoo Finance to fetch historical market data
    data = yf.download(tickers=ticker, period=period, interval=interval)
    data.reset_index(inplace=True)
    return data[['Datetime', 'Close']].set_index('Datetime')

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
    signals['MACD'] = macd
    signals['Signal Line'] = signal
    signals['RSI'] = rsi

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

def main():
    st.title('High-Frequency Trading Strategy Viewer')

    ticker = st.text_input('Enter Ticker Symbol', 'AAPL')
    interval = st.selectbox('Select Interval', ('1m', '5m', '15m', '30m', '1h'))

    data = get_real_time_data(ticker, interval)

    if data.empty:
        st.write("No data available for the given ticker and interval.")
        return

    signals = macd_rsi_divergence_strategy(data)
    current_signal = signals['Signal'].iloc[-1]

    if current_signal == 'BUY':
        st.markdown('**Current Signal: BUY**', unsafe_allow_html=True)
    elif current_signal == 'SELL':
        st.markdown('**Current Signal: SELL**', unsafe_allow_html=True)
    else:
        st.markdown('**Current Signal: HOLD**', unsafe_allow_html=True)

    st.subheader('Real-Time Data Histogram')
    st.bar_chart(data['Close'])

if __name__ == "__main__":
    main()
