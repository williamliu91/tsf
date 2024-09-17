import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Function to fetch stock data
def get_stock_data(ticker, period="1mo", interval="15m"):
    stock_data = yf.download(ticker, period=period, interval=interval)
    return stock_data

# Function to calculate Time Series Forecast (TSF) manually
def calculate_tsf(stock_data, tsf_period):
    close_prices = stock_data['Close'].values
    tsf_values = np.full_like(close_prices, np.nan)

    for i in range(tsf_period, len(close_prices)):
        y = close_prices[i-tsf_period:i].reshape(-1, 1)
        x = np.arange(tsf_period).reshape(-1, 1)
        
        # Linear regression to forecast the next value
        model = LinearRegression().fit(x, y)
        forecast_value = model.predict([[tsf_period]])[0][0]
        
        tsf_values[i] = forecast_value
    
    return tsf_values

# Function to generate buy and sell signals
def generate_signals(stock_data):
    signals = pd.DataFrame(index=stock_data.index)
    signals['Signal'] = 0
    
    # Buy signal: when the close price crosses above the TSF
    signals['Signal'][stock_data['Close'] > stock_data['TSF']] = 1
    
    # Sell signal: when the close price crosses below the TSF
    signals['Signal'][stock_data['Close'] < stock_data['TSF']] = -1
    
    # Shift signals to mark buy/sell at the time of crossing
    signals['Buy_Signal'] = signals['Signal'].diff().fillna(0) == 1
    signals['Sell_Signal'] = signals['Signal'].diff().fillna(0) == -1
    
    return signals

# Function to plot stock data, TSF, and signals
def plot_tsf_chart_with_signals(stock_data, tsf_period):
    # Calculate TSF using manual method
    stock_data['TSF'] = calculate_tsf(stock_data, tsf_period)
    
    # Generate buy/sell signals
    signals = generate_signals(stock_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the close price
    ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
    
    # Plot the TSF line
    ax.plot(stock_data.index, stock_data['TSF'], label=f'TSF ({tsf_period})', color='orange')
    
    # Plot buy signals
    ax.plot(stock_data.index[signals['Buy_Signal']], stock_data['Close'][signals['Buy_Signal']], 
            '^', markersize=10, color='green', lw=0, label='Buy Signal')
    
    # Plot sell signals
    ax.plot(stock_data.index[signals['Sell_Signal']], stock_data['Close'][signals['Sell_Signal']], 
            'v', markersize=10, color='red', lw=0, label='Sell Signal')
    
    ax.set_title(f'Stock Price, TSF, and Trading Signals for {ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

# Streamlit app interface
st.title("📈 Day Trading Chart with Time Series Forecast (TSF) and Signals")

# User inputs for stock ticker
ticker = st.text_input('Enter Stock Ticker Symbol:', 'AAPL')

# User inputs for TSF period
tsf_period = st.slider('Select TSF Period:', min_value=5, max_value=100, value=14)

# User inputs for stock data time frame
timeframe = st.selectbox('Select Time Frame:', ['1d', '5d', '1mo', '3mo', '6mo', '1y'])

# User inputs for stock data interval
interval = st.selectbox('Select Data Interval:', ['1m', '5m', '15m', '30m', '1h', '1d'])

# Display chart if the user enters a ticker
if ticker:
    try:
        # Fetch the stock data
        stock_data = get_stock_data(ticker, period=timeframe, interval=interval)
        
        # Show the data as a table (optional)
        st.subheader(f'Stock Data for {ticker} ({timeframe}, {interval})')
        st.dataframe(stock_data.tail())
        
        # Plot the chart with TSF and signals
        plot_tsf_chart_with_signals(stock_data, tsf_period)
    
    except Exception as e:
        st.error(f"Error fetching data: {e}")
