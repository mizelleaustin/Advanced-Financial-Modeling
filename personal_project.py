from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import wrds

# WRDS Data Fetcher
def fetch_wrds_data(tickers):
    """
    Fetch real-time stock price data from WRDS using the CRSP dataset.
    """
    # Connect to WRDS using credentials (replace with your username and password)
    conn = wrds.Connection(wrds_username="your_username", wrds_password="your_password")

    # Query to fetch stock data
    query = f"""
        SELECT permno, date, prc AS price
        FROM crsp.dsf
        WHERE date >= '2023-01-01'
          AND permno IN (
              SELECT permno FROM crsp.stocknames WHERE ticker IN ({",".join(["'" + t + "'" for t in tickers])})
          )
    """
    data = conn.raw_sql(query)

    # Map PERMNO back to tickers
    mapping_query = f"""
        SELECT ticker, permno
        FROM crsp.stocknames
        WHERE ticker IN ({",".join(["'" + t + "'" for t in tickers])})
    """
    ticker_mapping = conn.raw_sql(mapping_query)

    # Close connection
    conn.close()

    # Merge and clean data
    data = data.merge(ticker_mapping, on="permno", how="left")
    data = data.rename(columns={"date": "Date", "price": "Price", "ticker": "Ticker"})
    data["Date"] = pd.to_datetime(data["Date"])
    return data[["Date", "Ticker", "Price"]]

# Calculate Trading Signals
def calculate_trading_signals(market_data):
    """
    Adds moving average crossover, RSI, and Bollinger Bands signals to the data.
    """
    # Moving Averages
    market_data['Short_MA'] = market_data.groupby('Ticker')['Price'].transform(lambda x: x.rolling(window=10).mean())
    market_data['Long_MA'] = market_data.groupby('Ticker')['Price'].transform(lambda x: x.rolling(window=50).mean())
    market_data['MA_Signal'] = np.where(market_data['Short_MA'] > market_data['Long_MA'], 'Buy', 'Sell')

    # RSI
    window_length = 14
    delta = market_data.groupby('Ticker')['Price'].transform(lambda x: x.diff())
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window_length, min_periods=1).mean()
    avg_loss = loss.rolling(window=window_length, min_periods=1).mean()
    rs = avg_gain / avg_loss
    market_data['RSI'] = 100 - (100 / (1 + rs))
    market_data['RSI_Signal'] = np.where(market_data['RSI'] < 30, 'Buy', np.where(market_data['RSI'] > 70, 'Sell', 'Hold'))

    # Bollinger Bands
    market_data['Middle_Band'] = market_data.groupby('Ticker')['Price'].transform(lambda x: x.rolling(window=20).mean())
    market_data['Std_Dev'] = market_data.groupby('Ticker')['Price'].transform(lambda x: x.rolling(window=20).std())
    market_data['Upper_Band'] = market_data['Middle_Band'] + (2 * market_data['Std_Dev'])
    market_data['Lower_Band'] = market_data['Middle_Band'] - (2 * market_data['Std_Dev'])
    market_data['BB_Signal'] = np.where(market_data['Price'] > market_data['Upper_Band'], 'Sell', 
                                        np.where(market_data['Price'] < market_data['Lower_Band'], 'Buy', 'Hold'))

    return market_data

# Dash App
app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Real-Time Trading Signals Dashboard"),
    html.Div([
        html.Label("Enter tickers (comma-separated):"),
        dcc.Input(id="ticker-input", type="text", placeholder="AAPL, GOOGL", debounce=True),
        html.Button("Submit", id="submit-button", n_clicks=0),
    ], style={"margin-bottom": "20px"}),
    dcc.Graph(id="price-chart"),
    dcc.Graph(id="signals-chart"),
])

# Callbacks
@app.callback(
    [Output("price-chart", "figure"),
     Output("signals-chart", "figure")],
    [Input("submit-button", "n_clicks")],
    [State("ticker-input", "value")]
)
def update_dashboard(n_clicks, tickers):
    if not tickers:
        return {}, {}

    # Parse inputs
    tickers = [ticker.strip() for ticker in tickers.split(",")]

    # Fetch data from WRDS and calculate signals
    market_data = fetch_wrds_data(tickers)
    market_data = calculate_trading_signals(market_data)

    # Price Chart with Bollinger Bands
    price_chart = go.Figure()
    for ticker in tickers:
        ticker_data = market_data[market_data['Ticker'] == ticker]
        price_chart.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['Price'], mode='lines', name=f"{ticker} Price"))
        price_chart.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['Upper_Band'], mode='lines', name=f"{ticker} Upper Band", line=dict(dash='dot')))
        price_chart.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['Lower_Band'], mode='lines', name=f"{ticker} Lower Band", line=dict(dash='dot')))

    price_chart.update_layout(title="Price and Bollinger Bands", xaxis_title="Date", yaxis_title="Price")

    # Signals Chart
    signals_chart = go.Figure()
    for ticker in tickers:
        ticker_data = market_data[market_data['Ticker'] == ticker]

        # Plot Buy Signals
        buy_signals = ticker_data[ticker_data['MA_Signal'] == 'Buy']
        signals_chart.add_trace(go.Scatter(
            x=buy_signals['Date'], y=buy_signals['Price'], mode='markers',
            name=f"{ticker} Buy Signal", marker=dict(color='green', symbol='triangle-up', size=10)
        ))

        # Plot Sell Signals
        sell_signals = ticker_data[ticker_data['MA_Signal'] == 'Sell']
        signals_chart.add_trace(go.Scatter(
            x=sell_signals['Date'], y=sell_signals['Price'], mode='markers',
            name=f"{ticker} Sell Signal", marker=dict(color='red', symbol='triangle-down', size=10)
        ))

    signals_chart.update_layout(title="Buy and Sell Signals", xaxis_title="Date", yaxis_title="Price")

    return price_chart, signals_chart

# Run App
if __name__ == "__main__":
