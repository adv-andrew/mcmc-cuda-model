import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import numpy as np
from scipy.stats import norm
import pandas as pd

# Remove Alpaca API credentials and initialization
# API_KEY = 'PK2RFDCTW8EM2NNCVD5Y'
# SECRET_KEY = 'sjaQsdQausmx3APtMsN2uMbDmhIMDolqjPqrMlAr'
# BASE_URL = 'https://paper-api.alpaca.markets'  # For trading
# DATA_URL = 'https://data.alpaca.markets'  # For market data

# Initialize Alpaca API
# api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
# data_api = tradeapi.REST(API_KEY, SECRET_KEY, DATA_URL, api_version='v2')

class StockPriceMCMC:
    def __init__(
        self,
        prices,
        n_simulations=1000,
        n_steps=30,
        train_ratio=0.8,
        use_bootstrap=True,
        block_size=5,
        drift_window=200,
        volatility_window=500,
    ):
        self.prices = prices
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.use_bootstrap = use_bootstrap
        self.block_size = max(1, block_size)
        self.drift_window = max(1, drift_window)
        self.volatility_window = max(1, volatility_window)

        # Split data for backtesting
        split_idx = int(len(prices) * train_ratio)
        self.train_prices = prices[:split_idx]
        self.test_prices = prices[split_idx:]

    def _bootstrap_returns(self, returns_source):
        """Sample returns via (optionally block) bootstrap to preserve distributional features."""
        if hasattr(returns_source, 'values'):
            returns_array = returns_source.values
        else:
            returns_array = np.asarray(returns_source)

        if returns_array.size == 0:
            return np.zeros((self.n_simulations, self.n_steps))

        if self.block_size <= 1 or returns_array.size < self.block_size:
            idx = np.random.randint(0, returns_array.size, size=(self.n_simulations, self.n_steps))
            sampled = returns_array[idx]
        else:
            sampled = np.empty((self.n_simulations, self.n_steps))
            for sim in range(self.n_simulations):
                steps = 0
                while steps < self.n_steps:
                    start = np.random.randint(0, returns_array.size - self.block_size + 1)
                    block = returns_array[start:start + self.block_size]
                    block_len = min(self.n_steps - steps, self.block_size)
                    sampled[sim, steps:steps + block_len] = block[:block_len]
                    steps += block_len

        historical_mu = returns_array.mean()
        recent_slice = returns_array[-min(self.drift_window, returns_array.size):]
        recent_mu = recent_slice.mean()
        recent_vol_slice = returns_array[-min(self.volatility_window, returns_array.size):]
        historical_vol = returns_array.std(ddof=0)
        recent_vol = recent_vol_slice.std(ddof=0)

        centered = sampled - historical_mu
        if historical_vol > 0 and recent_vol > 0:
            scale = recent_vol / historical_vol
            centered = centered * scale

        adjusted = centered + recent_mu
        return adjusted

    def _parametric_returns(self, returns_source):
        returns_array = returns_source.values if hasattr(returns_source, 'values') else np.asarray(returns_source)
        if returns_array.size == 0:
            return np.zeros((self.n_simulations, self.n_steps))

        mu = returns_array.mean()
        sigma = returns_array.std(ddof=1) if returns_array.size > 1 else 0
        if sigma == 0:
            sigma = 1e-6
        return np.random.normal(loc=mu, scale=sigma, size=(self.n_simulations, self.n_steps))

    def _run_simulation(self, start_price, returns_source):
        """Simulate geometric price paths from sampled returns."""
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = start_price

        sampled_returns = (
            self._bootstrap_returns(returns_source)
            if self.use_bootstrap
            else self._parametric_returns(returns_source)
        )

        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(sampled_returns[:, t - 1])

        return paths

    def simulate_future_paths(self):
        """
        Simulate future price paths using the ENTIRE dataset for forecasting.
        """

        all_returns = np.log(self.prices / self.prices.shift(1)).dropna()
        last_price = self.prices.iloc[-1]

        return self._run_simulation(last_price, all_returns)

    def calculate_out_of_sample_error(self):
        """
        Calculate out-of-sample error by comparing a backtest simulation with the test set.
        """

        train_returns = np.log(self.train_prices / self.train_prices.shift(1)).dropna()
        
        if self.train_prices.empty:
            return 0, np.array([]), pd.Series(dtype='float64'), np.array([])
            
        train_last_price = self.train_prices.iloc[-1]

        backtest_paths = self._run_simulation(train_last_price, train_returns)

        test_length = min(len(self.test_prices), self.n_steps)
        if test_length == 0:
            return 0, np.array([]), pd.Series(dtype='float64'), np.array([])

        test_prices_actual = self.test_prices[:test_length]
        predicted_prices_steps = backtest_paths[:, 1:test_length + 1]
        
        rmse_per_path = np.sqrt(np.mean((predicted_prices_steps - test_prices_actual.values) ** 2, axis=1))
        avg_rmse = np.mean(rmse_per_path)
        
        return avg_rmse, rmse_per_path, test_prices_actual, backtest_paths

def calculate_ema_tradingview_style(prices, period):
    """
    Calculate EMA using TradingView style initialization
    - First EMA value = first price
    - Subsequent values use standard EMA formula
    """
    if len(prices) == 0:
        return pd.Series(dtype='float64')
    
    alpha = 2 / (period + 1)
    ema_values = np.zeros(len(prices))
    ema_values[0] = prices.iloc[0]
    
    for i in range(1, len(prices)):
        ema_values[i] = alpha * prices.iloc[i] + (1 - alpha) * ema_values[i-1]
    
    return pd.Series(ema_values, index=prices.index)

def fetch_stock_data(symbol, timeframe='1m', limit=100):
    """
    Fetch historical stock data from yfinance
    """
    try:
        ticker = yf.Ticker(symbol)

        if timeframe == '1m':
            period = '7d'
        elif timeframe == '5m':
            period = '60d'
        elif timeframe == '15m':
            period = '60d'
        elif timeframe == '30m':
            period = '60d'
        elif timeframe == '1h':
            period = '730d'
        elif timeframe == '1d':
            period = '5y'
        else:
            period = '1mo'  
        
        df = ticker.history(period=period, interval=timeframe)

        if len(df) > limit:
            df = df.tail(limit)
        
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

def create_candlestick_chart(symbol='AAPL', timeframe='1m', limit=100, mcmc_days=30):
    """
    Create an interactive candlestick chart using plotly with MCMC predictions
    """
    try:
        # Fetch data
        df = fetch_stock_data(symbol, timeframe, limit)
        
        if len(df) == 0:
            print(f"No data available for {symbol}. The market might be closed or the symbol might be invalid.")
            return
        
        # Create the candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_color='#26A69A',  # Green candles
            decreasing_line_color='#EF5350',  # Red candles,
            name='Historical Data'
        )])
        
        # Calculate and plot EMAs using TradingView style (proper initialization)
        if len(df) >= 8:
            ema_8 = calculate_ema_tradingview_style(df['close'], 8)
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=ema_8, 
                mode='lines', 
                name='EMA 8', 
                line=dict(color='yellow', width=2)
            ))
        
        if len(df) >= 21:
            ema_21 = calculate_ema_tradingview_style(df['close'], 21)
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=ema_21, 
                mode='lines', 
                name='EMA 21', 
                line=dict(color='orange', width=2)
            ))
        
        if len(df) >= 50:
            ema_50 = calculate_ema_tradingview_style(df['close'], 50)
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=ema_50, 
                mode='lines', 
                name='EMA 50', 
                line=dict(color='blue', width=2)
            ))
        
        if len(df) >= 200:
            ema_200 = calculate_ema_tradingview_style(df['close'], 200)
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=ema_200, 
                mode='lines', 
                name='EMA 200', 
                line=dict(color='purple', width=2)
            ))
        
        # Initialize the MCMC model
        mcmc = StockPriceMCMC(df['close'], n_simulations=100, n_steps=30)
        
        # --- Backtesting ---
        # Calculate out-of-sample error for statistics, but don't plot these paths
        oos_rmse, oos_rmse_per_path, test_prices, backtest_paths = mcmc.calculate_out_of_sample_error()
        
        # --- Forecasting ---
        # Run a new simulation from the last actual price for plotting
        forecast_paths = mcmc.simulate_future_paths()

        # Create future dates for the forecast paths
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=mcmc.n_steps + 1, freq='1min')

        # Plot the actual values from the test set for comparison
        if not test_prices.empty:
            test_dates = df.index[-len(test_prices):]
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=test_prices,
                mode='lines',
                line=dict(color='#00FF00', width=2, dash='dash'),
                name='Test Set Actual'
            ))

        # Plot the forecast paths
        for i in range(min(50, forecast_paths.shape[0])):
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast_paths[i],
                mode='lines',
                line=dict(color='rgba(255, 165, 0, 0.1)', width=1), 
                name=f'Forecast Path {i+1}',
                showlegend=False 
            ))
        

        percentiles = np.percentile(forecast_paths, [5, 25, 50, 75, 95], axis=0)
        median_path = percentiles[2]


        fig.add_trace(go.Scatter(
            x=future_dates,
            y=percentiles[1],
            fill=None,
            mode='lines',
            line=dict(
                color='rgba(255, 165, 0, 0.08)', 
                width=1
            ),
            name='25th Percentile (Likely Range)',
            legendgroup='confidence_intervals',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=percentiles[3],
            fill='tonexty',
            mode='lines',
            line=dict(
                color='rgba(255, 165, 0, 0.08)', 
                width=1
            ),
            name='75th Percentile (Likely Range)',
            legendgroup='confidence_intervals',
            showlegend=False
        ))


        fig.add_trace(go.Scatter(
            x=future_dates,
            y=percentiles[0],
            fill=None,
            mode='lines',
            line=dict(
                color='rgba(255, 0, 0, 0.03)',
                width=1
            ),
            name='5th Percentile (Extreme Range)',
            legendgroup='confidence_intervals',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=percentiles[4],
            fill='tonexty',
            mode='lines',
            line=dict(
                color='rgba(255, 0, 0, 0.03)',
                width=1
            ),
            name='95th Percentile (Extreme Range)',
            legendgroup='confidence_intervals',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=median_path,
            mode='lines',
            line=dict(color='#00BFFF', width=3),
            name='Median Forecast',
            legendgroup='median_forecast'
        ))
        
        # Update the layout
        fig.update_layout(
            title=f'{symbol} Stock Price with MCMC Forecast (1-Minute Data)',
            yaxis_title='Price',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            plot_bgcolor='#131722',
            paper_bgcolor='#131722',
            font=dict(color='#D9D9D9'),
            yaxis=dict(
                gridcolor='#232733',
                zerolinecolor='#232733',
            ),
            xaxis=dict(
                gridcolor='#232733',
                zerolinecolor='#232733',
                type='date',
                tickformat='%H:%M:%S'
            ),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)',
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            annotations=[
                dict(
                    text="Confidence Intervals (Forecast):<br>" +
                         "• Orange band: 25-75th percentile (likely range)<br>" +
                         "• Red band: 5-95th percentile (extreme range)<br>" +
                         "• Blue line: Median path (central tendency)",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=0.99,
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="rgba(255,255,255,0.2)",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(size=12)
                )
            ]
        )
        
        # # Add volume bars
        # fig.add_trace(go.Bar(
        #     x=df.index,
        #     y=df['volume'],
        #     name='Volume',
        #     marker_color='#2962FF',
        #     opacity=0.5,
        #     yaxis='y2'
        # ))
        
        # # Add secondary y-axis for volume
        # fig.update_layout(
        #     yaxis2=dict(
        #         title='Volume',
        #         overlaying='y',
        #         side='right',
        #         showgrid=False,
        #     )
        # )
        
        # Show the plot
        fig.show()

        print("\nMCMC Prediction Statistics (30 minutes):")
        print(f"Current Price: ${df['close'].iloc[-1]:.2f}")
        print(f"5th Percentile: ${percentiles[0, -1]:.2f}")
        print(f"25th Percentile: ${percentiles[1, -1]:.2f}")
        print(f"Median Forecast: ${percentiles[2, -1]:.2f}")
        print(f"75th Percentile: ${percentiles[3, -1]:.2f}")
        print(f"95th Percentile: ${percentiles[4, -1]:.2f}")
        print(f"Expected Price: ${np.mean(forecast_paths[:, -1]):.2f}")
        print("\nOut-of-Sample Error Statistics (from Backtest):")
        if oos_rmse > 0:
            print(f"Average OOS RMSE: ${oos_rmse:.2f}")
            print(f"Best Path OOS RMSE: ${np.min(oos_rmse_per_path):.2f}")
            print(f"Worst Path OOS RMSE: ${np.max(oos_rmse_per_path):.2f}")
        else:
            print("Not enough data to calculate out-of-sample error.")
        print(f"Training Set Size: {len(mcmc.train_prices)} minutes")
        print(f"Test Set Size: {len(mcmc.test_prices)} minutes")
        
    except Exception as e:
        print(f"Error creating chart: {str(e)}")

if __name__ == "__main__":
    create_candlestick_chart('SPY', '1m', 300, mcmc_days=30) 
