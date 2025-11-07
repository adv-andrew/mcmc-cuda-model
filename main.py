import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
import numpy as np
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

def fetch_stock_data(symbol, timeframe='1m', limit=100, start_date=None, end_date=None):
    """Fetch historical stock data from yfinance with optional explicit date window."""
    try:
        ticker = yf.Ticker(symbol)

        history_kwargs = {'interval': timeframe}

        if start_date or end_date:
            if start_date:
                start_ts = pd.to_datetime(start_date)
                if start_ts.tzinfo is not None:
                    start_ts = start_ts.tz_localize(None)
                history_kwargs['start'] = start_ts
            if end_date:
                end_ts = pd.to_datetime(end_date)
                if end_ts.tzinfo is not None:
                    end_ts = end_ts.tz_localize(None)
                history_kwargs['end'] = end_ts
        else:
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
            history_kwargs['period'] = period

        df = ticker.history(**history_kwargs)

        if limit and len(df) > limit:
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

def create_candlestick_chart(
    symbol='AAPL',
    timeframe='1m',
    limit=100,
    start_date=None,
    end_date=None,
    ax=None,
):
    """Render candlesticks + Monte Carlo fan chart either standalone or on a provided Matplotlib axis."""

    df = fetch_stock_data(symbol, timeframe, limit, start_date=start_date, end_date=end_date)

    if len(df) == 0:
        raise ValueError(
            f"No data available for {symbol} (timeframe={timeframe}). The market might be closed or the symbol might be invalid."
        )

    show_plot = ax is None
    if show_plot:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = ax.figure
        ax.clear()

    fig.patch.set_facecolor('#131722')
    ax.set_facecolor('#131722')
    ax.set_title(
        f'{symbol} Stock Price with Monte Carlo Forecast ({timeframe} data)',
        color='#D9D9D9',
        fontsize=14,
    )
    ax.set_ylabel('Price', color='#D9D9D9')

    date_nums = mdates.date2num(df.index.to_pydatetime())
    if len(date_nums) > 1:
        candle_width = (date_nums[1] - date_nums[0]) * 0.6
    else:
        candle_width = 0.0005

    for date_num, row in zip(date_nums, df.itertuples()):
        color = '#26A69A' if row.close >= row.open else '#EF5350'
        ax.plot([date_num, date_num], [row.low, row.high], color=color, linewidth=1.1, zorder=2)
        body_height = row.close - row.open
        body_height = body_height if body_height != 0 else 1e-6
        rect = Rectangle(
            (date_num - candle_width / 2, min(row.open, row.close)),
            candle_width,
            abs(body_height),
            facecolor=color,
            edgecolor=color,
            alpha=0.9,
            zorder=3,
        )
        ax.add_patch(rect)
    
    # Calculate and plot EMAs using TradingView style (proper initialization)
    if len(df) >= 8:
        ema_8 = calculate_ema_tradingview_style(df['close'], 8)
        ax.plot(df.index, ema_8, color='yellow', linewidth=2, label='EMA 8')
    
    if len(df) >= 21:
        ema_21 = calculate_ema_tradingview_style(df['close'], 21)
        ax.plot(df.index, ema_21, color='orange', linewidth=2, label='EMA 21')
    
    if len(df) >= 50:
        ema_50 = calculate_ema_tradingview_style(df['close'], 50)
        ax.plot(df.index, ema_50, color='blue', linewidth=2, label='EMA 50')
    
    if len(df) >= 200:
        ema_200 = calculate_ema_tradingview_style(df['close'], 200)
        ax.plot(df.index, ema_200, color='purple', linewidth=2, label='EMA 200')
    
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
    freq_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1h',
        '1d': '1d',
    }
    freq = freq_map.get(timeframe, '1min')
    future_dates = pd.date_range(start=last_date, periods=mcmc.n_steps + 1, freq=freq)

    # Plot the actual values from the test set for comparison
    if not test_prices.empty:
        test_dates = df.index[-len(test_prices):]
        ax.plot(test_dates, test_prices, color='#00FF00', linewidth=2, linestyle='--', label='Test Set Actual')

    # Plot the forecast paths
    for i in range(min(50, forecast_paths.shape[0])):
        ax.plot(future_dates, forecast_paths[i], color='#FFA500', alpha=0.08, linewidth=1)
    

    percentiles = np.percentile(forecast_paths, [5, 25, 50, 75, 95], axis=0)
    median_path = percentiles[2]


    ax.fill_between(
        future_dates,
        percentiles[1],
        percentiles[3],
        color='#FFA500',
        alpha=0.08,
        label='25th-75th Percentile (Likely)'
    )
    ax.fill_between(
        future_dates,
        percentiles[0],
        percentiles[4],
        color='#FF0000',
        alpha=0.02,
        label='5th-95th Percentile (Extreme)'
    )

    ax.plot(future_dates, median_path, color='#00BFFF', linewidth=3, label='Median Forecast', zorder=5)

    ax.grid(color='#232733')
    ax.tick_params(colors='#D9D9D9')
    for spine in ax.spines.values():
        spine.set_color('#232733')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()

    legend = ax.legend(loc='upper left', facecolor='none', edgecolor='none')
    if legend:
        for text in legend.get_texts():
            text.set_color('#D9D9D9')

    annotation_text = (
        "Confidence Intervals (Forecast):\n"
        "• Orange band: 25-75th percentile (likely range)\n"
        "• Red band: 5-95th percentile (extreme range)\n"
        "• Blue line: Median path (central tendency)"
    )
    ax.text(
        0.01,
        0.99,
        annotation_text,
        transform=ax.transAxes,
        fontsize=10,
        color='#D9D9D9',
        verticalalignment='top',
        bbox=dict(facecolor=(0, 0, 0, 0.5), edgecolor=(1, 1, 1, 0.2), boxstyle='round,pad=0.5')
    )

    fig.tight_layout()
    if show_plot:
        plt.show()

    stats = {
        'current_price': df['close'].iloc[-1],
        'percentiles': {
            5: percentiles[0, -1],
            25: percentiles[1, -1],
            50: percentiles[2, -1],
            75: percentiles[3, -1],
            95: percentiles[4, -1],
        },
        'expected_price': float(np.mean(forecast_paths[:, -1])),
        'oos_rmse': float(oos_rmse) if oos_rmse is not None else None,
        'oos_best': float(np.min(oos_rmse_per_path)) if len(oos_rmse_per_path) else None,
        'oos_worst': float(np.max(oos_rmse_per_path)) if len(oos_rmse_per_path) else None,
        'train_minutes': len(mcmc.train_prices),
        'test_minutes': len(mcmc.test_prices),
    }

    if show_plot:
        print("\nMCMC Prediction Statistics (30 minutes):")
        print(f"Current Price: ${stats['current_price']:.2f}")
        print(f"5th Percentile: ${stats['percentiles'][5]:.2f}")
        print(f"25th Percentile: ${stats['percentiles'][25]:.2f}")
        print(f"Median Forecast: ${stats['percentiles'][50]:.2f}")
        print(f"75th Percentile: ${stats['percentiles'][75]:.2f}")
        print(f"95th Percentile: ${stats['percentiles'][95]:.2f}")
        print(f"Expected Price: ${stats['expected_price']:.2f}")
        print("\nOut-of-Sample Error Statistics (from Backtest):")
        if stats['oos_rmse'] is not None and stats['oos_rmse'] > 0:
            print(f"Average OOS RMSE: ${stats['oos_rmse']:.2f}")
            print(f"Best Path OOS RMSE: ${stats['oos_best']:.2f}")
            print(f"Worst Path OOS RMSE: ${stats['oos_worst']:.2f}")
        else:
            print("Not enough data to calculate out-of-sample error.")
        print(f"Training Set Size: {stats['train_minutes']} minutes")
        print(f"Test Set Size: {stats['test_minutes']} minutes")

    return stats


class MCMCApp:
    """Tkinter front-end for interactive Monte Carlo visualization."""

    TIMEFRAME_CHOICES = ['1m', '5m', '15m', '30m', '1h', '1d']
    QUICK_RANGES = {
        'Last 1 Day': timedelta(days=1),
        'Last 5 Days': timedelta(days=5),
        'Last 1 Month': timedelta(days=30),
        'Last 3 Months': timedelta(days=90),
        'Last 6 Months': timedelta(days=180),
        'Last 1 Year': timedelta(days=365),
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Monte Carlo Stock Forecaster')

        self.ticker_var = tk.StringVar(value='NVDA')
        self.timeframe_var = tk.StringVar(value=self.TIMEFRAME_CHOICES[0])
        self.limit_var = tk.StringVar(value='500')
        self.start_date_var = tk.StringVar(value=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
        self.end_date_var = tk.StringVar(value='')
        self.quick_range_var = tk.StringVar(value='Last 7 Days')

        self._build_controls()

        self.figure, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.stats_var = tk.StringVar(value='Ready.')
        ttk.Label(self.root, textvariable=self.stats_var, padding=(10, 5)).pack(anchor='w')

        self.root.after(100, self.update_chart)

    def _build_controls(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)

        ttk.Label(control_frame, text='Ticker').grid(row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Entry(control_frame, textvariable=self.ticker_var, width=10).grid(row=1, column=0, padx=5)

        ttk.Label(control_frame, text='Timeframe').grid(row=0, column=1, padx=5, pady=2, sticky='w')
        timeframe_combo = ttk.Combobox(control_frame, textvariable=self.timeframe_var, values=self.TIMEFRAME_CHOICES, state='readonly', width=8)
        timeframe_combo.grid(row=1, column=1, padx=5)

        ttk.Label(control_frame, text='Max candles').grid(row=0, column=2, padx=5, pady=2, sticky='w')
        ttk.Entry(control_frame, textvariable=self.limit_var, width=10).grid(row=1, column=2, padx=5)

        ttk.Label(control_frame, text='Start date (YYYY-MM-DD)').grid(row=0, column=3, padx=5, pady=2, sticky='w')
        ttk.Entry(control_frame, textvariable=self.start_date_var, width=15).grid(row=1, column=3, padx=5)

        ttk.Label(control_frame, text='End date (optional)').grid(row=0, column=4, padx=5, pady=2, sticky='w')
        ttk.Entry(control_frame, textvariable=self.end_date_var, width=15).grid(row=1, column=4, padx=5)

        quick_values = ['Last 7 Days'] + list(self.QUICK_RANGES.keys())
        ttk.Label(control_frame, text='Quick range').grid(row=0, column=5, padx=5, pady=2, sticky='w')
        quick_combo = ttk.Combobox(control_frame, textvariable=self.quick_range_var, values=quick_values, state='readonly', width=15)
        quick_combo.grid(row=1, column=5, padx=5)
        quick_combo.bind('<<ComboboxSelected>>', self._apply_quick_range)

        ttk.Button(control_frame, text='Update Chart', command=self.update_chart).grid(row=1, column=6, padx=10)

        for col in range(7):
            control_frame.columnconfigure(col, weight=1)

    def _apply_quick_range(self, _event=None):
        selection = self.quick_range_var.get()
        if selection == 'Last 7 Days':
            start = datetime.now() - timedelta(days=7)
        else:
            delta = self.QUICK_RANGES.get(selection)
            if not delta:
                return
            start = datetime.now() - delta
        self.start_date_var.set(start.strftime('%Y-%m-%d'))
        self.end_date_var.set('')

    def update_chart(self):
        symbol = self.ticker_var.get().strip().upper() or 'NVDA'
        timeframe = self.timeframe_var.get()
        limit = self._parse_int(self.limit_var.get(), default=500)
        start_date = self.start_date_var.get().strip() or None
        end_date = self.end_date_var.get().strip() or None

        try:
            stats = create_candlestick_chart(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                start_date=start_date,
                end_date=end_date,
                ax=self.ax,
            )
            self.canvas.draw_idle()
            self.stats_var.set(self._format_stats(symbol, timeframe, stats))
        except Exception as exc:
            messagebox.showerror('Unable to update chart', str(exc))

    @staticmethod
    def _parse_int(value, default=500):
        try:
            parsed = int(value)
            return parsed if parsed > 0 else default
        except Exception:
            return default

    @staticmethod
    def _format_stats(symbol, timeframe, stats):
        pct = stats['percentiles']
        lines = [
            f"{symbol} ({timeframe})",
            f"Price: ${stats['current_price']:.2f} | Median: ${pct[50]:.2f} | Expected: ${stats['expected_price']:.2f}",
            f"Bands: 5% ${pct[5]:.2f} 25% ${pct[25]:.2f} 75% ${pct[75]:.2f} 95% ${pct[95]:.2f}",
        ]
        if stats['oos_rmse'] is not None and stats['oos_rmse'] > 0:
            lines.append(
                f"OOS RMSE avg ${stats['oos_rmse']:.2f} (best ${stats['oos_best']:.2f}, worst ${stats['oos_worst']:.2f})"
            )
        else:
            lines.append('OOS RMSE unavailable (insufficient test data)')
        lines.append(f"Train/Test minutes: {stats['train_minutes']}/{stats['test_minutes']}")
        return ' | '.join(lines)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = MCMCApp()
    app.run()
