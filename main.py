import os
import sys

# CUDA setup for Windows - this took forever to figure out lol
cuda_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64"
]
for p in cuda_paths:
    if os.path.exists(p) and p not in os.environ.get('PATH', ''):
        os.environ['PATH'] = p + os.pathsep + os.environ.get('PATH', '')

import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, time, timedelta
import pytz
import numpy as np
import pandas as pd

# Try to use GPU if available
try:
    import cupy as cp
    import warnings
    warnings.filterwarnings('ignore')

    # Quick test
    test = cp.zeros((100, 10))
    test2 = cp.exp(cp.array([[0.1, -0.2], [0.3, -0.1]]))
    cp.cuda.Device(0).synchronize()
    test2.get()

    # Custom CUDA kernel for Markov chains - way faster than Python loops
    markov_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void markov_chain_step(
        const float* trans_matrix,
        const float* rand_vals,
        int* states,
        const int n_sims,
        const int n_steps,
        const int n_states,
        const int init_state
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_sims) return;

        int curr = init_state;
        for (int step = 0; step < n_steps; step++) {
            float r = rand_vals[idx * n_steps + step];
            float cumsum = 0.0f;
            int next = 0;
            for (int s = 0; s < n_states; s++) {
                cumsum += trans_matrix[curr * n_states + s];
                if (r <= cumsum) {
                    next = s;
                    break;
                }
            }
            states[idx * n_steps + step] = next;
            curr = next;
        }
    }
    ''', 'markov_chain_step')

    print("GPU enabled (CuPy + CUDA kernels)")
    USE_GPU = True
except Exception as e:
    cp = None
    markov_kernel = None
    USE_GPU = False
    print("GPU disabled, using CPU")

EASTERN_TZ = pytz.timezone('US/Eastern')
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)


class StockPriceMCMC:
    def __init__(self, prices, n_simulations=1000, n_steps=30, train_ratio=0.8,
                 use_markov_chain=True, use_bootstrap=True, block_size=5,
                 drift_window=200, volatility_window=500, n_regimes=3,
                 transition_smoothing=1.0, enable_gpu=False):

        self.prices = prices
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.use_markov_chain = use_markov_chain
        self.use_bootstrap = use_bootstrap
        self.block_size = block_size
        self.drift_window = drift_window
        self.volatility_window = volatility_window
        self.n_regimes = n_regimes
        self.transition_smoothing = transition_smoothing
        self.enable_gpu = enable_gpu and USE_GPU

        # Use GPU array library if available
        self.xp = cp if self.enable_gpu else np

        # Split for backtesting
        split = int(len(prices) * train_ratio)
        self.train_prices = prices[:split]
        self.test_prices = prices[split:]

        # Store summaries
        self._markov_summary = None
        self._backtest_markov_summary = None

    def _bootstrap_returns(self, returns):
        """Bootstrap sampling - vectorized on GPU"""
        if hasattr(returns, 'values'):
            returns = returns.values
        else:
            returns = np.asarray(returns)

        if returns.size == 0:
            return np.zeros((self.n_simulations, self.n_steps))

        # GPU version
        if self.enable_gpu:
            xp = self.xp
            ret_gpu = xp.asarray(returns)

            if self.block_size <= 1 or returns.size < self.block_size:
                idx = xp.random.randint(0, returns.size, size=(self.n_simulations, self.n_steps))
                sampled = ret_gpu[idx]
            else:
                # Block bootstrap
                n_blocks = int(np.ceil(self.n_steps / self.block_size))
                starts = xp.random.randint(0, returns.size - self.block_size + 1,
                                          size=(self.n_simulations, n_blocks))
                offsets = xp.arange(self.block_size).reshape(1, 1, -1)
                all_idx = starts[:, :, xp.newaxis] + offsets
                all_idx = all_idx.reshape(self.n_simulations, -1)[:, :self.n_steps]
                sampled = ret_gpu[all_idx]

            # Adjust for recent drift and vol
            hist_mu = xp.mean(ret_gpu)
            recent_mu = xp.mean(ret_gpu[-self.drift_window:])
            hist_vol = xp.std(ret_gpu)
            recent_vol = xp.std(ret_gpu[-self.volatility_window:])

            centered = sampled - hist_mu
            if hist_vol > 0 and recent_vol > 0:
                centered = centered * (recent_vol / hist_vol)

            return centered + recent_mu

        # CPU version
        else:
            if self.block_size <= 1:
                idx = np.random.randint(0, returns.size, size=(self.n_simulations, self.n_steps))
                sampled = returns[idx]
            else:
                sampled = np.empty((self.n_simulations, self.n_steps))
                for sim in range(self.n_simulations):
                    pos = 0
                    while pos < self.n_steps:
                        start = np.random.randint(0, returns.size - self.block_size + 1)
                        block = returns[start:start + self.block_size]
                        length = min(self.n_steps - pos, self.block_size)
                        sampled[sim, pos:pos + length] = block[:length]
                        pos += length

            hist_mu = returns.mean()
            recent_mu = returns[-self.drift_window:].mean()
            hist_vol = returns.std()
            recent_vol = returns[-self.volatility_window:].std()

            centered = sampled - hist_mu
            if hist_vol > 0 and recent_vol > 0:
                centered = centered * (recent_vol / hist_vol)

            return centered + recent_mu

    def _parametric_returns(self, returns):
        """Simple normal distribution sampling"""
        if hasattr(returns, 'values'):
            returns = returns.values
        else:
            returns = np.asarray(returns)

        if returns.size == 0:
            return np.zeros((self.n_simulations, self.n_steps))

        mu = returns.mean()
        sigma = returns.std() if returns.size > 1 else 1e-6
        return np.random.normal(mu, sigma, size=(self.n_simulations, self.n_steps))

    def _infer_regimes(self, returns_series):
        """Figure out market regimes using quantiles"""
        if returns_series.empty or returns_series.nunique() < 2:
            return None, None, 0

        n = min(self.n_regimes, returns_series.nunique())
        if n < 2:
            return None, None, 0

        try:
            labels, edges = pd.qcut(returns_series, q=n, labels=False,
                                   retbins=True, duplicates='drop')
            labels = labels.astype(int)
            n_states = int(labels.max()) + 1
            return labels, edges, n_states if n_states >= 2 else (None, None, 0)
        except:
            return None, None, 0

    def _build_transition_matrix(self, states, n_states):
        """Build transition probability matrix"""
        counts = np.zeros((n_states, n_states))
        for i in range(len(states) - 1):
            counts[states[i], states[i+1]] += 1

        # Add smoothing
        if self.transition_smoothing > 0:
            counts += self.transition_smoothing

        # Normalize
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return counts / row_sums

    def _sample_markov_returns(self, returns):
        """Sample using Markov chain regime switching"""
        # Prep data
        if returns is None or len(returns) == 0:
            return None, None

        if isinstance(returns, pd.Series):
            ret_series = returns.dropna()
        else:
            ret_series = pd.Series(np.asarray(returns)).dropna()

        labels, edges, n_states = self._infer_regimes(ret_series)
        if labels is None:
            return None, None

        # Build transition matrix
        label_arr = labels.to_numpy()
        trans_mat = self._build_transition_matrix(label_arr, n_states)

        # Get return values for each state
        state_returns = []
        state_stats = {'means': [], 'stds': [], 'counts': []}
        for s in range(n_states):
            vals = ret_series[labels == s].values
            state_returns.append(vals)
            state_stats['means'].append(float(vals.mean()) if vals.size else 0.0)
            state_stats['stds'].append(float(vals.std()) if vals.size else 0.0)
            state_stats['counts'].append(len(vals))

        last_state = int(label_arr[-1])
        fallback = ret_series.values

        # GPU implementation
        if self.enable_gpu and markov_kernel is not None:
            xp = self.xp

            # Generate random values for state transitions
            rand_vals = xp.random.uniform(0, 1, (self.n_simulations, self.n_steps)).astype(xp.float32)
            trans_gpu = xp.asarray(trans_mat, dtype=xp.float32)
            states_gpu = xp.zeros((self.n_simulations, self.n_steps), dtype=xp.int32)

            # Launch kernel
            threads = 256
            blocks = (self.n_simulations + threads - 1) // threads
            markov_kernel((blocks,), (threads,),
                         (trans_gpu, rand_vals, states_gpu,
                          self.n_simulations, self.n_steps, n_states, last_state))
            xp.cuda.Stream.null.synchronize()

            # Map states to returns
            sampled = xp.zeros((self.n_simulations, self.n_steps), dtype=xp.float64)
            for s in range(n_states):
                vals = state_returns[s]
                if vals.size > 0:
                    mask = (states_gpu == s)
                    n_needed = int(xp.sum(mask))
                    if n_needed > 0:
                        idx = xp.random.randint(0, vals.size, size=n_needed)
                        sampled[mask] = xp.asarray(vals)[idx]

            summary = {
                'n_regimes': n_states,
                'state_means': state_stats['means'],
                'state_stds': state_stats['stds'],
                'state_counts': state_stats['counts'],
                'transition_matrix': trans_mat.tolist(),
                'last_state': last_state
            }
            return sampled, summary

        # CPU implementation
        else:
            sampled = np.zeros((self.n_simulations, self.n_steps))
            for sim in range(self.n_simulations):
                curr = last_state
                for step in range(self.n_steps):
                    probs = trans_mat[curr]
                    curr = np.random.choice(n_states, p=probs)
                    vals = state_returns[curr]
                    if vals.size > 0:
                        sampled[sim, step] = np.random.choice(vals)
                    elif fallback.size > 0:
                        sampled[sim, step] = np.random.choice(fallback)

            summary = {
                'n_regimes': n_states,
                'state_means': state_stats['means'],
                'state_stds': state_stats['stds'],
                'state_counts': state_stats['counts'],
                'transition_matrix': trans_mat.tolist(),
                'last_state': last_state
            }
            return sampled, summary

    def _generate_returns(self, returns, context='forecast'):
        """Generate returns using Markov or bootstrap"""
        summary = None
        sampled = None

        if self.use_markov_chain:
            sampled, summary = self._sample_markov_returns(returns)

        if sampled is None:
            sampled = self._bootstrap_returns(returns) if self.use_bootstrap else self._parametric_returns(returns)

        # Store summary
        if context == 'forecast':
            self._markov_summary = summary
        else:
            self._backtest_markov_summary = summary

        return sampled

    def _run_simulation(self, start_price, returns, context='forecast'):
        """Run Monte Carlo simulation"""
        xp = self.xp
        paths = xp.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = start_price

        sampled_returns = self._generate_returns(returns, context)

        # Convert to right format
        if self.enable_gpu and not isinstance(sampled_returns, cp.ndarray):
            sampled_returns = xp.asarray(sampled_returns)
        elif not self.enable_gpu and isinstance(sampled_returns, np.ndarray):
            pass
        else:
            sampled_returns = xp.asarray(sampled_returns)

        # Compute paths
        if self.enable_gpu:
            # Vectorized on GPU
            cum_rets = xp.exp(sampled_returns)
            cum_prod = xp.cumprod(cum_rets, axis=1)
            paths[:, 1:] = start_price * cum_prod
        else:
            # Loop on CPU
            for t in range(1, self.n_steps + 1):
                paths[:, t] = paths[:, t-1] * xp.exp(sampled_returns[:, t-1])

        # Convert to numpy if on GPU
        if self.enable_gpu:
            return cp.asnumpy(paths)
        return paths

    def simulate_future_paths(self):
        """Forecast future prices"""
        returns = np.log(self.prices / self.prices.shift(1)).dropna()
        last_price = self.prices.iloc[-1]
        return self._run_simulation(last_price, returns, 'forecast')

    def calculate_out_of_sample_error(self):
        """Backtest on test set"""
        if self.train_prices.empty:
            return 0, np.array([]), pd.Series(), np.array([])

        train_rets = np.log(self.train_prices / self.train_prices.shift(1)).dropna()
        last_train = self.train_prices.iloc[-1]
        paths = self._run_simulation(last_train, train_rets, 'backtest')

        test_len = min(len(self.test_prices), self.n_steps)
        if test_len == 0:
            return 0, np.array([]), pd.Series(), np.array([])

        test_actual = self.test_prices[:test_len]
        pred = paths[:, 1:test_len+1]

        rmse_per_path = np.sqrt(np.mean((pred - test_actual.values)**2, axis=1))
        avg_rmse = np.mean(rmse_per_path)

        return avg_rmse, rmse_per_path, test_actual, paths

    def get_markov_summary(self, context='forecast'):
        return self._backtest_markov_summary if context == 'backtest' else self._markov_summary


def fetch_stock_data(symbol, timeframe='1m', limit=100, start_date=None, end_date=None):
    """Get stock data from yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        kwargs = {'interval': timeframe}

        if start_date or end_date:
            if start_date:
                st = pd.to_datetime(start_date)
                if getattr(st, 'tzinfo', None):
                    st = st.tz_convert(None)
                kwargs['start'] = st
            if end_date:
                et = pd.to_datetime(end_date)
                if getattr(et, 'tzinfo', None):
                    et = et.tz_convert(None)
                kwargs['end'] = et
        else:
            # Default periods
            periods = {'1m': '7d', '5m': '60d', '15m': '60d', '30m': '60d',
                      '1h': '730d', '1d': '5y'}
            kwargs['period'] = periods.get(timeframe, '1mo')

        df = ticker.history(**kwargs)
        if limit and len(df) > limit:
            df = df.tail(limit)

        # Rename columns
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })

        return filter_rth(df)
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


def filter_rth(df):
    """Keep only regular trading hours"""
    if df.empty:
        return df

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(EASTERN_TZ)
    else:
        idx = idx.tz_convert(EASTERN_TZ)

    result = df.copy()
    result.index = idx
    result = result.between_time(RTH_OPEN.strftime('%H:%M'), RTH_CLOSE.strftime('%H:%M'))

    if not result.empty:
        result.index = result.index.tz_localize(None)
    return result


def create_chart(symbol='AAPL', timeframe='1m', limit=100, start_date=None,
                end_date=None, ax=None, n_simulations=None, use_gpu=None):
    """Create candlestick chart with Monte Carlo forecast"""

    effective_end = end_date or datetime.now()
    df = fetch_stock_data(symbol, timeframe, limit, start_date, effective_end)

    if len(df) == 0:
        raise ValueError(f"No data for {symbol}")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = ax.figure
        ax.clear()

    # Dark theme
    fig.patch.set_facecolor('#131722')
    ax.set_facecolor('#131722')
    ax.set_title(f'{symbol} with MCMC Forecast ({timeframe})', color='#D9D9D9', fontsize=14)
    ax.set_ylabel('Price', color='#D9D9D9')

    # Draw candles
    positions = np.arange(len(df))
    for pos, row in zip(positions, df.itertuples()):
        color = '#26A69A' if row.close >= row.open else '#EF5350'
        ax.plot([pos, pos], [row.low, row.high], color=color, linewidth=1.1, zorder=2)
        rect = Rectangle((pos - 0.3, min(row.open, row.close)), 0.6,
                        abs(row.close - row.open) or 1e-6,
                        facecolor=color, edgecolor=color, alpha=0.9, zorder=3)
        ax.add_patch(rect)

    # Setup MCMC
    use_gpu_flag = (use_gpu if use_gpu is not None else USE_GPU)
    n_sims = n_simulations if n_simulations else (50000 if use_gpu_flag else 1000)

    mcmc = StockPriceMCMC(df['close'], n_simulations=n_sims, n_steps=30,
                         use_markov_chain=True, n_regimes=3,
                         transition_smoothing=0.5, enable_gpu=use_gpu_flag)

    # Backtest
    oos_rmse, oos_rmse_paths, test_prices, backtest = mcmc.calculate_out_of_sample_error()

    # Forecast
    forecast = mcmc.simulate_future_paths()
    summary = mcmc.get_markov_summary()

    # Future time axis
    last_date = df.index[-1]
    freq_map = {'1m': '1min', '5m': '5min', '15m': '15min',
                '30m': '30min', '1h': '1h', '1d': '1d'}
    freq = freq_map.get(timeframe, '1min')
    future_pos = np.arange(len(df) - 1, len(df) - 1 + mcmc.n_steps + 1)

    # Plot test set if available
    if not test_prices.empty:
        test_pos = positions[-len(test_prices):]
        ax.plot(test_pos, test_prices, color='#00FF00', linewidth=2,
               linestyle='--', label='Test Actual')

    # Calculate percentiles on GPU if possible
    if use_gpu_flag and USE_GPU:
        forecast_gpu = cp.asarray(forecast)
        pct = cp.percentile(forecast_gpu, [5, 25, 50, 75, 95], axis=0)
        pct = cp.asnumpy(pct)
        exp_price = float(cp.asnumpy(cp.mean(forecast_gpu[:, -1])))

        # Plot sample paths
        sample_paths = cp.asnumpy(forecast_gpu[:50])
        for i in range(50):
            ax.plot(future_pos, sample_paths[i], color='#FFA500', alpha=0.08, linewidth=1)
    else:
        pct = np.percentile(forecast, [5, 25, 50, 75, 95], axis=0)
        exp_price = float(np.mean(forecast[:, -1]))

        for i in range(min(50, len(forecast))):
            ax.plot(future_pos, forecast[i], color='#FFA500', alpha=0.08, linewidth=1)

    # Plot bands
    ax.fill_between(future_pos, pct[1], pct[3], color='#FFA500', alpha=0.08,
                   label='25-75% Range')
    ax.fill_between(future_pos, pct[0], pct[4], color='#FF0000', alpha=0.02,
                   label='5-95% Range')
    ax.plot(future_pos, pct[2], color='#00BFFF', linewidth=3, label='Median', zorder=5)

    # Styling
    ax.grid(color='#232733')
    ax.tick_params(colors='#D9D9D9')
    for spine in ax.spines.values():
        spine.set_color('#232733')

    # X-axis labels
    tick_count = min(10, len(df))
    if tick_count > 0:
        tick_idx = np.linspace(0, len(df) - 1, tick_count, dtype=int)
        labels = [df.index[i].strftime('%Y-%m-%d %H:%M') for i in tick_idx]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(labels, rotation=45, ha='right', color='#D9D9D9')

    legend = ax.legend(loc='upper left', facecolor='none', edgecolor='none')
    if legend:
        for txt in legend.get_texts():
            txt.set_color('#D9D9D9')

    # Add info box
    info_lines = [
        "MCMC Forecast:",
        "- Orange: likely range",
        "- Red: extreme range",
        "- Blue: median path"
    ]
    if summary and summary.get('state_counts'):
        dom_idx = int(np.argmax(summary['state_counts']))
        dom_mean = summary['state_means'][dom_idx]
        info_lines.append(f"- Dominant regime: #{dom_idx} (μ={dom_mean:+.4f})")

    ax.text(0.01, 0.99, "\n".join(info_lines), transform=ax.transAxes,
           fontsize=10, color='#D9D9D9', verticalalignment='top',
           bbox=dict(facecolor=(0,0,0,0.5), edgecolor=(1,1,1,0.2), boxstyle='round'))

    fig.tight_layout()
    if standalone:
        plt.show()

    # Return stats
    stats = {
        'current_price': df['close'].iloc[-1],
        'percentiles': {5: pct[0,-1], 25: pct[1,-1], 50: pct[2,-1],
                       75: pct[3,-1], 95: pct[4,-1]},
        'expected_price': exp_price,
        'oos_rmse': float(oos_rmse) if oos_rmse else None,
        'oos_best': float(np.min(oos_rmse_paths)) if len(oos_rmse_paths) else None,
        'oos_worst': float(np.max(oos_rmse_paths)) if len(oos_rmse_paths) else None,
        'train_size': len(mcmc.train_prices),
        'test_size': len(mcmc.test_prices),
        'regime_summary': summary,
        'backend': 'cupy' if mcmc.enable_gpu else 'numpy',
        'n_simulations': mcmc.n_simulations,
    }

    if standalone:
        print(f"\nStats for {symbol} ({timeframe}):")
        print(f"Current: ${stats['current_price']:.2f}")
        print(f"5%: ${stats['percentiles'][5]:.2f}, 50%: ${stats['percentiles'][50]:.2f}, 95%: ${stats['percentiles'][95]:.2f}")
        print(f"Expected: ${stats['expected_price']:.2f}")
        print(f"Backend: {stats['backend']} ({stats['n_simulations']} sims)")
        if stats['oos_rmse']:
            print(f"OOS RMSE: ${stats['oos_rmse']:.2f}")

    return stats


class MCMCApp:
    """GUI for the forecaster"""

    TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '1d']
    PRESETS = {
        '1D': {'delta': timedelta(days=1), 'tf': '1m'},
        '5D': {'delta': timedelta(days=5), 'tf': '5m'},
        '1M': {'delta': timedelta(days=30), 'tf': '30m'},
        '3M': {'delta': timedelta(days=90), 'tf': '1h'},
        '6M': {'delta': timedelta(days=180), 'tf': '1h'},
        '1Y': {'delta': timedelta(days=365), 'tf': '1d'},
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('MCMC Stock Forecaster')

        self.ticker_var = tk.StringVar(value='NVDA')
        self.tf_var = tk.StringVar(value=self.TIMEFRAMES[0])
        self.limit_var = tk.StringVar(value='500')
        self.start_var = tk.StringVar(value=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))

        self._build_controls()
        self._build_presets()

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value='Ready')
        ttk.Label(self.root, textvariable=self.status_var, padding=(10, 5)).pack(anchor='w')

        self.root.after(100, self.update_chart)

    def _build_controls(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.X)

        ttk.Label(frame, text='Ticker').grid(row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Entry(frame, textvariable=self.ticker_var, width=10).grid(row=1, column=0, padx=5)

        ttk.Label(frame, text='Timeframe').grid(row=0, column=1, padx=5, pady=2, sticky='w')
        ttk.Combobox(frame, textvariable=self.tf_var, values=self.TIMEFRAMES,
                    state='readonly', width=8).grid(row=1, column=1, padx=5)

        ttk.Label(frame, text='Limit').grid(row=0, column=2, padx=5, pady=2, sticky='w')
        ttk.Entry(frame, textvariable=self.limit_var, width=12).grid(row=1, column=2, padx=5)

        ttk.Label(frame, text='Start Date').grid(row=0, column=3, padx=5, pady=2, sticky='w')
        ttk.Entry(frame, textvariable=self.start_var, width=15).grid(row=1, column=3, padx=5)

        ttk.Button(frame, text='Update', command=self.update_chart).grid(row=1, column=4, padx=10)

    def _build_presets(self):
        frame = ttk.Frame(self.root, padding=(10, 0))
        frame.pack(fill=tk.X)
        ttk.Label(frame, text='Quick ranges:').pack(side=tk.LEFT, padx=(0, 10))

        for label in self.PRESETS:
            ttk.Button(frame, text=label, command=lambda l=label: self._apply_preset(l),
                      width=6).pack(side=tk.LEFT, padx=3, pady=5)

    def _apply_preset(self, label):
        preset = self.PRESETS[label]
        start = datetime.now() - preset['delta']
        self.start_var.set(start.strftime('%Y-%m-%d'))
        self.tf_var.set(preset['tf'])
        self.limit_var.set('')
        self.update_chart()

    def update_chart(self):
        symbol = self.ticker_var.get().strip().upper() or 'NVDA'
        tf = self.tf_var.get()
        limit = self._parse_int(self.limit_var.get())
        start = self.start_var.get().strip() or None

        try:
            stats = create_chart(symbol=symbol, timeframe=tf, limit=limit,
                               start_date=start, end_date=datetime.now(), ax=self.ax)
            self.canvas.draw_idle()
            self.status_var.set(self._format_status(symbol, tf, stats))
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def _parse_int(self, val, default=None):
        if not val or not str(val).strip():
            return default
        try:
            n = int(val)
            return n if n > 0 else default
        except:
            return default

    def _format_status(self, sym, tf, stats):
        p = stats['percentiles']
        lines = [
            f"{sym} ({tf})",
            f"Price: ${stats['current_price']:.2f} | Med: ${p[50]:.2f} | Exp: ${stats['expected_price']:.2f}",
            f"Range: 5%=${p[5]:.2f} 25%=${p[25]:.2f} 75%=${p[75]:.2f} 95%=${p[95]:.2f}",
        ]
        if stats['oos_rmse']:
            lines.append(f"OOS RMSE: ${stats['oos_rmse']:.2f}")
        regime = stats.get('regime_summary')
        if regime and regime.get('state_counts'):
            counts = regime['state_counts']
            total = sum(counts) or 1
            dom = max(range(len(counts)), key=lambda i: counts[i])
            lines.append(f"Regimes: {len(counts)} (dom=#{dom}, μ={regime['state_means'][dom]:+.4f})")
        lines.append(f"{stats['backend']} ({stats['n_simulations']} sims) | train/test={stats['train_size']}/{stats['test_size']}")
        return ' | '.join(lines)

    def run(self):
        self.root.mainloop()


def benchmark(symbol='NVDA', tf='5m', n_sims=10000):
    """Compare GPU vs CPU"""
    import time

    print("=" * 60)
    print(f"Benchmarking {symbol} with {n_sims:,} simulations")
    print("=" * 60)

    df = fetch_stock_data(symbol, tf, limit=500)
    if len(df) == 0:
        print("No data - market closed?")
        return

    # CPU
    print("\nCPU test...")
    mcmc_cpu = StockPriceMCMC(df['close'], n_simulations=n_sims, n_steps=30,
                             use_markov_chain=True, enable_gpu=False)
    t0 = time.time()
    paths_cpu = mcmc_cpu.simulate_future_paths()
    cpu_time = time.time() - t0
    print(f"CPU: {cpu_time:.2f}s")

    # GPU
    if USE_GPU:
        print("\nGPU test...")
        mcmc_gpu = StockPriceMCMC(df['close'], n_simulations=n_sims, n_steps=30,
                                 use_markov_chain=True, enable_gpu=True)
        t0 = time.time()
        paths_gpu = mcmc_gpu.simulate_future_paths()
        if isinstance(paths_gpu, cp.ndarray):
            cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - t0
        print(f"GPU: {gpu_time:.2f}s")

        speedup = cpu_time / gpu_time
        print("\n" + "=" * 60)
        print(f"SPEEDUP: {speedup:.1f}x")
        print("=" * 60)
        print(f"Time saved: {cpu_time - gpu_time:.2f}s")
        print(f"Could run {int(speedup * n_sims):,} sims in same time on GPU!")
    else:
        print("\nGPU not available")


if __name__ == "__main__":
    # Uncomment to benchmark:
    # benchmark(symbol='NVDA', tf='5m', n_sims=10000)

    app = MCMCApp()
    app.run()
