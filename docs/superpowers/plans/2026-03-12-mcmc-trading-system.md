# MCMC Trading System Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GPU-accelerated trading system combining MCMC forecasting with LLM validation and parameter optimization.

**Architecture:** Modular pipeline - Signal Layer (MCMC + LLM) feeds Execution Layer (risk/position/broker) with parameters tuned by Optimization Layer (backtester + Optuna).

**Tech Stack:** Python 3.11, CuPy/CUDA, Optuna, PyTorch, yfinance, TradingAgents

---

## Chunk 1: Foundation & Data Layer

### Task 1: Project Setup

**Files:**
- Create: `trading/__init__.py`
- Create: `brokers/__init__.py`
- Create: `backtesting/__init__.py`
- Create: `optimization/__init__.py`
- Create: `integrations/__init__.py`
- Create: `config/default.yaml`
- Create: `config/tickers.yaml`
- Modify: `requirements.txt`
- Create: `CLAUDE.md`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p trading brokers backtesting optimization integrations config data/cache data/results scripts tests
touch tests/__init__.py
```

- [ ] **Step 2: Create package init files**

```python
# trading/__init__.py
"""Trading signal generation modules."""

# brokers/__init__.py
"""Broker interface implementations."""

# backtesting/__init__.py
"""GPU-accelerated backtesting engine."""

# optimization/__init__.py
"""Parameter optimization with Optuna."""

# integrations/__init__.py
"""External system integrations."""
```

- [ ] **Step 3: Create config/default.yaml**

```yaml
# MCMC Indicator Settings
mcmc:
  n_simulations: 25000
  n_steps: 30
  n_regimes: 3
  transition_smoothing: 0.5

# Signal Thresholds
signal:
  slope_threshold: 10.0  # degrees
  mtf_min_alignment: 3
  band_width_max: 0.05  # 5%
  regime_confidence_min: 0.6
  signal_strength_min: 0.7
  mtf_weight: 0.3
  regime_weight: 0.3

# Risk Management
risk:
  position_size_pct: 0.10  # 10% of portfolio
  max_position_pct: 0.25   # 25% max single position
  stop_loss_pct: 0.02      # 2%
  take_profit_pct: 0.05    # 5%
  daily_loss_limit_pct: 0.05  # 5% daily max loss

# Backtesting
backtest:
  train_months: 6
  test_months: 2
  slippage_pct: 0.0002  # 0.02%
  market_impact_pct: 0.0001  # 0.01%
  min_trades: 100

# LLM Integration
llm:
  daily_budget_usd: 2.0
  signal_threshold: 0.75  # Only call LLM above this
  max_calls_per_day: 3

# Timeframes
timeframes:
  - "1m"
  - "5m"
  - "15m"
  - "1h"
  - "1d"
```

- [ ] **Step 4: Create config/tickers.yaml**

```yaml
watchlist:
  - NVDA
  - AAPL
  - MSFT
  - GOOGL
  - AMZN
  - META
  - TSLA
  - SPY
```

- [ ] **Step 5: Update requirements.txt**

```
yfinance>=0.2.18
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
pytz>=2023.3
cupy-cuda12x>=12.0.0
streamlit>=1.28.0
optuna>=3.5.0
torch>=2.0.0
pyarrow>=14.0.0
pyyaml>=6.0.0
```

- [ ] **Step 6: Create CLAUDE.md**

```markdown
# MCMC Trading System

## Quick Start
- Run GUI: `python main.py`
- Run backtest: `python scripts/run_backtest.py`
- Optimize params: `python scripts/run_optimizer.py`
- Paper trade: `python scripts/run_paper_trader.py`

## Architecture
- `trading/` - Signal generation (MCMC indicator, combiner)
- `brokers/` - Execution (paper, alpaca)
- `backtesting/` - GPU-accelerated backtest engine
- `optimization/` - Optuna parameter search
- `integrations/` - TradingAgents LLM bridge

## Conventions
- GPU arrays: use CuPy, fall back to NumPy if unavailable
- Config: YAML for defaults, JSON for optimizer output
- Data: Parquet format, cached in data/cache/
- Dates: UTC internally, convert to Eastern for display

## Testing
- Run tests: `pytest tests/ -v`
- Backtest before changing params
- Walk-forward validation, not single split
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: project structure and config files"
```

---

### Task 2: Data Loader Module

**Files:**
- Create: `backtesting/data_loader.py`
- Create: `tests/test_data_loader.py`

- [ ] **Step 1: Write failing test for DataLoader**

```python
# tests/test_data_loader.py
import pytest
from datetime import datetime, timedelta
from backtesting.data_loader import DataLoader

def test_data_loader_fetch_single_ticker():
    loader = DataLoader(cache_dir="data/cache")
    df = loader.fetch("AAPL", "1d", days=30)

    assert not df.empty
    assert "open" in df.columns
    assert "close" in df.columns
    assert len(df) >= 15  # At least 15 trading days

def test_data_loader_caches_data():
    loader = DataLoader(cache_dir="data/cache")

    # First fetch
    df1 = loader.fetch("MSFT", "1d", days=30)

    # Second fetch should use cache
    df2 = loader.fetch("MSFT", "1d", days=30)

    assert len(df1) == len(df2)

def test_data_loader_multiple_timeframes():
    loader = DataLoader(cache_dir="data/cache")

    data = loader.fetch_multi_timeframe("SPY", ["1h", "1d"], days=30)

    assert "1h" in data
    assert "1d" in data
    assert not data["1h"].empty
    assert not data["1d"].empty
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_data_loader.py -v
```
Expected: FAIL - module not found

- [ ] **Step 3: Implement DataLoader**

```python
# backtesting/data_loader.py
"""Historical data fetching and caching."""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
import pytz

EASTERN_TZ = pytz.timezone('US/Eastern')


class DataLoader:
    """Fetches and caches historical price data."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str, timeframe: str) -> Path:
        return self.cache_dir / f"{ticker}_{timeframe}.parquet"

    def _load_cache(self, ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(ticker, timeframe)
        if path.exists():
            df = pd.read_parquet(path)
            # Check if cache is fresh (less than 1 hour old for intraday)
            if timeframe in ["1m", "5m", "15m", "30m", "1h"]:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if datetime.now() - mtime < timedelta(hours=1):
                    return df
            else:
                # Daily data: cache valid for 24 hours
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if datetime.now() - mtime < timedelta(hours=24):
                    return df
        return None

    def _save_cache(self, df: pd.DataFrame, ticker: str, timeframe: str):
        path = self._cache_path(ticker, timeframe)
        df.to_parquet(path)

    def fetch(
        self,
        ticker: str,
        timeframe: str = "1d",
        days: int = 365,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Fetch historical data for a single ticker."""

        # Try cache first
        if use_cache:
            cached = self._load_cache(ticker, timeframe)
            if cached is not None:
                return self._filter_date_range(cached, days, start_date, end_date)

        # Fetch from yfinance
        ticker_obj = yf.Ticker(ticker)

        # Determine period based on timeframe
        if start_date and end_date:
            df = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval=timeframe
            )
        else:
            period_map = {
                "1m": "7d",
                "5m": "60d",
                "15m": "60d",
                "30m": "60d",
                "1h": "730d",
                "1d": "max"
            }
            period = period_map.get(timeframe, "1y")
            df = ticker_obj.history(period=period, interval=timeframe)

        if df.empty:
            return df

        # Normalize column names
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Filter to RTH for intraday
        if timeframe in ["1m", "5m", "15m", "30m", "1h"]:
            df = self._filter_rth(df)

        # Cache the data
        if use_cache and not df.empty:
            self._save_cache(df, ticker, timeframe)

        return self._filter_date_range(df, days, start_date, end_date)

    def _filter_rth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to regular trading hours (9:30-16:00 ET)."""
        if df.empty:
            return df

        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize(EASTERN_TZ)
        else:
            idx = idx.tz_convert(EASTERN_TZ)

        result = df.copy()
        result.index = idx
        result = result.between_time("09:30", "16:00")

        if not result.empty:
            result.index = result.index.tz_localize(None)

        return result

    def _filter_date_range(
        self,
        df: pd.DataFrame,
        days: int,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Filter dataframe to date range."""
        if df.empty:
            return df

        if start_date and end_date:
            mask = (df.index >= start_date) & (df.index <= end_date)
            return df[mask]

        if days and len(df) > days:
            return df.tail(days)

        return df

    def fetch_multi_timeframe(
        self,
        ticker: str,
        timeframes: list[str],
        days: int = 365
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes."""
        return {tf: self.fetch(ticker, tf, days) for tf in timeframes}

    def fetch_multiple_tickers(
        self,
        tickers: list[str],
        timeframe: str = "1d",
        days: int = 365
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers."""
        return {t: self.fetch(t, timeframe, days) for t in tickers}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_data_loader.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backtesting/data_loader.py tests/test_data_loader.py
git commit -m "feat: data loader with caching"
```

---

### Task 3: Performance Metrics Module

**Files:**
- Create: `backtesting/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_metrics.py
import pytest
import numpy as np
from backtesting.metrics import calculate_metrics

def test_calculate_metrics_basic():
    # Simple equity curve: 100 -> 110 -> 105 -> 120
    equity_curve = np.array([100, 110, 105, 120])
    trades = [
        {"pnl": 10, "duration_minutes": 60},
        {"pnl": -5, "duration_minutes": 30},
        {"pnl": 15, "duration_minutes": 90},
    ]

    metrics = calculate_metrics(equity_curve, trades)

    assert metrics["total_return_pct"] == pytest.approx(20.0, rel=0.01)
    assert metrics["win_rate"] == pytest.approx(0.666, rel=0.01)
    assert metrics["n_trades"] == 3
    assert "sharpe_ratio" in metrics
    assert "max_drawdown_pct" in metrics

def test_calculate_metrics_drawdown():
    # Curve with 20% drawdown: 100 -> 120 -> 96 -> 110
    equity_curve = np.array([100, 120, 96, 110])
    trades = [{"pnl": 20, "duration_minutes": 60}, {"pnl": -24, "duration_minutes": 60}, {"pnl": 14, "duration_minutes": 60}]

    metrics = calculate_metrics(equity_curve, trades)

    # Max drawdown is (120-96)/120 = 20%
    assert metrics["max_drawdown_pct"] == pytest.approx(20.0, rel=0.01)
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_metrics.py -v
```

- [ ] **Step 3: Implement metrics module**

```python
# backtesting/metrics.py
"""Performance metrics calculation."""

import numpy as np
from typing import Optional


def calculate_metrics(
    equity_curve: np.ndarray,
    trades: list[dict],
    risk_free_rate: float = 0.0
) -> dict:
    """
    Calculate performance metrics from equity curve and trades.

    Args:
        equity_curve: Array of portfolio values over time
        trades: List of trade dicts with 'pnl' and 'duration_minutes'
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        Dict of performance metrics
    """
    if len(equity_curve) < 2:
        return _empty_metrics()

    # Basic returns
    initial = equity_curve[0]
    final = equity_curve[-1]
    total_return_pct = ((final - initial) / initial) * 100

    # Daily returns for Sharpe
    returns = np.diff(equity_curve) / equity_curve[:-1]

    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(returns) > 1 and np.std(returns) > 0:
        excess_returns = returns - (risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown_pct = np.max(drawdown) * 100

    # Trade statistics
    n_trades = len(trades)
    if n_trades > 0:
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / n_trades if n_trades > 0 else 0

        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf')

        durations = [t.get("duration_minutes", 0) for t in trades]
        avg_duration_minutes = np.mean(durations) if durations else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        avg_duration_minutes = 0

    return {
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_duration_minutes": avg_duration_minutes,
    }


def _empty_metrics() -> dict:
    """Return empty metrics dict."""
    return {
        "total_return_pct": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "n_trades": 0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "avg_duration_minutes": 0.0,
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backtesting/metrics.py tests/test_metrics.py
git commit -m "feat: performance metrics calculation"
```

---

## Chunk 2: MCMC Indicator & Signal Generation

### Task 4: MCMC Indicator Core

**Files:**
- Create: `trading/indicator.py`
- Create: `tests/test_indicator.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_indicator.py
import pytest
import numpy as np
import pandas as pd
from trading.indicator import MCMCIndicator

def test_indicator_generate_signal():
    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    df = pd.DataFrame({"close": prices}, index=dates)

    indicator = MCMCIndicator()
    signal = indicator.generate_signal("TEST", df)

    assert "ticker" in signal
    assert "slope_degrees" in signal
    assert "direction" in signal
    assert signal["direction"] in ["BULLISH", "BEARISH", "NEUTRAL"]
    assert "signal_strength" in signal
    assert 0 <= signal["signal_strength"] <= 1

def test_indicator_slope_calculation():
    # Uptrending data should give positive slope
    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    prices = np.linspace(100, 120, 100)  # Clear uptrend
    df = pd.DataFrame({"close": prices}, index=dates)

    indicator = MCMCIndicator()
    signal = indicator.generate_signal("TEST", df)

    assert signal["slope_degrees"] > 0
    assert signal["direction"] == "BULLISH"

def test_indicator_regime_detection():
    indicator = MCMCIndicator()

    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    df = pd.DataFrame({"close": prices}, index=dates)

    signal = indicator.generate_signal("TEST", df)

    assert "regime" in signal
    assert signal["regime"] in [0, 1, 2]
    assert "regime_confidence" in signal
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_indicator.py -v
```

- [ ] **Step 3: Implement MCMCIndicator**

```python
# trading/indicator.py
"""MCMC-based trading signal indicator."""

import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# GPU support
try:
    import cupy as cp
    USE_GPU = True
except ImportError:
    cp = None
    USE_GPU = False


class MCMCIndicator:
    """
    Generates trading signals from MCMC price forecasts.

    Extracts:
    - Slope angle of median forecast
    - Regime detection
    - Confidence bands
    - Signal strength composite
    """

    def __init__(
        self,
        n_simulations: int = 25000,
        n_steps: int = 30,
        n_regimes: int = 3,
        slope_threshold: float = 10.0,
        band_width_max: float = 0.05,
        regime_confidence_min: float = 0.6,
        mtf_weight: float = 0.3,
        regime_weight: float = 0.3,
        enable_gpu: bool = True
    ):
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.n_regimes = n_regimes
        self.slope_threshold = slope_threshold
        self.band_width_max = band_width_max
        self.regime_confidence_min = regime_confidence_min
        self.mtf_weight = mtf_weight
        self.regime_weight = regime_weight
        self.enable_gpu = enable_gpu and USE_GPU
        self.xp = cp if self.enable_gpu else np

    @classmethod
    def from_config(cls, config_path: str = "config/default.yaml") -> "MCMCIndicator":
        """Load indicator from config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return cls(
            n_simulations=config["mcmc"]["n_simulations"],
            n_steps=config["mcmc"]["n_steps"],
            n_regimes=config["mcmc"]["n_regimes"],
            slope_threshold=config["signal"]["slope_threshold"],
            band_width_max=config["signal"]["band_width_max"],
            regime_confidence_min=config["signal"]["regime_confidence_min"],
            mtf_weight=config["signal"]["mtf_weight"],
            regime_weight=config["signal"]["regime_weight"],
        )

    def generate_signal(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        timeframe: str = "1h"
    ) -> dict:
        """
        Generate trading signal from price data.

        Args:
            ticker: Stock symbol
            price_data: DataFrame with 'close' column
            timeframe: Data timeframe for labeling

        Returns:
            Signal dict with slope, direction, confidence, etc.
        """
        if price_data.empty or len(price_data) < 30:
            return self._empty_signal(ticker, timeframe)

        prices = price_data["close"].values
        current_price = prices[-1]

        # Run MCMC simulation
        forecast = self._run_mcmc(prices)

        # Calculate slope
        median_forecast = np.median(forecast[:, -1])
        slope_degrees = self._calculate_slope(current_price, median_forecast, prices)

        # Determine direction
        direction = self._get_direction(slope_degrees)

        # Calculate confidence metrics
        p5 = np.percentile(forecast[:, -1], 5)
        p95 = np.percentile(forecast[:, -1], 95)
        band_width_pct = (p95 - p5) / current_price

        # Regime detection
        regime, regime_confidence = self._detect_regime(prices)

        # Composite signal strength
        signal_strength = self._calculate_signal_strength(
            slope_degrees, band_width_pct, regime_confidence
        )

        # Suggested action
        suggested_action = self._get_action(direction, signal_strength)

        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "slope_degrees": slope_degrees,
            "direction": direction,
            "band_width_pct": band_width_pct,
            "regime": regime,
            "regime_confidence": regime_confidence,
            "signal_strength": signal_strength,
            "suggested_action": suggested_action,
            "forecast_median": median_forecast,
            "forecast_p5": p5,
            "forecast_p95": p95,
            "current_price": current_price,
        }

    def _run_mcmc(self, prices: np.ndarray) -> np.ndarray:
        """Run MCMC simulation and return forecast paths."""
        xp = self.xp

        # Calculate returns
        returns = np.log(prices[1:] / prices[:-1])

        # Regime-based sampling
        labels, n_states = self._infer_regimes(returns)

        if labels is None or n_states < 2:
            # Fallback to bootstrap
            return self._bootstrap_forecast(prices, returns)

        # Build transition matrix
        trans_mat = self._build_transition_matrix(labels, n_states)

        # Get returns per regime
        regime_returns = [returns[labels == s] for s in range(n_states)]

        # Simulate
        last_state = int(labels[-1])
        current_price = prices[-1]

        if self.enable_gpu:
            return self._gpu_simulate(
                current_price, trans_mat, regime_returns, last_state, returns
            )
        else:
            return self._cpu_simulate(
                current_price, trans_mat, regime_returns, last_state, returns
            )

    def _bootstrap_forecast(self, prices: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Simple bootstrap forecast fallback."""
        current_price = prices[-1]

        # Sample returns with replacement
        sampled = np.random.choice(returns, size=(self.n_simulations, self.n_steps))

        # Generate paths
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = current_price

        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(sampled[:, t-1])

        return paths

    def _infer_regimes(self, returns: np.ndarray) -> tuple:
        """Classify returns into regimes using quantiles."""
        if len(returns) < 10:
            return None, 0

        try:
            # Use pandas qcut for quantile-based binning
            series = pd.Series(returns)
            labels, _ = pd.qcut(series, q=self.n_regimes, labels=False, retbins=True, duplicates='drop')
            labels = labels.values.astype(int)
            n_states = int(labels.max()) + 1
            return labels, n_states
        except:
            return None, 0

    def _build_transition_matrix(self, states: np.ndarray, n_states: int) -> np.ndarray:
        """Build Markov transition matrix from state sequence."""
        counts = np.zeros((n_states, n_states))

        for i in range(len(states) - 1):
            counts[states[i], states[i+1]] += 1

        # Add smoothing
        counts += 0.5

        # Normalize rows
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1

        return counts / row_sums

    def _cpu_simulate(
        self,
        current_price: float,
        trans_mat: np.ndarray,
        regime_returns: list,
        last_state: int,
        all_returns: np.ndarray
    ) -> np.ndarray:
        """CPU-based MCMC simulation."""
        n_states = len(regime_returns)
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = current_price

        for sim in range(self.n_simulations):
            state = last_state
            for t in range(self.n_steps):
                # Transition
                state = np.random.choice(n_states, p=trans_mat[state])

                # Sample return
                if len(regime_returns[state]) > 0:
                    ret = np.random.choice(regime_returns[state])
                else:
                    ret = np.random.choice(all_returns)

                paths[sim, t+1] = paths[sim, t] * np.exp(ret)

        return paths

    def _gpu_simulate(
        self,
        current_price: float,
        trans_mat: np.ndarray,
        regime_returns: list,
        last_state: int,
        all_returns: np.ndarray
    ) -> np.ndarray:
        """GPU-accelerated MCMC simulation."""
        xp = self.xp
        n_states = len(regime_returns)

        # Transfer to GPU
        trans_gpu = xp.asarray(trans_mat, dtype=xp.float32)

        # Pre-generate random values
        rand_trans = xp.random.uniform(0, 1, (self.n_simulations, self.n_steps))

        # Simulate states
        states = xp.zeros((self.n_simulations, self.n_steps), dtype=xp.int32)
        current_states = xp.full(self.n_simulations, last_state, dtype=xp.int32)

        for t in range(self.n_steps):
            # Vectorized state transition
            cumsum = xp.cumsum(trans_gpu[current_states], axis=1)
            next_states = xp.argmax(rand_trans[:, t:t+1] <= cumsum, axis=1).astype(xp.int32)
            states[:, t] = next_states
            current_states = next_states

        # Map states to returns
        sampled_returns = xp.zeros((self.n_simulations, self.n_steps), dtype=xp.float64)

        for s in range(n_states):
            if len(regime_returns[s]) > 0:
                returns_gpu = xp.asarray(regime_returns[s])
                mask = (states == s)
                n_needed = int(xp.sum(mask))
                if n_needed > 0:
                    indices = xp.random.randint(0, len(returns_gpu), size=n_needed)
                    sampled_returns[mask] = returns_gpu[indices]

        # Generate paths
        paths = xp.zeros((self.n_simulations, self.n_steps + 1), dtype=xp.float64)
        paths[:, 0] = current_price

        cum_returns = xp.exp(sampled_returns)
        cum_prod = xp.cumprod(cum_returns, axis=1)
        paths[:, 1:] = current_price * cum_prod

        return xp.asnumpy(paths) if self.enable_gpu else paths

    def _calculate_slope(
        self,
        current_price: float,
        forecast_price: float,
        prices: np.ndarray
    ) -> float:
        """Calculate slope in degrees, normalized by volatility."""
        price_change = forecast_price - current_price

        # Normalize by recent volatility
        recent_std = np.std(prices[-20:]) if len(prices) >= 20 else np.std(prices)
        if recent_std == 0:
            recent_std = 1

        # Scale factor to make slope meaningful
        normalized_change = price_change / recent_std

        # Convert to degrees
        slope_radians = math.atan(normalized_change / self.n_steps)
        slope_degrees = math.degrees(slope_radians)

        return slope_degrees

    def _get_direction(self, slope_degrees: float) -> str:
        """Determine direction from slope."""
        if slope_degrees > self.slope_threshold:
            return "BULLISH"
        elif slope_degrees < -self.slope_threshold:
            return "BEARISH"
        return "NEUTRAL"

    def _detect_regime(self, prices: np.ndarray) -> tuple[int, float]:
        """Detect current market regime and confidence."""
        returns = np.diff(prices) / prices[:-1]

        if len(returns) < 10:
            return 1, 0.5  # Neutral, low confidence

        # Simple regime detection based on recent returns
        recent_mean = np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns)
        recent_std = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)

        if recent_std == 0:
            return 1, 0.5

        # Z-score of recent trend
        z_score = recent_mean / (recent_std / np.sqrt(min(20, len(returns))))

        if z_score > 1.5:
            regime = 2  # Bullish
            confidence = min(1.0, abs(z_score) / 3)
        elif z_score < -1.5:
            regime = 0  # Bearish
            confidence = min(1.0, abs(z_score) / 3)
        else:
            regime = 1  # Neutral
            confidence = 1 - abs(z_score) / 1.5

        return regime, confidence

    def _calculate_signal_strength(
        self,
        slope_degrees: float,
        band_width_pct: float,
        regime_confidence: float
    ) -> float:
        """Calculate composite signal strength 0-1."""
        # Slope component (stronger slope = stronger signal)
        slope_strength = min(1.0, abs(slope_degrees) / (self.slope_threshold * 2))

        # Band width component (tighter bands = more confident)
        band_strength = max(0, 1 - band_width_pct / self.band_width_max)

        # Weighted average
        slope_weight = 1 - self.mtf_weight - self.regime_weight

        strength = (
            slope_weight * slope_strength +
            self.regime_weight * regime_confidence +
            self.mtf_weight * band_strength
        )

        return min(1.0, max(0.0, strength))

    def _get_action(self, direction: str, signal_strength: float) -> str:
        """Determine suggested action."""
        if signal_strength < 0.5:
            return "HOLD"

        if direction == "BULLISH":
            return "BUY"
        elif direction == "BEARISH":
            return "SELL"

        return "HOLD"

    def _empty_signal(self, ticker: str, timeframe: str) -> dict:
        """Return empty signal for insufficient data."""
        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "slope_degrees": 0.0,
            "direction": "NEUTRAL",
            "band_width_pct": 0.0,
            "regime": 1,
            "regime_confidence": 0.0,
            "signal_strength": 0.0,
            "suggested_action": "HOLD",
            "forecast_median": 0.0,
            "forecast_p5": 0.0,
            "forecast_p95": 0.0,
            "current_price": 0.0,
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_indicator.py -v
```

- [ ] **Step 5: Commit**

```bash
git add trading/indicator.py tests/test_indicator.py
git commit -m "feat: MCMC indicator with slope and regime detection"
```

---

### Task 5: Multi-Timeframe Scanner

**Files:**
- Modify: `trading/indicator.py`
- Modify: `tests/test_indicator.py`

- [ ] **Step 1: Add MTF test**

```python
# Add to tests/test_indicator.py

def test_multi_timeframe_scan():
    indicator = MCMCIndicator()

    # Create data for multiple timeframes
    data = {}
    for tf in ["1m", "5m", "15m", "1h"]:
        dates = pd.date_range("2026-01-01", periods=100, freq="1h")
        prices = np.linspace(100, 110, 100)  # Uptrend
        data[tf] = pd.DataFrame({"close": prices}, index=dates)

    signal = indicator.generate_mtf_signal("TEST", data)

    assert "mtf_alignment" in signal
    assert "mtf_score" in signal
    assert signal["mtf_score"] >= 0
    assert signal["mtf_score"] <= 4
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_indicator.py::test_multi_timeframe_scan -v
```

- [ ] **Step 3: Add MTF method to MCMCIndicator**

```python
# Add to trading/indicator.py MCMCIndicator class

def generate_mtf_signal(
    self,
    ticker: str,
    timeframe_data: dict[str, pd.DataFrame]
) -> dict:
    """
    Generate signal with multi-timeframe confirmation.

    Args:
        ticker: Stock symbol
        timeframe_data: Dict mapping timeframe -> DataFrame

    Returns:
        Signal dict with MTF alignment info
    """
    signals = {}
    directions = {}

    # Generate signal for each timeframe
    for tf, data in timeframe_data.items():
        sig = self.generate_signal(ticker, data, tf)
        signals[tf] = sig
        directions[tf] = sig["direction"]

    # Calculate MTF alignment
    direction_counts = {}
    for d in directions.values():
        direction_counts[d] = direction_counts.get(d, 0) + 1

    # Dominant direction
    dominant_direction = max(direction_counts, key=direction_counts.get)
    mtf_score = direction_counts[dominant_direction]

    # Use the primary timeframe signal as base (first one)
    primary_tf = list(timeframe_data.keys())[0]
    result = signals[primary_tf].copy()

    # Add MTF info
    result["mtf_alignment"] = directions
    result["mtf_score"] = mtf_score
    result["mtf_dominant_direction"] = dominant_direction

    # Adjust signal strength based on MTF alignment
    alignment_bonus = (mtf_score - 2) * 0.1  # +0.1 for each aligned TF above 2
    result["signal_strength"] = min(1.0, result["signal_strength"] + alignment_bonus)

    # Update suggested action based on MTF
    if mtf_score >= 3 and dominant_direction != "NEUTRAL":
        result["suggested_action"] = "BUY" if dominant_direction == "BULLISH" else "SELL"
    elif mtf_score < 2:
        result["suggested_action"] = "HOLD"

    return result
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_indicator.py -v
```

- [ ] **Step 5: Commit**

```bash
git add trading/indicator.py tests/test_indicator.py
git commit -m "feat: multi-timeframe signal confirmation"
```

---

### Task 5.5: Signal Combiner

**Files:**
- Create: `trading/signal_combiner.py`
- Create: `tests/test_signal_combiner.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_signal_combiner.py
import pytest
from trading.signal_combiner import SignalCombiner

def test_signal_combiner_mcmc_only():
    combiner = SignalCombiner()

    mcmc_signal = {
        "direction": "BULLISH",
        "signal_strength": 0.8,
        "suggested_action": "BUY"
    }

    result = combiner.combine(mcmc_signal, llm_signal=None)

    assert result["action"] == "BUY"
    assert result["position_scale"] == 0.5  # Half position without LLM

def test_signal_combiner_agreement():
    combiner = SignalCombiner()

    mcmc_signal = {"direction": "BULLISH", "signal_strength": 0.8}
    llm_signal = {"direction": "BULLISH", "confidence": 0.75}

    result = combiner.combine(mcmc_signal, llm_signal)

    assert result["action"] == "BUY"
    assert result["position_scale"] == 1.0  # Full position

def test_signal_combiner_disagreement():
    combiner = SignalCombiner()

    mcmc_signal = {"direction": "BULLISH", "signal_strength": 0.8}
    llm_signal = {"direction": "BEARISH", "confidence": 0.75}

    result = combiner.combine(mcmc_signal, llm_signal)

    assert result["action"] == "HOLD"
    assert result["position_scale"] == 0.0
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_signal_combiner.py -v
```

- [ ] **Step 3: Implement SignalCombiner**

```python
# trading/signal_combiner.py
"""Combines MCMC and LLM signals into trading decisions."""

from typing import Optional


class SignalCombiner:
    """
    Merges MCMC indicator signals with TradingAgents LLM validation.

    Rules:
    - MCMC alone: small position (0.5x)
    - MCMC + LLM agree: full position (1.0x)
    - MCMC + LLM disagree: no trade (0x)
    """

    def __init__(
        self,
        mcmc_only_scale: float = 0.5,
        agreement_scale: float = 1.0,
        min_mcmc_strength: float = 0.6,
        min_llm_confidence: float = 0.5
    ):
        self.mcmc_only_scale = mcmc_only_scale
        self.agreement_scale = agreement_scale
        self.min_mcmc_strength = min_mcmc_strength
        self.min_llm_confidence = min_llm_confidence

    def combine(
        self,
        mcmc_signal: dict,
        llm_signal: Optional[dict] = None
    ) -> dict:
        """
        Combine MCMC and LLM signals.

        Args:
            mcmc_signal: Signal from MCMCIndicator
            llm_signal: Optional signal from TradingAgents

        Returns:
            Combined decision dict
        """
        mcmc_direction = mcmc_signal.get("direction", "NEUTRAL")
        mcmc_strength = mcmc_signal.get("signal_strength", 0)

        # Check minimum MCMC strength
        if mcmc_strength < self.min_mcmc_strength:
            return self._hold_result("MCMC strength below threshold")

        # No LLM signal - use MCMC only with reduced position
        if llm_signal is None:
            action = self._direction_to_action(mcmc_direction)
            return {
                "action": action,
                "position_scale": self.mcmc_only_scale if action != "HOLD" else 0.0,
                "reason": "MCMC only (no LLM validation)",
                "mcmc_direction": mcmc_direction,
                "llm_direction": None,
                "agreement": None
            }

        llm_direction = llm_signal.get("direction", "NEUTRAL")
        llm_confidence = llm_signal.get("confidence", 0)

        # Check LLM confidence
        if llm_confidence < self.min_llm_confidence:
            # Treat as MCMC-only
            action = self._direction_to_action(mcmc_direction)
            return {
                "action": action,
                "position_scale": self.mcmc_only_scale if action != "HOLD" else 0.0,
                "reason": "LLM confidence below threshold",
                "mcmc_direction": mcmc_direction,
                "llm_direction": llm_direction,
                "agreement": None
            }

        # Check agreement
        if mcmc_direction == llm_direction and mcmc_direction != "NEUTRAL":
            # Full agreement
            action = self._direction_to_action(mcmc_direction)
            return {
                "action": action,
                "position_scale": self.agreement_scale,
                "reason": "MCMC + LLM agreement",
                "mcmc_direction": mcmc_direction,
                "llm_direction": llm_direction,
                "agreement": True
            }

        elif mcmc_direction != llm_direction and llm_direction != "NEUTRAL" and mcmc_direction != "NEUTRAL":
            # Disagreement
            return {
                "action": "HOLD",
                "position_scale": 0.0,
                "reason": "MCMC + LLM disagreement",
                "mcmc_direction": mcmc_direction,
                "llm_direction": llm_direction,
                "agreement": False
            }

        else:
            # One is neutral - use the non-neutral one with reduced confidence
            active_direction = mcmc_direction if mcmc_direction != "NEUTRAL" else llm_direction
            action = self._direction_to_action(active_direction)
            return {
                "action": action,
                "position_scale": self.mcmc_only_scale if action != "HOLD" else 0.0,
                "reason": "Partial signal (one neutral)",
                "mcmc_direction": mcmc_direction,
                "llm_direction": llm_direction,
                "agreement": None
            }

    def _direction_to_action(self, direction: str) -> str:
        """Convert direction to action."""
        if direction == "BULLISH":
            return "BUY"
        elif direction == "BEARISH":
            return "SELL"
        return "HOLD"

    def _hold_result(self, reason: str) -> dict:
        """Return a HOLD result."""
        return {
            "action": "HOLD",
            "position_scale": 0.0,
            "reason": reason,
            "mcmc_direction": None,
            "llm_direction": None,
            "agreement": None
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_signal_combiner.py -v
```

- [ ] **Step 5: Commit**

```bash
git add trading/signal_combiner.py tests/test_signal_combiner.py
git commit -m "feat: signal combiner for MCMC + LLM fusion"
```

---

### Task 5.6: Position Manager

**Files:**
- Create: `trading/position_manager.py`
- Create: `tests/test_position_manager.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_position_manager.py
import pytest
from datetime import datetime, timedelta
from trading.position_manager import PositionManager

def test_position_manager_open_position():
    pm = PositionManager()

    pm.open_position("NVDA", quantity=10, price=500.0)

    pos = pm.get_position("NVDA")
    assert pos is not None
    assert pos["quantity"] == 10
    assert pos["avg_price"] == 500.0

def test_position_manager_close_position():
    pm = PositionManager()

    pm.open_position("AAPL", quantity=100, price=150.0)
    trade = pm.close_position("AAPL", quantity=100, price=160.0)

    assert trade["pnl"] == 1000.0  # (160-150) * 100
    assert pm.get_position("AAPL") is None

def test_position_manager_partial_close():
    pm = PositionManager()

    pm.open_position("MSFT", quantity=100, price=400.0)
    trade = pm.close_position("MSFT", quantity=50, price=420.0)

    assert trade["pnl"] == 1000.0  # (420-400) * 50

    pos = pm.get_position("MSFT")
    assert pos["quantity"] == 50

def test_position_manager_total_pnl():
    pm = PositionManager()

    pm.open_position("A", quantity=10, price=100.0)
    pm.close_position("A", quantity=10, price=110.0)  # +100

    pm.open_position("B", quantity=10, price=100.0)
    pm.close_position("B", quantity=10, price=90.0)  # -100

    assert pm.get_total_realized_pnl() == 0.0
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_position_manager.py -v
```

- [ ] **Step 3: Implement PositionManager**

```python
# trading/position_manager.py
"""Track positions and calculate P&L."""

from datetime import datetime
from typing import Optional


class PositionManager:
    """
    Tracks open positions and calculates P&L.

    Separate from broker to enable:
    - Cross-broker tracking
    - Detailed trade history
    - Performance analytics
    """

    def __init__(self):
        self._positions: dict[str, dict] = {}
        self._trades: list[dict] = []
        self._realized_pnl: float = 0.0

    def open_position(
        self,
        ticker: str,
        quantity: int,
        price: float,
        timestamp: Optional[datetime] = None
    ):
        """Open or add to a position."""
        timestamp = timestamp or datetime.now()

        if ticker in self._positions:
            # Add to existing position (average in)
            pos = self._positions[ticker]
            total_cost = pos["avg_price"] * pos["quantity"] + price * quantity
            new_quantity = pos["quantity"] + quantity
            new_avg_price = total_cost / new_quantity

            self._positions[ticker] = {
                "quantity": new_quantity,
                "avg_price": new_avg_price,
                "opened_at": pos["opened_at"],
                "last_updated": timestamp
            }
        else:
            self._positions[ticker] = {
                "quantity": quantity,
                "avg_price": price,
                "opened_at": timestamp,
                "last_updated": timestamp
            }

    def close_position(
        self,
        ticker: str,
        quantity: int,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> dict:
        """Close or reduce a position."""
        timestamp = timestamp or datetime.now()

        if ticker not in self._positions:
            raise ValueError(f"No position for {ticker}")

        pos = self._positions[ticker]
        if quantity > pos["quantity"]:
            raise ValueError(f"Cannot close {quantity} shares, only have {pos['quantity']}")

        # Calculate P&L
        pnl = (price - pos["avg_price"]) * quantity
        self._realized_pnl += pnl

        # Record trade
        duration = timestamp - pos["opened_at"]
        trade = {
            "ticker": ticker,
            "quantity": quantity,
            "entry_price": pos["avg_price"],
            "exit_price": price,
            "pnl": pnl,
            "pnl_pct": (price / pos["avg_price"] - 1) * 100,
            "duration_minutes": duration.total_seconds() / 60,
            "opened_at": pos["opened_at"],
            "closed_at": timestamp
        }
        self._trades.append(trade)

        # Update or remove position
        remaining = pos["quantity"] - quantity
        if remaining == 0:
            del self._positions[ticker]
        else:
            self._positions[ticker] = {
                "quantity": remaining,
                "avg_price": pos["avg_price"],
                "opened_at": pos["opened_at"],
                "last_updated": timestamp
            }

        return trade

    def get_position(self, ticker: str) -> Optional[dict]:
        """Get position for a ticker."""
        return self._positions.get(ticker)

    def get_all_positions(self) -> dict[str, dict]:
        """Get all open positions."""
        return self._positions.copy()

    def get_trades(self) -> list[dict]:
        """Get trade history."""
        return self._trades.copy()

    def get_total_realized_pnl(self) -> float:
        """Get total realized P&L."""
        return self._realized_pnl

    def get_unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        """Calculate unrealized P&L given current prices."""
        total = 0.0
        for ticker, pos in self._positions.items():
            if ticker in current_prices:
                current = current_prices[ticker]
                pnl = (current - pos["avg_price"]) * pos["quantity"]
                total += pnl
        return total

    def reset(self):
        """Reset all state."""
        self._positions.clear()
        self._trades.clear()
        self._realized_pnl = 0.0
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_position_manager.py -v
```

- [ ] **Step 5: Commit**

```bash
git add trading/position_manager.py tests/test_position_manager.py
git commit -m "feat: position manager with P&L tracking"
```

---

## Chunk 3: Broker & Execution Layer

### Task 6: Broker Base Interface

**Files:**
- Create: `brokers/base.py`
- Create: `tests/test_brokers.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_brokers.py
import pytest
from brokers.base import BaseBroker, Order, Position

def test_order_dataclass():
    order = Order(
        id="123",
        ticker="NVDA",
        side="BUY",
        quantity=10,
        price=500.0,
        status="FILLED"
    )
    assert order.id == "123"
    assert order.value == 5000.0

def test_position_dataclass():
    pos = Position(
        ticker="AAPL",
        quantity=100,
        avg_price=150.0,
        current_price=160.0
    )
    assert pos.market_value == 16000.0
    assert pos.unrealized_pnl == 1000.0
    assert pos.unrealized_pnl_pct == pytest.approx(6.67, rel=0.01)
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_brokers.py -v
```

- [ ] **Step 3: Implement base broker**

```python
# brokers/base.py
"""Abstract broker interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Order:
    """Represents a trade order."""
    id: str
    ticker: str
    side: str  # BUY or SELL
    quantity: int
    price: float
    status: str  # PENDING, FILLED, CANCELLED, REJECTED
    timestamp: datetime = None
    filled_price: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.filled_price is None:
            self.filled_price = self.price

    @property
    def value(self) -> float:
        return self.quantity * self.price


@dataclass
class Position:
    """Represents an open position."""
    ticker: str
    quantity: int
    avg_price: float
    current_price: float
    opened_at: datetime = None

    def __post_init__(self):
        if self.opened_at is None:
            self.opened_at = datetime.now()

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100


class BaseBroker(ABC):
    """Abstract base class for broker implementations."""

    @abstractmethod
    def buy(self, ticker: str, quantity: int, price: Optional[float] = None) -> Order:
        """Place a buy order."""
        pass

    @abstractmethod
    def sell(self, ticker: str, quantity: int, price: Optional[float] = None) -> Order:
        """Place a sell order."""
        pass

    @abstractmethod
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a specific ticker."""
        pass

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    def get_equity(self) -> float:
        """Get total account equity."""
        pass

    @abstractmethod
    def get_cash(self) -> float:
        """Get available cash."""
        pass

    @abstractmethod
    def get_orders(self) -> list[Order]:
        """Get order history."""
        pass
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_brokers.py -v
```

- [ ] **Step 5: Commit**

```bash
git add brokers/base.py tests/test_brokers.py
git commit -m "feat: broker base interface with Order and Position"
```

---

### Task 7: Paper Broker Implementation

**Files:**
- Create: `brokers/paper.py`
- Modify: `tests/test_brokers.py`

- [ ] **Step 1: Add paper broker tests**

```python
# Add to tests/test_brokers.py
from brokers.paper import PaperBroker

def test_paper_broker_initial_state():
    broker = PaperBroker(initial_cash=100000)

    assert broker.get_cash() == 100000
    assert broker.get_equity() == 100000
    assert len(broker.get_positions()) == 0

def test_paper_broker_buy():
    broker = PaperBroker(initial_cash=100000)

    order = broker.buy("NVDA", quantity=10, price=500.0)

    assert order.status == "FILLED"
    assert broker.get_cash() == 95000  # 100000 - 5000

    position = broker.get_position("NVDA")
    assert position is not None
    assert position.quantity == 10

def test_paper_broker_sell():
    broker = PaperBroker(initial_cash=100000)

    # Buy first
    broker.buy("AAPL", quantity=100, price=150.0)

    # Sell half
    order = broker.sell("AAPL", quantity=50, price=160.0)

    assert order.status == "FILLED"

    position = broker.get_position("AAPL")
    assert position.quantity == 50

def test_paper_broker_slippage():
    broker = PaperBroker(initial_cash=100000, slippage_pct=0.001)

    order = broker.buy("TEST", quantity=100, price=100.0)

    # With 0.1% slippage, filled price should be higher for buy
    assert order.filled_price > 100.0
    assert order.filled_price == pytest.approx(100.1, rel=0.01)
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_brokers.py::test_paper_broker_initial_state -v
```

- [ ] **Step 3: Implement PaperBroker**

```python
# brokers/paper.py
"""Paper trading broker for simulation."""

from datetime import datetime
from typing import Optional
import uuid

from brokers.base import BaseBroker, Order, Position


class PaperBroker(BaseBroker):
    """
    Simulated broker for paper trading.

    Features:
    - Realistic slippage simulation
    - Position tracking
    - P&L calculation
    """

    def __init__(
        self,
        initial_cash: float = 100000,
        slippage_pct: float = 0.0002,  # 0.02%
        market_impact_pct: float = 0.0001  # 0.01% for large orders
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.slippage_pct = slippage_pct
        self.market_impact_pct = market_impact_pct

        self._positions: dict[str, Position] = {}
        self._orders: list[Order] = []
        self._current_prices: dict[str, float] = {}

    def set_price(self, ticker: str, price: float):
        """Update current price for a ticker."""
        self._current_prices[ticker] = price

        # Update position if exists
        if ticker in self._positions:
            self._positions[ticker].current_price = price

    def buy(self, ticker: str, quantity: int, price: Optional[float] = None) -> Order:
        """Execute a buy order."""
        if price is None:
            price = self._current_prices.get(ticker, 0)

        if price <= 0:
            return self._rejected_order(ticker, "BUY", quantity, price, "Invalid price")

        # Calculate filled price with slippage (higher for buys)
        filled_price = price * (1 + self.slippage_pct)

        # Add market impact for large orders
        order_value = quantity * filled_price
        if order_value > 10000:
            filled_price *= (1 + self.market_impact_pct)

        total_cost = quantity * filled_price

        # Check sufficient cash
        if total_cost > self.cash:
            return self._rejected_order(ticker, "BUY", quantity, price, "Insufficient funds")

        # Execute order
        self.cash -= total_cost

        # Update or create position
        if ticker in self._positions:
            pos = self._positions[ticker]
            new_quantity = pos.quantity + quantity
            new_avg_price = (pos.cost_basis + total_cost) / new_quantity
            self._positions[ticker] = Position(
                ticker=ticker,
                quantity=new_quantity,
                avg_price=new_avg_price,
                current_price=filled_price,
                opened_at=pos.opened_at
            )
        else:
            self._positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity,
                avg_price=filled_price,
                current_price=filled_price
            )

        order = Order(
            id=str(uuid.uuid4()),
            ticker=ticker,
            side="BUY",
            quantity=quantity,
            price=price,
            status="FILLED",
            filled_price=filled_price
        )
        self._orders.append(order)

        return order

    def sell(self, ticker: str, quantity: int, price: Optional[float] = None) -> Order:
        """Execute a sell order."""
        if price is None:
            price = self._current_prices.get(ticker, 0)

        if price <= 0:
            return self._rejected_order(ticker, "SELL", quantity, price, "Invalid price")

        # Check position exists
        if ticker not in self._positions:
            return self._rejected_order(ticker, "SELL", quantity, price, "No position")

        pos = self._positions[ticker]
        if quantity > pos.quantity:
            return self._rejected_order(ticker, "SELL", quantity, price, "Insufficient shares")

        # Calculate filled price with slippage (lower for sells)
        filled_price = price * (1 - self.slippage_pct)

        # Market impact for large orders
        order_value = quantity * filled_price
        if order_value > 10000:
            filled_price *= (1 - self.market_impact_pct)

        # Execute order
        proceeds = quantity * filled_price
        self.cash += proceeds

        # Update position
        new_quantity = pos.quantity - quantity
        if new_quantity == 0:
            del self._positions[ticker]
        else:
            self._positions[ticker] = Position(
                ticker=ticker,
                quantity=new_quantity,
                avg_price=pos.avg_price,
                current_price=filled_price,
                opened_at=pos.opened_at
            )

        order = Order(
            id=str(uuid.uuid4()),
            ticker=ticker,
            side="SELL",
            quantity=quantity,
            price=price,
            status="FILLED",
            filled_price=filled_price
        )
        self._orders.append(order)

        return order

    def _rejected_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        price: float,
        reason: str
    ) -> Order:
        """Create a rejected order."""
        order = Order(
            id=str(uuid.uuid4()),
            ticker=ticker,
            side=side,
            quantity=quantity,
            price=price,
            status=f"REJECTED: {reason}"
        )
        self._orders.append(order)
        return order

    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a ticker."""
        return self._positions.get(ticker)

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self._positions.values())

    def get_equity(self) -> float:
        """Get total account equity (cash + positions)."""
        positions_value = sum(p.market_value for p in self._positions.values())
        return self.cash + positions_value

    def get_cash(self) -> float:
        """Get available cash."""
        return self.cash

    def get_orders(self) -> list[Order]:
        """Get order history."""
        return self._orders.copy()

    def reset(self):
        """Reset broker to initial state."""
        self.cash = self.initial_cash
        self._positions.clear()
        self._orders.clear()
        self._current_prices.clear()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_brokers.py -v
```

- [ ] **Step 5: Commit**

```bash
git add brokers/paper.py tests/test_brokers.py
git commit -m "feat: paper broker with slippage simulation"
```

---

### Task 8: Risk Manager

**Files:**
- Create: `trading/risk_manager.py`
- Create: `tests/test_risk_manager.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_risk_manager.py
import pytest
from trading.risk_manager import RiskManager
from brokers.paper import PaperBroker

def test_risk_manager_position_size():
    broker = PaperBroker(initial_cash=100000)
    rm = RiskManager(
        broker=broker,
        position_size_pct=0.10,
        max_position_pct=0.25
    )

    # 10% of 100k = 10k, at $500/share = 20 shares
    quantity = rm.calculate_position_size("NVDA", price=500.0)
    assert quantity == 20

def test_risk_manager_max_position_limit():
    broker = PaperBroker(initial_cash=100000)
    rm = RiskManager(
        broker=broker,
        position_size_pct=0.50,  # Would be 50k
        max_position_pct=0.25   # But max is 25k
    )

    # Max 25% of 100k = 25k, at $500/share = 50 shares
    quantity = rm.calculate_position_size("NVDA", price=500.0)
    assert quantity == 50

def test_risk_manager_stop_loss():
    broker = PaperBroker(initial_cash=100000)
    rm = RiskManager(broker=broker, stop_loss_pct=0.02)

    broker.buy("TEST", 100, 100.0)
    broker.set_price("TEST", 97.0)  # 3% loss

    assert rm.should_stop_loss("TEST") == True

def test_risk_manager_daily_loss_limit():
    broker = PaperBroker(initial_cash=100000)
    rm = RiskManager(broker=broker, daily_loss_limit_pct=0.05)

    # Simulate 6% loss
    broker.buy("TEST", 100, 100.0)
    broker.set_price("TEST", 40.0)  # Big loss

    assert rm.is_daily_loss_exceeded() == True
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_risk_manager.py -v
```

- [ ] **Step 3: Implement RiskManager**

```python
# trading/risk_manager.py
"""Risk management for position sizing and loss limits."""

from typing import Optional
from datetime import datetime, date

from brokers.base import BaseBroker


class RiskManager:
    """
    Manages trading risk through position sizing and loss limits.
    """

    def __init__(
        self,
        broker: BaseBroker,
        position_size_pct: float = 0.10,
        max_position_pct: float = 0.25,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.05,
        daily_loss_limit_pct: float = 0.05
    ):
        self.broker = broker
        self.position_size_pct = position_size_pct
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct

        self._daily_starting_equity: dict[date, float] = {}

    def calculate_position_size(
        self,
        ticker: str,
        price: float,
        signal_strength: float = 1.0
    ) -> int:
        """
        Calculate position size in shares.

        Args:
            ticker: Stock symbol
            price: Current price
            signal_strength: 0-1 multiplier for position size

        Returns:
            Number of shares to trade
        """
        if price <= 0:
            return 0

        equity = self.broker.get_equity()

        # Base position value
        target_value = equity * self.position_size_pct * signal_strength

        # Apply max position limit
        max_value = equity * self.max_position_pct
        position_value = min(target_value, max_value)

        # Check existing position
        existing = self.broker.get_position(ticker)
        if existing:
            current_value = existing.market_value
            remaining_value = max_value - current_value
            position_value = min(position_value, remaining_value)

        # Convert to shares
        quantity = int(position_value / price)

        # Check cash available
        cash = self.broker.get_cash()
        max_affordable = int(cash / price)

        return min(quantity, max_affordable)

    def should_stop_loss(self, ticker: str) -> bool:
        """Check if position should be stopped out."""
        position = self.broker.get_position(ticker)
        if position is None:
            return False

        loss_pct = -position.unrealized_pnl_pct / 100
        return loss_pct >= self.stop_loss_pct

    def should_take_profit(self, ticker: str) -> bool:
        """Check if position should take profit."""
        position = self.broker.get_position(ticker)
        if position is None:
            return False

        gain_pct = position.unrealized_pnl_pct / 100
        return gain_pct >= self.take_profit_pct

    def is_daily_loss_exceeded(self) -> bool:
        """Check if daily loss limit is exceeded."""
        today = date.today()

        # Record starting equity if not already
        if today not in self._daily_starting_equity:
            self._daily_starting_equity[today] = self.broker.get_equity()

        starting = self._daily_starting_equity[today]
        current = self.broker.get_equity()

        if starting == 0:
            return False

        loss_pct = (starting - current) / starting
        return loss_pct >= self.daily_loss_limit_pct

    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        return not self.is_daily_loss_exceeded()

    def get_exit_signals(self) -> list[dict]:
        """Get list of positions that should be exited."""
        exits = []

        for position in self.broker.get_positions():
            ticker = position.ticker

            if self.should_stop_loss(ticker):
                exits.append({
                    "ticker": ticker,
                    "reason": "STOP_LOSS",
                    "quantity": position.quantity,
                    "pnl_pct": position.unrealized_pnl_pct
                })
            elif self.should_take_profit(ticker):
                exits.append({
                    "ticker": ticker,
                    "reason": "TAKE_PROFIT",
                    "quantity": position.quantity,
                    "pnl_pct": position.unrealized_pnl_pct
                })

        return exits
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_risk_manager.py -v
```

- [ ] **Step 5: Commit**

```bash
git add trading/risk_manager.py tests/test_risk_manager.py
git commit -m "feat: risk manager with position sizing and stop loss"
```

---

## Chunk 4: Backtesting Engine

### Task 9: Backtesting Engine Core

**Files:**
- Create: `backtesting/engine.py`
- Create: `tests/test_backtest.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_backtest.py
import pytest
import numpy as np
import pandas as pd
from backtesting.engine import Backtester
from backtesting.data_loader import DataLoader

def test_backtester_simple_run():
    # Create simple test data
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    prices = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.01))
    data = {"SPY": pd.DataFrame({"close": prices, "open": prices, "high": prices, "low": prices}, index=dates)}

    backtester = Backtester(
        data=data,
        initial_cash=100000,
        params={"slope_threshold": 10.0}
    )

    result = backtester.run()

    assert "metrics" in result
    assert "equity_curve" in result
    assert "trades" in result
    assert result["metrics"]["n_trades"] >= 0

def test_backtester_walk_forward():
    dates = pd.date_range("2024-01-01", periods=504, freq="D")  # 2 years
    prices = 100 * np.exp(np.cumsum(np.random.randn(504) * 0.01))
    data = {"SPY": pd.DataFrame({"close": prices, "open": prices, "high": prices, "low": prices}, index=dates)}

    backtester = Backtester(data=data, initial_cash=100000)

    results = backtester.run_walk_forward(
        train_days=126,  # 6 months
        test_days=42     # 2 months
    )

    assert len(results) > 0
    assert all("metrics" in r for r in results)
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_backtest.py -v
```

- [ ] **Step 3: Implement Backtester**

```python
# backtesting/engine.py
"""GPU-accelerated backtesting engine."""

from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

from trading.indicator import MCMCIndicator
from trading.risk_manager import RiskManager
from brokers.paper import PaperBroker
from backtesting.metrics import calculate_metrics


class Backtester:
    """
    Backtesting engine for MCMC trading strategies.

    Features:
    - Walk-forward validation
    - Realistic cost modeling
    - Multi-ticker support
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        initial_cash: float = 100000,
        params: Optional[dict] = None,
        slippage_pct: float = 0.0002,
        market_impact_pct: float = 0.0001
    ):
        self.data = data
        self.initial_cash = initial_cash
        self.slippage_pct = slippage_pct
        self.market_impact_pct = market_impact_pct

        # Default params
        self.params = params or {
            "slope_threshold": 10.0,
            "mtf_min_alignment": 3,
            "signal_strength_min": 0.7,
            "position_size_pct": 0.10,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
            "n_simulations": 10000,
        }

    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> dict:
        """
        Run backtest on data.

        Returns:
            Dict with metrics, equity_curve, and trades
        """
        # Initialize components
        broker = PaperBroker(
            initial_cash=self.initial_cash,
            slippage_pct=self.slippage_pct,
            market_impact_pct=self.market_impact_pct
        )

        indicator = MCMCIndicator(
            n_simulations=self.params.get("n_simulations", 10000),
            slope_threshold=self.params.get("slope_threshold", 10.0),
        )

        risk_manager = RiskManager(
            broker=broker,
            position_size_pct=self.params.get("position_size_pct", 0.10),
            stop_loss_pct=self.params.get("stop_loss_pct", 0.02),
            take_profit_pct=self.params.get("take_profit_pct", 0.05),
        )

        # Get date range
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index.tolist())

        dates = sorted(all_dates)

        if start_date:
            dates = [d for d in dates if d >= pd.Timestamp(start_date)]
        if end_date:
            dates = [d for d in dates if d <= pd.Timestamp(end_date)]

        # Track results
        equity_curve = [self.initial_cash]
        trades = []

        # Lookback for indicator
        lookback = 100

        for i, current_date in enumerate(dates):
            if i < lookback:
                equity_curve.append(broker.get_equity())
                continue

            # Update prices
            for ticker, df in self.data.items():
                if current_date in df.index:
                    price = df.loc[current_date, "close"]
                    broker.set_price(ticker, price)

            # Check exits first
            for exit_signal in risk_manager.get_exit_signals():
                ticker = exit_signal["ticker"]
                pos = broker.get_position(ticker)
                if pos:
                    price = self.data[ticker].loc[current_date, "close"]
                    order = broker.sell(ticker, pos.quantity, price)
                    if order.status == "FILLED":
                        trades.append({
                            "ticker": ticker,
                            "side": "SELL",
                            "quantity": order.quantity,
                            "price": order.filled_price,
                            "pnl": pos.unrealized_pnl,
                            "reason": exit_signal["reason"],
                            "duration_minutes": (current_date - pos.opened_at).total_seconds() / 60 if pos.opened_at else 0,
                            "timestamp": current_date,
                        })

            # Check for entries
            if risk_manager.can_trade():
                for ticker, df in self.data.items():
                    if current_date not in df.index:
                        continue

                    # Get historical data for indicator
                    hist_idx = df.index.get_loc(current_date)
                    if hist_idx < lookback:
                        continue

                    hist_data = df.iloc[hist_idx - lookback:hist_idx + 1]

                    # Generate signal
                    signal = indicator.generate_signal(ticker, hist_data)

                    # Check signal strength
                    if signal["signal_strength"] < self.params.get("signal_strength_min", 0.7):
                        continue

                    price = df.loc[current_date, "close"]

                    # Execute based on signal
                    if signal["suggested_action"] == "BUY":
                        existing = broker.get_position(ticker)
                        if existing is None:
                            quantity = risk_manager.calculate_position_size(
                                ticker, price, signal["signal_strength"]
                            )
                            if quantity > 0:
                                order = broker.buy(ticker, quantity, price)
                                if order.status == "FILLED":
                                    trades.append({
                                        "ticker": ticker,
                                        "side": "BUY",
                                        "quantity": order.quantity,
                                        "price": order.filled_price,
                                        "pnl": 0,
                                        "reason": "SIGNAL",
                                        "duration_minutes": 0,
                                        "timestamp": current_date,
                                    })

            equity_curve.append(broker.get_equity())

        # Calculate metrics
        equity_array = np.array(equity_curve)
        metrics = calculate_metrics(equity_array, trades)

        return {
            "metrics": metrics,
            "equity_curve": equity_array,
            "trades": trades,
            "params": self.params,
        }

    def run_walk_forward(
        self,
        train_days: int = 126,
        test_days: int = 42
    ) -> list[dict]:
        """
        Run walk-forward backtest.

        Args:
            train_days: Training window size
            test_days: Testing window size

        Returns:
            List of test period results
        """
        # Get all dates
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index.tolist())

        dates = sorted(all_dates)

        results = []
        window_start = 0

        while window_start + train_days + test_days <= len(dates):
            train_end = window_start + train_days
            test_end = train_end + test_days

            train_start_date = dates[window_start]
            train_end_date = dates[train_end - 1]
            test_start_date = dates[train_end]
            test_end_date = dates[test_end - 1]

            # Run test period
            result = self.run(
                start_date=str(test_start_date),
                end_date=str(test_end_date)
            )

            result["period"] = {
                "train_start": str(train_start_date),
                "train_end": str(train_end_date),
                "test_start": str(test_start_date),
                "test_end": str(test_end_date),
            }

            results.append(result)

            # Slide window
            window_start += test_days

        return results
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_backtest.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backtesting/engine.py tests/test_backtest.py
git commit -m "feat: backtesting engine with walk-forward validation"
```

---

## Chunk 5: Optimization Layer

### Task 10: Optuna Optimizer

**Files:**
- Create: `optimization/optimizer.py`
- Create: `optimization/fitness.py`
- Create: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_optimizer.py
import pytest
import numpy as np
import pandas as pd
from optimization.optimizer import ParameterOptimizer

def test_optimizer_single_trial():
    # Create minimal test data
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    prices = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.01))
    data = {"SPY": pd.DataFrame({"close": prices, "open": prices, "high": prices, "low": prices}, index=dates)}

    optimizer = ParameterOptimizer(data=data, initial_cash=100000)

    # Run just 2 trials for speed
    best_params, best_value = optimizer.optimize(n_trials=2)

    assert "slope_threshold" in best_params
    assert "position_size_pct" in best_params
    assert isinstance(best_value, float)
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_optimizer.py -v
```

- [ ] **Step 3: Implement fitness function**

```python
# optimization/fitness.py
"""Fitness function for optimization."""

import numpy as np
from backtesting.engine import Backtester


def calculate_fitness(
    data: dict,
    params: dict,
    initial_cash: float = 100000,
    min_trades: int = 10
) -> float:
    """
    Calculate fitness score for a parameter set.

    Maximizes total return with penalties for:
    - Too few trades
    - Extreme drawdown

    Returns:
        Fitness score (higher is better)
    """
    backtester = Backtester(
        data=data,
        initial_cash=initial_cash,
        params=params
    )

    try:
        result = backtester.run()
        metrics = result["metrics"]

        # Base score: total return
        total_return = metrics["total_return_pct"]

        # Penalty for too few trades
        n_trades = metrics["n_trades"]
        if n_trades < min_trades:
            trade_penalty = (min_trades - n_trades) * 5
            total_return -= trade_penalty

        # Penalty for extreme drawdown
        max_dd = metrics["max_drawdown_pct"]
        if max_dd > 30:
            dd_penalty = (max_dd - 30) * 2
            total_return -= dd_penalty

        return total_return

    except Exception as e:
        # Return very negative score on error
        return -1000.0
```

- [ ] **Step 4: Implement optimizer**

```python
# optimization/optimizer.py
"""Parameter optimization using Optuna."""

import json
from pathlib import Path
from typing import Optional
import optuna
from optuna.samplers import TPESampler

from optimization.fitness import calculate_fitness


class ParameterOptimizer:
    """
    Optimizes trading parameters using Optuna.
    """

    def __init__(
        self,
        data: dict,
        initial_cash: float = 100000,
        min_trades: int = 100
    ):
        self.data = data
        self.initial_cash = initial_cash
        self.min_trades = min_trades

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        params = {
            # MCMC params
            "slope_threshold": trial.suggest_float("slope_threshold", 5.0, 20.0),
            "n_regimes": trial.suggest_int("n_regimes", 2, 4),
            "n_simulations": trial.suggest_int("n_simulations", 10000, 50000, step=10000),

            # Signal params
            "signal_strength_min": trial.suggest_float("signal_strength_min", 0.5, 0.9),
            "mtf_weight": trial.suggest_float("mtf_weight", 0.1, 0.5),
            "regime_weight": trial.suggest_float("regime_weight", 0.1, 0.5),

            # Risk params
            "position_size_pct": trial.suggest_float("position_size_pct", 0.05, 0.25),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.01, 0.05),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.02, 0.10),
        }

        return calculate_fitness(
            self.data,
            params,
            self.initial_cash,
            self.min_trades
        )

    def optimize(
        self,
        n_trials: int = 500,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ) -> tuple[dict, float]:
        """
        Run optimization.

        Args:
            n_trials: Number of trials to run
            timeout: Max seconds to run
            n_jobs: Parallel jobs (-1 for all cores)

        Returns:
            Tuple of (best_params, best_value)
        """
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        return study.best_params, study.best_value

    def save_best_params(
        self,
        params: dict,
        value: float,
        path: str = "config/best_params.json"
    ):
        """Save best parameters to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        output = {
            "params": params,
            "fitness": value,
            "min_trades": self.min_trades,
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_optimizer.py -v
```

- [ ] **Step 6: Commit**

```bash
git add optimization/optimizer.py optimization/fitness.py tests/test_optimizer.py
git commit -m "feat: Optuna parameter optimizer"
```

---

## Chunk 6: Scripts & Integration

### Task 11: Entry Point Scripts

**Files:**
- Create: `scripts/download_data.py`
- Create: `scripts/run_backtest.py`
- Create: `scripts/run_optimizer.py`
- Create: `scripts/run_paper_trader.py`

- [ ] **Step 1: Create download_data.py**

```python
# scripts/download_data.py
"""Download and cache historical data."""

import sys
sys.path.insert(0, ".")

import yaml
from backtesting.data_loader import DataLoader


def main():
    # Load tickers
    with open("config/tickers.yaml") as f:
        config = yaml.safe_load(f)

    tickers = config["watchlist"]
    timeframes = ["1m", "5m", "15m", "1h", "1d"]

    loader = DataLoader(cache_dir="data/cache")

    print(f"Downloading data for {len(tickers)} tickers...")

    for ticker in tickers:
        print(f"\n{ticker}:")
        for tf in timeframes:
            try:
                df = loader.fetch(ticker, tf, days=730, use_cache=False)
                print(f"  {tf}: {len(df)} bars")
            except Exception as e:
                print(f"  {tf}: ERROR - {e}")

    print("\nDone! Data cached in data/cache/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create run_backtest.py**

```python
# scripts/run_backtest.py
"""Run backtest with current parameters."""

import sys
sys.path.insert(0, ".")

import json
import yaml
from pathlib import Path

from backtesting.data_loader import DataLoader
from backtesting.engine import Backtester


def main():
    # Load config
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    with open("config/tickers.yaml") as f:
        tickers_config = yaml.safe_load(f)

    # Load best params if available
    params_path = Path("config/best_params.json")
    if params_path.exists():
        with open(params_path) as f:
            best = json.load(f)
            params = best["params"]
            print("Using optimized parameters")
    else:
        params = {
            "slope_threshold": config["signal"]["slope_threshold"],
            "position_size_pct": config["risk"]["position_size_pct"],
            "stop_loss_pct": config["risk"]["stop_loss_pct"],
            "take_profit_pct": config["risk"]["take_profit_pct"],
            "signal_strength_min": config["signal"]["signal_strength_min"],
            "n_simulations": config["mcmc"]["n_simulations"],
        }
        print("Using default parameters")

    # Load data
    loader = DataLoader()
    tickers = tickers_config["watchlist"]

    print(f"\nLoading data for {len(tickers)} tickers...")
    data = {}
    for ticker in tickers:
        df = loader.fetch(ticker, "1d", days=730)
        if not df.empty:
            data[ticker] = df
            print(f"  {ticker}: {len(df)} bars")

    # Run backtest
    print("\nRunning backtest...")
    backtester = Backtester(data=data, initial_cash=100000, params=params)

    results = backtester.run_walk_forward(train_days=126, test_days=42)

    # Aggregate results
    total_return = 1.0
    all_trades = []

    for r in results:
        period_return = 1 + r["metrics"]["total_return_pct"] / 100
        total_return *= period_return
        all_trades.extend(r["trades"])

    total_return_pct = (total_return - 1) * 100

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Total Trades: {len(all_trades)}")
    print(f"Periods: {len(results)}")

    if results:
        avg_sharpe = sum(r["metrics"]["sharpe_ratio"] for r in results) / len(results)
        max_dd = max(r["metrics"]["max_drawdown_pct"] for r in results)
        print(f"Avg Sharpe: {avg_sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2f}%")

    # Save results
    output = {
        "total_return_pct": total_return_pct,
        "n_trades": len(all_trades),
        "n_periods": len(results),
        "params": params,
    }

    Path("data/results").mkdir(parents=True, exist_ok=True)
    with open("data/results/backtest_latest.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to data/results/backtest_latest.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create run_optimizer.py**

```python
# scripts/run_optimizer.py
"""Run parameter optimization."""

import sys
sys.path.insert(0, ".")

import yaml
from backtesting.data_loader import DataLoader
from optimization.optimizer import ParameterOptimizer


def main():
    # Load tickers
    with open("config/tickers.yaml") as f:
        config = yaml.safe_load(f)

    tickers = config["watchlist"]

    # Load data
    loader = DataLoader()

    print(f"Loading data for {len(tickers)} tickers...")
    data = {}
    for ticker in tickers:
        df = loader.fetch(ticker, "1d", days=730)
        if not df.empty:
            data[ticker] = df
            print(f"  {ticker}: {len(df)} bars")

    # Run optimization
    print("\nStarting optimization (this may take a while)...")
    print("Press Ctrl+C to stop early\n")

    optimizer = ParameterOptimizer(data=data, initial_cash=100000, min_trades=50)

    try:
        best_params, best_value = optimizer.optimize(
            n_trials=500,
            timeout=14400,  # 4 hours max
            n_jobs=1
        )

        print("\n" + "=" * 50)
        print("OPTIMIZATION COMPLETE")
        print("=" * 50)
        print(f"Best fitness: {best_value:.2f}")
        print("\nBest parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        # Save
        optimizer.save_best_params(best_params, best_value)
        print("\nSaved to config/best_params.json")

    except KeyboardInterrupt:
        print("\nOptimization interrupted")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create run_paper_trader.py**

```python
# scripts/run_paper_trader.py
"""Run paper trading with live signals."""

import sys
sys.path.insert(0, ".")

import json
import time
import yaml
from datetime import datetime
from pathlib import Path

from backtesting.data_loader import DataLoader
from trading.indicator import MCMCIndicator
from trading.risk_manager import RiskManager
from brokers.paper import PaperBroker


def main():
    # Load config
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    with open("config/tickers.yaml") as f:
        tickers_config = yaml.safe_load(f)

    # Load best params if available
    params_path = Path("config/best_params.json")
    if params_path.exists():
        with open(params_path) as f:
            best = json.load(f)
            params = best["params"]
            print("Using optimized parameters")
    else:
        params = config["signal"]
        print("Using default parameters")

    tickers = tickers_config["watchlist"]

    # Initialize components
    loader = DataLoader()
    broker = PaperBroker(initial_cash=100000)

    indicator = MCMCIndicator(
        slope_threshold=params.get("slope_threshold", 10.0),
        n_simulations=params.get("n_simulations", 25000),
    )

    risk_manager = RiskManager(
        broker=broker,
        position_size_pct=params.get("position_size_pct", 0.10),
        stop_loss_pct=params.get("stop_loss_pct", 0.02),
        take_profit_pct=params.get("take_profit_pct", 0.05),
    )

    print("=" * 50)
    print("PAPER TRADER")
    print("=" * 50)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Initial cash: ${broker.get_cash():,.2f}")
    print("\nPress Ctrl+C to stop\n")

    try:
        while True:
            print(f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

            # Check exits
            for exit_signal in risk_manager.get_exit_signals():
                ticker = exit_signal["ticker"]
                pos = broker.get_position(ticker)
                if pos:
                    df = loader.fetch(ticker, "5m", days=1, use_cache=False)
                    if not df.empty:
                        price = df["close"].iloc[-1]
                        order = broker.sell(ticker, pos.quantity, price)
                        print(f"EXIT {ticker}: {exit_signal['reason']} @ ${price:.2f}")

            # Check entries
            if risk_manager.can_trade():
                for ticker in tickers:
                    # Skip if already have position
                    if broker.get_position(ticker):
                        continue

                    # Get data
                    df = loader.fetch(ticker, "5m", days=7, use_cache=False)
                    if df.empty or len(df) < 50:
                        continue

                    # Generate signal
                    signal = indicator.generate_signal(ticker, df, "5m")

                    # Log signal
                    direction = signal["direction"]
                    strength = signal["signal_strength"]

                    if strength > 0.5:
                        print(f"{ticker}: {direction} (strength={strength:.2f})")

                    # Execute if strong signal
                    if signal["suggested_action"] == "BUY" and strength >= params.get("signal_strength_min", 0.7):
                        price = df["close"].iloc[-1]
                        quantity = risk_manager.calculate_position_size(ticker, price, strength)

                        if quantity > 0:
                            order = broker.buy(ticker, quantity, price)
                            if order.status == "FILLED":
                                print(f"BUY {ticker}: {quantity} shares @ ${price:.2f}")

            # Print portfolio status
            equity = broker.get_equity()
            positions = broker.get_positions()

            print(f"\nEquity: ${equity:,.2f}")
            if positions:
                print("Positions:")
                for pos in positions:
                    print(f"  {pos.ticker}: {pos.quantity} @ ${pos.avg_price:.2f} (P&L: ${pos.unrealized_pnl:,.2f})")

            # Wait before next iteration
            print("\nWaiting 5 minutes...")
            time.sleep(300)

    except KeyboardInterrupt:
        print("\n\nStopping paper trader...")

        # Final summary
        print("\n" + "=" * 50)
        print("FINAL SUMMARY")
        print("=" * 50)
        print(f"Final equity: ${broker.get_equity():,.2f}")
        print(f"Total P&L: ${broker.get_equity() - 100000:,.2f}")
        print(f"Return: {((broker.get_equity() / 100000) - 1) * 100:.2f}%")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Commit**

```bash
git add scripts/
git commit -m "feat: entry point scripts for backtest, optimizer, paper trader"
```

---

### Task 12: TradingAgents Integration (Stub)

**Files:**
- Create: `integrations/tradingagents_bridge.py`
- Create: `tests/test_integrations.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_integrations.py
import pytest
from integrations.tradingagents_bridge import TradingAgentsBridge

def test_bridge_initialization():
    bridge = TradingAgentsBridge(daily_budget_usd=2.0)
    assert bridge.daily_budget_usd == 2.0
    assert bridge.calls_today == 0

def test_bridge_budget_tracking():
    bridge = TradingAgentsBridge(daily_budget_usd=2.0, max_calls_per_day=2)

    assert bridge.can_call() == True
    bridge.calls_today = 2
    assert bridge.can_call() == False
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_integrations.py -v
```

- [ ] **Step 3: Implement bridge stub**

```python
# integrations/tradingagents_bridge.py
"""TradingAgents LLM integration bridge."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional


class TradingAgentsBridge:
    """
    Bridge to TradingAgents LLM framework.

    Features:
    - Budget tracking ($1-2/day)
    - Fallback handling
    - Cost logging
    """

    def __init__(
        self,
        daily_budget_usd: float = 2.0,
        max_calls_per_day: int = 3,
        signal_threshold: float = 0.75
    ):
        self.daily_budget_usd = daily_budget_usd
        self.max_calls_per_day = max_calls_per_day
        self.signal_threshold = signal_threshold

        self.calls_today = 0
        self.spent_today = 0.0
        self._last_reset_date = date.today()

        self._usage_log_path = Path("data/results/llm_usage.json")

    def _reset_daily_counters(self):
        """Reset counters if new day."""
        today = date.today()
        if today != self._last_reset_date:
            self.calls_today = 0
            self.spent_today = 0.0
            self._last_reset_date = today

    def can_call(self) -> bool:
        """Check if LLM call is allowed."""
        self._reset_daily_counters()
        return (
            self.calls_today < self.max_calls_per_day and
            self.spent_today < self.daily_budget_usd
        )

    def should_validate(self, signal_strength: float) -> bool:
        """Check if signal warrants LLM validation."""
        return signal_strength >= self.signal_threshold and self.can_call()

    def validate_signal(
        self,
        ticker: str,
        mcmc_signal: dict,
        price_data: Optional[dict] = None
    ) -> dict:
        """
        Validate MCMC signal with TradingAgents.

        Returns:
            Validation result dict
        """
        if not self.can_call():
            return self._budget_exceeded_response()

        try:
            # TODO: Integrate actual TradingAgents call
            # For now, return stub response
            result = self._stub_validation(ticker, mcmc_signal)

            # Track usage
            self.calls_today += 1
            estimated_cost = 0.15  # Estimate per call
            self.spent_today += estimated_cost

            self._log_usage(ticker, estimated_cost, result)

            return result

        except Exception as e:
            return {
                "validated": False,
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
                "fallback": True
            }

    def _stub_validation(self, ticker: str, signal: dict) -> dict:
        """Stub validation - agrees with MCMC signal."""
        return {
            "validated": True,
            "direction": signal.get("direction", "NEUTRAL"),
            "confidence": signal.get("signal_strength", 0.5),
            "reasoning": "Stub validation - TradingAgents not yet integrated",
            "fallback": False
        }

    def _budget_exceeded_response(self) -> dict:
        """Response when budget is exceeded."""
        return {
            "validated": False,
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "error": "Daily budget exceeded",
            "fallback": True
        }

    def _log_usage(self, ticker: str, cost: float, result: dict):
        """Log LLM usage for tracking."""
        self._usage_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing log
        if self._usage_log_path.exists():
            with open(self._usage_log_path) as f:
                log = json.load(f)
        else:
            log = {"calls": []}

        # Append new entry
        log["calls"].append({
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "cost_usd": cost,
            "result": result.get("direction"),
        })

        # Save
        with open(self._usage_log_path, "w") as f:
            json.dump(log, f, indent=2)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_integrations.py -v
```

- [ ] **Step 5: Commit**

```bash
git add integrations/tradingagents_bridge.py tests/test_integrations.py
git commit -m "feat: TradingAgents integration stub with budget tracking"
```

---

### Task 13: Final Integration & Testing

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration tests."""

import pytest
import numpy as np
import pandas as pd

from trading.indicator import MCMCIndicator
from trading.risk_manager import RiskManager
from brokers.paper import PaperBroker
from backtesting.engine import Backtester
from backtesting.data_loader import DataLoader


def test_full_trading_loop():
    """Test complete signal -> trade -> exit flow."""
    # Create test data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    prices = 100 * np.exp(np.cumsum(np.random.randn(200) * 0.02))
    df = pd.DataFrame({
        "close": prices,
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99
    }, index=dates)

    # Initialize components
    broker = PaperBroker(initial_cash=100000)
    indicator = MCMCIndicator(n_simulations=1000)
    risk_manager = RiskManager(broker=broker)

    # Generate signal
    signal = indicator.generate_signal("TEST", df)

    # If bullish, execute trade
    if signal["direction"] == "BULLISH" and signal["signal_strength"] > 0.5:
        price = df["close"].iloc[-1]
        quantity = risk_manager.calculate_position_size("TEST", price)

        if quantity > 0:
            order = broker.buy("TEST", quantity, price)
            assert order.status == "FILLED"

            # Verify position
            position = broker.get_position("TEST")
            assert position is not None
            assert position.quantity == quantity


def test_backtest_produces_valid_results():
    """Test backtester produces valid metrics."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    prices = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.01))
    data = {
        "SPY": pd.DataFrame({
            "close": prices,
            "open": prices,
            "high": prices,
            "low": prices
        }, index=dates)
    }

    backtester = Backtester(data=data, initial_cash=100000)
    result = backtester.run()

    # Verify result structure
    assert "metrics" in result
    assert "equity_curve" in result
    assert "trades" in result

    # Verify metrics are reasonable
    metrics = result["metrics"]
    assert -100 < metrics["total_return_pct"] < 1000
    assert 0 <= metrics["win_rate"] <= 1
    assert metrics["max_drawdown_pct"] >= 0
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/ -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration tests"
```

- [ ] **Step 4: Final commit with all __init__.py updates**

```python
# Update all __init__.py files with exports

# trading/__init__.py
from trading.indicator import MCMCIndicator
from trading.risk_manager import RiskManager
from trading.signal_combiner import SignalCombiner
from trading.position_manager import PositionManager

__all__ = ["MCMCIndicator", "RiskManager", "SignalCombiner", "PositionManager"]

# brokers/__init__.py
from brokers.base import BaseBroker, Order, Position
from brokers.paper import PaperBroker

__all__ = ["BaseBroker", "Order", "Position", "PaperBroker"]

# backtesting/__init__.py
from backtesting.data_loader import DataLoader
from backtesting.engine import Backtester
from backtesting.metrics import calculate_metrics

__all__ = ["DataLoader", "Backtester", "calculate_metrics"]

# optimization/__init__.py
from optimization.optimizer import ParameterOptimizer
from optimization.fitness import calculate_fitness

__all__ = ["ParameterOptimizer", "calculate_fitness"]

# integrations/__init__.py
from integrations.tradingagents_bridge import TradingAgentsBridge

__all__ = ["TradingAgentsBridge"]
```

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete MCMC trading system implementation"
```

---

## Summary

**Total Tasks:** 15
**Estimated Time:** 5-7 hours

**Build Order:**
1. Foundation (Tasks 1-3): Project setup, data loader, metrics
2. Signals (Tasks 4-5.6): MCMC indicator, multi-timeframe, signal combiner, position manager
3. Execution (Tasks 6-8): Broker interface, paper broker, risk manager
4. Backtesting (Task 9): Backtest engine
5. Optimization (Task 10): Optuna optimizer
6. Scripts (Tasks 11-12): Entry points, TradingAgents stub
7. Testing (Task 13): Integration tests

**After Implementation:**
1. Run `python scripts/download_data.py` to cache data
2. Run `python scripts/run_optimizer.py` to find best params
3. Run `python scripts/run_backtest.py` to validate
4. Run `python scripts/run_paper_trader.py` to paper trade
