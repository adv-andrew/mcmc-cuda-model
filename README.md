# MCMC Trading System

Monte Carlo Markov Chain trading system with GPU acceleration. Uses regime-switching simulations to generate probabilistic price forecasts and trading signals.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest
python scripts/run_backtest.py

# Run parameter optimizer
python scripts/run_optimizer.py

# Run paper trader (live simulation)
python scripts/run_paper_trader.py

# Run GUI
python main.py
```

## Performance

Current optimized parameters achieve:
- **Avg Return:** +6.16% (walk-forward backtest)
- **Sharpe Ratio:** 2.36
- **Win/Loss Ratio:** 2.77

## System Architecture

```
config/
  default.yaml       # MCMC, signal, risk parameters
  tickers.yaml       # Watchlist
  best_params.json   # Optimized parameters (auto-generated)

trading/
  indicator.py       # MCMCIndicator - signal generation
  risk_manager.py    # Position sizing, stop-loss, take-profit

backtesting/
  engine.py          # Walk-forward backtester
  data_loader.py     # Yahoo Finance data fetching
  metrics.py         # Performance metrics

brokers/
  paper.py           # Paper trading broker

scripts/
  run_backtest.py    # Run walk-forward backtest
  run_optimizer.py   # Optuna parameter optimization
  run_paper_trader.py # Live paper trading loop (MTF enabled)
```

## Signal Generation

The MCMC indicator generates signals by:

1. **Monte Carlo Simulation:** 25,000 price paths using historical volatility
2. **Slope Calculation:** Forecast median vs current price, normalized by volatility
3. **Regime Detection:** Bull (2), Neutral (1), Bear (0) via quantile thresholds
4. **Signal Strength:** Composite score (0-1) based on slope magnitude

### Multi-Timeframe (MTF) Confirmation

The paper trader uses MTF confirmation requiring both 1h and 1d timeframes to align:

```
Ticker | 1h       | 1d       | Aligned | Action
-------|----------|----------|---------|-------
NVDA   | BULLISH  | BULLISH  | YES     | BUY
AAPL   | BEARISH  | BEARISH  | YES     | SHORT
GOOGL  | BULLISH  | BEARISH  | NO      | HOLD
```

Only aligned signals are traded, reducing false signals from short-term noise.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_simulations` | 25,000 | Monte Carlo paths |
| `n_steps` | 30 | Forecast horizon (bars) |
| `slope_threshold` | 15.0 | Degrees for BULLISH/BEARISH |
| `signal_strength_min` | 0.65 | Minimum strength to trade |
| `stop_loss_pct` | 0.05 | 5% stop loss |
| `take_profit_pct` | 0.12 | 12% take profit |
| `position_size_pct` | 0.08 | 8% of equity per position |

## GPU Acceleration

Uses CuPy for CUDA-accelerated simulations when available:

- **CPU:** ~1,000 simulations (NumPy)
- **GPU:** ~50,000 simulations (CuPy/CUDA)
- **Speedup:** ~20x

Falls back to NumPy automatically if no GPU detected.

## Usage Examples

### Get Current Signals

```python
from backtesting.data_loader import DataLoader
from trading.indicator import MCMCIndicator

loader = DataLoader()
indicator = MCMCIndicator.from_config('config/default.yaml')

data = loader.fetch('NVDA', timeframe='1d', days=100)
signal = indicator.generate_signal('NVDA', data, '1d')

print(f"Direction: {signal['direction']}")
print(f"Action: {signal['suggested_action']}")
print(f"Strength: {signal['signal_strength']}")
```

### Run Backtest

```python
from backtesting.data_loader import DataLoader
from backtesting.engine import Backtester

loader = DataLoader()
data = loader.fetch_multiple_tickers(['AAPL', 'MSFT', 'NVDA'], timeframe='1d', days=365)

bt = Backtester(data=data, params={'stop_loss_pct': 0.05, 'take_profit_pct': 0.12})
results = bt.run_walk_forward()

for r in results:
    print(f"Return: {r['metrics']['total_return_pct']:.2f}%")
```

## License

MIT License
