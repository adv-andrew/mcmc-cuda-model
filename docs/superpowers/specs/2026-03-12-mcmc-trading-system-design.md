# MCMC Trading System Design Specification

## Overview

A GPU-accelerated trading system that combines MCMC-based price forecasting with LLM-powered analysis (TradingAgents) to generate trading signals. The system uses Optuna to optimize parameters for maximum returns via walk-forward backtesting.

**Target Assets:** NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA, SPY

**Use Case:** Adaptive system for both day trading (1m-15m) and swing trading (1h-1d)

**Hardware:** NVIDIA 2080 Super (8GB VRAM)

---

## System Architecture

```
+-----------------------------------------------------------------------------+
|                              MCMC TRADING SYSTEM                             |
+-----------------------------------------------------------------------------+
|                                                                              |
|  +------------------------------------------------------------------------+ |
|  |                         1. SIGNAL LAYER                                 | |
|  |  +----------------------+     +--------------------------------------+ | |
|  |  |   MCMC Indicator     |     |      TradingAgents (LLM)             | | |
|  |  |                      |     |                                      | | |
|  |  |  - Slope angle       |     |  - Triggered only when:              | | |
|  |  |  - Multi-TF scan     |     |    - MCMC signal is strong           | | |
|  |  |  - Regime state      |     |    - Once per day max                | | |
|  |  |  - Confidence bands  |     |  - $1-2/day budget cap               | | |
|  |  |  - GPU-accelerated   |     |                                      | | |
|  |  +----------+-----------+     +------------------+-------------------+ | |
|  |             |                                    |                     | |
|  |             +----------------+-------------------+                     | |
|  |                              v                                         | |
|  |             +---------------------------+                              | |
|  |             |    Signal Combiner        |                              | |
|  |             |                           |                              | |
|  |             |  MCMC alone = small pos   |                              | |
|  |             |  MCMC + LLM = full pos    |                              | |
|  |             |  Disagreement = no trade  |                              | |
|  |             +-------------+-------------+                              | |
|  +---------------------------|--------------------------------------------+ |
|                              v                                              |
|  +------------------------------------------------------------------------+ |
|  |                      2. EXECUTION LAYER                                 | |
|  |  +----------------------+     +--------------------------------------+ | |
|  |  |   Risk Manager       |     |      Position Manager                | | |
|  |  |                      |     |                                      | | |
|  |  |  - Max position size |     |  - Track open positions              | | |
|  |  |  - Daily loss limit  |     |  - Entry/exit timestamps             | | |
|  |  |  - Drawdown monitor  |     |  - P&L calculation                   | | |
|  |  +----------------------+     +--------------------------------------+ | |
|  |                                                                         | |
|  |  +--------------------------------------------------------------------+ | |
|  |  |                      Broker Interface                               | | |
|  |  |  - PaperBroker (default)  - AlpacaBroker (future)                  | | |
|  |  +--------------------------------------------------------------------+ | |
|  +------------------------------------------------------------------------+ |
|                                                                              |
|  +------------------------------------------------------------------------+ |
|  |                    3. OPTIMIZATION LAYER                                | |
|  |  +----------------------+     +--------------------------------------+ | |
|  |  |   Backtester (GPU)   |     |      Optuna Optimizer                | | |
|  |  |                      |     |                                      | | |
|  |  |  - Walk-forward      |     |  - Search parameter space            | | |
|  |  |  - Realistic costs   |     |  - Maximize total returns            | | |
|  |  |  - Multi-ticker      |     |  - Prune bad trials early            | | |
|  |  |  - 2+ years history  |     |  - Save best params to config        | | |
|  |  +----------------------+     +--------------------------------------+ | |
|  +------------------------------------------------------------------------+ |
|                                                                              |
+-----------------------------------------------------------------------------+
```

---

## Component Specifications

### 1. MCMC Indicator Module

**Purpose:** Extract trading signals from MCMC price forecasts.

**Output Schema:**
```python
{
    "ticker": str,              # e.g., "NVDA"
    "timestamp": str,           # ISO format
    "timeframe": str,           # e.g., "5m"

    # Core signal
    "slope_degrees": float,     # Angle of median forecast line
    "direction": str,           # BULLISH / BEARISH / NEUTRAL

    # Confidence metrics
    "band_width_pct": float,    # 95th - 5th percentile as % of price
    "regime": int,              # Current detected regime (0=bear, 1=neutral, 2=bull)
    "regime_confidence": float, # How dominant the regime is (0-1)

    # Multi-timeframe
    "mtf_alignment": dict,      # Direction per timeframe
    "mtf_score": int,           # Count of aligned timeframes (0-4)

    # Final recommendation
    "signal_strength": float,   # 0-1 composite score
    "suggested_action": str     # BUY / SELL / HOLD
}
```

**Slope Calculation:**
- Compute median forecast price at final step
- Calculate angle: `theta = arctan((forecast - current) / n_steps)`
- Normalize by recent volatility to make comparable across assets
- Convert to degrees

**Default Thresholds (to be optimized):**
| Parameter | Default | Search Range |
|-----------|---------|--------------|
| slope_threshold | 10 deg | 5 - 20 deg |
| mtf_min_alignment | 3 | 2 - 4 |
| band_width_max | 5% | 2% - 10% |
| regime_confidence_min | 0.6 | 0.4 - 0.8 |

---

### 2. TradingAgents Integration

**Purpose:** LLM-powered validation of high-confidence MCMC signals.

**Budget:** $1-2/day (~$30-60/month)

**Trigger Conditions:**
1. MCMC signal_strength > 0.75
2. LLM not already called today
3. Sufficient portfolio value at risk

**Integration Flow:**
1. MCMC generates strong signal
2. Check daily budget
3. Call TradingAgents with context:
   - MCMC signal data
   - Current positions
   - Recent price action
4. TradingAgents returns: direction, confidence, reasoning
5. If MCMC and LLM agree: full position
6. If disagree: no trade

**Fallback:** If API fails or budget exceeded, continue with MCMC-only signals (smaller position sizes).

**Cost Tracking:** Log all LLM calls to `data/results/llm_usage.json`

---

### 3. Backtesting Engine

**Purpose:** GPU-accelerated simulation of trading strategies on historical data.

**Validation Method:** Walk-forward
- Train window: 6 months
- Test window: 2 months
- Roll forward, repeat
- Prevents lookahead bias

**Cost Model:**
| Cost Type | Value |
|-----------|-------|
| Slippage | 0.02% per trade |
| Commission | $0 |
| Spread | Estimated from data |
| Market Impact | 0.01% for positions > $10k |

**Output Metrics:**
- Primary: Total Return % (optimization target)
- Secondary: Sharpe Ratio, Max Drawdown %, Win Rate %, Profit Factor, Trade Count, Avg Duration

**Data Requirements:**
- 2+ years of historical data
- Timeframes: 1m, 5m, 15m, 1h, 1d
- Cached locally as Parquet files

---

### 4. Parameter Optimizer

**Purpose:** Find parameters that maximize returns.

**Algorithm:** Optuna with TPE (Tree-structured Parzen Estimator)
- Smarter than random/grid search
- Learns promising regions
- Early pruning of bad trials

**Search Space:**

| Category | Parameter | Range |
|----------|-----------|-------|
| MCMC | slope_threshold | [5, 20] degrees |
| MCMC | mtf_min_alignment | [2, 4] |
| MCMC | band_width_max | [2%, 10%] |
| MCMC | n_regimes | [2, 3, 4] |
| MCMC | n_simulations | [10000, 50000] |
| Signal | signal_strength_min | [0.5, 0.9] |
| Signal | mtf_weight | [0.1, 0.5] |
| Signal | regime_weight | [0.1, 0.5] |
| Risk | position_size_pct | [5%, 25%] |
| Risk | stop_loss_pct | [1%, 5%] |
| Risk | take_profit_pct | [2%, 10%] |

**Budget:** 500-1000 trials (~2-4 hours on 2080 Super)

**Anti-Overfitting Measures:**
- Walk-forward validation
- Minimum 100 trades required
- Penalize extreme parameter values

**Output:** `config/best_params.json` with best parameters and performance metrics

---

### 5. Broker Interface

**Purpose:** Abstract execution layer supporting paper and live trading.

**Interface:**
```python
class BaseBroker(ABC):
    def buy(self, ticker: str, quantity: int, price: float) -> Order
    def sell(self, ticker: str, quantity: int, price: float) -> Order
    def get_positions(self) -> list[Position]
    def get_equity(self) -> float
    def get_cash(self) -> float
```

**Implementations:**
- `PaperBroker`: Simulated trading with realistic fills
- `AlpacaBroker`: Future integration with Alpaca API

---

## File Structure

```
mcmc-cuda-model/
├── main.py                      # Existing GUI (unchanged)
├── simple_mcmc.py               # Existing demo (unchanged)
├── requirements.txt             # Updated with new deps
│
├── trading/                     # Core trading system
│   ├── __init__.py
│   ├── indicator.py             # MCMCIndicator class
│   ├── signal_combiner.py       # Merge MCMC + LLM signals
│   ├── risk_manager.py          # Position sizing, limits
│   └── position_manager.py      # Track positions, P&L
│
├── brokers/                     # Execution layer
│   ├── __init__.py
│   ├── base.py                  # Abstract broker interface
│   ├── paper.py                 # PaperBroker
│   └── alpaca.py                # AlpacaBroker (future)
│
├── backtesting/                 # Backtest engine
│   ├── __init__.py
│   ├── engine.py                # GPU-accelerated backtester
│   ├── data_loader.py           # Fetch & cache historical data
│   └── metrics.py               # Performance calculations
│
├── optimization/                # Parameter search
│   ├── __init__.py
│   ├── optimizer.py             # Optuna integration
│   └── fitness.py               # Objective function
│
├── integrations/                # External systems
│   ├── __init__.py
│   └── tradingagents_bridge.py  # TradingAgents wrapper
│
├── config/                      # Configuration
│   ├── default.yaml             # Default parameters
│   ├── best_params.json         # Optimizer output
│   └── tickers.yaml             # Watchlist
│
├── data/                        # Data storage
│   ├── cache/                   # Price data (parquet)
│   └── results/                 # Backtest results
│
├── scripts/                     # Entry points
│   ├── run_backtest.py
│   ├── run_optimizer.py
│   ├── run_paper_trader.py
│   └── download_data.py
│
└── CLAUDE.md                    # Project conventions
```

---

## Dependencies

**Existing:**
- yfinance>=0.2.18
- matplotlib>=3.7.0
- pandas>=2.0.0
- numpy>=1.24.0
- scipy>=1.10.0
- pytz>=2023.3
- cupy-cuda12x>=12.0.0

**New:**
- optuna>=3.5.0 (parameter optimization)
- torch>=2.0.0 (GPU compute, optional neural net)
- pyarrow>=14.0.0 (fast parquet I/O)
- pyyaml>=6.0.0 (config files)

**External:**
- TradingAgents (git clone from TauricResearch/TradingAgents)

---

## Success Criteria

1. MCMC indicator generates signals with measurable slope and confidence
2. Multi-timeframe scanner checks 4 timeframes in parallel
3. Backtester runs 500+ parameter combinations in <4 hours
4. Optimizer finds parameters with positive returns on walk-forward test
5. Paper trader executes signals with proper position sizing
6. LLM integration stays within $1-2/day budget
7. System beats buy-and-hold SPY on backtested period

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Overfitting to historical data | Walk-forward validation, min trade count |
| LLM API costs exceed budget | Daily cap, fallback to MCMC-only |
| GPU memory overflow | Batch simulations, monitor VRAM |
| yfinance rate limiting | Local caching, exponential backoff |
| Strategy doesn't generalize | Test across 8 tickers, multiple periods |

---

## Future Enhancements (Phase 2)

1. Neural network for dynamic parameter adjustment based on market regime
2. Reinforcement learning agent trained on backtest environment
3. Live broker integration (Alpaca)
4. Real-time streaming data
5. Portfolio-level optimization (correlation, diversification)
