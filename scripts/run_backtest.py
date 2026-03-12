"""Run walk-forward backtest using best_params.json if available, else default.yaml."""

import sys
sys.path.insert(0, ".")

import json
import logging
import os
import yaml

from backtesting.data_loader import DataLoader
from backtesting.engine import Backtester

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BEST_PARAMS_PATH = "config/best_params.json"
DEFAULT_CONFIG_PATH = "config/default.yaml"
TICKERS_CONFIG_PATH = "config/tickers.yaml"


def load_params() -> dict:
    if os.path.exists(BEST_PARAMS_PATH):
        logger.info("Loading params from %s", BEST_PARAMS_PATH)
        with open(BEST_PARAMS_PATH, "r") as fh:
            payload = json.load(fh)
        return payload.get("params", {})

    logger.info("best_params.json not found; using defaults from %s", DEFAULT_CONFIG_PATH)
    with open(DEFAULT_CONFIG_PATH, "r") as fh:
        cfg = yaml.safe_load(fh)

    params = {}
    params.update(cfg.get("mcmc", {}))
    params.update(cfg.get("signal", {}))
    params.update(cfg.get("risk", {}))
    return params


def load_tickers() -> list:
    with open(TICKERS_CONFIG_PATH, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("watchlist", [])


def main():
    params = load_params()
    tickers = load_tickers()

    loader = DataLoader(cache_dir="data/cache")
    logger.info("Loading daily data for %d tickers ...", len(tickers))
    data = loader.fetch_multiple_tickers(tickers, timeframe="1d", days=365)
    data = {k: v for k, v in data.items() if not v.empty}

    if not data:
        logger.error("No data loaded. Run download_data.py first.")
        sys.exit(1)

    backtester = Backtester(data=data, params=params)
    logger.info("Running walk-forward backtest ...")
    results = backtester.run_walk_forward()

    if not results:
        logger.warning("Walk-forward produced no results (insufficient data?).")
        return

    print(f"\n{'='*60}")
    print(f"Walk-Forward Backtest Results ({len(results)} windows)")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        m = r["metrics"]
        print(
            f"Window {i:2d} | {r['window_start']} -> {r['window_end']} | "
            f"Return: {m.get('total_return_pct', 0):.2f}% | "
            f"Sharpe: {m.get('sharpe_ratio', 0):.3f} | "
            f"MaxDD: {m.get('max_drawdown_pct', 0):.2f}% | "
            f"Trades: {m.get('total_trades', 0)}"
        )

    all_returns = [r["metrics"].get("total_return_pct", 0) for r in results]
    all_sharpe = [r["metrics"].get("sharpe_ratio", 0) for r in results]
    print(f"\nAvg Return: {sum(all_returns)/len(all_returns):.2f}%")
    print(f"Avg Sharpe: {sum(all_sharpe)/len(all_sharpe):.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
