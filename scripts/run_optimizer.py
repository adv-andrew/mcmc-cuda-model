"""Run Optuna parameter optimizer and save best params to config/best_params.json."""

import sys
sys.path.insert(0, ".")

import logging
import yaml

from backtesting.data_loader import DataLoader
from optimization.optimizer import ParameterOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TICKERS_CONFIG_PATH = "config/tickers.yaml"
BEST_PARAMS_OUTPUT = "config/best_params.json"


def load_tickers() -> list:
    with open(TICKERS_CONFIG_PATH, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("watchlist", [])


def main():
    tickers = load_tickers()

    loader = DataLoader(cache_dir="data/cache")
    logger.info("Loading daily data for %d tickers ...", len(tickers))
    data = loader.fetch_multiple_tickers(tickers, timeframe="1d", days=365)
    data = {k: v for k, v in data.items() if not v.empty}

    if not data:
        logger.error("No data loaded. Run download_data.py first.")
        sys.exit(1)

    optimizer = ParameterOptimizer(data=data)
    logger.info("Starting parameter optimization (n_trials=50) ...")
    best_params, best_value = optimizer.optimize(n_trials=50)

    logger.info("Best fitness value: %.4f", best_value)
    logger.info("Best params: %s", best_params)

    optimizer.save_best_params(best_params, best_value, path=BEST_PARAMS_OUTPUT)
    logger.info("Saved best params to %s", BEST_PARAMS_OUTPUT)

    print(f"\nOptimization complete. Best fitness: {best_value:.4f}")
    print(f"Best params saved to {BEST_PARAMS_OUTPUT}")


if __name__ == "__main__":
    main()
