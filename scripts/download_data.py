"""Download market data for all tickers and timeframes defined in config."""

import sys
sys.path.insert(0, ".")

import logging
import yaml

from backtesting.data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_tickers(path: str = "config/tickers.yaml") -> list:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("watchlist", [])


def load_timeframes(path: str = "config/default.yaml") -> list:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("timeframes", ["1d"])


def main():
    tickers = load_tickers()
    timeframes = load_timeframes()
    loader = DataLoader(cache_dir="data/cache")

    logger.info("Downloading data for %d tickers x %d timeframes", len(tickers), len(timeframes))

    for ticker in tickers:
        for tf in timeframes:
            logger.info("Fetching %s @ %s ...", ticker, tf)
            df = loader.fetch(ticker, timeframe=tf)
            if df.empty:
                logger.warning("  No data returned for %s @ %s", ticker, tf)
            else:
                logger.info("  Got %d rows for %s @ %s", len(df), ticker, tf)

    logger.info("Download complete.")


if __name__ == "__main__":
    main()
