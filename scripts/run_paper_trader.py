"""Paper trading loop: fetch data, generate signals, execute trades, sleep 5 minutes.

Uses multi-timeframe (MTF) confirmation: requires 1h and 1d signals to align
before taking a trade. This reduces false signals and improves win rate.
"""

import sys
sys.path.insert(0, ".")

import logging
import time
import yaml

from backtesting.data_loader import DataLoader
from brokers.paper import PaperBroker
from trading.indicator import MCMCIndicator
from trading.risk_manager import RiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TICKERS_CONFIG_PATH = "config/tickers.yaml"
DEFAULT_CONFIG_PATH = "config/default.yaml"
SLEEP_SECONDS = 300  # 5 minutes
MTF_TIMEFRAMES = ["1h", "1d"]  # Multi-timeframe: hourly + daily must align


def load_tickers() -> list:
    with open(TICKERS_CONFIG_PATH, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("watchlist", [])


def load_config() -> dict:
    with open(DEFAULT_CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)


def main():
    tickers = load_tickers()
    cfg = load_config()

    mcmc_cfg = cfg.get("mcmc", {})
    signal_cfg = cfg.get("signal", {})
    risk_cfg = cfg.get("risk", {})

    broker = PaperBroker(initial_cash=100_000.0)
    indicator = MCMCIndicator(
        n_simulations=mcmc_cfg.get("n_simulations", 25000),
        n_regimes=mcmc_cfg.get("n_regimes", 3),
        slope_threshold=signal_cfg.get("slope_threshold", 10.0),
    )
    risk = RiskManager(
        broker=broker,
        position_size_pct=risk_cfg.get("position_size_pct", 0.10),
        stop_loss_pct=risk_cfg.get("stop_loss_pct", 0.02),
        take_profit_pct=risk_cfg.get("take_profit_pct", 0.05),
    )
    loader = DataLoader(cache_dir="data/cache")
    signal_strength_min = signal_cfg.get("signal_strength_min", 0.7)

    iteration = 0
    logger.info("Starting paper trading loop for tickers: %s", tickers)

    while True:
        iteration += 1
        logger.info("--- Iteration %d ---", iteration)

        for ticker in tickers:
            df = loader.fetch(ticker, timeframe="1d", days=90, use_cache=False)
            if df.empty or len(df) < 5:
                logger.warning("Insufficient data for %s, skipping.", ticker)
                continue

            current_price = float(df["Close"].iloc[-1])
            broker.set_price(ticker, current_price)

            # Check exits
            position = broker.get_position(ticker)
            if position is not None and position.quantity > 0:
                if risk.should_stop_loss(ticker):
                    broker.sell(ticker, position.quantity, current_price)
                    logger.info("STOP LOSS exit: %s @ %.2f", ticker, current_price)
                    continue
                if risk.should_take_profit(ticker):
                    broker.sell(ticker, position.quantity, current_price)
                    logger.info("TAKE PROFIT exit: %s @ %.2f", ticker, current_price)
                    continue

            # Generate signal and enter if conditions met
            if not risk.can_trade():
                logger.info("Risk limits reached, skipping entries.")
                break

            # Fetch multi-timeframe data
            try:
                mtf_data = {}
                for tf in MTF_TIMEFRAMES:
                    days = 90 if tf == "1d" else 7  # 7 days of hourly data
                    tf_df = loader.fetch(ticker, timeframe=tf, days=days, use_cache=False)
                    if not tf_df.empty and len(tf_df) >= 5:
                        mtf_data[tf] = tf_df

                if len(mtf_data) < 2:
                    # Fall back to daily only if hourly unavailable
                    signal = indicator.generate_signal(ticker, df, "1d")
                    mtf_aligned = True  # No MTF check possible
                    mtf_info = "1d only"
                else:
                    # Use MTF signal with alignment requirement
                    signal = indicator.generate_mtf_signal(ticker, mtf_data)
                    mtf_alignment = signal.get("mtf_alignment", {})
                    mtf_score = signal.get("mtf_score", 0)

                    # Require ALL timeframes to agree for entry
                    directions = list(mtf_alignment.values())
                    mtf_aligned = len(set(directions)) == 1 and directions[0] != "NEUTRAL"
                    mtf_info = f"1h={mtf_alignment.get('1h', '?')} 1d={mtf_alignment.get('1d', '?')}"

            except Exception as exc:
                logger.warning("Signal generation failed for %s: %s", ticker, exc)
                continue

            logger.info(
                "%s: action=%s strength=%.3f regime=%s MTF=[%s] aligned=%s",
                ticker,
                signal["suggested_action"],
                signal["signal_strength"],
                signal.get("regime", "?"),
                mtf_info,
                mtf_aligned,
            )

            if (
                signal["suggested_action"] == "BUY"
                and signal["signal_strength"] >= signal_strength_min
                and mtf_aligned
                and broker.get_position(ticker) is None
            ):
                qty = risk.calculate_position_size(ticker, current_price, signal["signal_strength"])
                if qty > 0:
                    try:
                        broker.buy(ticker, qty, current_price)
                        logger.info("BUY %d shares of %s @ %.2f (MTF confirmed)", qty, ticker, current_price)
                    except ValueError as exc:
                        logger.warning("Buy failed for %s: %s", ticker, exc)

        equity = broker.get_equity()
        positions = broker.get_positions()
        logger.info(
            "Equity: $%.2f | Open positions: %d",
            equity,
            len(positions),
        )

        logger.info("Sleeping %d seconds ...", SLEEP_SECONDS)
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
