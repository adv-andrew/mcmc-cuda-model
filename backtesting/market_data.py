"""
Market data access for research and options backtests.

Yahoo Finance is not reachable from every environment (e.g. sandboxed CI),
so this module loads daily history from public GitHub-hosted mirrors first,
caches it locally as CSV, and only falls back to yfinance when a symbol is
not mirrored.

Sources
-------
- Equities/ETFs (SPY, QQQ, IWM, AAPL, NVDA; 2010-present, updated daily):
  github.com/abi1010-git/predicting-stock-market-personal-project
- CBOE VIX index (1990-present):
  github.com/datasets/finance-vix

Returned frames use a tz-naive ``DatetimeIndex`` and the columns
``Open, High, Low, Close, AdjClose, Volume``. ``Close`` is split-adjusted
but *not* dividend-adjusted, which is what listed options settle against.
"""

from __future__ import annotations

import io
import logging
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

GITHUB_EQUITY_URL = (
    "https://raw.githubusercontent.com/abi1010-git/"
    "predicting-stock-market-personal-project/main/data/{symbol}.csv"
)
GITHUB_VIX_URL = (
    "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"
)
GITHUB_SYMBOLS = frozenset({"SPY", "QQQ", "IWM", "AAPL", "NVDA"})

DEFAULT_CACHE_DIR = Path("data/cache/daily")
CACHE_TTL_HOURS = 12
COLUMNS = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]


def _http_get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def _cache_fresh(path: Path, ttl_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600


def _normalise_github_equity(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adjusted_close": "AdjClose",
            "volume": "Volume",
        }
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[COLUMNS].astype(float)


def _normalise_vix(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=str.title)  # DATE -> Date, OPEN -> Open ...
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["AdjClose"] = df["Close"]
    df["Volume"] = 0.0
    return df[COLUMNS].astype(float)


def _fetch_yfinance(symbol: str) -> pd.DataFrame:
    import yfinance as yf  # imported lazily: optional in sandboxed envs

    df = yf.download(symbol, start="2005-01-01", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Adj Close": "AdjClose"})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[COLUMNS].astype(float)


def load_daily(
    symbol: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    ttl_hours: float = CACHE_TTL_HOURS,
) -> pd.DataFrame:
    """Load full daily OHLCV history for ``symbol`` (``"VIX"`` or ``"^VIX"`` for VIX).

    Order of preference: fresh local cache -> GitHub mirror -> yfinance ->
    stale local cache. Raises ``RuntimeError`` if nothing is available.
    """
    symbol = symbol.upper().lstrip("^")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol}.csv"

    if not refresh and _cache_fresh(cache_path, ttl_hours):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    df: Optional[pd.DataFrame] = None
    try:
        if symbol == "VIX":
            df = _normalise_vix(pd.read_csv(io.BytesIO(_http_get(GITHUB_VIX_URL))))
        elif symbol in GITHUB_SYMBOLS:
            url = GITHUB_EQUITY_URL.format(symbol=symbol)
            df = _normalise_github_equity(pd.read_csv(io.BytesIO(_http_get(url))))
    except Exception as exc:  # network / parse failure
        logger.warning("GitHub mirror failed for %s: %s", symbol, exc)

    if df is None or df.empty:
        try:
            df = _fetch_yfinance(("^" + symbol) if symbol == "VIX" else symbol)
        except Exception as exc:
            logger.warning("yfinance failed for %s: %s", symbol, exc)

    if df is not None and not df.empty:
        df.to_csv(cache_path)
        return df

    if cache_path.exists():
        logger.warning("Using stale cache for %s", symbol)
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    raise RuntimeError(f"No data source available for {symbol}")


# Stock splits since 2010 for mirrored single names: (effective date, ratio).
# Mirror prices are split-adjusted; options trade on the *actual* price, and
# strike spacing / minimum ticks only make sense at that scale.
SPLITS = {
    "AAPL": [("2014-06-09", 7.0), ("2020-08-31", 4.0)],
    "NVDA": [("2021-07-20", 4.0), ("2024-06-10", 10.0)],
}


def split_factor(index: pd.DatetimeIndex, symbol: str) -> pd.Series:
    """Multiplier that converts split-adjusted prices back to traded prices."""
    factor = pd.Series(1.0, index=index)
    for date, ratio in SPLITS.get(symbol.upper(), []):
        factor[index < pd.Timestamp(date)] *= ratio
    return factor


def traded_close(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Close as it actually printed on the day (split-unadjusted)."""
    return df["Close"] * split_factor(df.index, symbol)


def load_universe(
    symbols: Iterable[str],
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load several symbols, skipping (and logging) any that fail."""
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[sym.upper()] = load_daily(sym, cache_dir=cache_dir, refresh=refresh)
        except RuntimeError as exc:
            logger.warning("%s", exc)
    return out
