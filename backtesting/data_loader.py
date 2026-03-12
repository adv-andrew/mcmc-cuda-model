"""
Data Loader Module for MCMC Trading System.

Fetches OHLCV data via yfinance with parquet-based caching and RTH filtering.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytz
import yfinance as yf

logger = logging.getLogger(__name__)

# Cache freshness thresholds
INTRADAY_CACHE_TTL_HOURS = 1
DAILY_CACHE_TTL_HOURS = 24

# Regular trading hours (Eastern Time)
RTH_START = "09:30"
RTH_END = "16:00"
EASTERN_TZ = pytz.timezone("America/New_York")

# yfinance interval strings treated as intraday
INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


def _is_intraday(timeframe: str) -> bool:
    return timeframe in INTRADAY_INTERVALS


class DataLoader:
    """Loads and caches market OHLCV data using yfinance."""

    def __init__(self, cache_dir: str = "data/cache") -> None:
        """
        Initialize DataLoader.

        Args:
            cache_dir: Directory for parquet cache files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        ticker: str,
        timeframe: str = "1d",
        days: int = 365,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            timeframe: yfinance interval string (e.g. "1d", "1h", "5m").
            days: Number of calendar days to look back when start_date is None.
            start_date: ISO date string for range start (overrides days).
            end_date: ISO date string for range end (defaults to today).
            use_cache: Whether to read/write the parquet cache.

        Returns:
            DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].
            Returns empty DataFrame on failure.
        """
        ticker = ticker.upper().strip()
        end_dt = (
            pd.Timestamp(end_date, tz="UTC") if end_date else pd.Timestamp.utcnow()
        )
        start_dt = (
            pd.Timestamp(start_date, tz="UTC")
            if start_date
            else end_dt - pd.Timedelta(days=days)
        )

        cache_path = self._cache_path(ticker, timeframe)

        if use_cache and self._cache_is_fresh(cache_path, timeframe):
            df = self._load_cache(cache_path)
            if df is not None and not df.empty:
                df = self._slice_date_range(df, start_dt, end_dt)
                if not df.empty:
                    logger.debug("Cache hit: %s %s", ticker, timeframe)
                    return df

        df = self._download(ticker, timeframe, start_dt, end_dt)
        if df.empty:
            logger.warning("No data returned for %s %s", ticker, timeframe)
            return df

        if use_cache:
            self._save_cache(df, cache_path)

        return df

    def fetch_multi_timeframe(
        self,
        ticker: str,
        timeframes: List[str],
        days: int = 365,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for a single ticker across multiple timeframes.

        Args:
            ticker: Ticker symbol.
            timeframes: List of yfinance interval strings.
            days: Look-back window in calendar days.

        Returns:
            Dict mapping timeframe -> DataFrame.
        """
        result: Dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            result[tf] = self.fetch(ticker, timeframe=tf, days=days)
        return result

    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        timeframe: str = "1d",
        days: int = 365,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers at a single timeframe.

        Args:
            tickers: List of ticker symbols.
            timeframe: yfinance interval string.
            days: Look-back window in calendar days.

        Returns:
            Dict mapping ticker -> DataFrame.
        """
        result: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            result[ticker.upper()] = self.fetch(ticker, timeframe=timeframe, days=days)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_rth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to Regular Trading Hours (9:30–16:00 ET).

        Only applied to intraday data that has a timezone-aware index.
        Daily data is returned unchanged.

        Args:
            df: Input OHLCV DataFrame with DatetimeIndex.

        Returns:
            Filtered DataFrame.
        """
        if df.empty:
            return df

        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            return df

        # Convert to Eastern time for RTH comparison
        if idx.tz is None:
            idx_et = idx.tz_localize("UTC").tz_convert(EASTERN_TZ)
        else:
            idx_et = idx.tz_convert(EASTERN_TZ)

        time_of_day = idx_et.time
        rth_start = datetime.strptime(RTH_START, "%H:%M").time()
        rth_end = datetime.strptime(RTH_END, "%H:%M").time()

        mask = (time_of_day >= rth_start) & (time_of_day <= rth_end)
        return df.loc[mask]

    def _download(
        self,
        ticker: str,
        timeframe: str,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
    ) -> pd.DataFrame:
        """Download data from yfinance and normalise columns."""
        try:
            raw = yf.download(
                ticker,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval=timeframe,
                progress=False,
                auto_adjust=True,
                multi_level_index=False,
            )
        except Exception as exc:
            logger.error("yfinance download failed for %s: %s", ticker, exc)
            return pd.DataFrame()

        if raw is None or raw.empty:
            return pd.DataFrame()

        # Normalise column names to title-case
        raw.columns = [c.strip().title() for c in raw.columns]

        # Keep only standard OHLCV columns that exist
        standard_cols = ["Open", "High", "Low", "Close", "Volume"]
        available = [c for c in standard_cols if c in raw.columns]
        df = raw[available].copy()

        # Ensure UTC-aware index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Filter to RTH for intraday
        if _is_intraday(timeframe):
            df = self._filter_rth(df)

        df.dropna(how="all", inplace=True)
        return df

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str, timeframe: str) -> Path:
        safe_tf = timeframe.replace("/", "_")
        return self.cache_dir / f"{ticker}_{safe_tf}.parquet"

    def _cache_is_fresh(self, path: Path, timeframe: str) -> bool:
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.now() - mtime
        ttl = (
            timedelta(hours=INTRADAY_CACHE_TTL_HOURS)
            if _is_intraday(timeframe)
            else timedelta(hours=DAILY_CACHE_TTL_HOURS)
        )
        return age < ttl

    def _load_cache(self, path: Path) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_parquet(path)
            # Ensure index is DatetimeIndex with UTC tz
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            elif df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            return df
        except Exception as exc:
            logger.warning("Failed to read cache %s: %s", path, exc)
            return None

    def _save_cache(self, df: pd.DataFrame, path: Path) -> None:
        try:
            df.to_parquet(path, engine="pyarrow")
        except Exception as exc:
            logger.warning("Failed to write cache %s: %s", path, exc)

    @staticmethod
    def _slice_date_range(
        df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp
    ) -> pd.DataFrame:
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
            df = df.copy()
            df.index = idx
        return df.loc[(idx >= start_dt) & (idx <= end_dt)]
