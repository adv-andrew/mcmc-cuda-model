"""
Tests for backtesting/data_loader.py
"""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backtesting.data_loader import DataLoader, _is_intraday


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 10, freq: str = "D", tz: str = "UTC") -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame with a timezone-aware DatetimeIndex.

    Uses a start date relative to today so cached data always falls within
    the dynamic date-range window used by DataLoader.fetch.
    """
    start = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=n + 5)
    idx = pd.date_range(start, periods=n, freq=freq, tz=tz)
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [105.0 + i for i in range(n)],
            "Low": [95.0 + i for i in range(n)],
            "Close": [102.0 + i for i in range(n)],
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )


def _make_intraday_ohlcv(n: int = 20) -> pd.DataFrame:
    """Intraday bars at 30-min intervals starting at 09:00 ET (converted to UTC)."""
    # 09:00 ET = 14:00 UTC
    idx = pd.date_range("2024-01-02 14:00", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [101.0] * n,
            "Low": [99.0] * n,
            "Close": [100.5] * n,
            "Volume": [50_000] * n,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# _is_intraday helper
# ---------------------------------------------------------------------------

class TestIsIntraday:
    def test_daily_not_intraday(self):
        assert _is_intraday("1d") is False

    def test_weekly_not_intraday(self):
        assert _is_intraday("1wk") is False

    def test_minute_is_intraday(self):
        assert _is_intraday("1m") is True

    def test_hourly_is_intraday(self):
        assert _is_intraday("1h") is True

    def test_5m_is_intraday(self):
        assert _is_intraday("5m") is True


# ---------------------------------------------------------------------------
# DataLoader.__init__
# ---------------------------------------------------------------------------

class TestDataLoaderInit:
    def test_default_cache_dir_created(self, tmp_path):
        cache = tmp_path / "cache"
        dl = DataLoader(cache_dir=str(cache))
        assert cache.exists()

    def test_custom_cache_dir(self, tmp_path):
        custom = tmp_path / "my" / "custom" / "dir"
        dl = DataLoader(cache_dir=str(custom))
        assert custom.exists()


# ---------------------------------------------------------------------------
# DataLoader.fetch – single ticker
# ---------------------------------------------------------------------------

class TestFetch:
    @patch("backtesting.data_loader.yf.download")
    def test_fetch_returns_dataframe(self, mock_dl, tmp_path):
        mock_dl.return_value = _make_ohlcv()
        dl = DataLoader(cache_dir=str(tmp_path))
        df = dl.fetch("AAPL", timeframe="1d", days=10, use_cache=False)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    @patch("backtesting.data_loader.yf.download")
    def test_fetch_normalises_ticker_to_uppercase(self, mock_dl, tmp_path):
        mock_dl.return_value = _make_ohlcv()
        dl = DataLoader(cache_dir=str(tmp_path))
        dl.fetch("aapl", timeframe="1d", days=10, use_cache=False)
        call_args = mock_dl.call_args
        assert call_args[0][0] == "AAPL"

    @patch("backtesting.data_loader.yf.download")
    def test_fetch_empty_dataframe_graceful(self, mock_dl, tmp_path):
        mock_dl.return_value = pd.DataFrame()
        dl = DataLoader(cache_dir=str(tmp_path))
        df = dl.fetch("FAKE", timeframe="1d", days=10, use_cache=False)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch("backtesting.data_loader.yf.download")
    def test_fetch_yfinance_exception_returns_empty(self, mock_dl, tmp_path):
        mock_dl.side_effect = Exception("network error")
        dl = DataLoader(cache_dir=str(tmp_path))
        df = dl.fetch("AAPL", timeframe="1d", days=10, use_cache=False)
        assert df.empty

    @patch("backtesting.data_loader.yf.download")
    def test_fetch_with_start_end_date(self, mock_dl, tmp_path):
        mock_dl.return_value = _make_ohlcv(5)
        dl = DataLoader(cache_dir=str(tmp_path))
        df = dl.fetch(
            "AAPL",
            timeframe="1d",
            start_date="2024-01-02",
            end_date="2024-01-06",
            use_cache=False,
        )
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# DataLoader – caching
# ---------------------------------------------------------------------------

class TestCaching:
    @patch("backtesting.data_loader.yf.download")
    def test_cache_file_created(self, mock_dl, tmp_path):
        mock_dl.return_value = _make_ohlcv()
        dl = DataLoader(cache_dir=str(tmp_path))
        dl.fetch("AAPL", timeframe="1d", days=30, use_cache=True)
        cache_file = tmp_path / "AAPL_1d.parquet"
        assert cache_file.exists()

    @patch("backtesting.data_loader.yf.download")
    def test_cache_used_on_second_call(self, mock_dl, tmp_path):
        mock_dl.return_value = _make_ohlcv(30)
        dl = DataLoader(cache_dir=str(tmp_path))
        # First call – writes cache
        dl.fetch("AAPL", timeframe="1d", days=30, use_cache=True)
        first_call_count = mock_dl.call_count
        # Second call – should use cache (no new download)
        dl.fetch("AAPL", timeframe="1d", days=30, use_cache=True)
        assert mock_dl.call_count == first_call_count, (
            "yf.download should not be called again when cache is fresh"
        )

    @patch("backtesting.data_loader.yf.download")
    def test_cache_bypassed_when_use_cache_false(self, mock_dl, tmp_path):
        mock_dl.return_value = _make_ohlcv(30)
        dl = DataLoader(cache_dir=str(tmp_path))
        dl.fetch("AAPL", timeframe="1d", days=30, use_cache=True)
        dl.fetch("AAPL", timeframe="1d", days=30, use_cache=False)
        assert mock_dl.call_count == 2

    @patch("backtesting.data_loader.yf.download")
    def test_stale_cache_triggers_redownload(self, mock_dl, tmp_path):
        """A cache file older than TTL should be ignored and data re-fetched."""
        mock_dl.return_value = _make_ohlcv(30)
        dl = DataLoader(cache_dir=str(tmp_path))
        dl.fetch("AAPL", timeframe="1d", days=30, use_cache=True)

        # Backdate the cache file modification time by 25 hours
        cache_file = tmp_path / "AAPL_1d.parquet"
        old_mtime = time.time() - 25 * 3600
        os.utime(cache_file, (old_mtime, old_mtime))

        dl.fetch("AAPL", timeframe="1d", days=30, use_cache=True)
        assert mock_dl.call_count == 2

    @patch("backtesting.data_loader.yf.download")
    def test_intraday_cache_ttl_is_1h(self, mock_dl, tmp_path):
        """Intraday cache stale after 1 h, not 24 h."""
        mock_dl.return_value = _make_intraday_ohlcv()
        dl = DataLoader(cache_dir=str(tmp_path))
        dl.fetch("AAPL", timeframe="5m", days=5, use_cache=True)

        # Backdate by 2 hours – should be stale for intraday
        cache_file = tmp_path / "AAPL_5m.parquet"
        old_mtime = time.time() - 2 * 3600
        os.utime(cache_file, (old_mtime, old_mtime))

        dl.fetch("AAPL", timeframe="5m", days=5, use_cache=True)
        assert mock_dl.call_count == 2


# ---------------------------------------------------------------------------
# DataLoader.fetch_multi_timeframe
# ---------------------------------------------------------------------------

class TestFetchMultiTimeframe:
    @patch("backtesting.data_loader.yf.download")
    def test_returns_dict_keyed_by_timeframe(self, mock_dl, tmp_path):
        mock_dl.return_value = _make_ohlcv()
        dl = DataLoader(cache_dir=str(tmp_path))
        timeframes = ["1d", "1h"]
        result = dl.fetch_multi_timeframe("AAPL", timeframes=timeframes, days=30)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(timeframes)

    @patch("backtesting.data_loader.yf.download")
    def test_each_value_is_dataframe(self, mock_dl, tmp_path):
        mock_dl.return_value = _make_ohlcv()
        dl = DataLoader(cache_dir=str(tmp_path))
        result = dl.fetch_multi_timeframe("AAPL", timeframes=["1d", "5m"], days=10)
        for tf, df in result.items():
            assert isinstance(df, pd.DataFrame), f"Expected DataFrame for {tf}"

    @patch("backtesting.data_loader.yf.download")
    def test_empty_timeframes_list(self, mock_dl, tmp_path):
        dl = DataLoader(cache_dir=str(tmp_path))
        result = dl.fetch_multi_timeframe("AAPL", timeframes=[], days=10)
        assert result == {}
        mock_dl.assert_not_called()


# ---------------------------------------------------------------------------
# DataLoader.fetch_multiple_tickers
# ---------------------------------------------------------------------------

class TestFetchMultipleTickers:
    @patch("backtesting.data_loader.yf.download")
    def test_returns_dict_keyed_by_ticker(self, mock_dl, tmp_path):
        mock_dl.return_value = _make_ohlcv()
        dl = DataLoader(cache_dir=str(tmp_path))
        tickers = ["AAPL", "MSFT"]
        result = dl.fetch_multiple_tickers(tickers, timeframe="1d", days=30)
        assert set(result.keys()) == {"AAPL", "MSFT"}

    @patch("backtesting.data_loader.yf.download")
    def test_lowercase_tickers_normalised(self, mock_dl, tmp_path):
        mock_dl.return_value = _make_ohlcv()
        dl = DataLoader(cache_dir=str(tmp_path))
        result = dl.fetch_multiple_tickers(["aapl", "msft"], timeframe="1d", days=10)
        assert "AAPL" in result
        assert "MSFT" in result

    @patch("backtesting.data_loader.yf.download")
    def test_failed_ticker_returns_empty_df(self, mock_dl, tmp_path):
        mock_dl.return_value = pd.DataFrame()
        dl = DataLoader(cache_dir=str(tmp_path))
        result = dl.fetch_multiple_tickers(["FAKE"], timeframe="1d", days=10)
        assert result["FAKE"].empty


# ---------------------------------------------------------------------------
# DataLoader._filter_rth
# ---------------------------------------------------------------------------

class TestFilterRth:
    def test_filters_to_rth_window(self, tmp_path):
        dl = DataLoader(cache_dir=str(tmp_path))
        # Create bars at every hour from 08:00 to 18:00 ET
        import pytz
        et = pytz.timezone("America/New_York")
        times_utc = pd.date_range("2024-01-02 13:00", periods=11, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1},
            index=times_utc,
        )
        filtered = dl._filter_rth(df)
        et_times = filtered.index.tz_convert(et)
        for t in et_times.time:
            assert t.strftime("%H:%M") >= "09:30"
            assert t.strftime("%H:%M") <= "16:00"

    def test_empty_df_passthrough(self, tmp_path):
        dl = DataLoader(cache_dir=str(tmp_path))
        df = pd.DataFrame()
        result = dl._filter_rth(df)
        assert result.empty
