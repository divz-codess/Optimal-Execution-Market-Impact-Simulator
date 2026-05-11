"""
fetch_data.py
=============
Downloads intraday OHLCV market data using yfinance and persists it to
data/market_data/<ticker>_<interval>.csv relative to the project root.

Public API
----------
fetch_intraday_data(ticker, period, interval) -> pd.DataFrame
    Download intraday data and cache it locally.

load_cached_data(ticker, interval) -> pd.DataFrame | None
    Load previously cached data, or None if not present.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent.parent   # project root
MARKET_DATA_DIR = _ROOT / "data" / "market_data"
MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def fetch_intraday_data(
    ticker: str = "AAPL",
    period: str = "5d",
    interval: str = "1m",
) -> pd.DataFrame:
    """Download intraday OHLCV data and save to CSV.

    Parameters
    ----------
    ticker   : Yahoo Finance ticker symbol, e.g. "AAPL"
    period   : Lookback window accepted by yfinance, e.g. "5d", "7d", "1mo"
    interval : Bar frequency accepted by yfinance, e.g. "1m", "5m", "1h"

    Returns
    -------
    pd.DataFrame with columns: datetime, open, high, low, close, volume
        Index is reset so datetime is a plain column.

    Notes
    -----
    yfinance intraday data is only available for the last 60 days (1m bars
    capped at 7 days).  The function falls back to cached data if the
    download fails.
    """
    log.info("Fetching %s %s bars for %s …", interval, period, ticker)

    ticker_obj = yf.Ticker(ticker)
    raw: pd.DataFrame = ticker_obj.history(period=period, interval=interval)

    if raw.empty:
        log.warning("yfinance returned empty DataFrame — trying cached data.")
        cached = load_cached_data(ticker, interval)
        if cached is not None:
            return cached
        raise ValueError(
            f"No data returned for {ticker} ({interval}, {period}) and no "
            "cached data found."
        )

    # Normalise column names to lowercase.
    raw.columns = [c.lower() for c in raw.columns]

    # Keep only the OHLCV columns we need.
    cols = ["open", "high", "low", "close", "volume"]
    df = raw[[c for c in cols if c in raw.columns]].copy()

    # Flatten the timezone-aware DatetimeIndex into a plain column.
    df.index.name = "datetime"
    df = df.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Remove rows with NaN prices.
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Persist to CSV.
    csv_path = _csv_path(ticker, interval)
    df.to_csv(csv_path, index=False)
    log.info("Saved %d rows → %s", len(df), csv_path)

    return df


def load_cached_data(ticker: str, interval: str = "1m") -> pd.DataFrame | None:
    """Load previously cached CSV, returning None if not found.

    Parameters
    ----------
    ticker   : Ticker symbol.
    interval : Bar interval used when the data was fetched.

    Returns
    -------
    pd.DataFrame or None
    """
    csv_path = _csv_path(ticker, interval)
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    log.info("Loaded %d cached rows from %s", len(df), csv_path)
    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _csv_path(ticker: str, interval: str) -> Path:
    """Return the canonical CSV path for (ticker, interval)."""
    return MARKET_DATA_DIR / f"{ticker.upper()}_{interval}.csv"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch intraday market data.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="5d")
    parser.add_argument("--interval", default="1m")
    args = parser.parse_args()

    df = fetch_intraday_data(args.ticker, args.period, args.interval)
    print(df.tail())
