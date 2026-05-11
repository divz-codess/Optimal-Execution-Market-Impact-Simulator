"""
calibration.py
==============
Calibrates Almgren-Chriss model parameters from historical intraday OHLCV
data.  All functions accept a pandas DataFrame with at minimum the columns
[datetime, close, volume] as returned by fetch_data.fetch_intraday_data().

Public API
----------
compute_log_returns(df)        -> pd.Series
estimate_volatility(df)        -> float
estimate_avg_volume(df)        -> float
estimate_market_impact(df)     -> dict[str, float]
calibrate_all(df)              -> dict[str, float]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """Compute log returns of the close price series.

    Log return at time t:   r_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    df : DataFrame with a 'close' column.

    Returns
    -------
    pd.Series of log returns (length = len(df) - 1).
    """
    closes = df["close"].dropna().values
    if len(closes) < 2:
        raise ValueError("Need at least 2 close prices to compute log returns.")

    log_returns = np.log(closes[1:] / closes[:-1])
    return pd.Series(log_returns, name="log_return")


def estimate_volatility(df: pd.DataFrame, annualise: bool = False) -> float:
    """Estimate per-bar historical volatility as the std dev of log returns.

    Parameters
    ----------
    df        : DataFrame with a 'close' column.
    annualise : If True, scale by sqrt(bars_per_year) using 252 trading days
                × 390 one-minute bars per day.

    Returns
    -------
    float — volatility in the same units as the bar frequency (or annualised).
    """
    log_returns = compute_log_returns(df)
    vol = float(log_returns.std(ddof=1))

    if annualise:
        bars_per_year = 252 * 390   # 1-minute bars
        vol *= np.sqrt(bars_per_year)

    return vol


def estimate_avg_volume(df: pd.DataFrame) -> float:
    """Compute the average traded volume per bar.

    Parameters
    ----------
    df : DataFrame with a 'volume' column.

    Returns
    -------
    float — mean volume per bar.
    """
    volumes = df["volume"].dropna()
    if volumes.empty:
        raise ValueError("Volume column is empty or all NaN.")
    return float(volumes.mean())


def estimate_market_impact(df: pd.DataFrame) -> dict[str, float]:
    """Estimate simplified Almgren-Chriss market impact coefficients.

    Temporary impact coefficient (eta)
    -----------------------------------
    In the Almgren-Chriss framework the temporary impact cost of trading v_k
    shares in one period is  eta * v_k.  A simple empirical proxy:

        eta ≈ (0.5 * median_spread) / avg_volume

    where  median_spread ≈ typical bid-ask spread estimated from OHLC.

    We use the Corwin-Schultz (2012) high-low spread estimator as a proxy:

        spread_proxy = (high - low) / ((high + low) / 2)   [relative]
    
    Then:  eta = 0.5 * median_relative_spread * avg_price / avg_volume

    Permanent impact coefficient (gamma)
    -------------------------------------
    Permanent impact is harder to estimate from OHLC data alone.  A common
    practitioner heuristic sets gamma ≈ eta / 2, reflecting that temporary
    impact is roughly twice the permanent footprint for typical large trades.

    Parameters
    ----------
    df : DataFrame with columns [high, low, close, volume].

    Returns
    -------
    dict with keys 'eta' (temporary) and 'gamma' (permanent).
    """
    required = {"high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    df = df.dropna(subset=list(required))

    avg_price = float(df["close"].mean())
    avg_vol = estimate_avg_volume(df)

    # Relative spread proxy (dimensionless).
    mid = (df["high"] + df["low"]) / 2.0
    rel_spread = (df["high"] - df["low"]) / mid
    median_rel_spread = float(rel_spread.median())

    # Convert to dollar-per-share temporary impact coefficient.
    # eta has units: $/share² — we divide out avg_volume (shares) so that
    # eta * v_k (shares) yields a $/share price move.
    if avg_vol > 0:
        eta = 0.5 * median_rel_spread * avg_price / avg_vol
    else:
        eta = 0.1   # sensible default

    # Cap eta at a reasonable range to avoid degenerate calibration.
    eta = float(np.clip(eta, 1e-6, 1.0))

    # Permanent impact ~ half of temporary (practitioner heuristic).
    gamma = eta / 2.0

    return {"eta": eta, "gamma": gamma}


def calibrate_all(df: pd.DataFrame) -> dict[str, float]:
    """Return a complete set of calibrated model parameters from market data.

    Combines volatility, volume, and impact estimation into a single dict
    compatible with the C++ simulator's ModelParams struct.

    Parameters
    ----------
    df : DataFrame from fetch_data.fetch_intraday_data().

    Returns
    -------
    dict with keys: sigma, avg_volume, eta, gamma.
        Note: S0, T, N, X, lambda are execution-specific and NOT calibrated here.
    """
    sigma = estimate_volatility(df, annualise=False)
    avg_volume = estimate_avg_volume(df)
    impact = estimate_market_impact(df)

    return {
        "sigma": sigma,
        "avg_volume": avg_volume,
        "eta": impact["eta"],
        "gamma": impact["gamma"],
    }
