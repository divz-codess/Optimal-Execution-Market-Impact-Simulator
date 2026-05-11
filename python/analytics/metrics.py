"""
metrics.py
==========
Execution quality metrics for post-trade analysis.

These functions accept DataFrames produced by simulator_interface or
execution_compare and compute standard institutional execution KPIs.

Public API
----------
implementation_shortfall(mean_cost, S0, X)      -> float  [bps]
arrival_price_slippage(mean_cost, S0, X)        -> float  [bps]
participation_rate(X, avg_volume, N, T)         -> float  [fraction]
vwap_shortfall(exec_prices, volumes, ref_vwap)  -> float  [bps]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def implementation_shortfall(
    mean_cost: float,
    S0: float,
    X: float,
) -> float:
    """Implementation shortfall in basis points.

    IS = E[Cost] / (S0 * X) * 10,000

    Parameters
    ----------
    mean_cost : Expected total execution cost in dollars.
    S0        : Arrival (decision) mid-price.
    X         : Total shares traded.

    Returns
    -------
    float — implementation shortfall in bps.
    """
    if S0 <= 0 or X <= 0:
        raise ValueError("S0 and X must be positive.")
    return mean_cost / (S0 * X) * 1e4


def arrival_price_slippage(
    mean_cost: float,
    S0: float,
    X: float,
) -> float:
    """Arrival price slippage in basis points (alias for IS in this model).

    Parameters
    ----------
    mean_cost : Expected total execution cost.
    S0        : Arrival mid-price.
    X         : Total shares.

    Returns
    -------
    float — slippage in bps.
    """
    return implementation_shortfall(mean_cost, S0, X)


def participation_rate(
    X: float,
    avg_volume: float,
    N: int,
    T: float,
) -> float:
    """Estimate average participation rate (order size / market volume per period).

    Parameters
    ----------
    X          : Total shares to liquidate.
    avg_volume : Average market volume per bar.
    N          : Number of trading periods.
    T          : Execution horizon.

    Returns
    -------
    float — fraction of market volume (0–1).  Values > 0.3 are considered
    aggressive and likely to cause significant market impact.
    """
    shares_per_period = X / N
    if avg_volume <= 0:
        return float("nan")
    return shares_per_period / avg_volume


def vwap_shortfall(
    exec_prices: np.ndarray,
    volumes: np.ndarray,
    ref_vwap: float,
) -> float:
    """VWAP shortfall: difference between realised VWAP and reference VWAP.

    Parameters
    ----------
    exec_prices : Array of execution prices per period.
    volumes     : Array of shares traded per period (same length).
    ref_vwap    : Reference VWAP (e.g. market VWAP over same window).

    Returns
    -------
    float — shortfall in bps (positive means we executed worse than VWAP).
    """
    exec_prices = np.asarray(exec_prices)
    volumes     = np.asarray(volumes)

    total_vol = volumes.sum()
    if total_vol <= 0:
        raise ValueError("Total volume must be positive.")

    realised_vwap = float(np.dot(exec_prices, volumes) / total_vol)
    if ref_vwap <= 0:
        raise ValueError("Reference VWAP must be positive.")

    return (ref_vwap - realised_vwap) / ref_vwap * 1e4


def build_summary_table(
    params: dict,
    strategy_results: dict[str, dict],
) -> pd.DataFrame:
    """Build a comprehensive execution summary table.

    Parameters
    ----------
    params           : Model parameter dict.
    strategy_results : Mapping of strategy_name → {mean_cost, std_cost, variance_cost}.

    Returns
    -------
    pd.DataFrame indexed by strategy name.
    """
    rows = []
    for name, res in strategy_results.items():
        mc   = res["mean_cost"]
        std  = res["std_cost"]
        var  = res["variance_cost"]
        rows.append({
            "Strategy":                       name,
            "Mean Cost ($)":                  round(mc, 2),
            "Std Dev ($)":                    round(std, 2),
            "Variance ($²)":                  round(var, 2),
            "Impl. Shortfall (bps)":          round(implementation_shortfall(mc, params["S0"], params["X"]), 4),
            "Cost per Share ($)":             round(mc / params["X"], 6),
        })
    return pd.DataFrame(rows).set_index("Strategy")
