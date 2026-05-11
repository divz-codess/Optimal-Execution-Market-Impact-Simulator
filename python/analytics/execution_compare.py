"""
execution_compare.py
====================
Strategy comparison layer for the Almgren-Chriss execution platform.

Implements three execution strategies in pure Python/NumPy:
  - TWAP   : Time-Weighted Average Price
  - VWAP   : Volume-Weighted Average Price
  - AC     : Almgren-Chriss optimal (closed-form closed-form trajectory)

For head-to-head comparison the module also delegates Monte Carlo simulations
to the C++ engine via simulator_interface (bridge layer).

Public API
----------
twap_trajectory(X, N)                           -> np.ndarray
vwap_trajectory(X, volume_profile)              -> np.ndarray
ac_trajectory(X, N, sigma, eta, lambda_, T)     -> np.ndarray

simulate_strategy_costs(trajectory, params)     -> np.ndarray   [pure-Python MC]
compare_strategies(params, num_paths, volume_profile) -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pure-Python trajectory generators
# ---------------------------------------------------------------------------


def twap_trajectory(X: float, N: int) -> np.ndarray:
    """Uniform TWAP inventory schedule.

    x_k = X * (N - k) / N  for k = 0 … N

    Parameters
    ----------
    X : Total shares to liquidate.
    N : Number of trading periods.

    Returns
    -------
    np.ndarray of shape (N+1,).  First element = X, last = 0.
    """
    k = np.arange(N + 1, dtype=float)
    return X * (N - k) / N


def vwap_trajectory(X: float, volume_profile: np.ndarray) -> np.ndarray:
    """Volume-proportional VWAP inventory schedule.

    Each period's trade size is proportional to the historical average volume
    fraction for that time-of-day bucket.

    Parameters
    ----------
    X              : Total shares to liquidate.
    volume_profile : 1-D array of length N — average volumes per period
                     (does NOT need to be normalised).

    Returns
    -------
    np.ndarray of shape (N+1,).  First element = X, last = 0.
    """
    N = len(volume_profile)
    total_vol = volume_profile.sum()
    if total_vol <= 0:
        # Fall back to TWAP if volume profile is degenerate.
        return twap_trajectory(X, N)

    # Fraction of X to trade each period.
    fractions = volume_profile / total_vol
    trades = X * fractions   # shape (N,)

    # Inventory x_k = X - cumulative_trades_up_to_k
    cum_trades = np.concatenate([[0.0], np.cumsum(trades)])
    inventory = X - cum_trades
    inventory[-1] = 0.0   # enforce terminal condition exactly
    return inventory


def ac_trajectory(
    X: float,
    N: int,
    sigma: float,
    eta: float,
    lambda_: float,
    T: float = 1.0,
) -> np.ndarray:
    """Almgren-Chriss closed-form optimal inventory schedule.

    Equation (18) of Almgren & Chriss (2000):

        x_k = X * sinh(kappa * (T - t_k)) / sinh(kappa * T)
        kappa = sqrt(lambda * sigma^2 / eta)

    Falls back to TWAP (l'Hôpital limit) when kappa → 0.

    Parameters
    ----------
    X       : Total shares.
    N       : Number of trading periods.
    sigma   : Per-period price volatility.
    eta     : Temporary impact coefficient.
    lambda_ : Risk-aversion parameter.
    T       : Total horizon.

    Returns
    -------
    np.ndarray of shape (N+1,).
    """
    dt = T / N
    kappa = np.sqrt(lambda_ * sigma**2 / eta) if eta > 0 else 0.0

    x = np.empty(N + 1)
    x[0] = X
    x[N] = 0.0

    if kappa < 1e-6:
        # l'Hôpital limit → TWAP.
        k = np.arange(1, N, dtype=float)
        x[1:N] = X * (T - k * dt) / T
    else:
        sinh_kappaT = np.sinh(kappa * T)
        k = np.arange(1, N, dtype=float)
        t_k = k * dt
        x[1:N] = X * np.sinh(kappa * (T - t_k)) / sinh_kappaT

    return x


# ---------------------------------------------------------------------------
# Pure-Python Monte Carlo (used as fallback / quick comparisons)
# ---------------------------------------------------------------------------


def simulate_strategy_costs(
    trajectory: np.ndarray,
    params: dict,
    num_paths: int = 5000,
    rng_seed: int = 42,
) -> np.ndarray:
    """Simulate execution costs over multiple Monte Carlo paths (pure Python).

    This mirrors the C++ logic exactly but runs in NumPy for flexibility.
    For production-scale simulations prefer the C++ engine via
    simulator_interface.

    Parameters
    ----------
    trajectory : Inventory schedule, shape (N+1,).
    params     : Dict with keys S0, sigma, T, N, eta, gamma.
    num_paths  : Number of independent paths.
    rng_seed   : Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (num_paths,) — one cost per path.
    """
    S0    = params["S0"]
    sigma = params["sigma"]
    T     = params["T"]
    N     = params["N"]
    eta   = params["eta"]
    gamma = params["gamma"]

    dt   = T / N
    sqdt = np.sqrt(dt)

    rng = np.random.default_rng(rng_seed)

    # Trade sizes v_k = x_k - x_{k+1}, shape (N,).
    v = np.diff(-trajectory)   # positive for selling (x decreasing)

    # Shape (num_paths, N) — standard normal shocks.
    Z = rng.standard_normal((num_paths, N))

    costs = np.zeros(num_paths)
    S = np.full(num_paths, S0)

    for k in range(N):
        vk = v[k]
        P_exec = S - eta * vk                      # temporary impact
        costs += vk * (S0 - P_exec)               # shortfall contribution
        S = S - gamma * vk + sigma * sqdt * Z[:, k]  # price update

    return costs


# ---------------------------------------------------------------------------
# Strategy comparison table
# ---------------------------------------------------------------------------


def compare_strategies(
    params: dict,
    num_paths: int = 5000,
    volume_profile: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run all three strategies and return a side-by-side comparison DataFrame.

    Uses the pure-Python Monte Carlo for portability (no C++ binary required).
    For large num_paths use simulator_interface instead.

    Parameters
    ----------
    params         : Dict with keys S0, sigma, T, N, X, eta, gamma, lambda.
    num_paths      : Number of MC paths per strategy.
    volume_profile : Optional volume fractions for VWAP (length N).
                     Falls back to TWAP-equivalent VWAP if None.

    Returns
    -------
    pd.DataFrame with columns:
        strategy, mean_cost, std_cost, variance_cost,
        implementation_shortfall, cost_per_share
    """
    X = params["X"]
    N = params["N"]

    # Build trajectories.
    traj_twap = twap_trajectory(X, N)

    if volume_profile is not None and len(volume_profile) == N:
        traj_vwap = vwap_trajectory(X, volume_profile)
    else:
        # Uniform volume → VWAP degenerates to TWAP.
        traj_vwap = twap_trajectory(X, N)

    traj_ac = ac_trajectory(
        X, N,
        sigma=params["sigma"],
        eta=params["eta"],
        lambda_=params["lambda"],
        T=params["T"],
    )

    strategies = {
        "TWAP": traj_twap,
        "VWAP": traj_vwap,
        "Almgren-Chriss": traj_ac,
    }

    rows: list[dict] = []
    for name, traj in strategies.items():
        costs = simulate_strategy_costs(traj, params, num_paths=num_paths)
        mean_c  = float(costs.mean())
        std_c   = float(costs.std(ddof=1))
        var_c   = float(costs.var(ddof=1))
        impl_sf = mean_c / (params["S0"] * X) * 1e4   # basis points
        cps     = mean_c / X                           # $/share

        rows.append({
            "Strategy":                      name,
            "Mean Cost ($)":                 round(mean_c, 2),
            "Std Dev ($)":                   round(std_c, 2),
            "Variance ($²)":                 round(var_c, 2),
            "Implementation Shortfall (bps)": round(impl_sf, 4),
            "Cost per Share ($)":             round(cps, 6),
        })

    return pd.DataFrame(rows).set_index("Strategy")
