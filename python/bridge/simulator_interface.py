"""
simulator_interface.py
======================
Python bridge to the compiled C++ Almgren-Chriss simulation binary.

All simulator invocations go through this module.  The C++ process is
launched via subprocess; stdout is captured and parsed into NumPy arrays /
pandas DataFrames.  No simulator logic is re-implemented here.

C++ CLI contract (see src/almgren_chriss.cpp main()):

    almgren_chriss <command> <strategy> <S0> <sigma> <T> <N> <X> <eta> <gamma> <lambda> [num_paths]

    command  : trajectory | montecarlo | paths
    strategy : optimal | twap | immediate

    trajectory  → N+1 inventory values, one float per line
    montecarlo  → "<mean_cost> <variance_cost>" on one line
    paths       → num_paths cost realisations, one float per line

Public API
----------
get_trajectory(params, strategy)              -> pd.DataFrame
run_montecarlo(params, strategy, num_paths)   -> dict[str, float]
get_cost_paths(params, strategy, num_paths)   -> np.ndarray
run_all_strategies(params, num_paths)         -> pd.DataFrame
efficient_frontier(params, lambdas, num_paths)-> pd.DataFrame
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

def _find_binary() -> Path:
    """Search common CMake output locations for the compiled binary."""
    here = Path(__file__).resolve().parent          # python/bridge/
    root = here.parent.parent                       # project root

    candidates = [
        root / "build" / "almgren_chriss",
        root / "build" / "almgren_chriss.exe",
        root / "build" / "Release" / "almgren_chriss.exe",
        root / "build" / "Debug"   / "almgren_chriss.exe",
        root / "build" / "Release" / "almgren_chriss",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "C++ binary 'almgren_chriss' not found.  Build it first:\n"
        "  cmake -B build\n"
        "  cmake --build build --config Release\n"
        "Searched:\n" + "\n".join(f"  {c}" for c in candidates)
    )


_BINARY: Path | None = None


def _binary() -> Path:
    global _BINARY
    if _BINARY is None:
        _BINARY = _find_binary()
    return _BINARY


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_args(params: dict) -> list[str]:
    """Serialise a params dict to the positional CLI arguments."""
    return [
        str(params["S0"]),
        str(params["sigma"]),
        str(params["T"]),
        str(int(params["N"])),
        str(params["X"]),
        str(params["eta"]),
        str(params["gamma"]),
        str(params["lambda"]),
    ]


def _run(command: str, strategy: str, params: dict, extra: list[str] | None = None) -> str:
    """Launch the C++ binary and return stdout as a string.

    Parameters
    ----------
    command  : "trajectory" | "montecarlo" | "paths"
    strategy : "optimal" | "twap" | "immediate"
    params   : Model parameter dict.
    extra    : Additional positional args (e.g. [str(num_paths)]).

    Returns
    -------
    stdout text from the subprocess.

    Raises
    ------
    RuntimeError if the process exits with a non-zero code.
    """
    args = [str(_binary()), command, strategy] + _build_args(params)
    if extra:
        args += extra

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"C++ simulator failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )

    return result.stdout


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_trajectory(params: dict, strategy: str = "optimal") -> pd.DataFrame:
    """Fetch the N+1 inventory schedule from the C++ binary.

    Parameters
    ----------
    params   : Model parameter dict (S0, sigma, T, N, X, eta, gamma, lambda).
    strategy : "optimal" | "twap" | "immediate"

    Returns
    -------
    pd.DataFrame with columns: period, time, inventory, trade_size
        period    : 0 … N
        time      : t_k = k * dt
        inventory : x_k (shares remaining)
        trade_size: v_k = x_k - x_{k+1} (shares traded this period)
    """
    stdout = _run("trajectory", strategy, params)
    inventory = np.array([float(line) for line in stdout.strip().split("\n")])

    N  = int(params["N"])
    T  = float(params["T"])
    dt = T / N

    periods = np.arange(N + 1)
    times   = periods * dt

    trades = np.diff(-inventory, prepend=0.0)
    trades[0] = 0.0   # no trade at t=0, first trade is x_0 - x_1

    # Recalculate properly: v_k = x_k - x_{k+1} for k = 0..N-1
    trade_sizes = np.append(np.diff(-inventory), 0.0)

    return pd.DataFrame({
        "period":     periods,
        "time":       times,
        "inventory":  inventory,
        "trade_size": trade_sizes,
    })


def run_montecarlo(
    params: dict,
    strategy: str = "optimal",
    num_paths: int = 10_000,
) -> dict[str, float]:
    """Run Monte Carlo and return mean/variance of execution cost.

    Parameters
    ----------
    params    : Model parameter dict.
    strategy  : "optimal" | "twap" | "immediate"
    num_paths : Number of MC paths.

    Returns
    -------
    dict with keys: mean_cost, variance_cost, std_cost, sharpe_cost
    """
    stdout = _run("montecarlo", strategy, params, extra=[str(num_paths)])
    parts = stdout.strip().split()
    mean_cost = float(parts[0])
    var_cost  = float(parts[1])
    std_cost  = float(np.sqrt(var_cost))

    return {
        "mean_cost":     mean_cost,
        "variance_cost": var_cost,
        "std_cost":      std_cost,
    }


def get_cost_paths(
    params: dict,
    strategy: str = "optimal",
    num_paths: int = 10_000,
) -> np.ndarray:
    """Return individual Monte Carlo cost realisations as a NumPy array.

    Parameters
    ----------
    params    : Model parameter dict.
    strategy  : "optimal" | "twap" | "immediate"
    num_paths : Number of paths.

    Returns
    -------
    np.ndarray of shape (num_paths,).
    """
    stdout = _run("paths", strategy, params, extra=[str(num_paths)])
    return np.array([float(line) for line in stdout.strip().split("\n")])


def run_all_strategies(
    params: dict,
    num_paths: int = 10_000,
) -> pd.DataFrame:
    """Run Monte Carlo for all three strategies and return a comparison table.

    Parameters
    ----------
    params    : Model parameter dict.
    num_paths : Number of MC paths per strategy.

    Returns
    -------
    pd.DataFrame indexed by strategy name with columns:
        mean_cost, std_cost, variance_cost,
        implementation_shortfall_bps, cost_per_share
    """
    strategies = ["optimal", "twap", "immediate"]
    rows: list[dict] = []

    for strat in strategies:
        mc = run_montecarlo(params, strat, num_paths)
        impl_sf = mc["mean_cost"] / (params["S0"] * params["X"]) * 1e4
        cps     = mc["mean_cost"] / params["X"]

        rows.append({
            "Strategy":                       strat.capitalize(),
            "Mean Cost ($)":                  round(mc["mean_cost"], 2),
            "Std Dev ($)":                    round(mc["std_cost"], 2),
            "Variance ($²)":                  round(mc["variance_cost"], 2),
            "Impl. Shortfall (bps)":          round(impl_sf, 4),
            "Cost per Share ($)":             round(cps, 6),
        })

    return pd.DataFrame(rows).set_index("Strategy")


def efficient_frontier(
    params: dict,
    lambdas: np.ndarray | None = None,
    num_paths: int = 5_000,
) -> pd.DataFrame:
    """Compute the efficient frontier by sweeping the risk-aversion parameter.

    For each lambda value, runs Monte Carlo for the optimal strategy and
    records (E[Cost], Var[Cost]).  The resulting curve traces the Pareto
    frontier between expected cost and cost variance.

    Parameters
    ----------
    params    : Base model parameter dict.  The 'lambda' key is overridden.
    lambdas   : 1-D array of lambda values to sweep.
                Defaults to 30 log-spaced values from 1e-4 to 10.
    num_paths : MC paths per lambda point.

    Returns
    -------
    pd.DataFrame with columns: lambda, mean_cost, variance_cost, std_cost
    """
    if lambdas is None:
        lambdas = np.logspace(-4, 1, 30)

    rows: list[dict] = []
    base = params.copy()

    for lam in lambdas:
        base["lambda"] = float(lam)
        try:
            mc = run_montecarlo(base, strategy="optimal", num_paths=num_paths)
            rows.append({
                "lambda":        lam,
                "mean_cost":     mc["mean_cost"],
                "variance_cost": mc["variance_cost"],
                "std_cost":      mc["std_cost"],
            })
        except RuntimeError:
            continue   # skip degenerate parameter combinations

    return pd.DataFrame(rows)
