"""
simulate.py
===========
Python wrapper around the C++ Almgren-Chriss simulation binary.

All calls to the C++ engine go through the three public functions below.
The binary is expected at  ../build/almgren_chriss[.exe]  relative to this
file; a secondary lookup covers MSVC multi-config layout (build/Release/).

Public API
----------
run_strategy(params, strategy_name, num_paths) -> (float, float)
    Return (mean_cost, variance_cost) from num_paths Monte Carlo paths.

get_all_costs(params, strategy_name, num_paths) -> np.ndarray
    Return a 1-D array of individual path costs (length num_paths).

get_trajectory(params, strategy_name) -> np.ndarray
    Return the N+1 inventory schedule {x_0, x_1, …, x_N}.
"""

import subprocess
import sys
import os
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Binary location
# ---------------------------------------------------------------------------

def _find_binary() -> Path:
    """Locate the compiled C++ binary, searching common CMake output paths."""
    here = Path(__file__).resolve().parent          # python/
    root = here.parent                              # project root

    candidates = [
        root / "build" / "almgren_chriss",
        root / "build" / "almgren_chriss.exe",
        root / "build" / "Release" / "almgren_chriss.exe",  # MSVC multi-config
        root / "build" / "Debug"   / "almgren_chriss.exe",
        root / "build" / "Release" / "almgren_chriss",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "C++ binary 'almgren_chriss' not found.\n"
        "Build it first:\n"
        "  cmake -B build\n"
        "  cmake --build build --config Release\n"
        f"Searched:\n" + "\n".join(f"  {c}" for c in candidates)
    )


_BINARY: Path | None = None   # cached after first call


def _binary() -> Path:
    global _BINARY
    if _BINARY is None:
        _BINARY = _find_binary()
    return _BINARY


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict = {
    "S0"    : 100.0,
    "sigma" : 0.02,
    "T"     : 1.0,
    "N"     : 20,
    "X"     : 1_000_000.0,
    "eta"   : 0.1,
    "gamma" : 0.05,
    "lambda": 0.1,
}


# ---------------------------------------------------------------------------
# Internal: build the CLI argument list from a params dict
# ---------------------------------------------------------------------------

def _build_args(params: dict) -> list[str]:
    """Convert a params dict into the positional CLI arguments expected by main()."""
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


def _run_binary(command: str, strategy: str, params: dict,
                num_paths: int | None = None) -> str:
    """
    Invoke the C++ binary with the given command and parameters.
    Returns the raw stdout string.
    Raises RuntimeError if the process exits with a non-zero code.
    """
    cmd = [str(_binary()), command, strategy] + _build_args(params)
    if num_paths is not None:
        cmd.append(str(num_paths))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"C++ binary failed (exit {result.returncode}):\n{result.stderr}"
        )
    return result.stdout


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_strategy(params: dict, strategy_name: str, num_paths: int = 10_000
                 ) -> tuple[float, float]:
    """
    Run num_paths Monte Carlo simulations for the given strategy and return
    (mean_cost, variance_of_cost).

    Parameters
    ----------
    params        : dict with keys S0, sigma, T, N, X, eta, gamma, lambda
    strategy_name : "optimal" | "twap" | "immediate"
    num_paths     : number of independent price paths to simulate
    """
    raw = _run_binary("montecarlo", strategy_name, params, num_paths)
    mean_str, var_str = raw.split()
    return float(mean_str), float(var_str)


def get_all_costs(params: dict, strategy_name: str, num_paths: int = 10_000
                  ) -> np.ndarray:
    """
    Return all individual path costs as a numpy array of shape (num_paths,).

    Useful for plotting histograms or computing quantiles.

    Parameters
    ----------
    params        : dict with keys S0, sigma, T, N, X, eta, gamma, lambda
    strategy_name : "optimal" | "twap" | "immediate"
    num_paths     : number of paths
    """
    raw = _run_binary("paths", strategy_name, params, num_paths)
    return np.fromstring(raw, sep='\n')


def get_trajectory(params: dict, strategy_name: str) -> np.ndarray:
    """
    Return the inventory schedule {x_0, x_1, …, x_N} as a numpy array.

    This calls the C++ trajectory calculator (no randomness involved).

    Parameters
    ----------
    params        : dict with keys S0, sigma, T, N, X, eta, gamma, lambda
    strategy_name : "optimal" | "twap" | "immediate"
    """
    raw = _run_binary("trajectory", strategy_name, params)
    return np.fromstring(raw, sep='\n')


# ---------------------------------------------------------------------------
# Quick smoke test when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Smoke test — default parameters")
    p = DEFAULT_PARAMS.copy()

    traj = get_trajectory(p, "optimal")
    print(f"Trajectory length: {len(traj)}  x[0]={traj[0]:.0f}  x[-1]={traj[-1]:.6f}")

    mean, var = run_strategy(p, "optimal", num_paths=1000)
    print(f"Optimal  MC(1000): mean={mean:.2e}  var={var:.2e}")

    mean, var = run_strategy(p, "twap", num_paths=1000)
    print(f"TWAP     MC(1000): mean={mean:.2e}  var={var:.2e}")

    mean, var = run_strategy(p, "immediate", num_paths=1000)
    print(f"Immediate MC(1000): mean={mean:.2e}  var={var:.2e}")
