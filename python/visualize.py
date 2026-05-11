"""
visualize.py
============
Generates all five diagnostic plots for the Almgren-Chriss optimal execution
simulator and saves them to ../plots/*.png.

Plots
-----
1. optimal_trajectory.png   – inventory schedules for Optimal (3 λ), TWAP, Immediate
2. efficient_frontier.png   – E[Cost] vs Var[Cost] as λ sweeps log-space
3. cost_distribution.png    – histogram of 10,000 MC paths per strategy
4. impact_decomposition.png – stacked bar: temporary vs permanent cost per period
5. kappa_sensitivity.png    – % of shares sold in first 25 % of horizon vs kappa

Run from the project root:
    python python/visualize.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Add the python/ directory to sys.path so we can import simulate.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import DEFAULT_PARAMS, run_strategy, get_all_costs, get_trajectory

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _save(name: str) -> None:
    """Save current figure to PLOTS_DIR and close it."""
    path = PLOTS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

STRATEGY_COLORS = {
    "optimal_low" : "#1f77b4",
    "optimal_mid" : "#ff7f0e",
    "optimal_high": "#2ca02c",
    "twap"        : "#9467bd",
    "immediate"   : "#d62728",
}


def _base_params(lambda_val: float | None = None) -> dict:
    p = DEFAULT_PARAMS.copy()
    if lambda_val is not None:
        p["lambda"] = lambda_val
    return p


# =============================================================================
# Graph 1 — Optimal Liquidation Trajectory vs Benchmarks
# =============================================================================

def plot_optimal_trajectory() -> None:
    """
    Shows how the inventory schedule x_k varies with the risk-aversion
    parameter lambda.

    - Low  lambda → slow, TWAP-like execution (minimises expected cost)
    - High lambda → fast, front-loaded execution (minimises timing risk)

    The TWAP and Immediate strategies serve as outer bounds.

    Lambda values are chosen to produce clearly distinct kappa values:
      kappa = sqrt(lambda * sigma^2 / eta) = sqrt(lambda * 0.0004 / 0.1) = sqrt(lambda * 0.004)

      lambda=2.5    → kappa ≈ 0.1  (near-TWAP)
      lambda=250    → kappa ≈ 1.0  (moderate front-loading)
      lambda=25000  → kappa ≈ 10.0 (heavy front-loading)

    With the default parameters (sigma=0.02, eta=0.1), visible front-loading
    only appears for lambda >> 1.  Values like lambda=0.001..5 all give kappa < 0.15
    and are visually indistinguishable from TWAP.
    """
    print("Graph 1: optimal_trajectory.png")

    lambdas = {"low": 2.5, "mid": 250.0, "high": 25000.0}
    N = int(DEFAULT_PARAMS["N"])
    periods = np.arange(N + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Compute kappa for each lambda (for informative legend labels)
    sigma = DEFAULT_PARAMS["sigma"]
    eta   = DEFAULT_PARAMS["eta"]

    # Optimal trajectories for three lambda values
    for key, lam in lambdas.items():
        p = _base_params(lam)
        kappa = (lam * sigma**2 / eta) ** 0.5
        traj = get_trajectory(p, "optimal")
        ax.plot(periods, traj / 1e6,
                color=STRATEGY_COLORS[f"optimal_{key}"],
                linewidth=2,
                label=f"Optimal  λ={lam:g}  (κ={kappa:.1f})")

    # TWAP
    traj_twap = get_trajectory(_base_params(), "twap")
    ax.plot(periods, traj_twap / 1e6,
            color=STRATEGY_COLORS["twap"],
            linewidth=2, linestyle="--",
            label="TWAP")

    # Immediate
    traj_imm = get_trajectory(_base_params(), "immediate")
    ax.plot(periods, traj_imm / 1e6,
            color=STRATEGY_COLORS["immediate"],
            linewidth=2, linestyle=":",
            label="Immediate Liquidation")

    ax.set_xlabel("Time period  k", fontsize=12)
    ax.set_ylabel("Remaining inventory  x_k  (millions of shares)", fontsize=12)
    ax.set_title("Optimal Liquidation Trajectory vs Benchmarks", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, N)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f M"))

    _save("optimal_trajectory.png")


# =============================================================================
# Graph 2 — Almgren-Chriss Efficient Frontier
# =============================================================================

def plot_efficient_frontier() -> None:
    """
    The efficient frontier traces the optimal (E[Cost], Var[Cost]) pairs as
    lambda sweeps from near-zero (risk-neutral, slow execution) to very large
    (highly risk-averse, near-immediate execution).

    Lambda range: 1e-4 to 1e5  (log-scale, 100 points)
    This gives kappa = sqrt(lambda * sigma^2/eta) in [0.0006, 20], covering
    the full spectrum from near-TWAP to near-Immediate.

    With the default parameters (sigma=0.02, eta=0.1), the user's original
    range of lambda ∈ [0.0001, 10] gives kappa ∈ [0.0006, 0.2] which is all
    near-TWAP and shows no visible frontier.  The range is extended here.

    TWAP and Immediate Liquidation are plotted as reference markers.
    """
    print("Graph 2: efficient_frontier.png  (100 lambda values × 1000 paths each — may take ~1–2 min)")

    # Extended range: kappa sweeps from 0.002 to 20 across these lambdas.
    lambdas = np.logspace(-4, 5, 100)
    means   = np.empty(len(lambdas))
    vars_   = np.empty(len(lambdas))

    for i, lam in enumerate(lambdas):
        p = _base_params(lam)
        m, v = run_strategy(p, "optimal", num_paths=1000)
        means[i] = m
        vars_[i] = v

    # Reference strategies (lambda does not affect TWAP/Immediate)
    p_ref = _base_params()
    twap_m, twap_v = run_strategy(p_ref, "twap",      num_paths=5000)
    imm_m,  imm_v  = run_strategy(p_ref, "immediate", num_paths=5000)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Express costs in billions of dollars, variance in millions of dollars²
    # so axes are in human-readable units.
    scale_y = 1e9    # → billions
    scale_x = 1e6   # → millions of (dollars²)

    # Frontier curve
    ax.plot(vars_ / scale_x, means / scale_y,
            color="#1f77b4", linewidth=2.5, label="Almgren-Chriss Frontier")

    # Annotate low/high lambda end-points
    ax.scatter([vars_[0]  / scale_x], [means[0]  / scale_y],
               color="#2ca02c", zorder=5, s=80,
               label=f"λ = {lambdas[0]:.4f}  (risk-neutral, slow)")
    ax.scatter([vars_[-1] / scale_x], [means[-1] / scale_y],
               color="#d62728", zorder=5, s=80,
               label=f"λ = {lambdas[-1]:.0f}  (highly risk-averse, fast)")

    # Reference markers
    ax.scatter([twap_v / scale_x], [twap_m / scale_y],
               marker="s", color=STRATEGY_COLORS["twap"],
               s=120, zorder=5, label="TWAP")
    ax.scatter([imm_v  / scale_x], [imm_m  / scale_y],
               marker="^", color=STRATEGY_COLORS["immediate"],
               s=120, zorder=5, label="Immediate Liquidation")

    ax.set_xlabel("Variance of execution cost  (millions of $²)", fontsize=12)
    ax.set_ylabel("Expected execution cost  (billions $)", fontsize=12)
    ax.set_title("Almgren-Chriss Efficient Frontier", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    _save("efficient_frontier.png")


# =============================================================================
# Graph 3 — Distribution of Execution Costs
# =============================================================================

def plot_cost_distribution() -> None:
    """
    The histogram of 10,000 execution-cost realisations reveals the
    risk profile of each strategy.

    Three separate subplots are used because the strategies have very
    different cost scales:
      - Optimal (λ=250, κ≈1.0): front-loaded, higher expected cost, lower variance
      - TWAP: uniform execution, moderate expected cost, higher variance
      - Immediate: everything sold at once, maximum temp impact, zero variance
    A single shared x-axis would compress the Optimal/TWAP distributions into
    invisible spikes next to the $100B immediate cost.

    Note: the Almgren-Chriss variance formula  Var = sigma² * Σ x_k² * dt
    gives Var[TWAP] ≈ 1.24e8 (std ≈ $11k on a $28.75B mean).  The distributions
    are plotted with individual x-axis ranges (±5σ per strategy) to make the
    stochastic spread visible.
    """
    print("Graph 3: cost_distribution.png  (3 × 10,000 paths)")

    # lambda=250 → kappa≈1.0: meaningful front-loading, clearly different from TWAP
    p_opt = _base_params(lambda_val=250.0)
    p_ref = _base_params()

    # Raw costs in dollars; convert to billions for readability
    costs_opt  = get_all_costs(p_opt, "optimal",    num_paths=10_000) / 1e9
    costs_twap = get_all_costs(p_ref, "twap",        num_paths=10_000) / 1e9
    costs_imm  = get_all_costs(p_ref, "immediate",   num_paths=10_000) / 1e9

    strategies = [
        (costs_opt,  STRATEGY_COLORS["optimal_mid"], "Optimal  (λ=250, κ≈1.0)"),
        (costs_twap, STRATEGY_COLORS["twap"],         "TWAP"),
        (costs_imm,  STRATEGY_COLORS["immediate"],    "Immediate Liquidation"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (costs, color, label) in zip(axes, strategies):
        mu_b  = costs.mean()         # mean in billions
        std_b = costs.std()          # std  in billions

        # Express deviation from mean in THOUSANDS of dollars so the axis tick
        # labels are human-readable without matplotlib's huge offset notation.
        # 1 billion * 1e6 = 1 million-thousands = 1e6 thousands
        # Equivalently: (costs - mean) in billions  * 1e9 / 1e3 = * 1e6  → k$
        dev_k = (costs - mu_b) * 1e6   # deviation in thousands of dollars
        std_k = std_b * 1e6            # std in thousands of dollars

        span_k = max(5.0 * std_k, 0.01)   # ±5σ window; at least 0.01 k$ wide
        bins   = np.linspace(-span_k, span_k, 80)

        ax.hist(dev_k, bins=bins, density=True, color=color, alpha=0.75)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Deviation from mean  (thousands $)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

        # Annotation shows the absolute mean and spread
        ax.text(0.97, 0.95,
                f"μ = ${mu_b:.4g}B\nσ = ${std_k:.3g}k",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("Distribution of Execution Costs: Optimal vs Benchmarks",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("cost_distribution.png")


# =============================================================================
# Graph 4 — Temporary vs Permanent Impact Decomposition
# =============================================================================

def plot_impact_decomposition() -> None:
    """
    Decomposes the per-period execution cost into two additive components:

    Temporary impact cost (k)  = eta * v_k²
        Reflects the adverse price concession made to absorb v_k shares
        in a single period; it does not persist into future prices.

    Permanent impact cost (k)  = v_k * gamma * (X - x_k)
        The cost attributable to the cumulative price depression that
        has already occurred by period k due to all prior trades.
        With S_k^drift = S0 - gamma*(X - x_k), perm_cost_k = v_k*(S0 - S_k^drift).

    Both components are always non-negative under the standard A-C sign
    convention (selling depresses prices: S_{k+1} = S_k - gamma*v_k).
    """
    print("Graph 4: impact_decomposition.png")

    p = _base_params(lambda_val=0.1)
    X      = p["X"]
    eta    = p["eta"]
    gamma  = p["gamma"]
    S0     = p["S0"]
    N      = int(p["N"])

    traj = get_trajectory(p, "optimal")
    periods = np.arange(N)

    temp_costs = np.empty(N)
    perm_costs = np.empty(N)

    # S_k^drift tracks where the mid-price drifts due to permanent impact alone
    # (i.e., ignoring the Brownian noise which averages to zero).
    S_drift = S0

    for k in range(N):
        v_k = traj[k] - traj[k + 1]

        # Temporary impact cost: quadratic in trade size.
        temp_costs[k] = eta * v_k ** 2

        # Permanent impact cost: the fraction of the cumulative price
        # depression (S0 - S_drift) that falls on period k's trade.
        # With S_drift = S0 - gamma*(X - x_k):
        #   S0 - S_drift = gamma*(X - x_k) = gamma * shares_already_sold
        perm_costs[k] = v_k * (S0 - S_drift)

        # Advance drift (permanent impact of this period's trade)
        S_drift -= gamma * v_k

    # Rescale to millions for readability
    temp_costs /= 1e6
    perm_costs /= 1e6

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(periods, temp_costs,
           color="#1f77b4", alpha=0.85, label="Temporary impact cost")
    ax.bar(periods, perm_costs, bottom=temp_costs,
           color="#ff7f0e", alpha=0.85, label="Permanent impact cost")

    ax.set_xlabel("Time period  k", fontsize=12)
    ax.set_ylabel("Cost contribution  (millions)", fontsize=12)
    ax.set_title("Temporary vs Permanent Impact Decomposition Over Execution Horizon",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(-0.5, N - 0.5)

    _save("impact_decomposition.png")


# =============================================================================
# Graph 5 — Execution Urgency vs Front-Loading: Effect of Kappa
# =============================================================================

def plot_kappa_sensitivity() -> None:
    """
    The urgency parameter kappa = sqrt(lambda * sigma^2 / eta) controls
    how front-loaded execution is:

    - kappa ≈ 0  → linear (TWAP) schedule
    - kappa large → concave schedule: most shares sold in the first few periods

    We measure front-loading as the percentage of X sold in the first 25 %
    of the time horizon, computed analytically from the closed-form trajectory:

        x(t = 0.25*T) = X * sinh(kappa * 0.75*T) / sinh(kappa * T)

        % front-loaded = (x_0 - x(0.25T)) / X * 100
                       = [1 - sinh(kappa*0.75T)/sinh(kappa*T)] * 100

    This is a pure mathematical curve (no Monte Carlo needed).
    """
    print("Graph 5: kappa_sensitivity.png")

    T = DEFAULT_PARAMS["T"]
    kappas = np.linspace(0.01, 5.0, 500)

    # Analytical fraction of inventory remaining at t = 0.25*T
    with np.errstate(over="ignore"):
        # For large kappa, sinh ratios may overflow; clip gracefully.
        ratio = np.where(
            kappas < 1e-6,
            0.75,   # l'Hôpital limit: (T - 0.25T)/T = 0.75
            np.sinh(kappas * 0.75 * T) / np.sinh(kappas * T)
        )

    pct_front_loaded = (1.0 - ratio) * 100.0

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(kappas, pct_front_loaded, color="#2ca02c", linewidth=2.5)
    ax.axhline(25, color="gray", linestyle="--", alpha=0.6, label="25 %  (TWAP baseline)")
    ax.axhline(100, color="gray", linestyle=":",  alpha=0.4)

    ax.set_xlabel("Urgency parameter  κ", fontsize=12)
    ax.set_ylabel("Shares sold in first 25 % of horizon  (%)", fontsize=12)
    ax.set_title("Execution Urgency vs Front-Loading: Effect of Kappa",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(kappas[0], kappas[-1])
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _save("kappa_sensitivity.png")


# =============================================================================
# main
# =============================================================================

def main() -> None:
    print(f"Output directory: {PLOTS_DIR}\n")

    plot_optimal_trajectory()
    plot_efficient_frontier()
    plot_cost_distribution()
    plot_impact_decomposition()
    plot_kappa_sensitivity()

    print("\nAll 5 plots generated successfully.")


if __name__ == "__main__":
    main()
