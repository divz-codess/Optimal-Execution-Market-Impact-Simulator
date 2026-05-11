// =============================================================================
// almgren_chriss.cpp
// =============================================================================
// Core simulation engine implementing the Almgren-Chriss (2000) optimal
// execution model.
//
// Compile:  g++ -O2 -std=c++17 src/almgren_chriss.cpp -o almgren_chriss
//           (or build via CMake — see CMakeLists.txt)
//
// CLI usage:
//   almgren_chriss <command> <strategy> <S0> <sigma> <T> <N> <X> <eta> <gamma> <lambda> [num_paths]
//
//   command  : trajectory | montecarlo | paths
//   strategy : optimal | twap | immediate
//
//   trajectory   → prints N+1 inventory values (one per line)
//   montecarlo   → prints "<mean_cost> <variance_cost>" on one line
//   paths        → prints num_paths individual cost realisations (one per line)
// =============================================================================

#include "almgren_chriss.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// compute_optimal_trajectory
// ─────────────────────────────────────────────────────────────────────────────
// Implements equation (18) of Almgren & Chriss (2000):
//
//   x_k = X * sinh(kappa * (T - t_k)) / sinh(kappa * T)
//
// kappa = sqrt(lambda * sigma^2 / eta)  is the "urgency" parameter.
//   High kappa → concave trajectory → front-loaded execution.
//   Low  kappa → nearly linear trajectory → approaches TWAP.
//
// Edge case: when kappa < 1e-6, sinh(kappa*u)/sinh(kappa*T) → u/T
// (l'Hôpital's rule), so we fall back to the exact TWAP formula to
// avoid numerical underflow / cancellation.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> compute_optimal_trajectory(const ModelParams &p)
{
    const int N = p.N;
    const double dt = p.T / static_cast<double>(N);

    // Urgency parameter kappa = sqrt(lambda * sigma^2 / eta)
    const double kappa = std::sqrt(p.lambda * p.sigma * p.sigma / p.eta);

    std::vector<double> x(N + 1);
    x[0] = p.X;
    x[N] = 0.0; // Enforce terminal condition exactly.

    if (kappa < 1e-6)
    {
        // l'Hôpital limit: reduces to TWAP when kappa → 0.
        for (int k = 1; k < N; ++k)
        {
            x[k] = p.X * (p.T - k * dt) / p.T;
        }
    }
    else
    {
        const double sinh_kappaT = std::sinh(kappa * p.T);
        for (int k = 1; k < N; ++k)
        {
            const double t_k = k * dt;
            x[k] = p.X * std::sinh(kappa * (p.T - t_k)) / sinh_kappaT;
        }
    }

    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_twap_trajectory
// ─────────────────────────────────────────────────────────────────────────────
// TWAP (Time-Weighted Average Price): sell exactly X/N shares every period.
//
//   x_k = X * (N - k) / N
//
// This is the risk-neutral benchmark — it minimises expected temporary
// impact cost in isolation but ignores timing risk.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> compute_twap_trajectory(const ModelParams &p)
{
    std::vector<double> x(p.N + 1);
    for (int k = 0; k <= p.N; ++k)
    {
        x[k] = p.X * static_cast<double>(p.N - k) / static_cast<double>(p.N);
    }
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_immediate_trajectory
// ─────────────────────────────────────────────────────────────────────────────
// Immediate liquidation: sell all X shares in the very first period.
//
//   x_0 = X,  x_k = 0 for k >= 1
//
// Minimises timing risk (zero exposure after t=0) but maximises temporary
// market impact cost in that single period.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> compute_immediate_trajectory(const ModelParams &p)
{
    std::vector<double> x(p.N + 1, 0.0);
    x[0] = p.X;
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// simulate_execution_cost  (single Monte Carlo path)
// ─────────────────────────────────────────────────────────────────────────────
// Simulates one price path and computes the total execution cost.
//
// Price dynamics — arithmetic Brownian motion with permanent impact:
//   S_{k+1} = S_k  -  gamma * v_k  +  sigma * sqrt(dt) * Z_k
//   Z_k ~ N(0,1)
//
//   The -gamma*v_k term: selling (v_k > 0) permanently depresses future
//   prices, reflecting the footprint of a large sell order in the market.
//   Temporary impact does NOT shift S_k; it only degrades the execution
//   price for the current trade.
//
// Execution price for trade v_k (temporary impact applies only to the
// price received for this period's trade):
//   P_exec_k = S_k - eta * v_k
//
// Cost contribution, measured as shortfall against the initial price S0:
//   cost_k = v_k * (S0 - P_exec_k)
//          = v_k * (S0 - S_k + eta * v_k)
//
//   This captures both price drift loss (S0 - S_k) and temporary impact (eta*v_k^2).
// ─────────────────────────────────────────────────────────────────────────────
double simulate_execution_cost(
    const std::vector<double> &trajectory,
    const ModelParams &p,
    std::mt19937 &rng)
{
    const int N = p.N;
    const double dt = p.T / static_cast<double>(N);
    const double sqdt = std::sqrt(dt);

    std::normal_distribution<double> normal(0.0, 1.0);

    double S = p.S0; // current mid-price
    double cost = 0.0;

    for (int k = 0; k < N; ++k)
    {
        // Shares traded this period: v_k = x_k - x_{k+1}  (positive for selling)
        const double v_k = trajectory[k] - trajectory[k + 1];

        // Execution price: temporary impact reduces the price we receive.
        const double P_exec = S - p.eta * v_k;

        // Implementation shortfall contribution for this period.
        cost += v_k * (p.S0 - P_exec);

        // Price update: permanent impact shifts the mid-price downward;
        // the stochastic term adds arithmetic Brownian noise.
        const double Z = normal(rng);
        S = S - p.gamma * v_k + p.sigma * sqdt * Z;
    }

    return cost;
}

// ─────────────────────────────────────────────────────────────────────────────
// run_monte_carlo
// ─────────────────────────────────────────────────────────────────────────────
// Runs num_paths independent simulations and returns:
//   { sample_mean_cost, sample_variance_of_cost }
//
// Uses Welford's online algorithm for numerically stable single-pass
// computation of mean and variance.
// ─────────────────────────────────────────────────────────────────────────────
std::pair<double, double> run_monte_carlo(
    const std::vector<double> &trajectory,
    const ModelParams &params,
    int num_paths)
{
    // Fixed seed for reproducibility across all strategies/runs.
    std::mt19937 rng(42);

    // Welford's online mean-variance algorithm (numerically stable).
    double mean = 0.0;
    double M2 = 0.0;

    for (int n = 1; n <= num_paths; ++n)
    {
        const double cost = simulate_execution_cost(trajectory, params, rng);
        const double delta = cost - mean;
        mean += delta / static_cast<double>(n);
        M2 += delta * (cost - mean); // uses updated mean
    }

    const double variance = (num_paths > 1)
                                ? M2 / static_cast<double>(num_paths - 1)
                                : 0.0;

    return {mean, variance};
}

// =============================================================================
// main — Command-line dispatcher
// =============================================================================
// Parses arguments, builds the requested trajectory, and runs the requested
// command, writing results to stdout for the Python layer to consume.
// =============================================================================
int main(int argc, char *argv[])
{
    if (argc < 11)
    {
        std::cerr << "Usage: almgren_chriss <command> <strategy>"
                     " <S0> <sigma> <T> <N> <X> <eta> <gamma> <lambda>"
                     " [num_paths]\n"
                     "  command  : trajectory | montecarlo | paths\n"
                     "  strategy : optimal | twap | immediate\n";
        return 1;
    }

    const std::string command = argv[1];
    const std::string strategy = argv[2];

    ModelParams p;
    p.S0 = std::stod(argv[3]);
    p.sigma = std::stod(argv[4]);
    p.T = std::stod(argv[5]);
    p.N = std::stoi(argv[6]);
    p.X = std::stod(argv[7]);
    p.eta = std::stod(argv[8]);
    p.gamma = std::stod(argv[9]);
    p.lambda = std::stod(argv[10]);

    int num_paths = 10000;
    if (argc >= 12)
        num_paths = std::stoi(argv[11]);

    // ── Build trajectory ──────────────────────────────────────────────────────
    std::vector<double> traj;
    if (strategy == "optimal")
    {
        traj = compute_optimal_trajectory(p);
    }
    else if (strategy == "twap")
    {
        traj = compute_twap_trajectory(p);
    }
    else if (strategy == "immediate")
    {
        traj = compute_immediate_trajectory(p);
    }
    else
    {
        std::cerr << "Unknown strategy: " << strategy
                  << "  (use: optimal | twap | immediate)\n";
        return 1;
    }

    // ── Execute command ───────────────────────────────────────────────────────
    if (command == "trajectory")
    {
        // Print N+1 inventory levels, one per line.
        std::cout.precision(6);
        std::cout << std::fixed;
        for (const double xi : traj)
        {
            std::cout << xi << '\n';
        }
    }
    else if (command == "montecarlo")
    {
        // Print mean and variance of total execution cost on one line.
        auto [mean, var] = run_monte_carlo(traj, p, num_paths);
        std::cout.precision(10);
        std::cout << std::scientific << mean << ' ' << var << '\n';
    }
    else if (command == "paths")
    {
        // Print one cost realisation per line (num_paths lines).
        // Each call uses a fresh mt19937 seeded with 42 so results are
        // reproducible, then advances the stream by sampling num_paths paths.
        std::mt19937 rng(42);
        std::cout.precision(10);
        std::cout << std::scientific;
        for (int i = 0; i < num_paths; ++i)
        {
            std::cout << simulate_execution_cost(traj, p, rng) << '\n';
        }
    }
    else
    {
        std::cerr << "Unknown command: " << command
                  << "  (use: trajectory | montecarlo | paths)\n";
        return 1;
    }

    return 0;
}
