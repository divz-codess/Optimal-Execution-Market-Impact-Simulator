#pragma once

#include <vector>
#include <utility>
#include <random>

// ---------------------------------------------------------------------------
// ModelParams
// ---------------------------------------------------------------------------
// Holds all parameters that define one instance of the Almgren-Chriss model.
//
//   S0     : initial mid-price
//   sigma  : volatility of arithmetic Brownian motion (price std-dev per unit time)
//   T      : total execution horizon (years, or any consistent time unit)
//   N      : number of equal-length trading periods
//   X      : total shares to liquidate (x_0 = X, x_N = 0)
//   eta    : temporary impact coefficient  (cost per share^2 per period)
//   gamma  : permanent impact coefficient  (price shift per share traded)
//   lambda : risk-aversion parameter controlling the E[Cost]-Var[Cost] tradeoff
//            lambda = 0  → minimise expected cost only (slow execution)
//            lambda → ∞  → minimise variance only    (fast execution)
// ---------------------------------------------------------------------------
struct ModelParams
{
    double S0 = 100.0;
    double sigma = 0.02;
    double T = 1.0;
    int N = 20;
    double X = 1'000'000.0;
    double eta = 0.1;
    double gamma = 0.05;
    double lambda = 0.1;
};

// ---------------------------------------------------------------------------
// compute_optimal_trajectory
// ---------------------------------------------------------------------------
// Returns the closed-form Almgren-Chriss optimal inventory schedule.
//
// Minimises  E[Cost] + lambda * Var[Cost]  subject to x_0 = X, x_N = 0.
//
// Closed form (Almgren & Chriss 2000, eq. 18):
//
//   x_k = X * sinh(kappa * (T - t_k)) / sinh(kappa * T)
//
// where  kappa = sqrt(lambda * sigma^2 / eta)
//        t_k   = k * dt,  dt = T / N
//
// When kappa → 0  (lambda → 0), l'Hôpital gives x_k → X*(T-t_k)/T  (TWAP).
//
// Returns: vector of size N+1.  result[0] = X, result[N] = 0.
// ---------------------------------------------------------------------------
std::vector<double> compute_optimal_trajectory(const ModelParams &params);

// ---------------------------------------------------------------------------
// compute_twap_trajectory
// ---------------------------------------------------------------------------
// Returns the TWAP (Time-Weighted Average Price) inventory schedule.
// Sells a uniform X/N shares each period.
//
//   x_k = X * (N - k) / N
//
// Returns: vector of size N+1.  result[0] = X, result[N] = 0.
// ---------------------------------------------------------------------------
std::vector<double> compute_twap_trajectory(const ModelParams &params);

// ---------------------------------------------------------------------------
// compute_immediate_trajectory
// ---------------------------------------------------------------------------
// Returns the immediate liquidation schedule:
// sell everything in the first period, hold 0 thereafter.
//
//   x_0 = X,  x_k = 0  for k >= 1
//
// Returns: vector of size N+1.
// ---------------------------------------------------------------------------
std::vector<double> compute_immediate_trajectory(const ModelParams &params);

// ---------------------------------------------------------------------------
// simulate_execution_cost  (single Monte Carlo path)
// ---------------------------------------------------------------------------
// Simulates one realisation of the liquidation process given a fixed
// inventory trajectory and returns the total execution cost.
//
// Price dynamics (arithmetic Brownian motion with permanent impact):
//   S_{k+1} = S_k - gamma * v_k + sigma * sqrt(dt) * Z_k
//   Z_k ~ N(0,1)
//
// Execution price for trade v_k (temporary impact reduces received price):
//   P_exec_k = S_k - eta * v_k
//
// Cost contribution (measured against initial price S0):
//   cost_k = v_k * (S0 - P_exec_k)
//          = v_k * (S0 - S_k + eta * v_k)
//
// Total cost = sum of cost_k over k = 0 … N-1.
//
// Parameters:
//   trajectory : inventory schedule {x_0, x_1, …, x_N}
//   params     : model parameters
//   rng        : caller-supplied Mersenne Twister (allows reproducible seeding)
// ---------------------------------------------------------------------------
double simulate_execution_cost(
    const std::vector<double> &trajectory,
    const ModelParams &params,
    std::mt19937 &rng);

// ---------------------------------------------------------------------------
// run_monte_carlo
// ---------------------------------------------------------------------------
// Runs num_paths independent simulations using simulate_execution_cost and
// returns the sample mean and sample variance of total execution cost.
//
// Returns: {mean_cost, variance_of_cost}
// ---------------------------------------------------------------------------
std::pair<double, double> run_monte_carlo(
    const std::vector<double> &trajectory,
    const ModelParams &params,
    int num_paths);
