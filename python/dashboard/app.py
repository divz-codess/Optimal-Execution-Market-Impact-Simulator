"""
app.py
======
Professional Almgren-Chriss Execution Analytics Dashboard — Streamlit.

Run from the project root:
    python -m streamlit run python/dashboard/app.py

The dashboard provides:
  - Live market data via yfinance
  - C++ Monte Carlo simulation engine (with pure-Python fallback)
  - Optimal liquidation trajectories: TWAP / VWAP / Almgren-Chriss
  - Execution cost distributions & VaR lines
  - Strategy comparison table
  - Efficient frontier visualisation
  - Execution analytics & KPI metrics

Architecture
------------
Path manipulation is handled via pathlib so the app works regardless of the
working directory from which streamlit is invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project subdirectories are importable regardless of cwd.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent   # project root
_PYTHON_DIR = _ROOT / "python"
for _subdir in ("bridge", "analytics", "data", ""):
    _p = _PYTHON_DIR / _subdir if _subdir else _PYTHON_DIR
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from bridge.simulator_interface import (
    get_trajectory,
    run_montecarlo,
    get_cost_paths,
    run_all_strategies,
    efficient_frontier,
    _find_binary,
)
from analytics.execution_compare import (
    twap_trajectory,
    vwap_trajectory,
    ac_trajectory,
    simulate_strategy_costs,
    compare_strategies,
)
from analytics.calibration import calibrate_all, estimate_volatility, estimate_avg_volume
from data.fetch_data import fetch_intraday_data, load_cached_data

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AC Execution Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark quant terminal aesthetic
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp { background-color: #0d1117; }
    .stAppHeader { background-color: #0d1117 !important; }
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }
    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
    }
    h2 { color: #e6edf3 !important; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
    thead tr th { background-color: #161b22 !important; color: #c9d1d9 !important; font-weight: 600 !important; }
    tbody tr:hover { background-color: #1c2128 !important; }
    hr { border-color: #21262d !important; }
    .stCaption { color: #8b949e !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Shared Plotly dark layout base (applied to every figure)
# ---------------------------------------------------------------------------
_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="Inter, Helvetica, Arial, sans-serif", size=12, color="#c9d1d9"),
    margin=dict(l=56, r=24, t=56, b=48),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="left",  x=0,
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
)

STRATEGY_COLORS = {
    "TWAP":           "#9467bd",
    "VWAP":           "#17becf",
    "Almgren-Chriss": "#ff7f0e",
}

# ---------------------------------------------------------------------------
# Sidebar — parameter controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Parameters")

    st.subheader("Market Data")
    ticker = st.text_input("Ticker", value="AAPL")
    data_period = st.selectbox("Period", ["5d", "7d", "1mo"], index=0)
    data_interval = st.selectbox("Interval", ["1m", "5m", "15m"], index=0)
    fetch_live = st.button("🔄 Fetch / Refresh Data")

    st.divider()
    st.subheader("Execution Parameters")

    order_size = st.number_input(
        "Order Size (shares)", min_value=1_000, max_value=10_000_000,
        value=1_000_000, step=10_000,
    )
    n_periods = st.slider("Liquidation Periods (N)", min_value=5, max_value=100, value=20)
    horizon_T = st.slider("Horizon T (years)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)

    st.divider()
    st.subheader("Model Parameters")

    use_calibrated = st.checkbox("Use Calibrated Parameters", value=False)

    sigma = st.number_input(
        "Volatility σ (per bar)", min_value=0.0001, max_value=0.5,
        value=0.02, format="%.4f",
    )
    eta = st.number_input(
        "Temporary Impact η", min_value=1e-6, max_value=10.0,
        value=0.1, format="%.6f",
    )
    gamma = st.number_input(
        "Permanent Impact γ", min_value=0.0, max_value=10.0,
        value=0.05, format="%.6f",
    )
    lambda_ = st.number_input(
        "Risk Aversion λ", min_value=0.0, max_value=100.0,
        value=0.1, format="%.4f",
    )
    s0 = st.number_input(
        "Initial Price S₀ ($)", min_value=1.0, max_value=100_000.0,
        value=100.0, format="%.2f",
    )

    st.divider()
    st.subheader("Monte Carlo")
    num_paths = st.slider("MC Paths", min_value=500, max_value=50_000, value=5_000, step=500)

# ---------------------------------------------------------------------------
# Session state — cache market data and calibration between reruns
# ---------------------------------------------------------------------------

if "market_df" not in st.session_state:
    st.session_state["market_df"] = None
if "calibrated_params" not in st.session_state:
    st.session_state["calibrated_params"] = {}

# ---------------------------------------------------------------------------
# Fetch market data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    try:
        return fetch_intraday_data(ticker, period, interval)
    except Exception:
        return load_cached_data(ticker, interval)


if fetch_live:
    with st.spinner(f"Fetching {ticker} {data_interval} data …"):
        df_market = _fetch_data(ticker, data_period, data_interval)
        st.session_state["market_df"] = df_market
        if df_market is not None and not df_market.empty:
            cal = calibrate_all(df_market)
            st.session_state["calibrated_params"] = cal
            st.sidebar.success(
                f"Calibrated: σ={cal['sigma']:.5f}  η={cal['eta']:.4e}  "
                f"γ={cal['gamma']:.4e}"
            )

df_market: pd.DataFrame | None = st.session_state["market_df"]
calibrated: dict = st.session_state["calibrated_params"]

# Apply calibrated values to parameter sliders if requested.
if use_calibrated and calibrated:
    sigma  = calibrated.get("sigma",  sigma)
    eta    = calibrated.get("eta",    eta)
    gamma  = calibrated.get("gamma",  gamma)
    if df_market is not None and "close" in df_market.columns:
        s0 = float(df_market["close"].iloc[-1])

# Detect whether the C++ binary is available.
try:
    _BINARY_PATH = _find_binary()
    _CPP_AVAILABLE = True
except FileNotFoundError:
    _CPP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Assemble the active parameter dict
# ---------------------------------------------------------------------------

params: dict = {
    "S0":     s0,
    "sigma":  sigma,
    "T":      horizon_T,
    "N":      n_periods,
    "X":      float(order_size),
    "eta":    eta,
    "gamma":  gamma,
    "lambda": lambda_,
}

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("📈 Almgren-Chriss Execution Analytics")
st.caption(
    "Optimal execution research platform · "
    "C++ Monte Carlo engine · "
    "Real market data · "
    "Strategy comparison"
)

if _CPP_AVAILABLE:
    st.success(f"C++ engine: `{_BINARY_PATH}`", icon="✅")
else:
    st.warning(
        "C++ binary not found — falling back to pure-Python simulation.  "
        "Build with `cmake -B build && cmake --build build --config Release`.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Helper: run simulation (C++ or Python fallback)
# ---------------------------------------------------------------------------

def _get_trajectories() -> dict[str, np.ndarray]:
    X = params["X"]
    N = params["N"]
    vol_profile = None
    if df_market is not None and "volume" in df_market.columns:
        raw_vol = df_market["volume"].values.astype(float)
        if len(raw_vol) >= N:
            # Use the last N bars as volume profile.
            vol_profile = raw_vol[-N:]

    return {
        "TWAP": twap_trajectory(X, N),
        "VWAP": vwap_trajectory(X, vol_profile) if vol_profile is not None else twap_trajectory(X, N),
        "Almgren-Chriss": ac_trajectory(
            X, N, sigma=sigma, eta=eta, lambda_=lambda_, T=horizon_T
        ),
    }


def _get_cost_paths(strategy_key: str, n_paths: int) -> np.ndarray:
    """Get cost paths — prefer C++ engine, fall back to NumPy."""
    cpp_strategy_map = {
        "TWAP": "twap",
        "Almgren-Chriss": "optimal",
        "VWAP": "twap",   # C++ doesn't have VWAP; use TWAP as proxy
    }
    if _CPP_AVAILABLE and strategy_key != "VWAP":
        try:
            return get_cost_paths(params, cpp_strategy_map[strategy_key], n_paths)
        except Exception:
            pass

    trajs = _get_trajectories()
    return simulate_strategy_costs(trajs[strategy_key], params, num_paths=n_paths)


# ===========================================================================
# SECTION 1 — Historical Market Data
# ===========================================================================

st.header("1 · Historical Market Data")

if df_market is not None and not df_market.empty:
    _latest      = float(df_market["close"].iloc[-1])
    _open_price  = float(df_market["close"].iloc[0])
    _log_rets    = np.log(df_market["close"] / df_market["close"].shift(1)).dropna()
    _realised_vol = float(_log_rets.std())
    _session_ret  = (_latest - _open_price) / _open_price * 100.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Price",         f"${_latest:,.2f}")
    col2.metric("Session Return",     f"{_session_ret:+.2f}%",  delta=f"{_session_ret:+.2f}%")
    col3.metric("Realised Vol (bar)", f"{_realised_vol:.5f}")
    col4.metric(
        "Avg Volume / bar",
        f"{estimate_avg_volume(df_market):,.0f}" if "volume" in df_market.columns else "N/A",
    )

    # Price + Volume subplot
    fig_price = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )
    fig_price.add_trace(
        go.Scatter(
            x=df_market["datetime"], y=df_market["close"],
            mode="lines", name="Close",
            line=dict(color="#58a6ff", width=1.4),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
        ),
        row=1, col=1,
    )
    # Bollinger Bands (20-bar ± 2σ)
    if len(df_market) >= 20:
        _roll_mean = df_market["close"].rolling(20).mean()
        _roll_std  = df_market["close"].rolling(20).std()
        fig_price.add_trace(
            go.Scatter(
                x=pd.concat([df_market["datetime"], df_market["datetime"][::-1]]),
                y=pd.concat([_roll_mean + 2 * _roll_std, (_roll_mean - 2 * _roll_std)[::-1]]),
                fill="toself", fillcolor="rgba(88,166,255,0.07)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Bollinger Bands (2σ)",
            ),
            row=1, col=1,
        )
        fig_price.add_trace(
            go.Scatter(
                x=df_market["datetime"], y=_roll_mean,
                mode="lines", name="MA-20",
                line=dict(color="#f0883e", width=1, dash="dot"),
            ),
            row=1, col=1,
        )
    if "volume" in df_market.columns:
        fig_price.add_trace(
            go.Bar(
                x=df_market["datetime"], y=df_market["volume"],
                name="Volume", marker_color="rgba(88,166,255,0.35)",
            ),
            row=2, col=1,
        )
    fig_price.update_layout(
        **_DARK_LAYOUT,
        title=f"{ticker.upper()} — {data_interval} OHLCV bars ({len(df_market):,} bars)",
        height=460,
        hovermode="x unified",
    )
    fig_price.update_yaxes(title_text="Price ($)",  row=1, col=1)
    fig_price.update_yaxes(title_text="Volume",     row=2, col=1)
    fig_price.update_xaxes(title_text="Date / Time", row=2, col=1)
    st.plotly_chart(fig_price, use_container_width=True)
else:
    st.info(
        "No market data loaded yet.  Enter a ticker in the sidebar and click "
        "**🔄 Fetch / Refresh Data**.",
        icon="ℹ️",
    )
    st.markdown("---")

# ===========================================================================
# SECTION 2 — Optimal Liquidation Trajectory
# ===========================================================================

st.header("2 · Optimal Liquidation Trajectories")

trajs = _get_trajectories()
dt = horizon_T / n_periods
times = np.arange(n_periods + 1) * dt
kappa = np.sqrt(lambda_ * sigma**2 / eta) if eta > 0 else 0.0

fig_traj = go.Figure()
for name, traj in trajs.items():
    pct = traj / order_size * 100.0
    fig_traj.add_trace(go.Scatter(
        x=times, y=pct,
        mode="lines+markers" if n_periods <= 40 else "lines",
        name=name,
        line=dict(color=STRATEGY_COLORS[name], width=2.2),
        marker=dict(size=5),
        hovertemplate=f"<b>{name}</b><br>t=%{{x:.3f}} yr<br>Inventory=%{{y:.1f}}%<extra></extra>",
    ))
# Shade AC vs TWAP difference region
_pct_ac   = trajs["Almgren-Chriss"] / order_size * 100.0
_pct_twap = trajs["TWAP"]           / order_size * 100.0
fig_traj.add_trace(go.Scatter(
    x=np.concatenate([times, times[::-1]]),
    y=np.concatenate([_pct_ac, _pct_twap[::-1]]),
    fill="toself", fillcolor="rgba(255,127,14,0.07)",
    line=dict(color="rgba(0,0,0,0)"),
    name="AC vs TWAP region", showlegend=False, hoverinfo="skip",
))
fig_traj.update_layout(
    **_DARK_LAYOUT,
    title="Inventory Schedule — percentage of initial position remaining",
    height=400,
)
fig_traj.update_xaxes(title_text="Time (years)")
fig_traj.update_yaxes(title_text="Inventory (%)", ticksuffix="%")
st.plotly_chart(fig_traj, use_container_width=True)

_ka1, _ka2, _ka3 = st.columns(3)
_ka1.metric("Urgency κ",  f"{kappa:.4f}",
            help="κ = √(λσ²/η).  Higher κ → more aggressive early trading.")
_ka2.metric("Periods N",  str(n_periods))
_ka3.metric("dt (years)", f"{dt:.4f}")
st.caption(
    "The Almgren-Chriss trajectory minimises **E[Cost] + λ·Var[Cost]**.  "
    "At κ → 0 it degenerates to TWAP; as κ increases execution is front-loaded."
)

# ===========================================================================
# SECTION 3 — Monte Carlo Price Paths
# ===========================================================================

st.header("3 · Monte Carlo Price Path Simulation")

_n_display_paths = min(num_paths, 200)   # cap displayed paths for browser performance

with st.spinner(f"Simulating {_n_display_paths} price paths …"):
    rng_vis = np.random.default_rng(42)
    dt_val  = horizon_T / n_periods
    sqdt    = np.sqrt(dt_val)
    ac_traj = trajs["Almgren-Chriss"]
    v_k     = np.diff(-ac_traj)

    Z       = rng_vis.standard_normal((_n_display_paths, n_periods))
    S_paths = np.zeros((_n_display_paths, n_periods + 1))
    S_paths[:, 0] = s0

    for k in range(n_periods):
        S_paths[:, k + 1] = (
            S_paths[:, k]
            - gamma * v_k[k]
            + sigma * sqdt * Z[:, k]
        )

fig_paths = go.Figure()

# Individual paths (very low opacity)
for i in range(min(_n_display_paths, 80)):
    fig_paths.add_trace(go.Scatter(
        x=times, y=S_paths[i],
        mode="lines",
        line=dict(color="rgba(88,166,255,0.07)", width=0.8),
        showlegend=False,
    ))

# Mean path.
mean_path = S_paths.mean(axis=0)
std_path  = S_paths.std(axis=0)

# ±2σ band
fig_paths.add_trace(go.Scatter(
    x=np.concatenate([times, times[::-1]]),
    y=np.concatenate([mean_path + 2 * std_path, (mean_path - 2 * std_path)[::-1]]),
    fill="toself", fillcolor="rgba(255,127,14,0.05)",
    line=dict(color="rgba(0,0,0,0)"),
    name="±2σ band", hoverinfo="skip",
))
# ±1σ band
fig_paths.add_trace(go.Scatter(
    x=np.concatenate([times, times[::-1]]),
    y=np.concatenate([mean_path + std_path, (mean_path - std_path)[::-1]]),
    fill="toself", fillcolor="rgba(255,127,14,0.12)",
    line=dict(color="rgba(0,0,0,0)"),
    name="±1σ band", hoverinfo="skip",
))
fig_paths.add_trace(go.Scatter(
    x=times, y=mean_path,
    mode="lines",
    line=dict(color="#ff7f0e", width=2.5),
    name="Mean path",
))

fig_paths.update_layout(
    **_DARK_LAYOUT,
    title=f"Monte Carlo Price Paths — AC optimal trajectory  ({_n_display_paths} paths shown)",
    height=400,
    hovermode="x unified",
)
fig_paths.update_xaxes(title_text="Time (years)")
fig_paths.update_yaxes(title_text="Mid Price ($)")
st.plotly_chart(fig_paths, use_container_width=True)

# ===========================================================================
# SECTION 4 — Cost Distribution
# ===========================================================================

st.header("4 · Execution Cost Distribution")

_path_cache: dict[str, np.ndarray] = {}
with st.spinner(f"Running {num_paths:,} Monte Carlo paths per strategy …"):
    _path_cache["TWAP"]           = _get_cost_paths("TWAP",           num_paths)
    _path_cache["Almgren-Chriss"] = _get_cost_paths("Almgren-Chriss", num_paths)
    # VWAP: pure-Python only (no C++ strategy); reuse cached VWAP trajectory
    _path_cache["VWAP"] = simulate_strategy_costs(trajs["VWAP"], params, num_paths=num_paths)

fig_hist = go.Figure()
_hist_colors = {"TWAP": "#9467bd", "VWAP": "#17becf", "Almgren-Chriss": "#ff7f0e"}

for strat_name, costs in _path_cache.items():
    fig_hist.add_trace(go.Histogram(
        x=costs,
        nbinsx=80,
        name=strat_name,
        opacity=0.65,
        marker_color=_hist_colors[strat_name],
        hovertemplate=f"<b>{strat_name}</b><br>Cost: $%{{x:,.0f}}<br>Count: %{{y}}<extra></extra>",
    ))

# Add 95th-percentile VaR vertical reference lines
for strat_name, costs in _path_cache.items():
    _var95 = float(np.percentile(costs, 95))
    fig_hist.add_vline(
        x=_var95,
        line=dict(color=_hist_colors[strat_name], dash="dot", width=1.2),
        annotation_text=f"{strat_name} VaR95",
        annotation_position="top",
        annotation_font_color=_hist_colors[strat_name],
    )

fig_hist.update_layout(
    **_DARK_LAYOUT,
    barmode="overlay",
    title=f"Execution Cost Distribution  ({num_paths:,} Monte Carlo paths per strategy)",
    height=420,
)
fig_hist.update_xaxes(title_text="Total Execution Cost ($)")
fig_hist.update_yaxes(title_text="Frequency")
st.plotly_chart(fig_hist, use_container_width=True)

# Summary stats below histogram
ac_costs   = _path_cache["Almgren-Chriss"]
twap_costs = _path_cache["TWAP"]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("AC Mean Cost",   f"${ac_costs.mean():,.0f}")
c2.metric("AC Std Dev",     f"${ac_costs.std():,.0f}")
c3.metric("TWAP Mean Cost", f"${twap_costs.mean():,.0f}")
c4.metric("TWAP Std Dev",   f"${twap_costs.std():,.0f}")
_savings_val = twap_costs.mean() - ac_costs.mean()
_savings_pct = _savings_val / twap_costs.mean() * 100.0 if twap_costs.mean() != 0 else 0.0
c5.metric(
    "AC vs TWAP Savings",
    f"${_savings_val:,.0f}",
    delta=f"{_savings_pct:+.1f}%",
)
c6.metric(
    "AC Variance Reduction",
    f"{(1 - ac_costs.var() / twap_costs.var()) * 100:.1f}%"
    if twap_costs.var() > 0 else "N/A",
)

# ===========================================================================
# SECTION 5 — Strategy Comparison Table
# ===========================================================================

st.header("5 · Strategy Comparison")

with st.spinner("Computing strategy comparison …"):
    vol_profile_arr = None
    if df_market is not None and "volume" in df_market.columns:
        raw_v = df_market["volume"].values.astype(float)
        if len(raw_v) >= n_periods:
            vol_profile_arr = raw_v[-n_periods:]

    comparison_df = compare_strategies(params, num_paths=num_paths, volume_profile=vol_profile_arr)

_fmt = {
    "Mean Cost ($)":                   "${:,.2f}",
    "Std Dev ($)":                     "${:,.2f}",
    "Variance ($²)":                   "{:,.2f}",
    "Implementation Shortfall (bps)":  "{:.4f}",
    "Cost per Share ($)":              "${:.6f}",
}
st.dataframe(
    comparison_df.style
        .format(_fmt)
        .highlight_min(subset=["Mean Cost ($)", "Variance ($²)", "Std Dev ($)"], color="#163b27")
        .highlight_max(subset=["Mean Cost ($)"],                                  color="#5c1a1a")
        .set_properties(**{"text-align": "right"}),
    use_container_width=True,
)

# Mean cost bar chart for visual comparison
_strat_names = comparison_df.index.tolist()
_mean_costs  = comparison_df["Mean Cost ($)"].values
_std_devs    = comparison_df["Std Dev ($)"].values
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=_strat_names,
    y=_mean_costs,
    error_y=dict(type="data", array=_std_devs, visible=True, color="#8b949e"),
    marker_color=[STRATEGY_COLORS.get(s, "#58a6ff") for s in _strat_names],
    text=[f"${v:,.0f}" for v in _mean_costs],
    textposition="outside",
    textfont=dict(color="#c9d1d9"),
    hovertemplate="<b>%{x}</b><br>Mean: $%{y:,.0f}<extra></extra>",
))
fig_bar.update_layout(
    **_DARK_LAYOUT,
    title="Mean Execution Cost Comparison  (error bars = ±1σ)",
    height=340,
    showlegend=False,
)
fig_bar.update_xaxes(title_text="Strategy")
fig_bar.update_yaxes(title_text="Mean Execution Cost ($)")
st.plotly_chart(fig_bar, use_container_width=True)

# ===========================================================================
# SECTION 6 — Efficient Frontier
# ===========================================================================

st.header("6 · Efficient Frontier  (E[Cost] vs Var[Cost])")

with st.spinner("Computing efficient frontier …"):
    lambdas_sweep = np.logspace(-4, 1, 25)

    frontier_rows: list[dict] = []
    for lam in lambdas_sweep:
        _p = params.copy()
        _p["lambda"] = float(lam)
        _traj = ac_trajectory(
            _p["X"], _p["N"],
            sigma=_p["sigma"], eta=_p["eta"],
            lambda_=lam, T=_p["T"],
        )
        costs_lam = simulate_strategy_costs(_traj, _p, num_paths=2000)
        frontier_rows.append({
            "lambda":    lam,
            "mean_cost": float(costs_lam.mean()),
            "var_cost":  float(costs_lam.var(ddof=1)),
            "std_cost":  float(costs_lam.std(ddof=1)),
        })

    frontier_df = pd.DataFrame(frontier_rows)

# Add TWAP & current AC as reference markers.
twap_traj_arr = twap_trajectory(params["X"], params["N"])
twap_costs_ef = simulate_strategy_costs(twap_traj_arr, params, num_paths=2000)
ac_traj_arr   = trajs["Almgren-Chriss"]
ac_costs_ef   = simulate_strategy_costs(ac_traj_arr, params, num_paths=2000)

fig_frontier = go.Figure()

fig_frontier.add_trace(go.Scatter(
    x=frontier_df["var_cost"], y=frontier_df["mean_cost"],
    mode="lines+markers",
    name="Efficient Frontier",
    line=dict(color="#58a6ff", width=2),
    marker=dict(size=5, color=np.log10(lambdas_sweep + 1e-9), colorscale="Viridis",
                showscale=True, colorbar=dict(title="log₁₀(λ)")),
))

fig_frontier.add_trace(go.Scatter(
    x=[twap_costs_ef.var(ddof=1)], y=[twap_costs_ef.mean()],
    mode="markers+text", name="TWAP",
    marker=dict(symbol="diamond", size=12, color="#9467bd"),
    text=["TWAP"], textposition="top right",
))
fig_frontier.add_trace(go.Scatter(
    x=[ac_costs_ef.var(ddof=1)], y=[ac_costs_ef.mean()],
    mode="markers+text", name=f"AC (λ={lambda_:.3f})",
    marker=dict(symbol="star", size=14, color="#ff7f0e"),
    text=[f"λ={lambda_:.3f}"], textposition="top right",
))

fig_frontier.update_layout(
    **_DARK_LAYOUT,
    title="Efficient Frontier — Expected Cost vs Variance  (Almgren & Chriss 2000)",
    height=440,
)
fig_frontier.update_xaxes(title_text="Variance of Execution Cost ($²)")
fig_frontier.update_yaxes(title_text="Expected Execution Cost ($)")
st.plotly_chart(fig_frontier, use_container_width=True)

st.caption(
    "Each point corresponds to a different risk-aversion λ.  "
    "Moving right along the frontier → lower variance, higher cost.  "
    "The optimal strategy for a given λ minimises the combined objective "
    "E[Cost] + λ·Var[Cost]."
)

# ===========================================================================
# SECTION 7 — Execution Analytics
# ===========================================================================

st.header("7 · Execution Analytics")

col_a, col_b = st.columns(2)

# — Trade schedule bar chart (all three strategies) —
with col_a:
    st.subheader("Trade Size Schedule")
    period_labels = np.arange(1, n_periods + 1)
    fig_trades = go.Figure()
    for _nm in ("Almgren-Chriss", "TWAP", "VWAP"):
        fig_trades.add_trace(go.Bar(
            x=period_labels, y=np.diff(-trajs[_nm]),
            name=_nm, marker_color=STRATEGY_COLORS[_nm], opacity=0.82,
            hovertemplate=f"<b>{_nm}</b><br>Period %{{x}}<br>%{{y:,.0f}} shares<extra></extra>",
        ))
    fig_trades.update_layout(
        **_DARK_LAYOUT,
        barmode="group",
        title="Shares Traded Per Period",
        height=340,
    )
    fig_trades.update_xaxes(title_text="Period")
    fig_trades.update_yaxes(title_text="Shares Traded")
    st.plotly_chart(fig_trades, use_container_width=True)

# — Cumulative cost build-up (all three strategies) —
with col_b:
    st.subheader("Cumulative Cost Build-Up")

    def _cumulative_costs(traj: np.ndarray, _params: dict, _n_mc: int = 500) -> np.ndarray:
        """Mean cumulative cost up to each period (small MC run)."""
        _n_p  = _params["N"]
        _dt_l = _params["T"] / _n_p
        _sqdt = np.sqrt(_dt_l)
        _v    = np.diff(-traj)
        _rng  = np.random.default_rng(0)
        _Z    = _rng.standard_normal((_n_mc, _n_p))
        _S    = np.full(_n_mc, _params["S0"])
        _cum  = np.zeros((_n_mc, _n_p))
        for _ki in range(_n_p):
            _P_exec       = _S - _params["eta"] * _v[_ki]
            _cum[:, _ki]  = _v[_ki] * (_params["S0"] - _P_exec)
            _S = _S - _params["gamma"] * _v[_ki] + _params["sigma"] * _sqdt * _Z[:, _ki]
        return _cum.mean(axis=0).cumsum()

    cum_times_arr = np.arange(1, n_periods + 1) * dt
    fig_cum = go.Figure()
    for _nm in ("Almgren-Chriss", "TWAP", "VWAP"):
        _cum_arr = _cumulative_costs(trajs[_nm], params)
        fig_cum.add_trace(go.Scatter(
            x=cum_times_arr, y=_cum_arr,
            mode="lines", name=_nm,
            line=dict(
                color=STRATEGY_COLORS[_nm], width=2,
                dash="solid" if _nm == "Almgren-Chriss" else "dash",
            ),
            hovertemplate=f"<b>{_nm}</b><br>t=%{{x:.3f}}<br>Cum. Cost: $%{{y:,.0f}}<extra></extra>",
        ))
    fig_cum.update_layout(
        **_DARK_LAYOUT,
        title="Mean Cumulative Cost Accrual",
        height=340,
    )
    fig_cum.update_xaxes(title_text="Time (years)")
    fig_cum.update_yaxes(title_text="Cumulative Cost ($)")
    st.plotly_chart(fig_cum, use_container_width=True)

# — Summary metrics row — (reuse already-computed _path_cache to avoid extra MC runs)
st.subheader("Key Execution Metrics")

# Reuse the cost samples computed in Section 4 (_path_cache is always present).
_ac_costs_s7   = _path_cache["Almgren-Chriss"]
_twap_costs_s7 = _path_cache["TWAP"]

_mean_cost_ac  = float(_ac_costs_s7.mean())
_std_cost_ac   = float(_ac_costs_s7.std())
_impl_sf_bps   = _mean_cost_ac / (s0 * order_size) * 1e4 if s0 > 0 and order_size > 0 else 0.0
_cost_per_share = _mean_cost_ac / order_size if order_size > 0 else 0.0
_var_ratio      = _ac_costs_s7.var() / _twap_costs_s7.var() if _twap_costs_s7.var() > 0 else 1.0

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("AC Expected Cost",  f"${_mean_cost_ac:,.0f}")
m2.metric("AC Std Dev",        f"${_std_cost_ac:,.0f}")
m3.metric("Impl. Shortfall",   f"{_impl_sf_bps:.2f} bps")
m4.metric("Cost / Share",      f"${_cost_per_share:.4f}")
m5.metric(
    "Urgency κ",
    f"{kappa:.4f}",
    help="κ = √(λσ²/η).  Higher κ → more aggressive front-loading.",
)
m6.metric(
    "AC Var / TWAP Var",
    f"{_var_ratio:.3f}",
    delta=f"{(_var_ratio - 1.0) * 100:+.1f}%",
    help="< 1.0 means AC has lower variance than TWAP.",
)

# ===========================================================================
# Footer
# ===========================================================================

# ===========================================================================
# SECTION 8 — Participation Rate & Market Impact
# ===========================================================================

st.header("8 · Market Impact & Participation Rate")

_avg_mkt_vol = (
    estimate_avg_volume(df_market)
    if df_market is not None and "volume" in df_market.columns
    else None
)

if _avg_mkt_vol and _avg_mkt_vol > 0:
    _shares_per_period = order_size / n_periods
    _part_rate         = _shares_per_period / _avg_mkt_vol * 100.0
    _pi1, _pi2, _pi3, _pi4 = st.columns(4)
    _pi1.metric("Avg Market Vol / bar",  f"{_avg_mkt_vol:,.0f} shares")
    _pi2.metric("Our Order / period",    f"{_shares_per_period:,.0f} shares")
    _pi3.metric("Participation Rate",    f"{_part_rate:.2f}%",
                help="Order size as % of average market volume per period. >20% is aggressive.")
    _temp_impact_est = eta * (_shares_per_period)
    _pi4.metric("Est. Temp Impact / period", f"${_temp_impact_est:,.4f} / share")
    if _part_rate > 30:
        st.warning(
            f"Participation rate of **{_part_rate:.1f}%** is high — significant market impact "
            "is likely. Consider reducing order size or extending the liquidation horizon.",
            icon="⚠️",
        )
    elif _part_rate > 10:
        st.info(
            f"Participation rate of {_part_rate:.1f}% is moderate. "
            "Monitor intraday volume closely.",
            icon="ℹ️",
        )
else:
    st.info(
        "Fetch market data to compute participation rate and live market impact estimates.",
        icon="ℹ️",
    )

# ===========================================================================
# Footer
# ===========================================================================

st.divider()
st.caption(
    "Almgren-Chriss Execution Analytics  ·  "
    "Based on Almgren & Chriss (2000) *Optimal Execution of Portfolio Transactions*  ·  "
    "C++ Monte Carlo engine · Streamlit · Plotly · yfinance"
)
