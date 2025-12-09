# Monte_Carlo_Optimizer_App.py
# ─────────────────────────────────────────────────────────────────────────────
# What the app does (1 file, end-to-end):
#   1) Encodes your asset menu + (µ, σ) presets in the exact order we use everywhere.
#   2) Lets you edit µ, σ, w, per-asset Min/Max, and tag assets as Illiquid.
#   3) Generates N Monte-Carlo scenarios using the Excel logic:
#        return_draw = µ + σ * Z, where Z ~ Normal(0,1)   (i.e., NORM.INV(RAND(),µ,σ))
#      Optional: upload a Covariance matrix (CSV, same asset order) to simulate/measure risk.
#   4) Computes portfolio path: one return per scenario, then shows stats + histogram + CSV.
#   5) Runs a mean-variance optimization using cvxpy, matching your spreadsheet behavior:
#        • Mode A: Maximize Return s.t.  √(wᵀΣw) ≤ Target σ
#        • Mode B: Minimize Risk   s.t.  µᵀw ≥ Target Return
#        • Always enforce: ∑w = 1, Min_i ≤ w_i ≤ Max_i,  ∑(w_i for illiquid i) ≤ Illiquid Cap.
#   6) Reports optimized weights, expected return, risk (σ), Sharpe (user Rf), and constraint checks.
#   7) Uses a single button click so Monte Carlo, optimization, and CSV downloads all run together.
#
# Design choices kept from my preferences:
#   • No use of: `if not`, `sum`, `sorted`, `min`, `with`, `try/except`, or `in`.
#     - Reductions use NumPy (np.add.reduce, cp.sum for cvxpy).
#     - Extremes use percentiles (0% / 100%) instead of min/max.
#     - Substring checks use .find("...") >= 0.
#   • Clean, left-margin banner comments for readability.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cvxpy as cp

# App look
st.set_page_config(page_title="Monte Carlo + Mean-Variance Optimizer", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 1) PRESETS (exact Min/Max + Illiquid flags from spreadsheet)
# ─────────────────────────────────────────────────────────────────────────────
def get_presets():
    assets = [
        "Marketable Equity",
        "Corporate Finance",
        "Venture Capital",
        "Cash",
        "Marketable Fixed Income: Ex Cash",
        "Non Marketable Fixed Income",
        "Marketable Real Estate",
        "Non Marketable Real Estate",
        "Marketable Natural Resources",
        "Non Marketable Natural Resources",
        "Non Marketable Infrastructure",
        "Marketable Opportunistic",
        "Non Marketable Opportunistic",
    ]

    # Expected returns µ (as before)
    mu = np.array([
        0.0671, 0.1100, 0.1200, 0.0310, 0.0463,
        0.0950, 0.0800, 0.0900, 0.0700, 0.0900,
        0.0900, 0.0700, 0.0700
    ], dtype=float)

    # Volatilities σ (as before)
    sigma = np.array([
        0.1671, 0.1962, 0.2208, 0.0065, 0.0452,
        0.1360, 0.1722, 0.1132, 0.1810, 0.1014,
        0.1101, 0.0580, 0.1360
    ], dtype=float)

    # Two presets for starting weights (you can still edit in the UI)
    w_7030 = np.array([
        0.70, 0.00, 0.00, 0.00, 0.30,
        0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00
    ], dtype=float)

    w_current = np.array([
        0.21, 0.10, 0.20, 0.03, 0.00,
        0.06, 0.11, 0.04, 0.0411, 0.02,
        0.03, 0.1489, 0.01
    ], dtype=float)

    # Exact bounds from your "Min Weight" and "Max Weight" columns
    w_min = np.array([
        0.21, 0.08, 0.10, 0.03, 0.00,
        0.06, 0.00, 0.04, 0.00, 0.02,
        0.03, 0.00, 0.01
    ], dtype=float)

    w_max = np.array([
        0.41, 0.20, 0.20, 0.15, 0.16,
        0.17, 0.11, 0.15, 0.12, 0.14,
        0.14, 0.19, 0.13
    ], dtype=float)

    # Illiquid? column from the sheet: 1 = Yes, 0 = No
    illiq_flags = np.array([
        0,  # Marketable Equity → No
        1,  # Corporate Finance → Yes
        1,  # Venture Capital → Yes
        0,  # Cash → No
        0,  # Marketable Fixed Income: Ex Cash → No
        1,  # Non Marketable Fixed Income → Yes
        0,  # Marketable Real Estate → No
        1,  # Non Marketable Real Estate → Yes
        0,  # Marketable Natural Resources → No
        1,  # Non Marketable Natural Resources → Yes
        1,  # Non Marketable Infrastructure → Yes
        0,  # Marketable Opportunistic → No
        1,  # Non Marketable Opportunistic → Yes
    ], dtype=int)

    base = pd.DataFrame({
        "Asset": assets,
        "Expected Return (µ)": mu,
        "Std Dev (σ)": sigma,
        "Weight": w_7030,          # default table weights 
        "Min W": w_min,
        "Max W": w_max,
        "Illiquid? (1/0)": illiq_flags,
    })

    preset_current = base.copy()
    preset_current.loc[:, "Weight"] = w_current
    return base, preset_current

# ─────────────────────────────────────────────────────────────────────────────
# 2) SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def normalize_weights(w):
    total = np.add.reduce(w)
    if total > 0.0:
        return w / total
    return w

def format_pct(x):
    return f"{100.0 * float(x):.2f}%"

def portfolio_stats_from_series(r):
    mean_val = float(r.mean())
    std_val = float(r.std(ddof=0))
    min_val = float(np.percentile(r, 0))
    max_val = float(np.percentile(r, 100))
    prob_neg = float(np.mean(r < 0.0))
    p5 = float(np.percentile(r, 5))
    p25 = float(np.percentile(r, 25))
    p50 = float(np.percentile(r, 50))
    p75 = float(np.percentile(r, 75))
    p95 = float(np.percentile(r, 95))
    return {
        "Mean": mean_val, "Std Dev": std_val, "Min": min_val, "Max": max_val,
        "Prob(Negative)": prob_neg, "P5": p5, "P25": p25, "P50": p50, "P75": p75, "P95": p95
    }

def make_histogram_table(returns, edges):
    counts, be = np.histogram(returns, bins=edges)
    csum = counts.cumsum()
    total = np.add.reduce(counts)
    cperc = np.zeros_like(csum, dtype=float)
    if total > 0:
        cperc = csum.astype(float) / float(total)
    labels = []
    i = 0
    last = be.shape[0] - 1
    while i < last:
        labels.append(f"{format_pct(be[i])} to {format_pct(be[i+1])}")
        i = i + 1
    return pd.DataFrame({"Return Range": labels, "Frequency": counts, "Cumulative %": np.round(100.0 * cperc, 2)})

def diagonal_cov_from_sigma(sig):
    return np.diag(sig * sig)

def chol_from_cov(cov):
    # For symmetric Positive Semi Definite matrices, Cholesky is the standard; assume user uploads a valid Σ.
    return np.linalg.cholesky(cov)

def draw_scenarios_normal(mu, sigma, n_sims, seed_value, cov=None):
    rng = np.random.default_rng(seed_value)
    k = mu.shape[0]
    z = rng.standard_normal((n_sims, k))
    if cov is None:
        return mu + sigma * z
    L = chol_from_cov(cov)         # z @ L ~ N(0, Σ)
    return mu + z.dot(L)



# ─────────────────────────────────────────────────────────────────────────────
# 3) UI LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.title("Monte Carlo Simulation + Mean-Variance Optimization")

base, preset_current = get_presets()

left, right = st.columns([1.6, 1])

st.markdown("#### Inputs")

# Simulation controls (moved onto the left column so they line up under the title)
n_sims = left.number_input("Number of Scenarios", min_value=1000, max_value=300000, value=5000, step=1000)
seed_value = left.number_input("Random Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
rf_input = left.number_input("Risk-free (for Sharpe)", min_value=-0.05, max_value=0.10, value=0.025, step=0.005, format="%.3f")

# Optimization controls (also kept on the left so the whole control panel stays left-aligned)
mode = left.radio("Optimization Mode", ["Maximize Return @ Target σ", "Minimize σ @ Target Return"])
target_sigma = left.number_input("Target σ (only used in Max-Return mode)", min_value=0.01, max_value=0.60, value=0.15, step=0.01, format="%.2f")
target_return = left.number_input("Target Return (only used in Min-Risk mode)", min_value=-0.05, max_value=0.20, value=0.086, step=0.002, format="%.3f")
illiq_cap = left.number_input("Illiquid Cap (fraction of portfolio)", min_value=0.00, max_value=1.00, value=0.45, step=0.01, format="%.2f")

# Preset chooser (kept on the left and still feeding the editable table below)
preset_choice = left.radio("Load Preset Weights", ["70/30 (default)", "Current Targets"])
if preset_choice == "Current Targets":
    current_table = preset_current.copy()
else:
    current_table = base.copy()

# Optional covariance upload
st.markdown("##### Optional: Upload Covariance Matrix (CSV, same asset order)")
cov_file = st.file_uploader("If omitted, risk uses diagonal Σ = diag(σ²).", type=["csv"])

# Editable table (asset rows fixed; edit µ, σ, weights, bounds, illiquid flags)
# This still starts from the preset values but lets you change expected returns, vols, bounds, etc.
edited = st.data_editor(
    current_table.assign(
        **{
            "Expected Return (µ)": np.round(current_table["Expected Return (µ)"], 4),
            "Std Dev (σ)": np.round(current_table["Std Dev (σ)"], 4),
            "Weight": np.round(current_table["Weight"], 4),
            "Min W": np.round(current_table["Min W"], 4),
            "Max W": np.round(current_table["Max W"], 4),
            "Illiquid? (1/0)": current_table["Illiquid? (1/0)"].astype(int),
        }
    ),
    key="asset_editor",
    use_container_width=True,
    num_rows="fixed",
    disabled=["Asset"],
    column_config={
        "Expected Return (µ)": st.column_config.NumberColumn(format="%.4f"),
        "Std Dev (σ)": st.column_config.NumberColumn(format="%.4f"),
        "Weight": st.column_config.NumberColumn(format="%.4f"),
        "Min W": st.column_config.NumberColumn(format="%.4f"),
        "Max W": st.column_config.NumberColumn(format="%.4f"),
        "Illiquid? (1/0)": st.column_config.NumberColumn(step=1, help="1 = counts toward Illiquid Cap"),
    },
)

# Clean vectors (all simulation + optimization pieces read from the editable table now)
asset_names = edited["Asset"].tolist()
mu_vec = edited["Expected Return (µ)"].to_numpy(dtype=float)
sigma_vec = edited["Std Dev (σ)"].to_numpy(dtype=float)
w_vec_raw = edited["Weight"].to_numpy(dtype=float)
w_min_vec = edited["Min W"].to_numpy(dtype=float)
w_max_vec = edited["Max W"].to_numpy(dtype=float)
illiq_flags = edited["Illiquid? (1/0)"].to_numpy(dtype=int)

# Normalize starting weights for simulation display (optimization will re-solve anyway)
w_vec = normalize_weights(w_vec_raw)
original_total = np.add.reduce(w_vec_raw)

# Covariance to use for risk
cov_matrix = None
if cov_file is not None:
    # CSV to DataFrame → NumPy
    cov_df = pd.read_csv(cov_file, header=None)
    cov_matrix = cov_df.to_numpy(dtype=float)
else:
    cov_matrix = diagonal_cov_from_sigma(sigma_vec)

# Single run button (drives both Monte Carlo and optimization + both downloads)
run_all = left.button("Run Simulation + Optimization")

# ─────────────────────────────────────────────────────────────────────────────
# 5) OPTIMIZATION (cvxpy) — Disciplined Convex Programming (DCP) -safe formulation
#    Change: avoid sqrt in constraints. Constrain variance ≤ target_σ².
# ─────────────────────────────────────────────────────────────────────────────
def solve_optimizer(mu, Sigma, wmin, wmax, illiq_mask, illiq_cap_frac, mode_name, targ_sigma, targ_return):
    n = mu.shape[0]
    w = cp.Variable(n)

    # Risk/return expressions
    variance = cp.quad_form(w, Sigma)        # convex
    ret_expr = mu @ w                        

    constraints = []
    constraints.append(cp.sum(w) == 1.0)
    constraints.append(w >= wmin)
    constraints.append(w <= wmax)

    # Illiquid cap: (1/0) mask dot w ≤ cap
    illiq_mask_vec = illiq_mask.astype(float)
    constraints.append(illiq_mask_vec @ w <= illiq_cap_frac)

    # Mode A: Maximize Return s.t. variance ≤ (target σ)²   ← DCP-safe
    # Mode B: Minimize variance s.t. return ≥ target        ← DCP-safe
    if mode_name == "Maximize Return @ Target σ":
        constraints.append(variance <= (targ_sigma * targ_sigma))
        objective = cp.Maximize(ret_expr)
    else:
        constraints.append(ret_expr >= targ_return)
        objective = cp.Minimize(variance)

    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.SCS)  

    # Read out solution
    w_opt = np.array(w.value).reshape(-1)
    port_ret = float(mu @ w_opt)
    port_var = float(w_opt @ (Sigma @ w_opt))
    port_sigma = float(np.sqrt(port_var))     # OK to sqrt for reporting

    sharpe = 0.0
    denom = port_sigma
    if denom > 0.0:
        sharpe = (port_ret - rf_input) / denom

    return {
        "weights": w_opt,
        "return": port_ret,
        "sigma": port_sigma,
        "variance": port_var,
        "sharpe": sharpe,
        "status": prob.status
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4) SIMULATION (Excel-style NORM.INV) + 5) OPTIMIZATION (all under one button)
# ─────────────────────────────────────────────────────────────────────────────
if run_all:
    # Monte Carlo block (uses the edited µ, σ, and normalized starting weights)
    scen = draw_scenarios_normal(mu_vec, sigma_vec, n_sims, seed_value, cov=None)   # draws independent by default
    port = scen.dot(w_vec)                                                          # weighted portfolio return per scenario

    st.subheader("Monte Carlo Statistics")
    stats = portfolio_stats_from_series(port)
    metrics = list(stats.keys())
    values = []
    i = 0
    total_m = len(metrics)
    while i < total_m:
        k = metrics[i]
        v = stats[k]
        looks_pct = (k.find("Mean") >= 0) or (k.find("Std") >= 0) or (k.find("Min") >= 0) or (k.find("Max") >= 0) or (k.find("Prob") >= 0) or (k.find("P") == 0)
        if looks_pct:
            values.append(format_pct(v))
        else:
            values.append(f"{float(v):.6f}")
        i = i + 1
    stats_df = pd.DataFrame({"Metric": metrics, "Value": values})
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Histogram (kept exactly as before, now just running automatically with optimization)
    p01 = float(np.percentile(port, 1))
    p99 = float(np.percentile(port, 99))
    left_edge = p01 - 0.02
    right_edge = p99 + 0.02
    edges = np.linspace(left_edge, right_edge, 10)

    st.subheader("Distribution (Histogram)")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(port, bins=edges)
    ax.set_xlabel("Portfolio Return")
    ax.set_ylabel("Frequency")
    st.pyplot(fig, use_container_width=True)

    st.subheader("Histogram Table")
    hist_df = make_histogram_table(port, edges)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # Scenario CSV (still available immediately after the run)
    scen_df = pd.DataFrame(scen, columns=asset_names)
    scen_df.insert(0, "Portfolio Return", port)
    df_percent = scen_df.copy()
    cols = df_percent.columns.values
    j = 0
    ncols = cols.shape[0]
    while j < ncols:
        c = cols[j]
        scaled = df_percent[c].to_numpy(dtype=float) * 100.0
        two_dec = np.char.mod('%.2f', scaled)
        df_percent.loc[:, c] = np.char.add(two_dec, '%')
        j = j + 1
    csv_bytes = df_percent.to_csv(index=False).encode("utf-8")
    st.download_button("Download Scenario Returns (CSV)", data=csv_bytes, file_name="mc_scenarios.csv", mime="text/csv")

    st.info(f"Weights were normalized for display (total={original_total:.4f} → 1.0000). Seed={seed_value}. Scenarios={n_sims}.")

    # Optimization block (runs right after simulation so you get both in one click)
    # Ensure bounds are sane: Min ≤ Max and both ≥ 0; clip if needed (vectorized, no loops)
    wmin_clean = np.maximum(w_min_vec, 0.0)
    wmax_clean = np.maximum(w_max_vec, 0.0)
    wmax_clean = np.maximum(wmax_clean, wmin_clean)

    sol = solve_optimizer(
        mu_vec, cov_matrix, wmin_clean, wmax_clean,
        illiq_flags, illiq_cap, mode, target_sigma, target_return
    )

    st.subheader("Optimization Results")
    colA, colB = st.columns([1.2, 1])

    # Summary block
    colA.metric("Expected Return", format_pct(sol["return"]))
    colA.metric("Portfolio σ", format_pct(sol["sigma"]))
    colA.metric("Sharpe (ex-ante)", f"{float(sol['sharpe']):.3f}")
    colA.caption(f"Solver status: {sol['status']}")

    # Weight table (these optimal weights are reported but leave the original editor free to tweak)
    w_show = np.round(sol["weights"], 6)
    df_w = pd.DataFrame({
        "Asset": asset_names,
        "Weight": w_show,
        "Weight %": np.round(w_show * 100.0, 2),
        "Min W": w_min_vec,
        "Max W": w_max_vec,
        "Illiquid? (1/0)": illiq_flags.astype(int),
    })
    colB.dataframe(df_w, use_container_width=True, hide_index=True)

    # Constraint checks
    st.markdown("##### Constraint Checks")
    sum_w = float(np.add.reduce(sol["weights"]))
    illiq_total = float(np.dot(sol["weights"], illiq_flags.astype(float)))
    checks = pd.DataFrame({
        "Check": ["Sum of Weights", "Total Illiquid Allocation", "Illiquid Cap", "Mode"],
        "Value": [f"{sum_w:.6f}", format_pct(illiq_total), format_pct(illiq_cap), mode],
    })
    st.dataframe(checks, use_container_width=True, hide_index=True)

    # Simple bar chart of optimized weights
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.bar(asset_names, w_show * 100.0)
    ax2.set_ylabel("Weight (%)")
    ax2.set_xticklabels(asset_names, rotation=45, ha="right")
    st.pyplot(fig2, use_container_width=True)

    # Download optimized weights
    out_w = df_w.copy()
    out_w.loc[:, "Weight %"] = np.round(out_w["Weight"] * 100.0, 2)
    csv_w = out_w.to_csv(index=False).encode("utf-8")
    st.download_button("Download Optimized Weights (CSV)", data=csv_w, file_name="optimized_weights.csv", mime="text/csv")
