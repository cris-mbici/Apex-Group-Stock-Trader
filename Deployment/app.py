# app.py
# Run: streamlit run app.py
# Requirements: pip install streamlit pandas numpy plotly

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="NSE Monthly Model Dashboard", layout="wide")

# -----------------------------
# Loaders
# -----------------------------
@st.cache_data
def load_csv(path, parse_month=True):
    df = pd.read_csv(path)
    if parse_month and "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"])
    return df

try:
    alloc   = load_csv("allocations.csv")
    bt      = load_csv("backtest_monthly.csv")
    prices  = load_csv("prices_monthly.csv")   # used for random benchmark
except Exception as e:
    st.error(f"Error loading CSVs: {e}")
    st.stop()

st.title("NSE Monthly Model Dashboard")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

# Model-valid window: trained up to 2023 => only allow 2024-01 .. 2025-10
allowed_start = pd.to_datetime("2024-01-01")
allowed_end   = pd.to_datetime("2025-10-01")

avail_months = pd.to_datetime(sorted(bt["month"].unique()))
avail_months = avail_months[(avail_months >= allowed_start) & (avail_months <= allowed_end)]
if len(avail_months) == 0:
    st.warning("No backtest months available in 2024-01 .. 2025-10.")
    st.stop()

start_month = st.sidebar.selectbox("Start month", options=avail_months, index=0)
end_choices = [m for m in avail_months if m >= start_month]
end_month = st.sidebar.selectbox("End month", options=end_choices, index=len(end_choices) - 1)

initial_capital = st.sidebar.number_input(
    "Starting Capital (KES)", min_value=100_000.0, value=1_000_000.0, step=100_000.0, format="%.0f"
)

# Fees/slippage per side (bps) and withholding tax (on positive monthly profits)
fee_bps_per_side  = st.sidebar.number_input("Fees (per side, bps)", min_value=0.0, value=30.0, step=5.0)
slip_bps_per_side = st.sidebar.number_input("Slippage (per side, bps)", min_value=0.0, value=20.0, step=5.0)
withholding_tax_rate = st.sidebar.number_input("Withholding Tax on Profits (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0) / 100.0
roundtrip_bps = 2.0 * (fee_bps_per_side + slip_bps_per_side)

st.caption(
    f"Costs applied: Fees {fee_bps_per_side:.0f} bps/side, Slippage {slip_bps_per_side:.0f} bps/side "
    f"(round-trip {roundtrip_bps:.0f} bps). Withholding tax {withholding_tax_rate*100:.0f}% on positive monthly profits."
)

# -----------------------------
# Random benchmark only
# -----------------------------
def normalize_to_equity(returns_df, month_col, ret_col, start_month, end_month, start_capital):
    s = returns_df[(returns_df[month_col] >= start_month) & (returns_df[month_col] <= end_month)].copy().sort_values(month_col)
    if s.empty:
        return pd.DataFrame()
    eq = start_capital
    equities = []
    for r in s[ret_col].fillna(0.0):
        eq *= (1.0 + float(r))
        equities.append(eq)
    s["equity"] = equities
    return s[[month_col, "equity"]]

def compute_random_average_benchmark(prices_df, alloc_df, start_month, end_month, start_capital,
                                     roundtrip_bps, withholding_tax_rate, volume_threshold=50_000, seed=41):
    rng = np.random.default_rng(seed)
    p = prices_df.copy()
    p["ret"] = (p["close"] - p["open"]) / p["open"]
    p = p.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret"])
    p = p[p["volume"].fillna(0) >= volume_threshold]

    months = pd.to_datetime(sorted(p["month"].unique()))
    months = months[(months >= start_month) & (months <= end_month)]

    rows = []
    for m in months:
        k = len(alloc_df[alloc_df["month"] == m]) if not alloc_df.empty else 0
        if k <= 0:
            k = min(10, len(p[p["month"] == m]["ticker"].unique()))
        if k <= 0:
            continue

        pm = p[p["month"] == m]
        tickers = pm["ticker"].unique().tolist()
        if not tickers:
            continue

        pick = rng.choice(tickers, size=min(k, len(tickers)), replace=False)
        pm_pick = pm[pm["ticker"].isin(pick)].copy()
        if pm_pick.empty:
            continue

        gross_ret = pm_pick["ret"].mean()
        monthly_cost_frac = (roundtrip_bps / 10_000.0)
        ret_after_cost = gross_ret - monthly_cost_frac
        tax = max(ret_after_cost, 0.0) * withholding_tax_rate
        ret_after_tax = ret_after_cost - tax

        rows.append({"month": m, "bench_ret": ret_after_tax})

    if not rows:
        return pd.DataFrame()

    r = pd.DataFrame(rows).sort_values("month")
    r = normalize_to_equity(r, "month", "bench_ret", start_month, end_month, start_capital)
    r["label"] = "Benchmark: Average investor choice (Random EW)"
    return r

# -----------------------------
# Slice backtest and compute equity
# -----------------------------
bt_slice = bt[(bt["month"] >= start_month) & (bt["month"] <= end_month)].copy().sort_values("month")
if bt_slice.empty:
    st.info("No backtest rows in the selected range.")
    st.stop()

if {"gross_ret", "turnover"}.issubset(bt_slice.columns):
    monthly_cost_frac = bt_slice["turnover"].fillna(0.0) * (roundtrip_bps / 10_000.0)
    bt_slice["monthly_ret_costed"] = bt_slice["gross_ret"].fillna(0.0) - monthly_cost_frac
    base_ret_col = "monthly_ret_costed"
elif "monthly_ret" in bt_slice.columns:
    base_ret_col = "monthly_ret"
else:
    bt_slice["monthly_ret"] = 0.0
    base_ret_col = "monthly_ret"

bt_slice["monthly_tax"] = bt_slice[base_ret_col].clip(lower=0.0) * withholding_tax_rate
bt_slice["monthly_ret_after_tax"] = bt_slice[base_ret_col] - bt_slice["monthly_tax"]
use_ret_col = "monthly_ret_after_tax"

equity = initial_capital
equities = []
for r in bt_slice[use_ret_col].fillna(0.0):
    equity *= (1.0 + float(r))
    equities.append(equity)
bt_slice["equity_scaled"] = equities

# -----------------------------
# Compute random investor benchmark (seed 41)
# -----------------------------
rand_eq = compute_random_average_benchmark(
    prices_df=prices,
    alloc_df=alloc[(alloc["month"] >= start_month) & (alloc["month"] <= end_month)] if not alloc.empty else alloc,
    start_month=start_month,
    end_month=end_month,
    start_capital=initial_capital,
    roundtrip_bps=roundtrip_bps,
    withholding_tax_rate=withholding_tax_rate,
    volume_threshold=50_000,
    seed=41
)

# -----------------------------
# Layout
# -----------------------------
tab1, tab2 = st.tabs(["Backtest", "Predictive (Next Month)"])

with tab1:
    c1, c2 = st.columns([2, 1])

    with c1:
        # Equity curve with random benchmark
        fig = px.line(bt_slice, x="month", y="equity_scaled", title="Equity Curve")

        # Add random investor benchmark line
        if not rand_eq.empty:
            label = rand_eq["label"].iloc[0]
            fig.add_scatter(x=rand_eq["month"], y=rand_eq["equity"], mode="lines", name=str(label))

        # Move legend below the chart
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,           # push legend below plot area
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=50, b=100)  # add space below for legend
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Layman-friendly KPIs
        monthly = bt_slice[use_ret_col].fillna(0.0)
        final_value = bt_slice["equity_scaled"].iloc[-1]
        total_return = final_value / initial_capital - 1.0
        avg_monthly = monthly.mean() if len(monthly) else 0.0
        best = monthly.max() if len(monthly) else 0.0
        worst = monthly.min() if len(monthly) else 0.0
        win_rate = monthly.gt(0).mean() if len(monthly) else 0.0
        eq_series = bt_slice["equity_scaled"]
        maxdd = (eq_series / eq_series.cummax() - 1).min() if not eq_series.empty else 0.0

        st.metric("Final Portfolio Value", f"KES {final_value:,.0f}")
        st.metric("Total Return", f"{total_return:.2%}")
        st.metric("Average Monthly Return", f"{avg_monthly:.2%}")
        st.metric("Winning Months", f"{win_rate:.0%}")
        st.metric("Best Month", f"{best:.2%}")
        st.metric("Worst Month", f"{worst:.2%}")
        st.metric("Max Drawdown", f"{maxdd:.2%}")

    st.markdown("### Monthly Net Returns (after fees, slippage, tax)")
    fig2 = px.bar(bt_slice, x="month", y=use_ret_col, title="Monthly Net Returns")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Holdings & Realized P&L (Selected Month)")
    alloc_range = alloc[(alloc["month"] >= start_month) & (alloc["month"] <= end_month)].copy()
    if not alloc_range.empty:
        sel_month = st.selectbox("Month", options=sorted(alloc_range["month"].unique()))
        a = alloc_range[alloc_range["month"] == sel_month].copy()
        if not a.empty:
            if {"exec_open", "exec_close"}.issubset(a.columns):
                a["gross_ret"] = (a["exec_close"] - a["exec_open"]) / a["exec_open"]
                a["weighted_pnl"] = a.get("weight", 0.0) * a["gross_ret"]
            a = a.sort_values("weight", ascending=False) if "weight" in a.columns else a
            cols = ["ticker", "weight", "proba", "composite", "exec_open", "exec_close", "gross_ret", "weighted_pnl"]
            st.dataframe(a[[c for c in cols if c in a.columns]].fillna(""), use_container_width=True)