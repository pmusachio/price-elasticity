"""Price Elasticity Dashboard — interactive simulator for e-commerce pricing decisions."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from price_elasticity.analysis import run_analysis
from price_elasticity.config import load_config
from price_elasticity.data import normalize_columns


st.set_page_config(
    page_title="Price Elasticity Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_report() -> pd.DataFrame:
    report_path = ROOT / "reports" / "price_elasticity.csv"
    if not report_path.exists():
        with st.spinner("Running elasticity pipeline for the first time…"):
            run_analysis(ROOT / "configs" / "project.toml")
    return pd.read_csv(report_path)


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    config = load_config(ROOT / "configs" / "project.toml")
    raw_path = ROOT / config["data"]["train_file"]
    return normalize_columns(pd.read_csv(raw_path, low_memory=False))


@st.cache_data(show_spinner=False)
def load_cross_price() -> pd.DataFrame | None:
    path = ROOT / "data/raw/df_crossprice_previous.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def product_history(raw: pd.DataFrame, product: str) -> pd.DataFrame:
    config = load_config(ROOT / "configs" / "project.toml")
    data = config["data"]
    product_col = normalize_columns(pd.DataFrame(columns=[data.get("product_column", "name")])).columns[0]
    price_col = normalize_columns(pd.DataFrame(columns=[data.get("price_column", "disc_price")])).columns[0]
    demand_col = normalize_columns(pd.DataFrame(columns=[data.get("demand_column", "imp_count")])).columns[0]

    history = raw.loc[raw[product_col] == product].copy()
    if history.empty:
        return history

    history[price_col] = pd.to_numeric(history[price_col], errors="coerce")
    history[demand_col] = pd.to_numeric(history[demand_col], errors="coerce")
    history = history[(history[price_col] > 0) & (history[demand_col] > 0)]

    if "week_number" in history.columns:
        period_col = "week_number"
    elif "date_imp_d" in history.columns:
        history["period"] = pd.to_datetime(history["date_imp_d"], errors="coerce").dt.to_period("W").astype(str)
        period_col = "period"
    else:
        history["period"] = np.arange(len(history))
        period_col = "period"

    grouped = (
        history.groupby(period_col)
        .agg(avg_price=(price_col, "mean"), demand=(demand_col, "sum"), observations=(demand_col, "size"))
        .reset_index()
        .rename(columns={period_col: "period"})
    )
    return grouped


def regression_figure(history: pd.DataFrame, row: pd.Series) -> go.Figure:
    elasticity = float(row["price_elasticity"])
    r2 = float(row["r2"])

    ci_lo_cols = [c for c in row.index if "ci_" in c and "lower" in c]
    ci_hi_cols = [c for c in row.index if "ci_" in c and "upper" in c]
    ci_text = ""
    if ci_lo_cols and ci_hi_cols:
        lo = float(row[ci_lo_cols[0]])
        hi = float(row[ci_hi_cols[0]])
        ci_text = f"  |  95% CI: [{lo:.2f}, {hi:.2f}]"

    fig = px.scatter(
        history,
        x="avg_price",
        y="demand",
        size="observations",
        hover_data=["period"],
        labels={"avg_price": "Average Price ($)", "demand": "Observed Demand"},
        title=f"Price × Demand  |  ε = {elasticity:.2f}{ci_text}  |  R² = {r2:.2f}",
    )

    if len(history) >= 2 and history["avg_price"].nunique() > 1:
        slope, intercept = np.polyfit(history["avg_price"], history["demand"], deg=1)
        x_line = np.linspace(history["avg_price"].min(), history["avg_price"].max(), 100)
        y_line = intercept + slope * x_line
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Linear trend", line=dict(color="red", dash="dash")))

    fig.update_layout(height=460, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def simulate_row(row: pd.Series, price_change_pct: float) -> dict:
    elasticity = float(row["price_elasticity"])
    current_price = float(row["avg_price"])
    current_demand = float(row["avg_demand"])
    new_price = current_price * (1 + price_change_pct)
    demand_change_pct = elasticity * price_change_pct
    new_demand = max(0.0, current_demand * (1 + demand_change_pct))
    current_revenue = current_price * current_demand
    new_revenue = new_price * new_demand
    return {
        "current_price": current_price,
        "new_price": new_price,
        "current_demand": current_demand,
        "new_demand": new_demand,
        "demand_change_pct": demand_change_pct,
        "current_revenue": current_revenue,
        "new_revenue": new_revenue,
        "revenue_delta": new_revenue - current_revenue,
        "revenue_delta_pct": (new_revenue - current_revenue) / current_revenue if current_revenue else 0.0,
    }


def simulate_portfolio(df: pd.DataFrame, price_change_pct: float) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        result = simulate_row(row, price_change_pct)
        rows.append({
            "product": row["product"],
            "price_elasticity": row["price_elasticity"],
            "r2": row["r2"],
            "current_revenue": result["current_revenue"],
            "new_revenue": result["new_revenue"],
            "revenue_delta": result["revenue_delta"],
            "revenue_delta_pct": result["revenue_delta_pct"],
        })
    return pd.DataFrame(rows).sort_values("revenue_delta", ascending=False)


def cross_price_table(cross: pd.DataFrame | None, product: str) -> pd.DataFrame | None:
    if cross is None:
        return None
    column = f"{product} CPE"
    if column not in cross.columns or "name" not in cross.columns:
        return None
    result = cross[["name", column]].rename(columns={column: "cross_price_elasticity"})
    result["cross_price_elasticity"] = pd.to_numeric(result["cross_price_elasticity"], errors="coerce")
    return result.dropna().sort_values("cross_price_elasticity", ascending=False)


def elasticity_label(e: float) -> str:
    if e < -1:
        return "Elastic — discount increases revenue"
    if e < 0:
        return "Inelastic — discount reduces revenue"
    if e >= 0:
        return "Positive — unusual (Giffen / prestige)"
    return "—"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Price Elasticity")
    st.caption("E-commerce pricing intelligence")
    st.markdown("---")
    st.markdown(
        "**How it works:**\n\n"
        "For each product, we fit a log-log OLS regression:\n\n"
        "```\nln(demand) = α + β · ln(price)\n```\n\n"
        "The slope **β** is the price elasticity."
    )
    st.markdown("---")
    st.markdown("**Elasticity guide:**")
    st.markdown("- **ε < −1:** elastic → discount ↑ revenue\n- **−1 < ε < 0:** inelastic → discount ↓ revenue\n- **ε = −1:** unit elastic → revenue unchanged")
    st.markdown("---")
    st.caption("Built following *Hands-On ML with Scikit-Learn and PyTorch* — Géron (2025)")


# ── Main layout ───────────────────────────────────────────────────────────────

st.title("Price Elasticity Dashboard")
st.caption("Estimate price sensitivity and simulate revenue impact of pricing decisions")

with st.spinner("Loading data…"):
    df = load_report()

if df.empty:
    st.warning("No products met the minimum observation threshold. Check your data and config.")
    st.stop()

df = df.sort_values("price_elasticity")

with st.spinner("Loading raw data…"):
    raw = load_raw_data()
    cross = load_cross_price()

# Controls row
col_product, col_r2 = st.columns([3, 1])
with col_product:
    product = st.selectbox("Select a product", df["product"].tolist())
with col_r2:
    min_r2 = st.slider("Min R² for ranking/simulation", 0.0, 1.0, 0.10, 0.05)

row = df.loc[df["product"] == product].iloc[0]
history = product_history(raw, product)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Elasticity", "Simulation", "Ranking", "Cross-Elasticity"])

# ── Tab 0: Elasticity ─────────────────────────────────────────────────────────
with tabs[0]:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Elasticity (ε)", f"{row['price_elasticity']:.2f}")
    m2.metric("R²", f"{row['r2']:.2f}")
    m3.metric("Avg Price", f"${row['avg_price']:,.2f}")
    m4.metric("Avg Demand", f"{row['avg_demand']:,.1f}")
    m5.metric("Observations", f"{int(row['observations']):,}")

    # Show CI if available
    ci_lo_cols = [c for c in df.columns if "ci_" in c and "lower" in c]
    ci_hi_cols = [c for c in df.columns if "ci_" in c and "upper" in c]
    if ci_lo_cols and ci_hi_cols:
        lo, hi = float(row[ci_lo_cols[0]]), float(row[ci_hi_cols[0]])
        st.info(
            f"**95% Confidence Interval:** [{lo:.3f}, {hi:.3f}]  |  "
            f"**p-value:** {row.get('p_value', float('nan')):.4f}  |  "
            f"**Interpretation:** {elasticity_label(row['price_elasticity'])}"
        )

    if "p_value" in row and not pd.isna(row["p_value"]):
        sig = "✅ Statistically significant (p < 0.05)" if row["p_value"] < 0.05 else "⚠️ Not significant (p ≥ 0.05)"
        st.caption(sig)

    if history.empty:
        st.info("Product history not found in the analytical dataset.")
    else:
        st.plotly_chart(regression_figure(history, row), use_container_width=True)
        with st.expander("Raw data for this product"):
            st.dataframe(history, use_container_width=True)

# ── Tab 1: Simulation ─────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Revenue Simulator")
    change = st.slider("Price change (%)", min_value=-60, max_value=60, value=-10, step=1)
    price_change_pct = change / 100
    result = simulate_row(row, price_change_pct)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${result['current_price']:,.2f}")
    c2.metric("New Price", f"${result['new_price']:,.2f}", f"{change:+.0f}%")
    c3.metric("Projected Demand", f"{result['new_demand']:,.1f}", f"{result['demand_change_pct']:.1%}")
    c4.metric("Projected Revenue", f"${result['new_revenue']:,.2f}", f"{result['revenue_delta_pct']:.1%}")

    st.markdown("---")
    st.subheader("Portfolio Impact")
    filtered_df = df.loc[df["r2"] >= min_r2]
    simulation = simulate_portfolio(filtered_df, price_change_pct)
    total_current = simulation["current_revenue"].sum()
    total_new = simulation["new_revenue"].sum()
    total_delta = total_new - total_current

    st.markdown(
        f"With a **{change:+.0f}%** price change across **{len(simulation)}** products (R² ≥ {min_r2}), "
        f"projected revenue changes from **${total_current:,.0f}** to **${total_new:,.0f}** "
        f"(**{total_delta:+,.0f}**, {(total_new/total_current - 1):.1%})."
    )

    top = simulation.head(20)
    fig = px.bar(
        top.sort_values("revenue_delta"),
        x="revenue_delta",
        y="product",
        orientation="h",
        color="revenue_delta",
        color_continuous_scale=["coral", "lightgray", "steelblue"],
        title=f"Top 20 Products by Revenue Impact at {change:+.0f}% Price Change",
        labels={"revenue_delta": "Revenue Delta ($)", "product": "Product"},
    )
    fig.update_layout(height=620, margin=dict(l=20, r=20, t=60, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full simulation table"):
        st.dataframe(simulation, use_container_width=True)

# ── Tab 2: Ranking ────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader(f"Elasticity Ranking (R² ≥ {min_r2})")
    filtered = df.loc[df["r2"] >= min_r2].copy()
    filtered["elasticity_abs"] = filtered["price_elasticity"].abs()
    filtered["type"] = filtered["price_elasticity"].apply(
        lambda e: "Elastic (ε < -1)" if e < -1 else ("Inelastic" if e < 0 else "Positive")
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Products (filtered)", len(filtered))
    m2.metric("Elastic (ε < -1)", (filtered["price_elasticity"] < -1).sum())
    m3.metric("Statistically significant", (filtered.get("p_value", pd.Series(dtype=float)) < 0.05).sum() if "p_value" in filtered.columns else "N/A")

    fig = px.bar(
        filtered.sort_values("price_elasticity").head(30),
        x="price_elasticity",
        y="product",
        orientation="h",
        color="price_elasticity",
        color_continuous_scale=["steelblue", "lightgray", "coral"],
        title="Most Price-Sensitive Products",
        labels={"price_elasticity": "Price Elasticity", "product": "Product"},
    )
    fig.add_vline(x=-1, line_dash="dash", line_color="red", annotation_text="ε = -1")
    fig.update_layout(height=760, margin=dict(l=20, r=20, t=60, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full ranked table"):
        st.dataframe(
            filtered.sort_values("elasticity_abs", ascending=False).drop(columns=["elasticity_abs"]),
            use_container_width=True,
        )

# ── Tab 3: Cross-Elasticity ───────────────────────────────────────────────────
with tabs[3]:
    st.subheader(f"Cross-Price Elasticity for: {product[:60]}")
    cross_table = cross_price_table(cross, product)
    if cross_table is None:
        st.info(
            "Cross-price elasticity matrix not available for this product.  \n"
            "Place `data/raw/df_crossprice_previous.csv` to enable this feature."
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Potential substitutes** (positive CPE — competing products)")
            st.dataframe(cross_table.head(15), use_container_width=True)
        with c2:
            st.markdown("**Potential complements** (negative CPE — bundle candidates)")
            st.dataframe(cross_table.tail(15).sort_values("cross_price_elasticity"), use_container_width=True)
