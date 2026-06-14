"""Interactive price-elasticity dashboard.

Shows the estimated price elasticity for a product and simulates the demand and
revenue impact of a price change, on the subset of products where elasticity is
economically identifiable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402

D = config.DRACULA
st.set_page_config(page_title="Price Elasticity", layout="wide")
st.markdown(
    f"""<style>
    .stApp {{ background-color: {D['background']}; color: {D['foreground']}; }}
    section[data-testid="stSidebar"] {{ background-color: {D['current_line']}; }}
    h1, h2, h3 {{ color: {D['purple']}; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


def style_axes(ax):
    ax.set_facecolor(D["background"])
    for s in ax.spines.values():
        s.set_color(D["current_line"])
    ax.tick_params(colors=D["foreground"])
    ax.xaxis.label.set_color(D["foreground"])
    ax.yaxis.label.set_color(D["foreground"])
    ax.grid(True, color=D["current_line"], linestyle="--", alpha=0.4)


def revenue_curve(predictor, product, chosen):
    changes = np.linspace(-0.5, 0.2, 60)
    rev = [predictor.simulate(product, c)["revenue_change_pct"] for c in changes]
    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor=D["background"])
    ax.plot(changes * 100, rev, color=D["green"], linewidth=2)
    ax.axhline(0, color=D["comment"], linestyle="--", linewidth=1)
    ax.axvline(chosen * 100, color=D["pink"], linestyle=":", linewidth=1.5)
    ax.set_xlabel("Price change (%)")
    ax.set_ylabel("Revenue change (%)")
    style_axes(ax)
    fig.tight_layout()
    return fig


def main():
    try:
        predictor = load_predictor()
    except FileNotFoundError:
        st.error("Model artifact not found. Run the pipeline before launching the app.")
        return

    s = predictor.summary
    st.title("Price Elasticity — Discount Revenue Simulator")
    st.markdown(
        "Estimates how demand responds to price per product and simulates the revenue impact of a "
        "discount, for the products where elasticity is statistically identifiable."
    )

    products = predictor.products()
    with st.sidebar:
        st.header("Product")
        product = st.selectbox("Choose a product", products)
        price_change = st.slider("Price change (%)", -50, 20, -10, 5) / 100
        st.caption(f"{s['reliable_products']} of {s['total_products']} products have an "
                   f"identifiable (negative, well-fit) elasticity.")

    e = predictor.elasticity(product)
    res = predictor.simulate(product, price_change)

    st.subheader("Elasticity")
    c = st.columns(4)
    c[0].metric("Elasticity (beta)", f"{e['elasticity']:.2f}")
    c[1].metric("Classification", res["classification"].capitalize())
    c[2].metric("Fit R2", f"{e['r2']:.2f}")
    c[3].metric("Avg price", f"${e['avg_price']:.0f}")
    st.markdown(
        f"95% CI for elasticity: [{e['ci95'][0]:.2f}, {e['ci95'][1]:.2f}]. "
        f"A 1% price rise moves demand about {e['elasticity']:.1f}%."
    )

    st.subheader(f"Simulated {abs(price_change)*100:.0f}% {'discount' if price_change < 0 else 'increase'}")
    m = st.columns(3)
    m[0].metric("New price", f"${res['new_price']:.0f}")
    m[1].metric("Demand change", f"{res['demand_change_pct']:+.0f}%")
    m[2].metric("Revenue change", f"{res['revenue_change_pct']:+.0f}%")
    st.pyplot(revenue_curve(predictor, product, price_change))

    st.caption(
        "Note: demand is proxied by impression counts and prices are observational, not "
        "randomized. Only the negative, well-fit subset is shown; positive or extreme naive "
        "estimates are treated as confounded and excluded."
    )


if __name__ == "__main__":
    main()
