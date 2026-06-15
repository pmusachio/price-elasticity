# Price Elasticity — Discount Revenue Simulator

> Econometrics · Log-log demand regression · Per-product elasticity · Revenue simulation

## Business Problem

An electronics retailer wants to know which products to discount. The decision the analysis
informs is **where a price cut pays for itself**: discounting an elastic product wins enough extra
volume to grow revenue, while discounting an inelastic product just gives away margin.

The quantity that drives this decision is the **price elasticity of demand** per product — the
percent change in demand per percent change in price. The cost of error is direct: discount the
wrong products and revenue falls; raise prices on elastic ones and volume collapses. Because the
data is observational (prices were not randomized) and demand is proxied by impressions, the harder
problem is not fitting a line but knowing **which estimates are trustworthy**.

A single blanket discount was rejected: elasticity varies enormously across products, so the
profitable policy is product-specific.

## Dataset

A prepared electronics-pricing panel (product, category, price, impression-based demand), derived
from a public retail catalogue and versioned with the project.

| Property | Value |
|----------|-------|
| Observations | 23,151 price-demand records |
| Qualifying products | 647 (>=12 observations and >=3 distinct prices) |
| Price field | discounted price |
| Demand proxy | impression count |

## Solution Strategy

1. **Acquisition** — the prepared panel is versioned and loaded directly.
2. **Qualification** — keep only products with enough observations and price variation to fit a slope.
3. **Estimation** — fit `log(demand) ~ log(price)` per product; the slope is the elasticity, with a 95% confidence interval and R-squared.
4. **Identifiability filter** — flag an estimate reliable only if it is negative (economically sensible) and not an extreme outlier from thin price variation; the rest are treated as confounded.
5. **Simulation** — translate elasticity into the demand and revenue impact of any price change.

## Top Insights & Hypotheses

- **Most naive estimates are not economically identifiable.** Only **31% of products show the expected negative price-demand relationship**; the rest come out positive, a signature of confounding (impressions track promotion and demand, not a controlled price experiment).
- **Among the 108 identifiable products, demand is highly elastic** (median elasticity **-2.0**; 82% are elastic), so targeted discounts can grow revenue there.
- **Premium and unique items are inelastic** — an iPhone SE estimates near -0.15, so discounting it mostly sacrifices margin.
- **Commodity storage and accessories are the most elastic** (some below -8), the best discount candidates.

## Model

A per-product log-log ordinary-least-squares regression, with an identifiability filter that
separates trustworthy elasticities from confounded ones.

| Quantity | Value |
|----------|------:|
| Qualifying products | 647 |
| Well-fit (R-squared >= 0.05) | 459 |
| Identifiable (negative, bounded) | 108 |
| Products with negative sign | 31% |
| Median reliable elasticity | -2.03 |

## Business Results

For the identifiable subset, the simulator converts elasticity into a revenue decision. For an
elastic product (elasticity -2), a 10% discount lifts demand about 20% and raises revenue ~8%; for
an inelastic one the same discount loses money. The dashboard lets a category manager pick a
product, choose a price change, and read the demand and revenue impact and the break-even point off
the curve.

The headline action: **concentrate discounts on the elastic, well-identified products and hold or
raise prices on inelastic ones**, rather than running a uniform promotion.

## How to Run

1. **Clone**
   ```
   git clone https://github.com/pmusachio/price-elasticity.git
   cd price-elasticity
   ```
2. **Environment**
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Data** — the prepared panel is versioned under `data/sample/`.
4. **Run the pipeline**
   ```
   python -m src.pipeline
   ```
5. **Tests**
   ```
   pytest tests/
   ```
6. **App (local)**
   ```
   streamlit run app/streamlit_app.py
   ```
7. **Live app** — [price-elasticity-bqul.onrender.com](https://price-elasticity-bqul.onrender.com) — pick a product and simulate a discount.

## Next Steps

- The core limitation is identification: with observational prices and an impression-based demand
  proxy, two-thirds of products yield uninterpretable signs. A proper price experiment, or an
  instrument / control for promotion and seasonality, would make far more products usable.
- Replace per-product OLS with a hierarchical (partial-pooling) model so products with thin data
  borrow strength from their category instead of producing extreme slopes; deferred until the
  identification issue above is addressed, since it is the binding constraint.
- Add cross-price elasticities to capture cannibalization between substitute products.
