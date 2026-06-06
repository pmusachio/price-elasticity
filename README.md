# Price Elasticity — End-to-End Data Science Project

> **Estimating price sensitivity with log-log regression and simulating discount impact on revenue.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)](https://streamlit.io/)

---

## 1. Business Problem

An e-commerce operation needs to understand how price changes affect demand and revenue at the product level. A single blanket discount rule destroys margin for inelastic products while under-investing in elastic ones.

**Goal:** Estimate the price-demand sensitivity coefficient for each product and simulate the revenue impact of any price change scenario.

**Primary metric:** Log-log elasticity coefficient (β), coefficient of determination (R²), and confidence interval width per product.

---

## 2. Business Assumptions

- Elasticity varies per product; a single discount rule can destroy margin.
- Products with fewer than 12 price–demand observations are excluded from conclusions.
- The simulation combines demand response and average price to project revenue delta.
- Products with R² < 0.10 are flagged as statistically unreliable.

---

## 3. Solution Strategy

Following the end-to-end ML project workflow from *Hands-On Machine Learning with Scikit-Learn and PyTorch* (Aurélien Géron, 2025):

| Step | Description |
|------|-------------|
| 01 | **Data Understanding** — schema validation, missing values, data types, granularity |
| 02 | **Exploratory Analysis** — price & demand distributions, temporal trends, discount patterns |
| 03 | **Feature Engineering** — log transformation, discount rate, weekly aggregation |
| 04 | **Modeling** — per-product OLS log-log regression with confidence intervals and bootstrap |
| 05 | **Business Translation** — elasticity ranking, revenue simulation, cross-elasticity matrix |
| 06 | **Deployment** — Streamlit dashboard with interactive price simulator |

---

## 4. Data Source

Dataset: [Ecommerce Behavior Data from Multi Category Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) — Kaggle.

Expected files in `data/raw/`:

| File | Description |
|------|-------------|
| `df_ready.csv` | Main analytical table: one row per product-date observation with price, discounted price, and demand count |
| `df_crossprice_previous.csv` | *(optional)* Pre-computed cross-price elasticity matrix |
| `df_elasticity_previous.csv` | *(optional)* Reference elasticity results for comparison |

> The project expects `df_ready.csv` already at product-date level. If you start from raw Kaggle event logs, aggregate them per the schema described in `data/raw/README.md`.

---

## 5. Notebooks — Analysis Journey

The notebooks are the best entry point to understand the full project narrative:

| Notebook | Description |
|----------|-------------|
| `00_business_understanding.ipynb` | Problem framing, success criteria, analytical plan |
| `01_data_understanding.ipynb` | Data profiling, schema, missing values, sample inspection |
| `02_exploratory_analysis.ipynb` | Price/demand distributions, temporal trends, hypotheses |
| `03_feature_engineering.ipynb` | Log transformation, discount rate, sklearn Pipeline |
| `04_modeling_and_business_results.ipynb` | OLS log-log per product, CI, R², business translation |
| `05_deployment_and_consumption.ipynb` | Streamlit overview, pipeline execution, output artifacts |

---

## 6. Key Insights

- **Most products are price-inelastic** (|ε| < 1): demand does not respond proportionally to price changes.
- **High-discount products** provide more price variation, yielding more stable estimates.
- **Demand gains from discounts rarely offset revenue losses** for inelastic products — R²-filtered ranking is essential before acting.
- **Cross-elasticity** reveals substitutes (positive CPE) and complements (negative CPE) enabling bundle and assortment decisions.

---

## 7. Model

**Log-log OLS regression per product:**

```
ln(demand) = α + β · ln(price) + ε
```

- **β (elasticity coefficient):** 1% price increase → β% demand change.
- **β < −1:** elastic (discount profitable for revenue).
- **−1 < β < 0:** inelastic (discount hurts revenue).
- **Confidence intervals** via OLS standard errors (scipy.stats) and bootstrap resampling.

---

## 8. Main Outputs

| Artifact | Description |
|----------|-------------|
| `reports/price_elasticity.csv` | Per-product: elasticity, R², CI lower/upper, avg price, avg demand |
| `app/streamlit_app.py` | Interactive dashboard: product explorer, price simulator, ranking, cross-elasticity |

---

## 9. How to Run

### Option A — Local (recommended for Streamlit)

```bash
# 1. Clone the repository
git clone <REPO_URL> price-elasticity
cd price-elasticity

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt -r requirements-app.txt

# 4. Run the analysis pipeline (generates reports/price_elasticity.csv)
PYTHONPATH=src python -m price_elasticity.cli analyze

# 5. Launch the Streamlit dashboard
PYTHONPATH=src streamlit run app/streamlit_app.py
```

The dashboard will open at **http://localhost:8501**.

---

### Option B — Google Colab (analysis only)

```python
# Step 1 — Clone and install
REPO_URL = "https://github.com/<your-username>/price-elasticity.git"
!git clone {REPO_URL} project
%cd project
!pip install -q -r requirements.txt
```

```python
# Step 2 — Upload data (if not versioned)
from google.colab import files
files.upload()  # upload df_ready.csv
!mv df_ready.csv data/raw/df_ready.csv
```

```python
# Step 3 — Run the pipeline
!PYTHONPATH=src python -m price_elasticity.cli profile
!PYTHONPATH=src python -m price_elasticity.cli analyze
```

```python
# Step 4 — Inspect results
import pandas as pd
pd.read_csv("reports/price_elasticity.csv").sort_values("price_elasticity").head(20)
```

---

### Option C — Jupyter Notebooks (step-by-step walkthrough)

```bash
pip install -r requirements.txt jupyter
cd notebooks
jupyter notebook
```

Open notebooks in order `00` → `05` to follow the full analysis narrative.

---

## 10. Repository Structure

```
price-elasticity/
├── app/
│   └── streamlit_app.py          # Interactive dashboard
├── configs/
│   └── project.toml              # Project contract: data, columns, parameters
├── data/
│   └── raw/                      # Source data (not versioned)
├── docs/
│   └── deployment.md             # Deployment notes
├── notebooks/                    # Analysis journey (00–05)
├── reports/
│   └── price_elasticity.csv      # Main output: elasticity per product
├── src/
│   └── price_elasticity/
│       ├── analysis.py           # Log-log OLS + CI + bootstrap
│       ├── config.py             # Config loader
│       ├── data.py               # Data loading utilities
│       ├── features.py           # Feature engineering
│       ├── models.py             # sklearn Pipeline for supervised tasks
│       └── cli.py                # CLI entry point
├── tests/
│   └── test_project_contract.py
├── requirements.txt
├── requirements-app.txt
└── pyproject.toml
```

---

## 11. Next Steps

- Add gross margin per product to simulate **profit** (not just revenue).
- Implement **category-level elasticity** for products with insufficient individual observations.
- Publish the Streamlit dashboard to **Streamlit Cloud** for public access.
- Add **time-series decomposition** to separate seasonal demand from price effects.

---

## 12. Tests

```bash
python -m pytest
```

---

## References

- Géron, Aurélien. *Hands-On Machine Learning with Scikit-Learn and PyTorch*, O'Reilly, 2025 — Chapter 2 (End-to-End ML Projects), Chapter 4 (Training Models).
- Varian, Hal R. *Intermediate Microeconomics*, 9th ed. — Price Elasticity of Demand.
