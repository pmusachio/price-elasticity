# Notebooks — Analysis Journey

Each notebook covers one phase of the end-to-end ML project workflow, following *Hands-On Machine Learning with Scikit-Learn and PyTorch* (Géron, 2025).

| Notebook | Phase | Key content |
|----------|-------|-------------|
| `00_business_understanding.ipynb` | Problem framing | Business problem, ML task definition, success criteria, hypotheses |
| `01_data_understanding.ipynb` | Get the data | Schema validation, missing values, product coverage, eligible products |
| `02_exploratory_analysis.ipynb` | Explore & visualise | Price/demand scatter, log-log linearity, discount distribution, temporal trends |
| `03_feature_engineering.ipynb` | Prepare data | Log transformation, discount rate, sklearn Pipeline demonstration |
| `04_modeling_and_business_results.ipynb` | Train & evaluate | OLS log-log per product, R², confidence intervals, revenue simulation |
| `05_deployment_and_consumption.ipynb` | Deploy | Pipeline artifacts, Streamlit overview, one-click run verification |

## Running the Notebooks

```bash
# From the project root
pip install -r requirements.txt jupyter
jupyter notebook notebooks/
```

Open notebooks in order `00` → `05` to follow the full narrative.

## Project root resolution

All notebooks auto-detect the project root:

```python
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
```

This means they work whether you launch Jupyter from `notebooks/` or from the project root.
