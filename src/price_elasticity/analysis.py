"""Project-specific analytical routines for non-standard ML tasks."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .config import load_config, resolve_project_path
from .data import load_training_frame, normalize_columns


def _normal_two_sided_pvalue(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def analyze_ab_test(config: dict) -> dict:
    df = normalize_columns(load_training_frame(config))
    data = config.get("data", {})
    group_col = data.get("group_column", "group")
    metric_col = data.get("metric_column", "converted")
    control = data.get("control_value", "control")
    treatment = data.get("treatment_value", "treatment")

    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    if metric_col not in df.columns:
        if "purchases" in df.columns:
            metric_col = "purchases"
        else:
            raise ValueError(f"Missing metric column: {metric_col}")

    summary = df.groupby(group_col)[metric_col].agg(["count", "mean", "std", "sum"]).reset_index()
    rates = dict(zip(summary[group_col], summary["mean"]))
    counts = dict(zip(summary[group_col], summary["count"]))

    if control not in rates or treatment not in rates:
        groups = list(rates)
        control, treatment = groups[0], groups[-1]

    p1, p2 = float(rates[control]), float(rates[treatment])
    n1, n2 = int(counts[control]), int(counts[treatment])
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2)) if n1 and n2 else float("nan")
    z = (p2 - p1) / se if se else float("nan")
    result = {
        "control": control,
        "treatment": treatment,
        "control_rate": p1,
        "treatment_rate": p2,
        "absolute_lift": p2 - p1,
        "relative_lift": (p2 - p1) / p1 if p1 else None,
        "z_stat": z,
        "p_value": _normal_two_sided_pvalue(z) if not math.isnan(z) else None,
        "summary": summary.to_dict(orient="records"),
    }
    output = resolve_project_path(config, "reports/ab_test_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"results_path": str(output), "results": result}


def _ols_elasticity(x, y):
    """Fit log-log OLS and return slope, intercept, R², SE, t-stat, p-value.

    Uses closed-form OLS: beta = Cov(x,y)/Var(x), with t-distribution for inference.
    This mirrors the approach in Géron Ch.4 (Training Models — Normal Equation / OLS).
    """
    import math as _math
    import numpy as np
    from scipy import stats

    n = len(x)
    x_mean, y_mean = x.mean(), y.mean()
    ss_xx = float(((x - x_mean) ** 2).sum())
    ss_xy = float(((x - x_mean) * (y - y_mean)).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())

    beta = ss_xy / ss_xx if ss_xx else 0.0
    alpha = y_mean - beta * x_mean
    y_hat = alpha + beta * x
    ss_res = float(((y - y_hat) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0

    # Standard error of slope
    mse = ss_res / max(n - 2, 1)
    se_beta = _math.sqrt(mse / ss_xx) if ss_xx else float("nan")
    t_stat = beta / se_beta if se_beta and not _math.isnan(se_beta) else float("nan")
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 2)) if not _math.isnan(t_stat) else float("nan")

    return {
        "beta": float(beta),
        "alpha": float(alpha),
        "r2": float(r2),
        "se_beta": float(se_beta),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n": n,
        "ss_xx": float(ss_xx),
        "mse": float(mse),
    }


def _confidence_interval(ols: dict, confidence_level: float = 0.95):
    """Compute t-distribution CI for the slope coefficient."""
    import math as _math
    from scipy import stats

    alpha = 1.0 - confidence_level
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=max(ols["n"] - 2, 1)))
    margin = t_crit * ols["se_beta"]
    return float(ols["beta"] - margin), float(ols["beta"] + margin)


def _bootstrap_elasticity(x, y, n_iter: int = 500, confidence_level: float = 0.95, rng_seed: int = 42):
    """Bootstrap CI for slope — provides distribution-free uncertainty estimate."""
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    slopes = []
    n = len(x)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        if xi.std() < 1e-10:
            continue
        b = float(np.cov(xi, yi)[0, 1] / np.var(xi))
        slopes.append(b)
    if not slopes:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence_level
    lo = float(np.percentile(slopes, 100 * alpha / 2))
    hi = float(np.percentile(slopes, 100 * (1 - alpha / 2)))
    return lo, hi


def analyze_price_elasticity(config: dict) -> dict:
    """Estimate per-product price elasticity using log-log OLS with CI and bootstrap.

    Follows the end-to-end ML pipeline from Géron Ch.2:
    - Frame the problem (price sensitivity estimation)
    - Get and prepare the data (log transformation)
    - Fit model (OLS log-log per product)
    - Evaluate (R², p-value, CI width)
    - Present results (sorted CSV with uncertainty bands)
    """
    import numpy as np
    import pandas as pd

    df = normalize_columns(load_training_frame(config))
    data = config.get("data", {})
    modeling = config.get("modeling", {})

    product_col = data.get("product_column", "name")
    price_col = data.get("price_column", "disc_price")
    demand_col = data.get("demand_column", "imp_count")
    min_observations = int(modeling.get("min_observations", 12))
    min_price_points = int(modeling.get("min_price_points", 3))
    confidence_level = float(modeling.get("confidence_level", 0.95))
    bootstrap_iter = int(modeling.get("bootstrap_iterations", 500))
    rng_seed = int(modeling.get("random_state", 42))

    for col in [price_col, demand_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df[price_col] > 0) & (df[demand_col] > 0)].dropna(
        subset=[product_col, price_col, demand_col]
    )

    rows = []
    for product, group in df.groupby(product_col):
        if len(group) < min_observations:
            continue
        if group[price_col].nunique() < min_price_points:
            continue

        x = np.log(group[price_col].to_numpy(dtype=float))
        y = np.log(group[demand_col].to_numpy(dtype=float))

        ols = _ols_elasticity(x, y)
        ci_lo, ci_hi = _confidence_interval(ols, confidence_level)
        bs_lo, bs_hi = _bootstrap_elasticity(x, y, bootstrap_iter, confidence_level, rng_seed)

        rows.append(
            {
                "product": product,
                "observations": ols["n"],
                "price_elasticity": ols["beta"],
                "r2": ols["r2"],
                "se": ols["se_beta"],
                "t_stat": ols["t_stat"],
                "p_value": ols["p_value"],
                f"ci_{int(confidence_level*100)}_lower": ci_lo,
                f"ci_{int(confidence_level*100)}_upper": ci_hi,
                "bootstrap_lower": bs_lo,
                "bootstrap_upper": bs_hi,
                "avg_price": float(group[price_col].mean()),
                "avg_demand": float(group[demand_col].mean()),
                "min_price": float(group[price_col].min()),
                "max_price": float(group[price_col].max()),
            }
        )

    result = pd.DataFrame(rows).sort_values("price_elasticity")
    output = resolve_project_path(config, "reports/price_elasticity.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    summary = {
        "n_products_analyzed": int(len(result)),
        "n_elastic": int((result["price_elasticity"] < -1).sum()),
        "n_inelastic": int(((result["price_elasticity"] >= -1) & (result["price_elasticity"] < 0)).sum()),
        "median_elasticity": float(result["price_elasticity"].median()),
        "median_r2": float(result["r2"].median()),
        "high_quality_r2_gt_0.10": int((result["r2"] >= 0.10).sum()),
        "significant_p_lt_0.05": int((result["p_value"] < 0.05).sum()),
    }
    summary_path = resolve_project_path(config, "reports/elasticity_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "results_path": str(output),
        "n_products": int(len(result)),
        "summary": summary,
    }


def run_analysis(config_path: str | Path | None = None):
    config = load_config(config_path)
    problem_type = config.get("project", {}).get("problem_type")
    if problem_type == "ab_testing":
        return analyze_ab_test(config)
    if problem_type == "price_elasticity":
        return analyze_price_elasticity(config)
    raise ValueError(f"No special analysis routine for problem_type={problem_type!r}. Use train instead.")
