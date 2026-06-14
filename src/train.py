"""Estimation layer: fit a log-log demand model per product to recover its price
elasticity with a confidence interval and fit quality, summarize the catalogue, and
serialize a self-contained artifact plus a model card.

elasticity beta is the slope of log(demand) ~ log(price): a 1% price rise changes
demand by beta percent. |beta| > 1 is elastic, |beta| < 1 inelastic.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from src import config

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "1.0"


class ElasticityEstimator:
    def __init__(self, qualifying: pd.DataFrame, data_source: Path | None = None) -> None:
        self.df = qualifying
        self.data_source = data_source
        self.elasticities: Dict[str, Any] = {}
        self.summary: Dict[str, Any] = {}

    def fit(self) -> Dict[str, Any]:
        t_crit = stats.t.ppf(1 - (1 - config.CONFIDENCE) / 2, df=10_000)
        for product, grp in self.df.groupby(config.PRODUCT_COL):
            x = np.log(grp[config.PRICE_COL].to_numpy())
            y = np.log(grp[config.DEMAND_COL].to_numpy())
            if np.ptp(x) == 0:
                continue
            reg = stats.linregress(x, y)
            beta = float(reg.slope)
            se = float(reg.stderr)
            r2 = float(reg.rvalue ** 2)
            self.elasticities[str(product)] = {
                "elasticity": round(beta, 4),
                "ci95": [round(beta - t_crit * se, 4), round(beta + t_crit * se, 4)],
                "r2": round(r2, 4),
                "n": int(len(grp)),
                "avg_price": round(float(grp[config.PRICE_COL].mean()), 2),
                "avg_demand": round(float(grp[config.DEMAND_COL].mean()), 2),
                "category": str(grp[config.CATEGORY_COL].mode().iloc[0]) if config.CATEGORY_COL in grp else "",
                "well_fit": bool(r2 >= config.MIN_R2),
                "elastic": bool(abs(beta) > config.ELASTIC_THRESHOLD),
                "reliable": bool(r2 >= config.MIN_R2
                                 and config.RELIABLE_BOUNDS[0] <= beta <= config.RELIABLE_BOUNDS[1]),
            }
        self._summarize()
        logger.info("Estimated %d products: %d well-fit, %d reliable (negative, bounded)",
                    len(self.elasticities), self.summary["well_fit_products"], self.summary["reliable_products"])
        return self.elasticities

    def _summarize(self) -> None:
        all_e = list(self.elasticities.values())
        reliable = [e for e in all_e if e["reliable"]]
        betas = [e["elasticity"] for e in reliable]
        self.summary = {
            "total_products": len(all_e),
            "well_fit_products": sum(e["well_fit"] for e in all_e),
            "reliable_products": len(reliable),
            "pct_negative_sign": round(100 * np.mean([e["elasticity"] < 0 for e in all_e]), 1) if all_e else 0.0,
            "median_reliable_elasticity": round(float(np.median(betas)), 4) if betas else None,
            "pct_elastic_among_reliable": round(100 * np.mean([e["elastic"] for e in reliable]), 1) if reliable else 0.0,
        }

    def save(self) -> None:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"schema_version": SCHEMA_VERSION, "elasticities": self.elasticities,
                     "summary": self.summary}, config.PIPELINE_PATH)
        logger.info("Artifact written to %s", config.PIPELINE_PATH)
        # examples from the economically identifiable (reliable) set
        rel = {k: v for k, v in self.elasticities.items() if v["reliable"]}
        by_beta = sorted(rel.items(), key=lambda kv: kv[1]["elasticity"])
        card = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": config.KAGGLE_DATASET, "data_sha256": self._hash(),
            "problem": "per-product price elasticity (log-log demand regression)",
            "summary": self.summary,
            "most_elastic": [{"product": k, **v} for k, v in by_beta[:5]],
            "least_elastic": [{"product": k, **v} for k, v in by_beta[-5:]],
        }
        config.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2))
        logger.info("Model card written to %s", config.MODEL_CARD_PATH)

    def _hash(self) -> str:
        src = self.data_source or config.SAMPLE_PATH
        return hashlib.sha256(Path(src).read_bytes()).hexdigest() if src and Path(src).exists() else "unknown"
