"""Serving layer: load the elasticity artifact and simulate the revenue impact of a
price change for a product.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from src import config

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, artifact_path: Path = config.PIPELINE_PATH) -> None:
        import joblib

        if not Path(artifact_path).exists():
            raise FileNotFoundError(f"No artifact at {artifact_path}. Run `python -m src.pipeline` first.")
        art = joblib.load(artifact_path)
        self.elasticities: Dict[str, Any] = art["elasticities"]
        self.summary: Dict[str, Any] = art["summary"]

    def products(self, reliable_only: bool = True) -> List[str]:
        return sorted(k for k, v in self.elasticities.items() if v.get("reliable") or not reliable_only)

    def elasticity(self, product: str) -> Dict[str, Any]:
        return self.elasticities[product]

    def simulate(self, product: str, price_change_pct: float) -> Dict[str, Any]:
        """Revenue impact of an absolute price change (e.g. -0.10 for a 10% discount)."""
        e = self.elasticities[product]
        beta = e["elasticity"]
        price_ratio = 1 + price_change_pct
        if price_ratio <= 0:
            price_ratio = 1e-6
        demand_ratio = price_ratio ** beta
        revenue_ratio = price_ratio * demand_ratio
        return {
            "elasticity": beta,
            "classification": "elastic" if e["elastic"] else "inelastic",
            "new_price": round(e["avg_price"] * price_ratio, 2),
            "demand_change_pct": round((demand_ratio - 1) * 100, 1),
            "revenue_change_pct": round((revenue_ratio - 1) * 100, 1),
        }
