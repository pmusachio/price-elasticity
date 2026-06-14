"""Transformation layer: keep clean, positive price/demand observations and retain
only products with enough price variation to estimate an elasticity.
"""
from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self, processed_path=config.PROCESSED_PATH) -> None:
        self.processed_path = processed_path

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        d = df.copy()
        d[config.PRICE_COL] = pd.to_numeric(d[config.PRICE_COL], errors="coerce")
        d[config.DEMAND_COL] = pd.to_numeric(d[config.DEMAND_COL], errors="coerce")
        d = d.dropna(subset=[config.PRODUCT_COL, config.PRICE_COL, config.DEMAND_COL])
        d = d[(d[config.PRICE_COL] > 0) & (d[config.DEMAND_COL] > 0)]

        g = d.groupby(config.PRODUCT_COL)
        qualifying = g.filter(
            lambda x: len(x) >= config.MIN_OBSERVATIONS
            and x[config.PRICE_COL].nunique() >= config.MIN_PRICE_POINTS)
        logger.info("Qualifying: %d products, %d observations",
                    qualifying[config.PRODUCT_COL].nunique(), len(qualifying))
        return qualifying.reset_index(drop=True), d
