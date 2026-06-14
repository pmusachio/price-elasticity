"""Smoke tests for the qualification contract and the elasticity serving surface."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402
from src.preprocessing import Preprocessor  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(config.SAMPLE_PATH)


def test_preprocessing_keeps_only_qualifying_products(sample):
    qualifying, _ = Preprocessor().run(sample)
    counts = qualifying.groupby(config.PRODUCT_COL)[config.PRICE_COL]
    assert (counts.count() >= config.MIN_OBSERVATIONS).all()
    assert (counts.nunique() >= config.MIN_PRICE_POINTS).all()


def test_reliable_products_have_negative_bounded_elasticity():
    pred = Predictor()
    for p in pred.products():
        e = pred.elasticity(p)
        assert config.RELIABLE_BOUNDS[0] <= e["elasticity"] <= config.RELIABLE_BOUNDS[1]
        assert e["r2"] >= config.MIN_R2


def test_simulate_discount_increases_demand_for_elastic_product():
    pred = Predictor()
    p = pred.products()[0]
    res = pred.simulate(p, -0.10)  # 10% discount
    assert res["demand_change_pct"] > 0  # negative elasticity => discount lifts demand
    assert "revenue_change_pct" in res


def test_summary_flags_confounding():
    pred = Predictor()
    assert pred.summary["reliable_products"] >= 1
    assert pred.summary["median_reliable_elasticity"] < 0
