"""Central configuration: paths, dataset identity, estimation constants and the
Dracula palette shared by the pipeline, the serving layer and the dashboard.
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLE_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = BASE_DIR / "models"

PIPELINE_PATH: Path = MODELS_DIR / "pipeline.joblib"
MODEL_CARD_PATH: Path = MODELS_DIR / "model_card.json"
PROCESSED_PATH: Path = PROCESSED_DIR / "elasticities.parquet"

# The project input is a prepared electronics-pricing extract, versioned here.
SAMPLE_FILENAME: str = "price_observations.csv"
SAMPLE_PATH: Path = SAMPLE_DIR / SAMPLE_FILENAME
KAGGLE_DATASET: str = "datafiniti/electronic-products-prices"
RAW_FILENAME: str = "price_observations.csv"

PRODUCT_COL: str = "product"
CATEGORY_COL: str = "category"
PRICE_COL: str = "price"
DEMAND_COL: str = "demand"

MIN_OBSERVATIONS: int = 12
MIN_PRICE_POINTS: int = 3
MIN_R2: float = 0.05
CONFIDENCE: float = 0.95
ELASTIC_THRESHOLD: float = 1.0  # |beta| > 1 => elastic
# An economically identifiable elasticity is negative (demand falls as price rises)
# and not an extreme outlier from thin price variation. Estimates outside this band
# are flagged unreliable: impressions proxy demand and observational prices are
# confounded, so many naive log-log slopes come out positive or implausibly large.
RELIABLE_BOUNDS: tuple[float, float] = (-10.0, -0.05)
SEED: int = 42

DRACULA = {
    "background": "#282a36", "current_line": "#44475a", "foreground": "#f8f8f2",
    "comment": "#6272a4", "cyan": "#8be9fd", "green": "#50fa7b", "orange": "#ffb86c",
    "pink": "#ff79c6", "purple": "#bd93f9", "red": "#ff5555", "yellow": "#f1fa8c",
}
