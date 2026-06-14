"""Single entrypoint: load -> preprocess -> estimate -> serialize. Idempotent.

    python -m src.pipeline
"""
from __future__ import annotations

import logging

from src import config
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.train import ElasticityEstimator


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def run() -> None:
    configure_logging()
    log = logging.getLogger("pipeline")
    log.info("Stage 1/4 - acquisition")
    loader = DataLoader()
    raw_path = loader.download()
    df = loader.load()
    log.info("Stage 2/4 - preprocessing")
    qualifying, _ = Preprocessor().run(df)
    log.info("Stage 3/4 - elasticity estimation")
    est = ElasticityEstimator(qualifying, data_source=raw_path)
    est.fit()
    log.info("Stage 4/4 - serialization")
    est.save()
    log.info("Done. Artifact: %s | Card: %s", config.PIPELINE_PATH, config.MODEL_CARD_PATH)


if __name__ == "__main__":
    run()
