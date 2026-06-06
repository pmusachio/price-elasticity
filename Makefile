.PHONY: install profile train analyze test api

install:
	python -m pip install -r requirements.txt

profile:
	PYTHONPATH=src python -m price_elasticity.cli profile

train:
	PYTHONPATH=src python -m price_elasticity.cli train

analyze:
	PYTHONPATH=src python -m price_elasticity.cli analyze

test:
	python -m pytest

api:
	PYTHONPATH=src uvicorn price_elasticity.api:app --reload
