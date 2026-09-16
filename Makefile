PY ?= .venv/bin/python
CONFIG ?= configs/train.yaml
# Most recent run unless given: make gate RUN=<run_id>
RUN ?= $(shell ls -t outputs 2>/dev/null | head -1)
# Host port for the inference service (8000 is often taken by other stacks).
PORT ?= 8080
# Pin the served weights to a Hub commit: make docker-run MODEL_REVISION=<sha>
MODEL_REVISION ?=

.PHONY: setup test smoke train evaluate gate push ui serve docker-build docker-run

setup:
	python3.12 -m venv .venv
	$(PY) -m pip install -r requirements-train.txt

test:
	$(PY) -m pytest -q

# Tiny end-to-end run (a few rows, one epoch) to check the pipeline wiring.
smoke:
	$(PY) -m bsa.train --config $(CONFIG) -o experiment_name=smoke \
		-o data.subsample=48 -o training.epochs=1 -o log_model_artifact=false

train:
	$(PY) -m bsa.train --config $(CONFIG)

evaluate:
	$(PY) -m bsa.evaluate --config $(CONFIG) --model outputs/$(RUN)/model

gate:
	$(PY) -m bsa.gate --config $(CONFIG) --run-dir outputs/$(RUN)

push:
	$(PY) -m bsa.gate --config $(CONFIG) --run-dir outputs/$(RUN) --push

ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

serve:
	$(PY) -m uvicorn bsa.serve:app --host 0.0.0.0 --port $(PORT) --reload

docker-build:
	docker build -t bengali-sentiment .

docker-run:
	docker run --rm -p $(PORT):8000 -e MODEL_REVISION=$(MODEL_REVISION) bengali-sentiment
