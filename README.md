# Bengali Book Review Sentiment

[![CI](https://github.com/Kushal-2002/Bengali-Sentiment-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Kushal-2002/Bengali-Sentiment-Analysis/actions/workflows/ci.yml)

Sentiment classification for Bengali book reviews: IndicBERT fine-tuned on
[IndicSentiment](https://huggingface.co/datasets/ai4bharat/IndicSentiment), wrapped in a
pipeline that makes every reported number reproducible and blocks a bad model from
reaching production.

- **Model:** https://huggingface.co/k040902/bengali_book_review_sentiment
- **Demo:** https://smai-assignment-3.streamlit.app/
- **API:** FastAPI + Docker (`make docker-run`)

## Results

Held-out test set (156 reviews), evaluated once, after training:

| Metric | Value |
|---|---|
| Accuracy | **78.85%** |
| Macro F1 | **0.7880** |
| Macro precision / recall | 0.7882 / 0.7879 |

Run `b85e3fce`, 8 epochs, ~4.5 minutes on an M-series laptop. Best epoch selected on a
validation split; per-epoch curves live in MLflow.

### Why this number is lower than it used to be

An earlier version of this project reported **82.69%**. That run selected its best epoch by
scoring on the test set, so the test set had already influenced model selection and the
score was optimistic. This pipeline carves a validation split out of the training pool,
selects on it, and touches the test set exactly once. Accuracy dropped ~4 points and the
number became trustworthy — which is the whole point of the rest of this repo.

## Architecture

```
configs/train.yaml
        │
        ▼
   bsa.train ──► outputs/<run_id>/{model, metrics.json} ──► bsa.gate ──┬──► MLflow registry @champion
        │                                                              │
        │                                                              └──► (--push) HF Hub + commit tag
        └──────────► MLflow run: params, per-epoch curves, git commit, data fingerprints
                                                                       │
                                             bsa.serve (FastAPI/Docker) ◄── pinned MODEL_REVISION
```

| Stage | What it guarantees |
|---|---|
| **Data** (`src/bsa/data.py`) | Dataset read from a pinned Hub commit; stratified train/val split; a content fingerprint per split, so "same data?" is a checkable claim. |
| **Train** (`src/bsa/train.py`) | Base model pinned by commit; seeded; epoch chosen on validation only; every run logs its config, git commit, dirty-tree flag, data lineage and per-epoch metrics to MLflow. |
| **Evaluate** (`src/bsa/evaluate.py`) | One final pass on the untouched test set → `metrics.json` with metrics, confusion matrix and classification report. Works on any local or Hub model. |
| **Gate** (`src/bsa/gate.py`) | A candidate must clear absolute floors **and** not regress against the champion — and if the champion was scored on a different test set, the gate refuses to compare rather than comparing incomparable numbers. Passing runs are registered and aliased `@champion`; failures exit non-zero. |
| **Serve** (`src/bsa/serve.py`) | FastAPI service loading a pinned Hub revision, containerized on CPU-only torch, running as a non-root user, with request-level logging that never records review text. |

Splits follow the dataset's own convention for this task: the Hub `test` split (998 rows)
is the training pool, split 898/100 into train and validation; the Hub `validation` split
(156 rows) is the held-out test set.

## Quickstart

```bash
make setup
make test
make smoke
```

`make smoke` runs the entire pipeline — data, training, evaluation, gate — on 48 rows in
under a minute. If it passes, the wiring is sound.

## Training

```bash
make train
make gate
make push
make ui
```

Override any config value without editing the file:

```bash
.venv/bin/python -m bsa.train -o training.epochs=4 -o training.learning_rate=3.0e-5
```

Floats need a decimal point (`3.0e-5`); YAML reads `3e-5` as a string.

`make gate` promotes only if the candidate clears the floors in `configs/train.yaml` and
does not regress against the current champion. `make push` additionally publishes to the
Hub and tags the commit, writing the Hub revision back onto the MLflow run — so a tracked
experiment points at the exact weights a container can serve.

## Inference API

```bash
make serve
make docker-build && make docker-run
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness, plus the model id and revision actually loaded |
| `POST /predict` | One review → label, confidence, per-class scores |
| `POST /predict/batch` | Up to `MAX_BATCH` (256) reviews per call |

```bash
curl -sS -X POST http://localhost:8080/predict \
  -H 'content-type: application/json' \
  -d '{"text": "বইটি অসাধারণ ছিল"}'
```

```bash
make docker-run PORT=9000 MODEL_REVISION=<hub-commit-sha>
```

Pinning `MODEL_REVISION` is what keeps a deployed container on known weights instead of
whatever the Hub repo points at today. Startup blocks on the model load, so the container
is either not-ready or fully working — `HEALTHCHECK` reports which.

## Continuous integration

| Job | When | What it proves |
|---|---|---|
| Unit tests | every push and PR | 25 tests across config, data, gate, metrics and the API — the API tests stub the model, so CI needs no weights |
| Build inference image | every push and PR | the Dockerfile builds, the container boots and answers `/health` |
| End-to-end smoke run | manual and weekly | the full pipeline still runs against the live Hub, catching upstream dataset or model drift |

## Layout

```
src/bsa/        config, data, train, evaluate, gate, serve
tests/          unit tests (no network, no weights)
configs/        train.yaml — every hyperparameter and gate threshold
Dockerfile      CPU-only inference image
app.py          Streamlit demo
report.tex      write-up, with the method and results
```

## Streamlit demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

This began as a course assignment and was rebuilt as an MLOps project: the modelling is
unchanged, everything around it — reproducibility, tracking, gating, serving, CI — is the
work.
