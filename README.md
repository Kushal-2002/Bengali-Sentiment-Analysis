# Bengali Book Review Sentiment

IndicBERT fine-tuned for Bengali review sentiment, with a reproducible training
pipeline, experiment tracking and a quality-gated model registry.

- Model: https://huggingface.co/k040902/bengali_book_review_sentiment
- App: https://smai-assignment-3.streamlit.app/

## Pipeline

```
configs/train.yaml ──► bsa.train ──► outputs/<run_id>/{model, metrics.json} ──► bsa.gate ──► MLflow @champion ──► (--push) HF Hub
                          │                                                        │
                          └──────────── MLflow run: params, curves, lineage ◄──────┘
```

| Step | What it guarantees |
|---|---|
| **Data** (`bsa/data.py`) | Dataset read from a pinned Hub commit; test reviews removed from the training pool; stratified train/val split; content fingerprint per split. |
| **Train** (`bsa/train.py`) | Base model pinned by commit; epoch chosen on the **validation** split only; every run logs config, git commit, data lineage and per-epoch metrics to MLflow. |
| **Evaluate** (`bsa/evaluate.py`) | One final pass on the untouched test set → `metrics.json` (metrics, confusion matrix, classification report). Also works on any saved or Hub model. |
| **Gate** (`bsa/gate.py`) | Candidate must clear absolute floors and not regress vs. the current champion on the same test set. Passing runs are registered and aliased `@champion`; failing runs exit non-zero. `--push` publishes the model with its metrics to the Hub and tags the commit. |

Data convention (from the assignment): the Hub `test` split is the training pool,
the Hub `validation` split is the held-out test set.

## Usage

```bash
make setup      # .venv with pinned deps (requirements-train.txt)
make test       # unit tests
make smoke      # tiny end-to-end run to check wiring
make train      # full run
make gate       # gate + register the latest run   (RUN=<run_id> to choose)
make push       # gate + register + publish to the Hub (needs `huggingface-cli login`)
make ui         # MLflow UI at http://127.0.0.1:5000
```

Override any config value without editing the file:

```bash
.venv/bin/python -m bsa.train -o training.epochs=4 -o training.learning_rate=3.0e-5
```

Write floats with a decimal point (`3.0e-5`); YAML reads `3e-5` as a string.

## Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py
```
