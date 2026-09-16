"""FastAPI service exposing the promoted Bengali sentiment model.

    uvicorn bsa.serve:app --host 0.0.0.0 --port 8000

The model is resolved from ``MODEL_ID`` (a Hub repo id or a local directory) and
loaded once at startup; ``MODEL_REVISION`` pins it to a Hub commit so a running
container serves a known set of weights rather than "whatever main is today".
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = os.getenv("MODEL_ID", "k040902/bengali_book_review_sentiment")
MODEL_REVISION = os.getenv("MODEL_REVISION") or None
MAX_BATCH = int(os.getenv("MAX_BATCH", "256"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("bsa.serve")

state: dict = {"classifier": None}


def load_classifier():
    from transformers import pipeline

    kwargs = {"model": MODEL_ID, "tokenizer": MODEL_ID, "top_k": None}
    if MODEL_REVISION:
        kwargs["revision"] = MODEL_REVISION
    return pipeline("text-classification", **kwargs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests set BSA_SKIP_MODEL_LOAD so the suite never downloads weights.
    if os.getenv("BSA_SKIP_MODEL_LOAD") != "1":
        started = time.perf_counter()
        state["classifier"] = load_classifier()
        logger.info(
            "model_loaded model_id=%s revision=%s seconds=%.1f",
            MODEL_ID, MODEL_REVISION or "latest", time.perf_counter() - started,
        )
    yield
    state["classifier"] = None


app = FastAPI(
    title="Bengali Review Sentiment",
    description="Sentiment classification for Bengali book reviews (IndicBERT).",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000, examples=["বইটি অসাধারণ ছিল"])


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    count: int
    latency_ms: float


def _normalise(raw) -> list[list[dict]]:
    """transformers returns list[dict] or list[list[dict]] depending on input shape."""
    return [raw] if raw and isinstance(raw[0], dict) else raw


def _predict(texts: list[str]) -> tuple[list[Prediction], float]:
    classifier = state["classifier"]
    if classifier is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    stripped = [text.strip() for text in texts]
    if any(not text for text in stripped):
        raise HTTPException(status_code=422, detail="text must not be blank")

    started = time.perf_counter()
    results = _normalise(classifier(stripped))
    latency_ms = (time.perf_counter() - started) * 1000

    predictions = []
    for scores in results:
        best = max(scores, key=lambda score: score["score"])
        predictions.append(Prediction(
            label=best["label"],
            confidence=float(best["score"]),
            scores={score["label"]: float(score["score"]) for score in scores},
        ))

    # Log shape and outcome, never the review text itself.
    logger.info(
        "prediction n=%d latency_ms=%.1f labels=%s mean_confidence=%.3f mean_chars=%d",
        len(predictions), latency_ms,
        ",".join(sorted({p.label for p in predictions})),
        sum(p.confidence for p in predictions) / len(predictions),
        sum(len(text) for text in stripped) // len(stripped),
    )
    return predictions, latency_ms


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if state["classifier"] is not None else "model_unavailable",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION or "latest",
        "max_batch": MAX_BATCH,
    }


@app.post("/predict", response_model=Prediction)
def predict(request: PredictRequest) -> Prediction:
    predictions, _ = _predict([request.text])
    return predictions[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    predictions, latency_ms = _predict(request.texts)
    return BatchResponse(predictions=predictions, count=len(predictions), latency_ms=round(latency_ms, 1))
