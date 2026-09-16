import os

os.environ["BSA_SKIP_MODEL_LOAD"] = "1"

import pytest
from fastapi.testclient import TestClient

from bsa import serve


class FakeClassifier:
    """Stands in for the transformers pipeline so tests need no weights."""

    def __call__(self, texts):
        return [
            [{"label": "Positive", "score": 0.91}, {"label": "Negative", "score": 0.09}]
            for _ in texts
        ]


@pytest.fixture
def client():
    with TestClient(serve.app) as test_client:
        serve.state["classifier"] = FakeClassifier()
        yield test_client
        serve.state["classifier"] = None


def test_health_reports_model_identity(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_id"] == serve.MODEL_ID


def test_health_flags_missing_model(client):
    serve.state["classifier"] = None
    assert client.get("/health").json()["status"] == "model_unavailable"


def test_predict_returns_label_and_scores(client):
    body = client.post("/predict", json={"text": "বইটি অসাধারণ ছিল"}).json()
    assert body["label"] == "Positive"
    assert body["confidence"] == pytest.approx(0.91)
    assert set(body["scores"]) == {"Positive", "Negative"}


def test_batch_predicts_every_row(client):
    body = client.post("/predict/batch", json={"texts": ["ভালো", "খারাপ", "চমৎকার"]}).json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3


def test_blank_text_is_rejected(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_oversized_batch_is_rejected(client):
    texts = ["ভালো"] * (serve.MAX_BATCH + 1)
    assert client.post("/predict/batch", json={"texts": texts}).status_code == 422


def test_requests_fail_loudly_when_model_is_unavailable(client):
    serve.state["classifier"] = None
    assert client.post("/predict", json={"text": "ভালো"}).status_code == 503
