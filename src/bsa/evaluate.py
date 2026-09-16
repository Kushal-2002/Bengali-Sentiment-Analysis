"""Evaluate a saved model (local directory or Hub repo id) on the held-out test set.

    python -m bsa.evaluate --model outputs/<run_id>/model
    python -m bsa.evaluate --model k040902/bengali_book_review_sentiment --out baseline.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bsa.config import load_config
from bsa.data import make_splits
from bsa.metrics import classification_metrics, detailed_report


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_logits(model, tokenizer, texts: list[str], max_length: int, batch_size: int) -> torch.Tensor:
    device = pick_device()
    model.to(device).eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start:start + batch_size], truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        ).to(device)
        chunks.append(model(**batch).logits.float().cpu())
    return torch.cat(chunks)


def evaluate_model(model_ref: str, test_df: pd.DataFrame, max_length: int, batch_size: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForSequenceClassification.from_pretrained(model_ref)
    label2id = model.config.label2id
    unknown = set(test_df["label"]) - set(label2id)
    if unknown:
        raise ValueError(f"model {model_ref} has no id for labels {sorted(unknown)}")
    label_names = [model.config.id2label[i] for i in range(model.config.num_labels)]

    y_true = torch.tensor([label2id[label] for label in test_df["label"]])
    logits = predict_logits(model, tokenizer, test_df["text"].tolist(), max_length, batch_size)
    y_pred = logits.argmax(dim=-1)

    test = classification_metrics(y_true.numpy(), y_pred.numpy())
    test["loss"] = float(F.cross_entropy(logits, y_true))
    test["n"] = len(test_df)
    return {"test": test, **detailed_report(y_true.numpy(), y_pred.numpy(), label_names)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on the held-out test set.")
    parser.add_argument("--model", required=True, help="local model directory or Hub repo id")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, help="write the report as JSON here")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.override)
    splits = make_splits(cfg["data"], cfg["seed"])
    report = evaluate_model(
        args.model, splits.test, cfg["model"]["max_length"], cfg["training"]["eval_batch_size"]
    )
    report["dataset"] = {"revision": cfg["data"]["revision"], **splits.stats()}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
