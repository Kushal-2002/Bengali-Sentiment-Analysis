"""Fine-tune IndicBERT with MLflow experiment tracking.

    python -m bsa.train --config configs/train.yaml [-o training.epochs=4 ...]

Each run is an MLflow run plus a local directory ``outputs/<run_id>/`` holding the
selected model and ``metrics.json`` (the input to ``bsa.gate``).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from bsa.config import config_hash, flatten, load_config
from bsa.data import make_splits
from bsa.evaluate import evaluate_model
from bsa.metrics import classification_metrics


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, label2id: dict[str, int], max_length: int):
        self.encodings = tokenizer(df["text"].tolist(), truncation=True, max_length=max_length)
        self.labels = [label2id[label] for label in df["label"]]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {key: values[i] for key, values in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


class MLflowMetricsCallback(TrainerCallback):
    """Forward Trainer logs (train loss, per-epoch validation metrics) to the active run."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        metrics = {
            key: float(value) for key, value in (logs or {}).items()
            if isinstance(value, (int, float)) and key != "epoch"
        }
        if metrics:
            mlflow.log_metrics(metrics, step=state.global_step)


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    return classification_metrics(labels, np.argmax(logits, axis=-1))


def git_state() -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}
    return {"git_commit": commit, "git_dirty": str(bool(status)).lower()}


def run(cfg: dict) -> Path:
    set_seed(cfg["seed"])
    splits = make_splits(cfg["data"], cfg["seed"])
    label2id = {name: i for i, name in enumerate(splits.label_names)}
    id2label = {i: name for name, i in label2id.items()}
    model_cfg, tr = cfg["model"], cfg["training"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base"], revision=model_cfg["revision"])
    train_ds = EncodedDataset(splits.train, tokenizer, label2id, model_cfg["max_length"])
    val_ds = EncodedDataset(splits.val, tokenizer, label2id, model_cfg["max_length"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["base"], revision=model_cfg["revision"],
        num_labels=len(label2id), id2label=id2label, label2id=label2id,
    )

    data_info = {"hf_repo": cfg["data"]["hf_repo"], "revision": cfg["data"]["revision"], **splits.stats()}
    lineage = {**git_state(), "config_hash": config_hash(cfg)}

    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run() as active:
        run_id = active.info.run_id
        run_dir = Path(cfg["output_dir"]) / run_id
        mlflow.set_tags({**lineage, "test_fingerprint": data_info["fingerprints"]["test"]})
        mlflow.log_params(flatten(cfg))
        mlflow.log_params({f"n_{name}": n for name, n in data_info["sizes"].items()})
        mlflow.log_dict(cfg, "config.yaml")
        mlflow.log_dict(data_info, "data_info.json")

        args = TrainingArguments(
            output_dir=str(run_dir / "checkpoints"),
            learning_rate=float(tr["learning_rate"]),
            per_device_train_batch_size=tr["batch_size"],
            per_device_eval_batch_size=tr["eval_batch_size"],
            num_train_epochs=tr["epochs"],
            weight_decay=float(tr["weight_decay"]),
            warmup_ratio=float(tr["warmup_ratio"]),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model=tr["metric_for_best_model"],
            greater_is_better=True,
            seed=cfg["seed"],
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[MLflowMetricsCallback()],
        )
        trainer.train()

        model_dir = run_dir / "model"
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)

        # Same rule as load_best_model_at_end: the earliest epoch with the best score.
        key = f"eval_{tr['metric_for_best_model']}"
        best = max((log for log in trainer.state.log_history if key in log), key=lambda log: log[key])
        best_val = {k.removeprefix("eval_"): v for k, v in best.items() if k.startswith("eval_")}

        report = evaluate_model(str(model_dir), splits.test, model_cfg["max_length"], tr["eval_batch_size"])
        metrics = {
            "run_id": run_id,
            **lineage,
            "base_model": f"{model_cfg['base']}@{model_cfg['revision']}",
            "dataset": data_info,
            "best_epoch": best["epoch"],
            "val": best_val,
            **report,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        mlflow.log_metrics({f"best_val_{k}": v for k, v in best_val.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in report["test"].items()})
        mlflow.log_artifact(str(run_dir / "metrics.json"))
        if cfg["log_model_artifact"]:
            mlflow.log_artifacts(str(model_dir), artifact_path="model")

    test = report["test"]
    print(
        f"Run {run_id}: best epoch {best['epoch']:.0f}, "
        f"test accuracy {test['accuracy']:.4f}, test macro-F1 {test['f1_macro']:.4f}"
    )
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune IndicBERT with MLflow tracking.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument(
        "-o", "--override", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. -o training.epochs=4",
    )
    args = parser.parse_args(argv)
    run_dir = run(load_config(args.config, args.override))
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
