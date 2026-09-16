"""Load IndicSentiment from a pinned Hub revision and build train/val/test splits.

Assignment convention: the Hub ``test`` split is the training pool and the Hub
``validation`` split is the held-out test set. Epoch selection uses a validation
slice carved out of the training pool, so the test set is only ever touched by
the final evaluation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    label_names: list[str]
    removed_test_overlap: int

    def frames(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def stats(self) -> dict:
        """JSON-serialisable summary used for lineage (sizes, balance, fingerprints)."""
        frames = self.frames()
        return {
            "label_names": list(self.label_names),
            "sizes": {name: len(df) for name, df in frames.items()},
            "label_counts": {
                name: {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}
                for name, df in frames.items()
            },
            "fingerprints": {name: fingerprint(df) for name, df in frames.items()},
            "removed_test_overlap": self.removed_test_overlap,
        }


def source_url(data_cfg: dict, split: str) -> str:
    return (
        f"https://huggingface.co/datasets/{data_cfg['hf_repo']}/resolve/"
        f"{data_cfg['revision']}/data/{split}/{data_cfg['language']}.json"
    )


def clean_frame(raw: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """Keep the text/label columns as ``text``/``label``; drop missing or blank rows."""
    missing = {text_col, label_col} - set(raw.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    df = raw[[text_col, label_col]].dropna().rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str)
    return df[df["text"] != ""].reset_index(drop=True)


def load_split(data_cfg: dict, split: str) -> pd.DataFrame:
    raw = pd.read_json(source_url(data_cfg, split), lines=True)
    return clean_frame(raw, data_cfg["text_col"], data_cfg["label_col"])


def fingerprint(df: pd.DataFrame) -> str:
    """Content hash of an (ordered) split, so runs can prove they used the same data."""
    h = hashlib.sha256()
    for text, label in zip(df["text"], df["label"]):
        h.update(f"{label}\t{text}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def split_frames(
    pool: pd.DataFrame,
    test: pd.DataFrame,
    val_fraction: float,
    seed: int,
    subsample: int | None = None,
) -> Splits:
    label_names = sorted(pool["label"].unique())
    unknown = set(test["label"]) - set(label_names)
    if unknown:
        raise ValueError(f"test set has labels not present in training data: {sorted(unknown)}")

    # A review that appears in the test set must never be trained on.
    in_test = pool["text"].isin(set(test["text"]))
    removed = int(in_test.sum())
    pool = pool[~in_test]

    train, val = train_test_split(
        pool, test_size=val_fraction, stratify=pool["label"], random_state=seed
    )
    if subsample:
        train, val, test = (
            df.sample(n=min(subsample, len(df)), random_state=seed) for df in (train, val, test)
        )
    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        label_names=label_names,
        removed_test_overlap=removed,
    )


def make_splits(data_cfg: dict, seed: int) -> Splits:
    pool = load_split(data_cfg, data_cfg["train_source_split"])
    test = load_split(data_cfg, data_cfg["test_source_split"])
    return split_frames(pool, test, data_cfg["val_fraction"], seed, data_cfg.get("subsample"))
