import json

import pandas as pd
import pytest

from bsa.data import clean_frame, fingerprint, split_frames


def make_pool(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "text": [f"review {i}" for i in range(n)],
        "label": ["Positive" if i % 2 else "Negative" for i in range(n)],
    })


HELD_OUT = pd.DataFrame({"text": ["held out"], "label": ["Positive"]})


def test_clean_frame_drops_missing_and_blank():
    raw = pd.DataFrame({
        "INDIC REVIEW": ["ভালো", None, "  ", "খারাপ"],
        "LABEL": ["Positive", "Negative", "Positive", None],
    })
    df = clean_frame(raw, "INDIC REVIEW", "LABEL")
    assert df.to_dict("records") == [{"text": "ভালো", "label": "Positive"}]


def test_clean_frame_requires_columns():
    with pytest.raises(ValueError):
        clean_frame(pd.DataFrame({"x": [1]}), "INDIC REVIEW", "LABEL")


def test_split_is_stratified_disjoint_and_deterministic():
    a = split_frames(make_pool(), HELD_OUT, 0.2, seed=42)
    b = split_frames(make_pool(), HELD_OUT, 0.2, seed=42)
    assert fingerprint(a.train) == fingerprint(b.train)
    assert len(a.val) == 20
    assert set(a.val["label"].value_counts()) == {10}
    assert not set(a.train["text"]) & set(a.val["text"])


def test_test_reviews_are_removed_from_training_pool():
    pool = make_pool()
    test = pool.iloc[:4].copy()
    splits = split_frames(pool, test, 0.2, seed=0)
    assert splits.removed_test_overlap == 4
    assert not set(test["text"]) & (set(splits.train["text"]) | set(splits.val["text"]))


def test_unknown_test_label_is_rejected():
    test = pd.DataFrame({"text": ["x"], "label": ["Neutral"]})
    with pytest.raises(ValueError):
        split_frames(make_pool(), test, 0.2, seed=0)


def test_stats_are_json_serialisable():
    stats = split_frames(make_pool(), HELD_OUT, 0.2, seed=0).stats()
    assert json.loads(json.dumps(stats))["sizes"] == {"train": 80, "val": 20, "test": 1}
