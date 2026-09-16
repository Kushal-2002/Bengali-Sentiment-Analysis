import pytest

from bsa.metrics import classification_metrics, detailed_report


def test_macro_metrics():
    metrics = classification_metrics([0, 0, 1, 1], [0, 0, 1, 0])
    assert metrics["accuracy"] == 0.75
    # class 0: P=2/3 R=1 F1=0.8; class 1: P=1 R=0.5 F1=2/3
    assert metrics["f1_macro"] == pytest.approx((0.8 + 2 / 3) / 2)


def test_detailed_report_confusion_matrix():
    report = detailed_report([0, 0, 1, 1], [0, 0, 1, 0], ["Negative", "Positive"])
    assert report["confusion_matrix"] == {"labels": ["Negative", "Positive"], "matrix": [[2, 0], [1, 1]]}
    assert report["classification_report"]["Positive"]["recall"] == 0.5
