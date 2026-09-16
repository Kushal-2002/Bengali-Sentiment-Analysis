"""Classification metrics shared by per-epoch validation and final test evaluation."""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def detailed_report(y_true, y_pred, label_names: list[str]) -> dict:
    labels = list(range(len(label_names)))
    return {
        "confusion_matrix": {
            "labels": list(label_names),
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=list(label_names),
            output_dict=True, zero_division=0,
        ),
    }
