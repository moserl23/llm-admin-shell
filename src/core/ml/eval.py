"""Evaluation helpers for classifier predictions.

This module computes aggregate and per-class metrics used in the benchmark
pipeline and keeps the reporting format consistent across experiments.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

@dataclass
class EvalResult:
    accuracy: float
    balanced_accuracy: float
    mcc: float
    cohen_kappa: float
    f1_macro: float
    f1_weighted: float

    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float

    per_class_report: str
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion: np.ndarray

def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, *, labels: Optional[List[str]] = None) -> EvalResult:
    """Compute a compact set of classification metrics from predictions.

    Metrics are derived from a shared label set so aggregate scores, per-class
    summaries, and the confusion matrix remain aligned. Returns an `EvalResult`.
    """
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    # Keep only the true class entries; sklearn also injects aggregate rows.
    per_class_metrics = {
        cls: {k: float(v) for k, v in metrics.items()}
        for cls, metrics in report_dict.items()
        if cls not in {"accuracy", "macro avg", "weighted avg"}
        and isinstance(metrics, dict)
    }

    return EvalResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        cohen_kappa=float(cohen_kappa_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        precision_weighted=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_weighted=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),

        per_class_report=classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0),
        per_class_metrics=per_class_metrics,
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
    )
