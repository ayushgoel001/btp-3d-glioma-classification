from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .utils import INDEX_TO_LABEL


def classification_metrics(y_true: np.ndarray, probabilities_hgg: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities_hgg = np.asarray(probabilities_hgg, dtype=np.float64)
    predictions = (probabilities_hgg >= 0.5).astype(np.int64)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=[0, 1], zero_division=0
    )
    per_class = {
        INDEX_TO_LABEL[index]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in (0, 1)
    }
    return {
        "n_patients": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "sensitivity_hgg": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity_lgg": float(tn / (tn + fp)) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(y_true, probabilities_hgg))
        if len(np.unique(y_true)) == 2
        else None,
        "confusion_matrix_lgg_hgg": matrix.tolist(),
        "per_class": per_class,
    }
