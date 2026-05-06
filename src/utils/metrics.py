"""Metrik Hesaplama"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_bin = (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred.astype(int)
    return {
        "f1":        round(f1_score(y_true, y_bin, zero_division=0), 4),
        "accuracy":  round(accuracy_score(y_true, y_bin), 4),
        "precision": round(precision_score(y_true, y_bin, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_bin, zero_division=0), 4),
        "cm":        confusion_matrix(y_true, y_bin).tolist(),
    }
