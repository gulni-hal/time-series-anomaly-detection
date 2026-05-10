import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


def compute_metrics(y_true, y_pred, threshold=0.03):
    if hasattr(y_pred, 'dtype') and np.issubdtype(y_pred.dtype, np.floating):
        y_bin = (y_pred > threshold).astype(int)
    else:
        y_bin = y_pred.astype(int)
    return {
        "f1":        round(f1_score(y_true, y_bin, zero_division=0), 4),
        "accuracy":  round(accuracy_score(y_true, y_bin), 4),
        "precision": round(precision_score(y_true, y_bin, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_bin, zero_division=0), 4),
        "cm":        confusion_matrix(y_true, y_bin, labels=[0,1]).tolist(),
    }