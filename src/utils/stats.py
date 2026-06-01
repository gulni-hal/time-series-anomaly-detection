import numpy as np
from scipy.stats import wilcoxon as _wilcoxon
from scipy.stats import chi2


def wilcoxon_test(scores_a, scores_b):
    """
    Wilcoxon işaretli sıra testi: İki modelin performans skorlarını karşılaştırır.
    Dönüş: {"statistic": float, "p_value": float}
    """
    scores_a = np.array(scores_a, dtype=float)
    scores_b = np.array(scores_b, dtype=float)
    diff = scores_a - scores_b
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0}
    try:
        stat, p_value = _wilcoxon(scores_a, scores_b)
        return {"statistic": float(stat), "p_value": float(p_value)}
    except Exception:
        return {"statistic": 0.0, "p_value": 1.0}


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar testi: İki sınıflandırıcının hatalarını karşılaştırır.
    b = A doğru, B yanlış; c = A yanlış, B doğru.
    Dönüş: {"statistic": float, "p_value": float}
    """
    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)

    b = int(np.sum((y_pred_a == y_true) & (y_pred_b != y_true)))
    c = int(np.sum((y_pred_a != y_true) & (y_pred_b == y_true)))

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    statistic = float((abs(b - c) - 1) ** 2 / (b + c))
    p_value = float(1.0 - chi2.cdf(statistic, df=1))
    return {"statistic": statistic, "p_value": p_value}
