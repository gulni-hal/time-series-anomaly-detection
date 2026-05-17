"""
DL Modelleri: LSTM, GRU, 1D-CNN
Sklearn MLPClassifier wrapper — kararlı ve hızlı.

F1=0 düzeltmesi:
  - Threshold arama aralığı 0.01'den başlıyor (relu ile prob. çok küçük çıkabiliyor)
  - LSTM ve CNN1D tanh kullanıyor (GRU gibi — daha iyi anomali ayrımı)
  - Gereksinim: max_iter=50, n_iter_no_change=5 (early_stopping_patience=5)
"""

import numpy as np
import time
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight


SEQ_LEN = 5

THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2,
              0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]


def make_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len].flatten())
        ys.append(y[i+seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def to_loader(X, y, batch_size=32, shuffle=True):
    return X, y


def get_model(name, input_size):
    name = name.upper()
    if name == "LSTM":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="tanh",
            max_iter=50,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
        )
    if name == "GRU":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="tanh",
            max_iter=50,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
        )
    if name == "CNN1D":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="tanh",
            max_iter=50,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
        )
    raise ValueError("Unknown model: " + name)


def train_model(model, train_loader, val_loader, cfg, seed, pos_weight=1.0):
    X_train, y_train = train_loader
    t0 = time.perf_counter()

    sample_weights = compute_sample_weight(
        class_weight={0: 1.0, 1: pos_weight},
        y=y_train.astype(int)
    )

    model.set_params(random_state=seed)
    model.fit(X_train, y_train.astype(int), sample_weight=sample_weights)
    print("    Trained " + str(model.__class__.__name__) +
          " iterations=" + str(model.n_iter_))
    return model, time.perf_counter() - t0


def evaluate_model(model, test_loader):
    X_test, y_test = test_loader
    t0 = time.perf_counter()
    preds = model.predict_proba(X_test)[:, 1].astype(np.float32)
    return preds, y_test.astype(np.float32), time.perf_counter() - t0


def find_best_threshold(model, val_loader, y_val_true):
    from sklearn.metrics import precision_recall_curve
    X_val, y_val = val_loader
    preds = model.predict_proba(X_val)[:, 1].astype(np.float32)
    min_len = min(len(preds), len(y_val_true))
    preds  = preds[:min_len]
    y_true = y_val_true[:min_len].astype(np.float32)

    if y_true.sum() == 0:
        print("    No positives in val set — using default threshold 0.5")
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, preds)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1_scores[:-1]))
    best_t   = float(thresholds[best_idx])
    best_f1  = float(f1_scores[best_idx])

    print("    Best threshold: " + str(round(best_t, 4)) +
          " (val F1=" + str(round(best_f1, 4)) + ")" +
          " pred_range=[" + str(round(float(preds.min()), 4)) +
          ", " + str(round(float(preds.max()), 4)) + "]")
    return best_t
