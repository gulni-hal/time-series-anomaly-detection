"""
Merkezi Konfigürasyon — Tüm parametreler buradan yönetilir.
Hard-coded değer YASAK. Değiştirmek istediğin her şey buradadır.
"""

# ─── Veri Seti ────────────────────────────────────────────────────────────────
DATASETS = {
    "SWAT": {
        "train": "data/raw/SWAT_train.csv",
        "test":  "data/raw/SWAT_test.csv",
        "label_col": "Normal/Attack",
        "attack_label": "Attack",
    },
    "WADI": {
        "train": "data/raw/WADI_train.csv",
        "test":  "data/raw/WADI_test.csv",
        "label_col": "Attack",
        "attack_label": 1,
    },
}

# Aktif veri setleri (grup 1-20: SWAT + WADI)
ACTIVE_DATASETS = ["SWAT", "WADI"]

# ─── Veri Bölme (Zorunlu: 60/20/20) ──────────────────────────────────────────
SPLIT = {
    "train": 0.60,
    "val":   0.20,
    "test":  0.20,
}

# ─── Random Seed'ler (Zorunlu: 5 farklı) ─────────────────────────────────────
SEEDS = [42, 123, 2026, 7, 999]

# ─── Derin Öğrenme Parametreleri ──────────────────────────────────────────────
DL = {
    "epochs":        50,
    "batch_size":    32,
    "patience":      5,          # early stopping
    "learning_rate": 1e-3,
    "hidden_size":   64,
    "num_layers":    2,
    "dropout":       0.2,
    "models":        ["LSTM", "GRU", "CNN1D"],  # en az 2 zorunlu
}

# ─── Otomata Parametreleri ────────────────────────────────────────────────────
AUTOMATA = {
    # Karşılaştırma için sabit (zorunlu)
    "window_size":   4,
    "alphabet_size": 3,

    # Parametre tarama aralığı (Tablo 4 için)
    "window_sizes":   [3, 4, 5, 6],
    "alphabet_sizes": [3, 4, 5, 6],
}

# ─── Gürültü Parametresi ──────────────────────────────────────────────────────
NOISE = {
    "type":   "gaussian",
    "mean":   0.0,
    "std":    0.05,   # ihtiyaca göre ayarla
}

# ─── Çıktı Yolları ────────────────────────────────────────────────────────────
PATHS = {
    "logs":     "logs/",
    "figures":  "outputs/figures/",
    "results":  "outputs/results/",
    "processed":"data/processed/",
}
