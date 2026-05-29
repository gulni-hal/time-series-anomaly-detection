"""
Merkezi Konfigürasyon — Tüm parametreler buradan yönetilir.
Hard-coded değer YASAK. Değiştirmek istediğin her şey buradadır.
"""

# ─── Veri Seti ────────────────────────────────────────────────────────────────
DATASETS = {
    "SKAB": {
        "folder_path": "data/raw/SKAB/", # İçinde valve1 ve valve2 klasörleri olmalı
        "label_col": "anomaly",          # SKAB'da hedef değişken
    },
    "BATADAL": {
        "file_path": "data/raw/BATADAL/BATADAL_Training2.csv",
        "label_col": "ATT_FLAG",         # Etiket sütunu veri setinden kontrol edilmeli (Genelde ATT_FLAG)
    },
}

# Aktif veri setleri
ACTIVE_DATASETS = ["SKAB", "BATADAL"]

# ─── Veri Bölme (Zorunlu: 60/20/20) ──────────────────────────────────────────
SPLIT = {
    "train": 0.60,
    "val":   0.20,
    "test":  0.20,
}

# ─── Random Seed'ler (Zorunlu: 5 farklı) ─────────────────────────────────────
SEEDS = [42, 123, 2026, 7, 999]

# ─── Derin Öğrenme Parametreleri (Zorunlu sabitler) ───────────────────────────
DL = {
    "epochs":        50,         # Epoch üst sınırı: 50
    "batch_size":    32,         # Batch size: 32
    "patience":      5,          # Early stopping: validation loss (patience = 5)
    "learning_rate": 1e-3,
    "hidden_size":   64,
    "num_layers":    2,
    "dropout":       0.2,
    "models":        ["LSTM", "GRU", "CNN1D"],  # En az 2'si uygulanmalı
}

# ─── Otomata Parametreleri ────────────────────────────────────────────────────
AUTOMATA = {
    # Karşılaştırma için sabit (zorunlu)
    "window_size":   4,
    "alphabet_size": 3,

    # Parametre varyasyonu (Analiz için)
    "window_sizes":   [3, 4, 5, 6],
    "alphabet_sizes": [3, 4, 5, 6],
}

# ─── Gürültü Parametresi ──────────────────────────────────────────────────────
NOISE = {
    "type":   "gaussian",
    "mean":   0.0,
    "std":    0.05,
}

# ─── Çıktı Yolları ────────────────────────────────────────────────────────────
PATHS = {
    "logs":     "logs/",
    "figures":  "outputs/figures/",
    "results":  "outputs/results/",
    "processed":"data/processed/",
}