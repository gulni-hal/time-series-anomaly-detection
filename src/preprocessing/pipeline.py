"""
Veri Ön İşleme Pipeline'ı
- Normalizasyon (sadece train'e fit)
- PCA (sadece train'e fit)
- Train/Val/Test bölme
- Gürültü ekleme (senaryo 2)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import SPLIT, NOISE


def load_raw_data(train_path: str, test_path: str, label_col: str):
    """
    CSV dosyalarını yükler.
    Veri seti gelmeden önce MOCK data üretir (geliştirme için).
    """
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)
        print(f"[✓] Gerçek veri yüklendi: {train_path}")
    else:
        print(f"[!] Veri seti bulunamadı → Mock data üretiliyor: {train_path}")
        train_df, test_df = _generate_mock_data(label_col)
    return train_df, test_df


def _generate_mock_data(label_col: str, n_train=5000, n_test=2000, n_features=10):
    """
    Veri seti olmadan pipeline'ı test etmek için sentetik veri üretir.
    Gerçek veri gelince bu fonksiyon çağrılmaz.
    """
    np.random.seed(42)
    # Normal veri
    normal_train = np.random.randn(int(n_train * 0.9), n_features)
    anomaly_train = np.random.randn(int(n_train * 0.1), n_features) + 3.0

    X_train = np.vstack([normal_train, anomaly_train])
    y_train = np.array([0] * len(normal_train) + [1] * len(anomaly_train))

    normal_test = np.random.randn(int(n_test * 0.85), n_features)
    anomaly_test = np.random.randn(int(n_test * 0.15), n_features) + 3.0
    X_test = np.vstack([normal_test, anomaly_test])
    y_test = np.array([0] * len(normal_test) + [1] * len(anomaly_test))

    cols = [f"sensor_{i}" for i in range(n_features)]
    train_df = pd.DataFrame(X_train, columns=cols)
    train_df[label_col] = y_train
    test_df  = pd.DataFrame(X_test,  columns=cols)
    test_df[label_col]  = y_test
    return train_df, test_df


def split_train_val(df: pd.DataFrame, label_col: str):
    """
    Train DataFrame'ini %60/%20 oranında Train/Val'e böler.
    (Toplam veri: Train %60, Val %20, Test %20)
    Zaman serisi olduğu için shuffle YOK — sıra korunur.
    """
    val_ratio_of_train = SPLIT["val"] / (SPLIT["train"] + SPLIT["val"])
    n = len(df)
    train_end = int(n * (1 - val_ratio_of_train))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:].copy()
    return train, val


def preprocess(train_df, val_df, test_df, label_col: str, use_pca: bool = True, pca_components: int = 1):
    """
    Pipeline:
    1. Label'ı ayır
    2. StandardScaler → sadece train'e fit
    3. PCA → sadece train'e fit (otomata için tek boyut)
    4. Sonuçları döndür

    UYARI: scaler ve pca objeleri döndürülüyor — test sırasında
    aynı objeyle transform yapılacak (data leakage önlemi).
    """
    feature_cols = [c for c in train_df.columns if c != label_col]

    X_train = train_df[feature_cols].values
    X_val   = val_df[feature_cols].values
    X_test  = test_df[feature_cols].values

    y_train = train_df[label_col].values
    y_val   = val_df[label_col].values
    y_test  = test_df[label_col].values

    # ── Normalizasyon (sadece train'e fit) ────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit + transform
    X_val   = scaler.transform(X_val)          # sadece transform
    X_test  = scaler.transform(X_test)         # sadece transform

    # ── PCA (sadece train'e fit) ──────────────────────────────────────────────
    pca = None
    if use_pca and X_train.shape[1] > 1:
        pca = PCA(n_components=pca_components)
        X_train_pca = pca.fit_transform(X_train)   # fit + transform
        X_val_pca   = pca.transform(X_val)          # sadece transform
        X_test_pca  = pca.transform(X_test)         # sadece transform
        print(f"[✓] PCA uygulandı → {X_train.shape[1]} özellik → {pca_components} bileşen")
        print(f"    Açıklanan varyans (PC1): {pca.explained_variance_ratio_[0]:.3f}")
    else:
        X_train_pca, X_val_pca, X_test_pca = X_train, X_val, X_test

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "X_train_pca": X_train_pca, "X_val_pca": X_val_pca, "X_test_pca": X_test_pca,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler, "pca": pca,
        "feature_cols": feature_cols,
    }


def add_gaussian_noise(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Senaryo 2: Gaussian gürültü ekler (config'den parametreler alınır).
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(NOISE["mean"], NOISE["std"], X.shape)
    return X + noise


def encode_labels(y: np.ndarray, attack_label) -> np.ndarray:
    """String veya int etiketleri 0/1'e çevirir."""
    return (y == attack_label).astype(int)


if __name__ == "__main__":
    # Hızlı test — gerçek veri olmadan çalışır
    from configs.config import DATASETS, ACTIVE_DATASETS

    for ds_name in ACTIVE_DATASETS:
        cfg = DATASETS[ds_name]
        print(f"\n{'='*50}")
        print(f"[{ds_name}] Pipeline testi başlıyor...")

        train_df, test_df = load_raw_data(cfg["train"], cfg["test"], cfg["label_col"])

        # Etiketleri sayısallaştır
        train_df[cfg["label_col"]] = encode_labels(
            train_df[cfg["label_col"]].values, cfg["attack_label"]
        )
        test_df[cfg["label_col"]] = encode_labels(
            test_df[cfg["label_col"]].values, cfg["attack_label"]
        )

        train_part, val_part = split_train_val(train_df, cfg["label_col"])
        data = preprocess(train_part, val_part, test_df, cfg["label_col"])

        print(f"[✓] Train: {data['X_train'].shape}, Val: {data['X_val'].shape}, Test: {data['X_test'].shape}")
        print(f"[✓] Anomali oranı (test): {data['y_test'].mean():.2%}")

        # Gürültü testi
        X_noisy = add_gaussian_noise(data["X_test_pca"])
        print(f"[✓] Gürültülü test verisi hazır: {X_noisy.shape}")
