"""
Veri Yukleme ve On Isleme Pipeline
SWAT  : 11-Mar-2026_0900_1700.csv  -> label YOK (tamami normal veri)
WADI  : WADI_14days_new.csv        -> train (label yok)
        WADI_attackdataLABLE.csv   -> test  (label: 1=normal, -1=attack)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yaml, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def load_config(config_path: str = "configs/default_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_swat(cfg: dict) -> pd.DataFrame:
    path = os.path.join("data", cfg["datasets"]["swat"])
    df = pd.read_csv(path, low_memory=False)

    drop_cols = ["t_stamp"]
    df = df.drop(columns=drop_cols, errors="ignore")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map({"Inactive": 0, "Active": 1, "Bad Input": 0}).fillna(0)

    df["label"] = 0

    print(f"[SWAT] Yuklendi: {df.shape[0]} satir, {df.shape[1]-1} ozellik")
    print(f"[SWAT] Anomali orani: 0% (label yok, normal veri)")
    return df


def load_wadi_train(cfg: dict, max_rows: int = 100000) -> pd.DataFrame:
    path = os.path.join("data", cfg["datasets"]["wadi_train"])
    df = pd.read_csv(path, nrows=max_rows)

    drop_cols = ["Row", "Date", "Time"]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)
    df["label"] = 0

    print(f"[WADI-Train] Yuklendi: {df.shape[0]} satir, {df.shape[1]-1} ozellik")
    return df


def load_wadi_test(cfg: dict) -> pd.DataFrame:
    path = os.path.join("data", cfg["datasets"]["wadi_test"])
    df = pd.read_csv(path, header=1)

    label_col = "Attack LABLE (1:No Attack, -1:Attack)"
    drop_cols = ["Row ", "Date ", "Time"]
    labels = (df[label_col] == -1).astype(int)

    df = df.drop(columns=drop_cols + [label_col], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)
    df["label"] = labels.values

    print(f"[WADI-Test] Yuklendi: {df.shape[0]} satir, {df.shape[1]-1} ozellik")
    print(f"[WADI-Test] Anomali orani: {labels.mean():.2%}")
    return df


def split_data(df: pd.DataFrame, cfg: dict):
    n = len(df)
    train_end = int(n * cfg["split_ratios"]["train"])
    val_end   = int(n * (cfg["split_ratios"]["train"] + cfg["split_ratios"]["validation"]))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    print(f"  Bolme: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test


def preprocess(train_df, val_df, test_df, n_pca_components: int = 1):
    feature_cols = [c for c in train_df.columns if c != "label"]

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val   = val_df[feature_cols].values.astype(np.float32)
    X_test  = test_df[feature_cols].values.astype(np.float32)

    y_train = train_df["label"].values.astype(int)
    y_val   = val_df["label"].values.astype(int)
    y_test  = test_df["label"].values.astype(int)

    # Normalizasyon - sadece train'e fit
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # PCA - sadece train'e fit
    pca = PCA(n_components=n_pca_components)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_val_pca   = pca.transform(X_val_sc)
    X_test_pca  = pca.transform(X_test_sc)

    var_explained = pca.explained_variance_ratio_[0]
    print(f"  PCA: {X_train_sc.shape[1]} ozellik -> {n_pca_components} bilesen "
          f"(PC1 varyans: {var_explained:.3f})")

    return {
        "X_train": X_train_sc, "X_val": X_val_sc, "X_test": X_test_sc,
        "X_train_pca": X_train_pca, "X_val_pca": X_val_pca, "X_test_pca": X_test_pca,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler, "pca": pca,
        "feature_cols": feature_cols,
    }


def add_gaussian_noise(X: np.ndarray, cfg: dict, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(cfg["noise"]["mean"], cfg["noise"]["std_dev"], X.shape)
    return (X + noise).astype(np.float32)


def prepare_swat(cfg: dict):
    print("\n" + "="*50)
    print("[SWAT] Pipeline basliyor...")
    df = load_swat(cfg)
    train_df, val_df, test_df = split_data(df, cfg)
    data = preprocess(train_df, val_df, test_df)
    return data


def prepare_wadi(cfg: dict):
    print("\n" + "="*50)
    print("[WADI] Pipeline basliyor...")

    train_full = load_wadi_train(cfg)
    test_df    = load_wadi_test(cfg)

    n = len(train_full)
    val_start = int(n * 0.75)
    train_df = train_full.iloc[:val_start].copy()
    val_df   = train_full.iloc[val_start:].copy()

    print(f"  WADI Bolme: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    data = preprocess(train_df, val_df, test_df)
    return data


if __name__ == "__main__":
    cfg = load_config("configs/default_config.yaml")

    data_swat = prepare_swat(cfg)
    print(f"  SWAT Train: {data_swat['X_train'].shape}, Test: {data_swat['X_test'].shape}")

    data_wadi = prepare_wadi(cfg)
    print(f"  WADI Train: {data_wadi['X_train'].shape}, Test: {data_wadi['X_test'].shape}")
    print(f"  WADI Test anomali: {data_wadi['y_test'].mean():.2%}")

    print("\n[OK] Pipeline testi basarili!")