import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yaml, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def load_config(config_path="configs/default_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_swat(cfg):
    path = os.path.join("data", cfg["datasets"]["swat"])
    df = pd.read_csv(path, low_memory=False)
    df = df.drop(columns=["t_stamp"], errors="ignore")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map({"Inactive": 0, "Active": 1, "Bad Input": 0}).fillna(0)
    df["label"] = 0
    print("[SWAT] Yuklendi: " + str(df.shape[0]) + " satir, " + str(df.shape[1]-1) + " ozellik")
    return df


def load_wadi_combined(cfg, max_train_rows=100000):
    # Normal veri
    train_path = os.path.join("data", cfg["datasets"]["wadi_train"])
    df_train = pd.read_csv(train_path, nrows=max_train_rows)
    df_train = df_train.drop(columns=["Row", "Date", "Time"], errors="ignore")
    df_train = df_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    df_train["label"] = 0

    # Anomalili veri
    test_path = os.path.join("data", cfg["datasets"]["wadi_test"])
    df_test = pd.read_csv(test_path, header=1)
    label_col = "Attack LABLE (1:No Attack, -1:Attack)"
    labels = (df_test[label_col] == -1).astype(int)
    df_test = df_test.drop(columns=["Row ", "Date ", "Time ", label_col], errors="ignore")
    df_test = df_test.apply(pd.to_numeric, errors="coerce").fillna(0)
    df_test["label"] = labels.values

    # Ortak kolonlar
    common_cols = sorted(list(set(df_train.columns) & set(df_test.columns)))
    df_combined = pd.concat([df_train[common_cols], df_test[common_cols]], ignore_index=True)

    total_anomaly = int(df_combined["label"].sum())
    print("[WADI] Birlestirildi: " + str(len(df_combined)) + " satir, " +
          str(len(common_cols)-1) + " ozellik, anomali=" + str(total_anomaly))
    return df_combined


def split_data_stratified(df, cfg, seed=42):
    """
    Stratified split: anomali ornekleri her bolumde dengeli dagitilir.
    Zaman serisi sirasi her sinif icinde korunur.
    """
    tr_r = cfg["split_ratios"]["train"]
    vl_r = cfg["split_ratios"]["validation"]

    normal_idx  = df[df["label"] == 0].index.tolist()
    anomaly_idx = df[df["label"] == 1].index.tolist()

    def split_indices(idx, tr_r, vl_r):
        n = len(idx)
        t = int(n * tr_r)
        v = int(n * (tr_r + vl_r))
        return idx[:t], idx[t:v], idx[v:]

    n_tr, n_vl, n_te = split_indices(normal_idx, tr_r, vl_r)
    a_tr, a_vl, a_te = split_indices(anomaly_idx, tr_r, vl_r)

    train_idx = sorted(n_tr + a_tr)
    val_idx   = sorted(n_vl + a_vl)
    test_idx  = sorted(n_te + a_te)

    train = df.loc[train_idx].reset_index(drop=True)
    val   = df.loc[val_idx].reset_index(drop=True)
    test  = df.loc[test_idx].reset_index(drop=True)

    print("  Bolme: Train=" + str(len(train)) + " (anomali=" + str(int(train['label'].sum())) +
          "), Val=" + str(len(val)) + " (anomali=" + str(int(val['label'].sum())) +
          "), Test=" + str(len(test)) + " (anomali=" + str(int(test['label'].sum())) + ")")
    return train, val, test


def preprocess(train_df, val_df, test_df, n_pca_components=1):
    feature_cols = [c for c in train_df.columns if c != "label"]

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val   = val_df[feature_cols].values.astype(np.float32)
    X_test  = test_df[feature_cols].values.astype(np.float32)

    y_train = train_df["label"].values.astype(int)
    y_val   = val_df["label"].values.astype(int)
    y_test  = test_df["label"].values.astype(int)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    pca = PCA(n_components=n_pca_components)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_val_pca   = pca.transform(X_val_sc)
    X_test_pca  = pca.transform(X_test_sc)

    print("  PCA: " + str(X_train_sc.shape[1]) + " -> " + str(n_pca_components) +
          " (var=" + str(round(pca.explained_variance_ratio_[0], 3)) + ")")

    return {
        "X_train": X_train_sc, "X_val": X_val_sc, "X_test": X_test_sc,
        "X_train_pca": X_train_pca, "X_val_pca": X_val_pca, "X_test_pca": X_test_pca,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler, "pca": pca, "feature_cols": feature_cols,
    }


def add_gaussian_noise(X, cfg, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(cfg["noise"]["mean"], cfg["noise"]["std_dev"], X.shape)
    return (X + noise).astype(np.float32)


def prepare_swat(cfg):
    print("\n" + "="*50)
    print("[SWAT] Pipeline basliyor...")
    df = load_swat(cfg)
    train_df, val_df, test_df = split_data_stratified(df, cfg)
    return preprocess(train_df, val_df, test_df)


def prepare_wadi(cfg):
    print("\n" + "="*50)
    print("[WADI] Pipeline basliyor...")
    df = load_wadi_combined(cfg)
    train_df, val_df, test_df = split_data_stratified(df, cfg)
    return preprocess(train_df, val_df, test_df)


if __name__ == "__main__":
    cfg = load_config("configs/default_config.yaml")
    data = prepare_wadi(cfg)
    print("Train anomali: " + str(data['y_train'].mean().round(4)))
    print("Val anomali:   " + str(data['y_val'].mean().round(4)))
    print("Test anomali:  " + str(data['y_test'].mean().round(4)))
    print("[OK] Done!")