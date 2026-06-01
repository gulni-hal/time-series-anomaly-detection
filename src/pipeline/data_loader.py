import os
import glob
import pandas as pd
import numpy as np

def load_skab(data_dir):
    """
    SKAB veri setini yükler. valve1 ve valve2 klasörlerindeki tüm CSV'leri birleştirir.
    Dosyalar sıralı yüklenir (deterministik sıra — temporal split için zorunlu).
    """
    all_data = []
    folders = sorted(['valve1', 'valve2'])  # deterministik klasör sırası

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Uyarı: {folder_path} bulunamadı!")
            continue

        # Sıralanmış dosya listesi: OS bağımsız deterministik sıra
        csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        for file_path in csv_files:
            df = pd.read_csv(file_path, sep=';')
            df['source_group'] = folder
            df['source_file'] = os.path.basename(file_path)
            all_data.append(df)

    if not all_data:
        raise ValueError(f"SKAB verisi {data_dir} dizininde bulunamadı!")

    combined_df = pd.concat(all_data, ignore_index=True)

    cols_to_drop = ['datetime', 'changepoint']
    cols_to_drop = [c for c in cols_to_drop if c in combined_df.columns]
    combined_df = combined_df.drop(columns=cols_to_drop)

    return combined_df

def load_batadal(file_path):
    """
    BATADAL veri setini yükler ve gereksiz zaman sütunlarını kaldırır.
    Etiketleri kesin olarak 0 ve 1 formatına dönüştürür.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"BATADAL verisi {file_path} konumunda bulunamadı!")
        
    df = pd.read_csv(file_path)
    
    # Zaman sütunu model girdisi olmamalı
    if 'DATETIME' in df.columns:
        df = df.drop(columns=['DATETIME'])
    
    # BATADAL etiket sütununun ismindeki olası boşlukları temizle
    df.columns = df.columns.str.strip()
    
    # HATA ÇÖZÜMÜ: PyTorch BCELoss sadece 0 ve 1 bekler.
    # BATADAL'daki -999 gibi "Normal" veri etiketlerini 0'a çeviriyoruz.
    # Sadece değeri tam 1 olanları (Anomali) 1 bırakıyoruz.
    if 'ATT_FLAG' in df.columns:
        df['ATT_FLAG'] = df['ATT_FLAG'].apply(lambda x: 1 if x == 1 else 0)
        
    return df

def load_swat(file_path):
    """
    SWaT (Secure Water Treatment) veri setini yükler.
    Bu dosya normal çalışma verisini içerir; tüm satırlar etiket=0 (Normal) olarak işaretlenir.
    Karışık tip sütunlar pd.to_numeric ile sayısala dönüştürülür.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"SWAT verisi {file_path} konumunda bulunamadı!")

    df = pd.read_csv(file_path, low_memory=False)

    # Zaman damgası sütununu kaldır
    drop_cols = [c for c in ['t_stamp', 'Timestamp'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Karışık (string/numeric) sütunları sayısala çevir; "Bad Input" gibi değerler NaN olur
    # Tüm sütunlara uygulanır — pandas 2.x'de str dtype, object yerine ayrı bir tip olabilir
    df = df.apply(pd.to_numeric, errors='coerce')

    # NaN değerleri ileri dolgu (ffill) ve sıfır ile doldur
    df = df.ffill().fillna(0)

    # Normal çalışma verisi: tüm satırlar anomalisiz (0)
    df['Normal/Attack'] = 0

    return df


def load_wadi(file_path):
    """
    WADI (Water Distribution) saldırı veri setini yükler.
    header=1 ile okunur (ilk satır meta veri), etiket -1→1, 1→0 olarak dönüştürülür.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"WADI verisi {file_path} konumunda bulunamadı!")

    df = pd.read_csv(file_path, header=1, low_memory=False)

    # Gereksiz sütunları kaldır: sıra numarası, tarih, saat
    drop_cols = [c for c in ['Row ', 'Date ', 'Time'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Tamamen NaN olan sütunları kaldır (2_LS_001_AL, 2_P_001_STATUS vb.)
    all_nan_cols = [c for c in df.columns if df[c].isnull().all()]
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)

    # Etiket sütununu ikili (0/1) formata dönüştür: 1=Normal→0, -1=Saldırı→1
    label_col = 'Attack LABLE (1:No Attack, -1:Attack)'
    if label_col in df.columns:
        attack_labels = (df[label_col].astype(float) != 1).astype(int).values
        df = df.drop(columns=[label_col])
        df = df.copy()  # fragmantasyonu önlemek için tek seferde kopyala
        df['attack_label'] = attack_labels
    else:
        df = df.copy()
        df['attack_label'] = 0

    # Kalan NaN değerleri ileri dolgu ve sıfır ile doldur
    df = df.ffill().fillna(0)

    return df


def load_dataset(dataset_name, config_paths):
    """Config'den gelen isme göre ilgili veri setini yükler."""
    data_dir = config_paths['data_raw_dir']

    if dataset_name == "SKAB":
        return load_skab(os.path.join(data_dir, "SKAB"))
    elif dataset_name == "BATADAL":
        return load_batadal(os.path.join(data_dir, "BATADAL/BATADAL_Training2.csv"))
    elif dataset_name == "SWAT":
        fname = config_paths.get('swat_file', '11-Mar-2026_0900_1700.csv')
        return load_swat(os.path.join(data_dir, fname))
    elif dataset_name == "WADI":
        fname = config_paths.get('wadi_file', 'WADI_attackdataLABLE.csv')
        return load_wadi(os.path.join(data_dir, fname))
    else:
        raise ValueError(f"Bilinmeyen veri seti: {dataset_name}")
    


def split_skab_data(df, target_col='anomaly', train_ratio=0.6, val_ratio=0.2, **kwargs):
    """
    SKAB verisini zamansal (temporal) sırayla böler: ilk %60 train, sonraki %20 val, son %20 test.
    GroupShuffleSplit yerine sıralı bölme kullanılır — gelecek verinin eğitime sızmasını önler.
    Dosyalar load_skab() içinde sıralı yüklendiği için satır sırası zamansal sırayı korur.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    def separate_X_y(data_df):
        y = data_df[target_col]
        X = data_df.drop(columns=[target_col, 'source_file', 'source_group'], errors='ignore')
        return X, y

    X_train, y_train = separate_X_y(train_df)
    X_val, y_val = separate_X_y(val_df)
    X_test, y_test = separate_X_y(test_df)

    return X_train, y_train, X_val, y_val, X_test, y_test

def split_batadal_data(df, target_col='ATT_FLAG', train_ratio=0.6, val_ratio=0.2):
    """
    BATADAL verisini zaman sırasına göre sıralı olarak böler.
    Zaman serisi örüntüsünün ardışıklığı kesinlikle bozulmaz.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    def separate_X_y(data_df):
        y = data_df[target_col]
        X = data_df.drop(columns=[target_col], errors='ignore')
        return X, y

    X_train, y_train = separate_X_y(train_df)
    X_val, y_val = separate_X_y(val_df)
    X_test, y_test = separate_X_y(test_df)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_train_val_test_splits(dataset_name, df, config_split, label_col, random_state=42):
    """
    Ana yönlendirici fonksiyon. Veri seti ismine göre ilgili bölme stratejisini çağırır.
    SWAT ve WADI sıralı (temporal) bölme kullanır.
    """
    test_ratio = config_split.get('test', 0.2)
    val_ratio = config_split.get('val', 0.2)
    train_ratio = config_split.get('train', 0.6)

    if dataset_name == "SKAB":
        return split_skab_data(df, target_col=label_col,
                               train_ratio=train_ratio, val_ratio=val_ratio)
    elif dataset_name in ("BATADAL", "SWAT", "WADI"):
        return split_batadal_data(df, target_col=label_col,
                                  train_ratio=train_ratio, val_ratio=val_ratio)
    else:
        raise ValueError(f"Bölme stratejisi tanımlanmamış veri seti: {dataset_name}")