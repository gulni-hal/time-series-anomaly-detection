import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def load_skab(data_dir):
    """
    SKAB veri setini yükler. valve1 ve valve2 klasörlerindeki tüm CSV'leri birleştirir.
    Hangi klasör ve dosyadan geldiğini belirten ek sütunlar oluşturur.
    """
    all_data = []
    folders = ['valve1', 'valve2']
    
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Uyarı: {folder_path} bulunamadı!")
            continue
            
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        for file_path in csv_files:
            # SKAB genellikle noktalı virgül (;) ile ayrılmıştır, index datetime'dır.
            df = pd.read_csv(file_path, sep=';')
            
            # Hangi verinin nereden geldiğini izlemek için (GroupKFold için gerekli olacak)
            df['source_group'] = folder
            df['source_file'] = os.path.basename(file_path)
            all_data.append(df)
            
    if not all_data:
        raise ValueError(f"SKAB verisi {data_dir} dizininde bulunamadı!")
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Modellemeye girmeyecek gereksiz sütunları (varsa) temizle
    # Not: 'anomaly' sütununu etiket olarak ayıracağımız için onu bırakıyoruz.
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

def load_dataset(dataset_name, config_paths):
    """Config'den gelen isme göre ilgili veri setini yükler."""
    if dataset_name == "SKAB":
        data_dir = os.path.join(config_paths['data_raw_dir'], "SKAB")
        return load_skab(data_dir)
    elif dataset_name == "BATADAL":
        file_path = os.path.join(config_paths['data_raw_dir'], "BATADAL/BATADAL_Training2.csv")
        return load_batadal(file_path)
    else:
        raise ValueError(f"Bilinmeyen veri seti: {dataset_name}")
    


def split_skab_data(df, target_col='anomaly', test_size=0.2, val_size=0.2, random_state=42):
    """
    SKAB verisini grup (source_file) bazlı böler. 
    Aynı dosyadan (deneyden) gelen veriler farklı setlere dağılmaz, bütünlük korunur.
    """
    # 1. Önce Test setini ayır
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['source_file']))
    
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]
    
    # 2. Kalan veriden Validation setini ayır (Kalanın içindeki orana göre hesapla)
    # Eğer test %20 ise, val'in %20 olması için kalanın %25'i ayrılmalıdır (0.20 / 0.80 = 0.25)
    relative_val_size = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['source_file']))
    
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]
    
    # Yardımcı fonksiyon: Etiketleri ayır ve modelde kullanılmayacak kolonları temizle
    def separate_X_y(data_df):
        y = data_df[target_col]
        # source_file ve source_group sadece bölme işlemi içindi, artık çöpe gidebilir
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
    """
    test_ratio = config_split.get('test', 0.2)
    val_ratio = config_split.get('val', 0.2)
    train_ratio = config_split.get('train', 0.6)
    
    if dataset_name == "SKAB":
        return split_skab_data(df, target_col=label_col, test_size=test_ratio, val_size=val_ratio, random_state=random_state)
    elif dataset_name == "BATADAL":
        return split_batadal_data(df, target_col=label_col, train_ratio=train_ratio, val_ratio=val_ratio)
    else:
        raise ValueError("Bölme stratejisi tanımlanmamış veri seti!")