import os
import glob
import pandas as pd
import numpy as np

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
    """
    if not os.path.exists(file_path):
        raise ValueError(f"BATADAL verisi {file_path} konumunda bulunamadı!")
        
    df = pd.read_csv(file_path)
    
    # Zaman sütunu model girdisi olmamalı
    if 'DATETIME' in df.columns:
        df = df.drop(columns=['DATETIME'])
    
    # BATADAL etiket sütununun ismindeki olası boşlukları temizle
    df.columns = df.columns.str.strip()
    
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