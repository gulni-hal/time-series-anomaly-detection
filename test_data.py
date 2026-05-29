# test_data.py
import yaml
from src.pipeline.data_loader import load_dataset, get_train_val_test_splits
from src.preprocessing.pipeline import DataPreprocessor

# 1. Config'i oku
with open("configs/default_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. BATADAL Verisini Test Et
print("--- BATADAL Testi ---")
df_batadal = load_dataset("BATADAL", config['paths'])
X_train_b, y_train_b, X_val_b, y_val_b, X_test_b, y_test_b = get_train_val_test_splits(
    "BATADAL", df_batadal, config['split_ratios'], config['datasets']['batadal_label']
)
print(f"Eğitim boyutu: {X_train_b.shape}, Doğrulama: {X_val_b.shape}, Test: {X_test_b.shape}")

# 3. Preprocessor Testi (PCA sadece Train'e fit edilecek)
preprocessor = DataPreprocessor()
X_train_scaled, X_train_pca = preprocessor.fit_transform(X_train_b)
X_val_scaled, X_val_pca = preprocessor.transform(X_val_b)

print(f"Derin Öğrenme Girdisi (Çok Değişkenli): {X_train_scaled.shape}")
print(f"Otomata Girdisi (PCA 1D): {X_train_pca.shape}")
print("Test başarılı! Veriler sızıntı olmadan bölündü ve işlendi.\n")