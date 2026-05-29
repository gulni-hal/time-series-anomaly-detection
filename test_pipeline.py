import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pipeline.data_loader import load_dataset, get_train_val_test_splits
from src.preprocessing.pipeline import DataPreprocessor
from src.models.deep_learning import LSTMAutoencoder
from src.models.automata import ProbabilisticAutomata
from src.pipeline.train import train_deep_learning, train_automata
from src.pipeline.evaluate import evaluate_models_over_seeds

print("1. Config ve Veri Yükleniyor...")
with open("configs/default_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Hızlı test için epoch sayısını 1'e düşür
config['training']['max_epochs'] = 1

# BATADAL verisini yükle
df = load_dataset("BATADAL", config['paths'])

# Hızlı olması için verinin sadece ilk 1000 satırını alalım
df = df.iloc[:1000]

X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_splits(
    "BATADAL", df, config['split_ratios'], config['datasets']['batadal_label']
)

print("2. Veri Ön İşleme (PCA ve Ölçeklendirme)...")
preprocessor = DataPreprocessor()
X_train_scaled, X_train_pca = preprocessor.fit_transform(X_train)
X_val_scaled, X_val_pca = preprocessor.transform(X_val)
X_test_scaled, X_test_pca = preprocessor.transform(X_test)

# PyTorch DataLoader hazırlığı (Derin öğrenme için)
train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1) # (batch, seq_len, features)
val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).unsqueeze(1)
train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=32)
val_loader = DataLoader(TensorDataset(val_tensor, val_tensor), batch_size=32)

print("\n3. Modeller Başlatılıyor...")
input_dim = X_train_scaled.shape[1]
dl_model = LSTMAutoencoder(input_dim=input_dim)
automata = ProbabilisticAutomata(
    window_size=config['automata']['window_size'], 
    alphabet_size=config['automata']['alphabet_size']
)

print("4. Modeller Eğitiliyor (Sadece 1 Epoch)...")
dl_model = train_deep_learning(dl_model, train_loader, val_loader, {'epochs': 1, 'patience': 1, 'learning_rate': 0.01})
print(" - Derin Öğrenme (LSTM) eğitimi tamamlandı.")

automata = train_automata(automata, X_train_pca)
print(" - Otomata eğitimi (Durumlar ve Olasılıklar) tamamlandı.")

print("\n5. Senaryolu Değerlendirme Testi Başlıyor...")
# Test fonksiyonumuzun doğru çalışıp çalışmadığını mock'luyoruz
def dummy_eval_fn(models, data):
    # Gerçek metrik hesaplaması yerine sahte bir skor dönüyoruz
    return {"Accuracy": 0.95, "F1": 0.92}

data_dict = {'train': X_train, 'test': X_test_pca}

# Seed loop test ediliyor
evaluate_models_over_seeds(
    seeds=config['training']['random_seeds'][:2], # Sadece ilk 2 seed'i test edelim
    config=config,
    data_dict=data_dict,
    train_pipeline_fn=None, 
    eval_fn=dummy_eval_fn
)

print("\nBÜTÜN PIPELINE TESTİ BAŞARILI! Kodlar hatasız çalışıyor.")