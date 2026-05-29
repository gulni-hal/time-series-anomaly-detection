import yaml
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from src.pipeline.data_loader import load_dataset, get_train_val_test_splits
from src.preprocessing.pipeline import DataPreprocessor
from src.models.deep_learning import LSTMClassifier
from src.models.automata import ProbabilisticAutomata
from src.pipeline.train import train_deep_learning, train_automata
from src.utils.metrics import calculate_metrics
from src.utils.visualize import plot_confusion_matrix, plot_automata_transitions
from src.explainability.unseen_handler import UnseenHandler
from src.explainability.explainer import AutomataExplainer

def create_sequences(data, target, window_size):
    """Zaman serisini kayan pencerelere (sliding windows) ve hedeflere böler."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, window_size):
    """Verileri PyTorch Tensor'lerine çevirir ve DataLoader oluşturur."""
    X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, window_size)
    X_v_seq, y_v_seq = create_sequences(X_val, y_val, window_size)
    X_te_seq, y_te_seq = create_sequences(X_test, y_test, window_size)

    train_tensor_x = torch.tensor(X_tr_seq, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_tr_seq, dtype=torch.float32)
    val_tensor_x = torch.tensor(X_v_seq, dtype=torch.float32)
    val_tensor_y = torch.tensor(y_v_seq, dtype=torch.float32)
    test_tensor_x = torch.tensor(X_te_seq, dtype=torch.float32)
    test_tensor_y = torch.tensor(y_te_seq, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(train_tensor_x, train_tensor_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor_x, val_tensor_y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_tensor_x, test_tensor_y), batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, y_te_seq

def get_dl_predictions(model, test_loader):
    """Supervised (Gözetimli) modelden olasılıkları alır ve 0.5 eşiğiyle sınıflandırır."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch_x = batch[0]
            output = model(batch_x).squeeze() # Sigmoid'den gelen 0-1 arası olasılık
            
            # Tekil eleman olma durumu kontrolü
            if output.ndim == 0:
                preds = [1 if output.item() > 0.5 else 0]
            else:
                preds = (output > 0.5).int().numpy().tolist()
                
            predictions.extend(preds)
    return np.array(predictions)

def main():
    print("="*50)
    print("Zaman Serisi Anomali Tespiti - Eğitim Başlıyor")
    print("="*50)

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_name = "BATADAL"
    print(f"\n[1] {dataset_name} Veri Seti Yükleniyor...")
    df = load_dataset(dataset_name, config['paths'])
    label_col = config['datasets']['batadal_label']
    
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_splits(
        dataset_name, df, config['split_ratios'], label_col
    )
    
    print("\n[2] Veri Ön İşleme Uygulanıyor (Standartlaştırma & PCA)...")
    preprocessor = DataPreprocessor()
    X_train_scaled, X_train_pca = preprocessor.fit_transform(X_train)
    X_val_scaled, X_val_pca = preprocessor.transform(X_val)
    X_test_scaled, X_test_pca = preprocessor.transform(X_test)

    win_size = config['automata']['window_size']
    train_loader, val_loader, test_loader, y_test_seq = create_dataloaders(
        X_train_scaled, y_train.values, 
        X_val_scaled, y_val.values, 
        X_test_scaled, y_test.values, 
        config['training']['batch_size'], win_size
    )

    print(f"\n[3] Derin Öğrenme Modeli Eğitiliyor (Supervised LSTM)...")
    input_dim = X_train_scaled.shape[1]
    dl_model = LSTMClassifier(input_dim=input_dim)
    
    dl_model = train_deep_learning(dl_model, train_loader, val_loader, config['training'])
    print("Derin Öğrenme eğitimi tamamlandı.")
    
    y_pred_dl = get_dl_predictions(dl_model, test_loader)
    metrics_dl, cm_dl = calculate_metrics(y_test_seq, y_pred_dl)
    
    print("\n--- Derin Öğrenme Test Sonuçları ---")
    for k, v in metrics_dl.items():
        print(f"{k}: {v:.4f}")
        
    plot_confusion_matrix(cm_dl, classes=["Normal", "Anomaly"], 
                          title="LSTM Confusion Matrix", 
                          save_path="outputs/figures/dl_confusion_matrix.png")

    print("\n[4] Otomata Modeli Eğitiliyor (PAA, SAX, Durum Geçişleri)...")
    automata = ProbabilisticAutomata(
        window_size=win_size, 
        alphabet_size=config['automata']['alphabet_size']
    )
    automata = train_automata(automata, X_train_pca)
    print(f"Toplam {len(automata.transition_matrix)} benzersiz durum (state) öğrenildi.")

    known_patterns = list(automata.transition_matrix.keys())
    unseen_handler = UnseenHandler(known_patterns)
    explainer = AutomataExplainer(automata, unseen_handler)

    print("\n[5] Otomata Açıklanabilirlik Raporu Üretiliyor...")
    test_sequence_states = automata.extract_patterns(X_test_pca.flatten()[:100])
    report_df = explainer.explain_sequence(test_sequence_states)
    report_df.to_csv("outputs/results/automata_explanation_report.csv", index=False)

    print("\n[6] Durum Geçiş Diyagramı (State Transition Graph) Çiziliyor...")
    plot_automata_transitions(automata.transition_matrix, threshold=0.1, 
                              save_path="outputs/figures/automata_graph.png")

    print("\n[7] Parametre Duyarlılık Grafiği Çiziliyor...")
    from src.utils.visualize import plot_parameter_sensitivity
    plot_parameter_sensitivity(
        window_sizes=config['automata']['experiment_window_sizes'],
        alphabet_sizes=config['automata']['experiment_alphabet_sizes'],
        data_1d=X_train_pca.flatten()
    )

    print("\n"+"="*50)
    print("EĞİTİM VE TEST SÜRECİ BAŞARIYLA TAMAMLANDI!")
    print("="*50)

if __name__ == "__main__":
    main()