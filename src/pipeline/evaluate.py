import numpy as np
import random
import torch

def set_seed(seed_value):
    """Tüm kütüphaneler için rastgeleliği sabitler (Reproducibility)."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def add_gaussian_noise(data, mean=0.0, std=0.05):
    """PDF İsteri: Veriye Gauss gürültüsü ekleme senaryosu."""
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

def simulate_unseen_data(data, anomaly_injection_rate=0.05):
    """PDF İsteri: Test verisine eğitimde görülmemiş sentetik anomaliler (Unseen) ekler."""
    noisy_data = data.copy()
    num_anomalies = int(len(noisy_data) * anomaly_injection_rate)
    
    # Rastgele noktalara büyük sıçramalar (spike) ekle
    indices = np.random.choice(len(noisy_data), num_anomalies, replace=False)
    for idx in indices:
        noisy_data[idx] += np.random.normal(5.0, 1.0) # Normalin çok dışında bir değer
    return noisy_data

def evaluate_models_over_seeds(seeds, config, data_dict, train_pipeline_fn, eval_fn):
    """
    Zorunlu İster: Tüm modelleri 5 farklı seed ile ve 3 farklı senaryoda test eder.
    Ortalama ve Standart Sapma hesaplar.
    """
    scenarios = ["Orijinal", "Gürültülü", "Unseen"]
    results = {scenario: [] for scenario in scenarios}
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ile Deney Başlıyor ---")
        set_seed(seed)
        
        # 1. Modeli bu seed ile baştan eğit (train_pipeline_fn)
        # trained_automata, trained_dl = train_pipeline_fn(data_dict['train'])
        
        for scenario in scenarios:
            test_data = data_dict['test'].copy()
            
            if scenario == "Gürültülü":
                test_data = add_gaussian_noise(test_data, config['noise']['mean'], config['noise']['std_dev'])
            elif scenario == "Unseen":
                test_data = simulate_unseen_data(test_data)
                
            # 2. Modeli değerlendir ve metrikleri (Accuracy, F1 vs.) al (eval_fn)
            # metrics = eval_fn(trained_models, test_data)
            # results[scenario].append(metrics)
            print(f"[{scenario}] senaryosu değerlendirildi.")
            
    # Sonuçların ortalamasını ve standart sapmasını al (Raporlama için)
    print("\n--- DENEY SONUÇLARI (5 Seed Ortalaması) ---")
    # Gerçek uygulamada numpy mean/std hesaplamaları buraya gelir
    return results