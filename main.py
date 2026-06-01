import yaml
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Küçük batch boyutlarında çoklu thread overhead'i önlemek için tek thread kullan
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from src.pipeline.data_loader import load_dataset, get_train_val_test_splits
from src.preprocessing.pipeline import DataPreprocessor
from src.models.deep_learning import LSTMClassifier, GRUClassifier, CNN1DClassifier
from src.models.automata import ProbabilisticAutomata
from src.pipeline.train import train_deep_learning, train_automata
from src.pipeline.evaluate import set_seed, add_gaussian_noise, simulate_unseen_data
from src.utils.metrics import calculate_metrics
from src.utils.logger import ExperimentLogger
from src.utils.stats import wilcoxon_test, mcnemar_test
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

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_seq, dtype=torch.float32),
                      torch.tensor(y_tr_seq, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_v_seq, dtype=torch.float32),
                      torch.tensor(y_v_seq, dtype=torch.float32)),
        batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_te_seq, dtype=torch.float32),
                      torch.tensor(y_te_seq, dtype=torch.float32)),
        batch_size=batch_size
    )
    return train_loader, val_loader, test_loader, y_te_seq


def get_dl_predictions(model, test_loader, threshold=0.5):
    """Modelden logit çıktıları alır, sigmoid uygular ve eşikle sınıflandırır."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch_x = batch[0]
            output = torch.sigmoid(model(batch_x)).squeeze(-1)
            if output.ndim == 0:
                preds = [1 if output.item() > threshold else 0]
            else:
                preds = (output > threshold).int().numpy().tolist()
            predictions.extend(preds)
    return np.array(predictions)


def to_log_metrics(m):
    """calculate_metrics() çıktısını ExperimentLogger formatına çevirir."""
    return {
        "f1": m["F1_Score"],
        "accuracy": m["Accuracy"],
        "precision": m["Precision"],
        "recall": m["Recall"],
    }


def get_prediction_threshold(config, dataset_name):
    """Veri setine özgü tahmin eşiğini döndürür; yoksa 'default' değeri kullanılır."""
    thresholds = config['training'].get('prediction_thresholds', {})
    return thresholds.get(dataset_name, thresholds.get('default', 0.5))


def get_label_col(dataset_name, config):
    """Veri setine göre etiket sütunu adını döndürür."""
    mapping = {
        "SKAB":    config['datasets']['skab_label'],
        "BATADAL": config['datasets']['batadal_label'],
        "SWAT":    config['datasets'].get('swat_label', 'Normal/Attack'),
        "WADI":    config['datasets'].get('wadi_label', 'attack_label'),
    }
    if dataset_name not in mapping:
        raise ValueError(f"Bilinmeyen veri seti etiket sütunu: {dataset_name}")
    return mapping[dataset_name]


def run_dataset_pipeline(dataset_name, config, logger):
    """Bir veri seti için tam pipeline çalıştırır (tüm modeller, tüm seed'ler, tüm senaryolar)."""
    print(f"\n{'='*55}")
    print(f"[Veri Seti] {dataset_name} işleniyor...")
    print(f"{'='*55}")

    label_col = get_label_col(dataset_name, config)
    df = load_dataset(dataset_name, config['paths'])

    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_splits(
        dataset_name, df, config['split_ratios'], label_col
    )

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Veri ön işleme: scaler ve PCA sadece eğitim verisinde fit edilir (data leakage yok)
    preprocessor = DataPreprocessor()
    X_train_scaled, X_train_pca = preprocessor.fit_transform(X_train)
    X_val_scaled, X_val_pca = preprocessor.transform(X_val)
    X_test_scaled, X_test_pca = preprocessor.transform(X_test)

    threshold = get_prediction_threshold(config, dataset_name)
    win_size = config['automata']['window_size']
    cnn_win_size = 2  # CNN1D daha kısa dizi uzunluğuyla çalışır (hız optimizasyonu)
    input_dim = X_train_scaled.shape[1]
    seeds = config['training']['random_seeds']

    # Seed bazlı F1 skorlarını ve tahminleri istatistiksel testler için sakla
    seed_f1 = {name: [] for name in ["LSTM", "GRU", "CNN1D", "Automata"]}
    last_preds = {}   # Son seed tahminleri (McNemar için)
    last_y_true = None

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        set_seed(seed)

        # LSTM/GRU için dataloaderlar (win_size=4)
        train_loader, val_loader, test_loader, y_test_seq = create_dataloaders(
            X_train_scaled, y_train.values,
            X_val_scaled, y_val.values,
            X_test_scaled, y_test.values,
            config['training']['batch_size'], win_size
        )

        # Gürültülü ve unseen test verisi (DL için scaled)
        X_test_noisy = add_gaussian_noise(
            X_test_scaled, config['noise']['mean'], config['noise']['std_dev']
        )
        X_test_unseen = simulate_unseen_data(X_test_scaled)

        # Her senaryo için LSTM/GRU test loader'larını hazırla
        noisy_test_loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    create_sequences(X_test_noisy, y_test.values, win_size)[0],
                    dtype=torch.float32
                ),
                torch.tensor(y_test_seq, dtype=torch.float32)
            ), batch_size=config['training']['batch_size']
        )
        unseen_test_loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    create_sequences(X_test_unseen, y_test.values, win_size)[0],
                    dtype=torch.float32
                ),
                torch.tensor(y_test_seq, dtype=torch.float32)
            ), batch_size=config['training']['batch_size']
        )

        scenario_loaders = {
            "original": test_loader,
            "noisy": noisy_test_loader,
            "unseen": unseen_test_loader,
        }

        # CNN1D için ayrı loaderlar: daha kısa dizi (cnn_win_size=2)
        cnn_train_loader, cnn_val_loader, cnn_test_loader, cnn_y_test_seq = create_dataloaders(
            X_train_scaled, y_train.values,
            X_val_scaled, y_val.values,
            X_test_scaled, y_test.values,
            config['training']['batch_size'], cnn_win_size
        )
        cnn_noisy_test_loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    create_sequences(X_test_noisy, y_test.values, cnn_win_size)[0],
                    dtype=torch.float32
                ),
                torch.tensor(cnn_y_test_seq, dtype=torch.float32)
            ), batch_size=config['training']['batch_size']
        )
        cnn_unseen_test_loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    create_sequences(X_test_unseen, y_test.values, cnn_win_size)[0],
                    dtype=torch.float32
                ),
                torch.tensor(cnn_y_test_seq, dtype=torch.float32)
            ), batch_size=config['training']['batch_size']
        )
        cnn_scenario_loaders = {
            "original": cnn_test_loader,
            "noisy": cnn_noisy_test_loader,
            "unseen": cnn_unseen_test_loader,
        }

        # DL modelleri: LSTM, GRU, CNN1D
        dl_models = [
            ("LSTM", LSTMClassifier(input_dim=input_dim)),
            ("GRU", GRUClassifier(input_dim=input_dim)),
            ("CNN1D", CNN1DClassifier(input_dim=input_dim, seq_len=cnn_win_size)),
        ]

        for model_name, model in dl_models:
            # CNN1D kendi kısa pencere loaderlarını kullanır
            if model_name == "CNN1D":
                tr_loader, vl_loader = cnn_train_loader, cnn_val_loader
                sc_loaders = cnn_scenario_loaders
                y_seq = cnn_y_test_seq
            else:
                tr_loader, vl_loader = train_loader, val_loader
                sc_loaders = scenario_loaders
                y_seq = y_test_seq

            t0 = time.time()
            trained_model = train_deep_learning(model, tr_loader, vl_loader, config['training'], y_train=y_train.values)
            train_time = time.time() - t0

            for scenario, sc_loader in sc_loaders.items():
                t1 = time.time()
                y_pred = get_dl_predictions(trained_model, sc_loader, threshold=threshold)
                infer_time = time.time() - t1

                metrics, cm = calculate_metrics(y_seq, y_pred)
                logger.log(model_name, dataset_name, seed, scenario,
                           to_log_metrics(metrics),
                           train_time=train_time, infer_time=infer_time)

                if scenario == "original":
                    seed_f1[model_name].append(metrics["F1_Score"])
                    if seed == seeds[-1]:
                        last_preds[model_name] = y_pred
                        if model_name != "CNN1D":
                            last_y_true = y_test_seq

        # Otomata modeli
        automata = ProbabilisticAutomata(
            window_size=win_size,
            alphabet_size=config['automata']['alphabet_size']
        )
        t0 = time.time()
        automata.fit(X_train_pca)
        train_time = time.time() - t0
        automata.set_threshold(X_val_pca, threshold_override=threshold)

        known_patterns = list(automata.transition_matrix.keys())
        unseen_handler = UnseenHandler(known_patterns)

        # Otomata için PCA test verileri (1D)
        test_1d = X_test_pca.flatten()
        test_noisy_1d = add_gaussian_noise(
            X_test_pca, config['noise']['mean'], config['noise']['std_dev']
        ).flatten()
        test_unseen_1d = simulate_unseen_data(X_test_pca).flatten()

        auto_scenarios = {
            "original": test_1d,
            "noisy": test_noisy_1d,
            "unseen": test_unseen_1d,
        }

        for scenario, test_data_1d in auto_scenarios.items():
            t1 = time.time()
            y_pred_auto = automata.predict(test_data_1d)
            infer_time = time.time() - t1

            min_len = min(len(y_pred_auto), len(y_test_seq))
            if min_len > 0:
                metrics, _ = calculate_metrics(y_test_seq[:min_len], y_pred_auto[:min_len])
            else:
                metrics = {"F1_Score": 0.0, "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0}

            extra = None
            if scenario == "unseen":
                extra = {"unseen_rate": round(unseen_handler.unseen_rate(
                    automata.extract_patterns(test_data_1d)
                ), 4)}

            logger.log("Automata", dataset_name, seed, scenario,
                       to_log_metrics(metrics),
                       train_time=train_time, infer_time=infer_time, extra=extra)

            if scenario == "original":
                seed_f1["Automata"].append(metrics["F1_Score"])
                if seed == seeds[-1]:
                    last_preds["Automata"] = y_pred_auto[:min_len]

    # İstatistiksel testler (seed'ler arası F1 karşılaştırması)
    print(f"\n  [İstatistiksel Testler] {dataset_name}")
    model_pairs = [
        ("LSTM", "GRU"),
        ("LSTM", "Automata"),
        ("GRU", "Automata"),
        ("LSTM", "CNN1D"),
    ]
    for m_a, m_b in model_pairs:
        if seed_f1[m_a] and seed_f1[m_b]:
            w_res = wilcoxon_test(seed_f1[m_a], seed_f1[m_b])
            print(f"    Wilcoxon {m_a} vs {m_b}: stat={w_res['statistic']:.4f}, p={w_res['p_value']:.4f}")
            logger.log("Wilcoxon_" + m_a + "_vs_" + m_b, dataset_name, "all", "stat_test",
                       {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0},
                       extra={"statistic": w_res["statistic"], "p_value": w_res["p_value"]})

    if last_y_true is not None:
        for m_a, m_b in [("LSTM", "GRU"), ("LSTM", "Automata")]:
            if m_a in last_preds and m_b in last_preds:
                min_len = min(len(last_y_true), len(last_preds[m_a]), len(last_preds[m_b]))
                mc_res = mcnemar_test(
                    last_y_true[:min_len],
                    last_preds[m_a][:min_len],
                    last_preds[m_b][:min_len],
                )
                print(f"    McNemar {m_a} vs {m_b}: stat={mc_res['statistic']:.4f}, p={mc_res['p_value']:.4f}")
                logger.log("McNemar_" + m_a + "_vs_" + m_b, dataset_name, seeds[-1], "stat_test",
                           {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0},
                           extra={"statistic": mc_res["statistic"], "p_value": mc_res["p_value"]})

    # Son seed'den açıklanabilirlik raporu
    automata_last = ProbabilisticAutomata(win_size, config['automata']['alphabet_size'])
    automata_last.fit(X_train_pca)
    automata_last.set_threshold(X_val_pca, threshold_override=threshold)
    uh_last = UnseenHandler(list(automata_last.transition_matrix.keys()))
    explainer = AutomataExplainer(automata_last, uh_last)
    test_states = automata_last.extract_patterns(X_test_pca.flatten()[:100])
    report_df = explainer.explain_sequence(test_states)
    os.makedirs("outputs/results", exist_ok=True)
    report_df.to_csv(
        f"outputs/results/automata_explanation_{dataset_name}.csv", index=False
    )

    # Parametre duyarlılık grafikleri
    from src.utils.visualize import plot_parameter_sensitivity
    plot_parameter_sensitivity(
        window_sizes=config['automata']['experiment_window_sizes'],
        alphabet_sizes=config['automata']['experiment_alphabet_sizes'],
        data_1d=X_train_pca.flatten(),
        save_path=f"outputs/figures/param_sensitivity_{dataset_name}.png"
    )

    print(f"\n  [OK] {dataset_name} pipeline tamamlandı.")


def run_cross_dataset_evaluation(config, logger):
    """
    Çapraz veri seti değerlendirmesi.
    Boyut uyumsuzluğunu aşmak için her veri seti PCA ile 1D'ye indirgenir.
    Train on source (1D), test on target (1D).
    """
    datasets = config['active_datasets']
    if len(datasets) < 2:
        print("[Uyarı] Çapraz değerlendirme için en az 2 veri seti gereklidir.")
        return

    print(f"\n{'='*55}")
    print("[Çapraz Veri Seti] Cross-dataset Değerlendirmesi")
    print(f"{'='*55}")

    seed = config['training']['random_seeds'][0]
    set_seed(seed)
    win_size = config['automata']['window_size']

    for src_name, tgt_name in [(datasets[0], datasets[1]), (datasets[1], datasets[0])]:
        cross_label = f"{src_name}->{tgt_name}"
        print(f"\n  {cross_label} değerlendiriliyor...")

        # Kaynak veri yükleme
        src_label = get_label_col(src_name, config)
        src_df = load_dataset(src_name, config['paths'])
        X_src_train, y_src_train, X_src_val, y_src_val, _, _ = get_train_val_test_splits(
            src_name, src_df, config['split_ratios'], src_label
        )
        src_prep = DataPreprocessor()
        _, X_src_train_pca = src_prep.fit_transform(X_src_train)
        _, X_src_val_pca = src_prep.transform(X_src_val)

        # Hedef veri yükleme (kendi preprocessor'ı ile 1D'ye indir)
        tgt_label = get_label_col(tgt_name, config)
        tgt_df = load_dataset(tgt_name, config['paths'])
        X_tgt_train, y_tgt_train, _, _, X_tgt_test, y_tgt_test = get_train_val_test_splits(
            tgt_name, tgt_df, config['split_ratios'], tgt_label
        )
        tgt_prep = DataPreprocessor()
        _, X_tgt_train_pca = tgt_prep.fit_transform(X_tgt_train)
        _, X_tgt_test_pca = tgt_prep.transform(X_tgt_test)

        # Hedef test için hizalanmış etiketler (DL create_sequences ile aynı hizalama)
        y_tgt_seq = y_tgt_test.values[win_size:]

        # --- Otomata ---
        automata = ProbabilisticAutomata(win_size, config['automata']['alphabet_size'])
        t0 = time.time()
        automata.fit(X_src_train_pca.flatten())
        train_time = time.time() - t0
        automata.set_threshold(X_src_val_pca.flatten(), threshold_override=get_prediction_threshold(config, src_name))

        t1 = time.time()
        y_pred_auto = automata.predict(X_tgt_test_pca.flatten())
        infer_time = time.time() - t1

        min_len = min(len(y_pred_auto), len(y_tgt_seq))
        if min_len > 0:
            metrics, _ = calculate_metrics(y_tgt_seq[:min_len], y_pred_auto[:min_len])
            logger.log("Automata", cross_label, seed, "cross_dataset",
                       to_log_metrics(metrics),
                       train_time=train_time, infer_time=infer_time)

        # --- DL modeller (1D PCA girişiyle) ---
        X_src_train_pca_seq, y_src_train_seq = create_sequences(
            X_src_train_pca, y_src_train.values, win_size
        )
        X_src_val_pca_seq, y_src_val_seq = create_sequences(
            X_src_val_pca, y_src_val.values, win_size
        )
        X_tgt_test_pca_seq, _ = create_sequences(
            X_tgt_test_pca, np.zeros(len(X_tgt_test_pca)), win_size
        )

        if len(X_src_train_pca_seq) == 0 or len(X_tgt_test_pca_seq) == 0:
            print(f"  [Uyarı] {cross_label}: Yeterli veri yok, DL atlanıyor.")
            continue

        src_train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_src_train_pca_seq, dtype=torch.float32),
                torch.tensor(y_src_train_seq, dtype=torch.float32)
            ), batch_size=config['training']['batch_size'], shuffle=True
        )
        src_val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_src_val_pca_seq, dtype=torch.float32),
                torch.tensor(y_src_val_seq, dtype=torch.float32)
            ), batch_size=config['training']['batch_size']
        )
        tgt_test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_tgt_test_pca_seq, dtype=torch.float32),
                torch.tensor(np.zeros(len(X_tgt_test_pca_seq)), dtype=torch.float32)
            ), batch_size=config['training']['batch_size']
        )

        dl_models = [
            ("LSTM", LSTMClassifier(input_dim=1)),
            ("GRU", GRUClassifier(input_dim=1)),
            ("CNN1D", CNN1DClassifier(input_dim=1, seq_len=win_size)),
        ]

        for model_name, model in dl_models:
            t0 = time.time()
            trained = train_deep_learning(model, src_train_loader, src_val_loader, config['training'], y_train=y_src_train.values)
            train_time = time.time() - t0

            t1 = time.time()
            y_pred_dl = get_dl_predictions(trained, tgt_test_loader, threshold=get_prediction_threshold(config, src_name))
            infer_time = time.time() - t1

            min_len = min(len(y_pred_dl), len(y_tgt_seq))
            if min_len > 0:
                metrics, _ = calculate_metrics(y_tgt_seq[:min_len], y_pred_dl[:min_len])
                logger.log(model_name, cross_label, seed, "cross_dataset",
                           to_log_metrics(metrics),
                           train_time=train_time, infer_time=infer_time)

    print("\n  [OK] Çapraz veri seti değerlendirmesi tamamlandı.")


def apply_smoke_test_overrides(config):
    """
    --smoke-test modu: yalnızca 1 seed, 1 epoch, 1 veri seti (son aktif veri seti).
    Çapraz veri seti değerlendirmesi de atlanır.
    Hızlı doğrulama için idealdir (~2 dakika).
    """
    config = dict(config)
    config['training'] = dict(config['training'])
    config['training']['random_seeds'] = [config['training']['random_seeds'][0]]
    config['training']['max_epochs'] = 1
    config['training']['early_stopping_patience'] = 1
    # Sadece son veri setini çalıştır (genellikle WADI gibi etiketli olan)
    config['active_datasets'] = [config['active_datasets'][-1]]
    config['_smoke_test'] = True
    print("[SMOKE-TEST] 1 seed, 1 epoch, sadece:", config['active_datasets'])
    return config


def main():
    smoke_test = "--smoke-test" in sys.argv

    print("=" * 55)
    if smoke_test:
        print("Zaman Serisi Anomali Tespiti - SMOKE TEST Modu")
    else:
        print("Zaman Serisi Anomali Tespiti - Pipeline Başlıyor")
    print("=" * 55)

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if smoke_test:
        config = apply_smoke_test_overrides(config)

    logger = ExperimentLogger(log_dir=config['paths']['logs_dir'])

    # Her veri seti için tam pipeline çalıştır
    for dataset_name in config['active_datasets']:
        run_dataset_pipeline(dataset_name, config, logger)

    # Çapraz veri seti değerlendirmesi (smoke-test modunda atlanır)
    if not config.get('_smoke_test'):
        run_cross_dataset_evaluation(config, logger)

    # Sonuçları kaydet ve özetle
    logger.save()
    logger.print_summary()

    print("\n" + "=" * 55)
    print("EĞİTİM VE TEST SÜRECİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 55)


if __name__ == "__main__":
    main()
