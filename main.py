"""
Ana Deney Koşucusu
Çalıştır: py main.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.data_loader import load_config, prepare_swat, prepare_wadi, add_gaussian_noise
from src.models.automata import ProbabilisticAutomata, extract_patterns
from src.models.deep_learning import get_model, make_sequences, to_loader, train_model, evaluate_model
from src.utils.metrics import compute_metrics
from src.utils.logger import ExperimentLogger
from src.explainability.explainer import AutomataExplainer


def run_automata(data: dict, scenario: str, seed: int, ds_name: str,
                 cfg: dict, logger: ExperimentLogger):
    np.random.seed(seed)
    X_train = data["X_train_pca"]
    X_val   = data["X_val_pca"]
    X_test  = data["X_test_pca"].copy()
    y_test  = data["y_test"]

    ws = cfg["automata"]["window_size"]
    ab = cfg["automata"]["alphabet_size"]

    if scenario == "noisy":
        X_test = add_gaussian_noise(X_test, cfg, seed)

    model = ProbabilisticAutomata(ws, ab)
    model.fit(X_train)
    model.set_threshold(X_val)

    preds, scores, explanations = model.predict_with_scores(X_test)

    # Boyut eşitle
    min_len = min(len(preds), len(y_test))
    metrics = compute_metrics(y_test[:min_len], preds[:min_len])

    extra = {}
    if scenario == "unseen":
        test_patterns = extract_patterns(X_test.flatten(), ws, ab)
        unseen = sum(1 for p in test_patterns if p not in model.vocabulary)
        extra["unseen_rate"] = round(unseen / max(len(test_patterns), 1), 4)

    logger.log("Automata", ds_name, seed, scenario, metrics,
               model.train_time, model.infer_time, extra)

    # İlk seed'de açıklama örneği göster
    if seed == cfg["training"]["random_seeds"][0] and scenario == "original":
        explainer = AutomataExplainer(model)
        exp = explainer.explain_window(X_test[:model.window_size*2])
        explainer.print_explanation(exp)

    return model


def run_dl(model_name: str, data: dict, scenario: str, seed: int,
           ds_name: str, cfg: dict, logger: ExperimentLogger):
    np.random.seed(seed)
    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"].copy()

    if scenario == "noisy":
        X_test = add_gaussian_noise(X_test, cfg, seed)

    Xs_tr, ys_tr = make_sequences(X_train, data["y_train"])
    Xs_vl, ys_vl = make_sequences(X_val,   data["y_val"])
    Xs_te, ys_te = make_sequences(X_test,  data["y_test"])

    bs   = cfg["training"]["batch_size"]
    tr_l = to_loader(Xs_tr, ys_tr, bs)
    vl_l = to_loader(Xs_vl, ys_vl, bs, shuffle=False)
    te_l = to_loader(Xs_te, ys_te, bs, shuffle=False)

    model = get_model(model_name, X_train.shape[1])
    model, train_time = train_model(model, tr_l, vl_l, cfg, seed)
    preds, y_true, infer_time = evaluate_model(model, te_l)

    metrics = compute_metrics(y_true, preds)
    logger.log(model_name, ds_name, seed, scenario, metrics, train_time, infer_time)


def main():
    cfg    = load_config("configs/default_config.yaml")
    logger = ExperimentLogger(cfg["paths"]["logs_dir"])
    seeds  = cfg["training"]["random_seeds"]

    datasets = {
        "SWAT": prepare_swat,
        "WADI": prepare_wadi,
    }

    dl_models = ["LSTM", "GRU", "CNN1D"]
    scenarios = ["original", "noisy"]

    for ds_name, prepare_fn in datasets.items():
        data = prepare_fn(cfg)

        for seed in seeds:
            print(f"\n  → {ds_name} | Seed: {seed}")

            # DL modelleri
            for model_name in dl_models:
                for scenario in scenarios:
                    print(f"    [{model_name}] {scenario}...")
                    run_dl(model_name, data, scenario, seed, ds_name, cfg, logger)

            # Automata (original + noisy + unseen)
            for scenario in scenarios + ["unseen"]:
                print(f"    [Automata] {scenario}...")
                run_automata(data, scenario, seed, ds_name, cfg, logger)

    logger.save()
    logger.print_summary()
    print("\n[✓] Tüm deneyler tamamlandı!")


if __name__ == "__main__":
    main()
