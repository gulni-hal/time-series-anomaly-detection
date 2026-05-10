import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.data_loader import load_config, prepare_swat, prepare_wadi, add_gaussian_noise
from src.models.automata import ProbabilisticAutomata, extract_patterns
from src.models.deep_learning import get_model, make_sequences, train_model, evaluate_model, find_best_threshold
from src.utils.metrics import compute_metrics
from src.utils.logger import ExperimentLogger


def get_pos_weight(y_train):
    n_normal  = int((y_train == 0).sum())
    n_anomaly = int((y_train == 1).sum())
    if n_anomaly == 0:
        return 1.0
    w = float(n_normal) / float(n_anomaly)
    print("    Class weight: " + str(round(w, 2)))
    return w


def run_automata(data, scenario, seed, ds_name, cfg, logger):
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
    model.set_threshold(X_val, percentile=10.0)

    preds, scores, explanations = model.predict_with_scores(X_test)
    min_len = min(len(preds), len(y_test))
    metrics = compute_metrics(y_test[:min_len], preds[:min_len])

    extra = {}
    if scenario == "unseen":
        test_patterns = extract_patterns(X_test.flatten(), ws, ab)
        unseen = sum(1 for p in test_patterns if p not in model.vocabulary)
        extra["unseen_rate"] = round(unseen / max(len(test_patterns), 1), 4)

    logger.log("Automata", ds_name, seed, scenario, metrics,
               model.train_time, model.infer_time, extra)
    return model


def run_dl(model_name, data, scenario, seed, ds_name, cfg, logger):
    np.random.seed(seed)
    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"].copy()

    if scenario == "noisy":
        X_test = add_gaussian_noise(X_test, cfg, seed)

    Xs_tr, ys_tr = make_sequences(X_train, data["y_train"])
    Xs_vl, ys_vl = make_sequences(X_val,   data["y_val"])
    Xs_te, ys_te = make_sequences(X_test,  data["y_test"])

    pos_weight = get_pos_weight(data["y_train"])
    model = get_model(model_name, X_train.shape[1])
    model, train_time = train_model(model, (Xs_tr, ys_tr), (Xs_vl, ys_vl), cfg, seed, pos_weight)

    best_t = find_best_threshold(model, (Xs_vl, ys_vl), ys_vl)

    preds, y_true, infer_time = evaluate_model(model, (Xs_te, ys_te))
    print("    Test pred min=" + str(round(float(preds.min()), 4)) +
          " max=" + str(round(float(preds.max()), 4)))

    metrics = compute_metrics(y_true, preds, threshold=best_t)
    logger.log(model_name, ds_name, seed, scenario, metrics, train_time, infer_time)


def main():
    cfg    = load_config("configs/default_config.yaml")
    logger = ExperimentLogger(cfg["paths"]["logs_dir"])
    seeds  = cfg["training"]["random_seeds"]

    datasets = {"SWAT": prepare_swat, "WADI": prepare_wadi}
    dl_models = ["LSTM", "GRU", "CNN1D"]
    scenarios = ["original", "noisy"]

    for ds_name, prepare_fn in datasets.items():
        data = prepare_fn(cfg)
        for seed in seeds:
            print("\n  -> " + ds_name + " | Seed: " + str(seed))
            for model_name in dl_models:
                for scenario in scenarios:
                    print("    [" + model_name + "] " + scenario + "...")
                    run_dl(model_name, data, scenario, seed, ds_name, cfg, logger)
            for scenario in scenarios + ["unseen"]:
                print("    [Automata] " + scenario + "...")
                run_automata(data, scenario, seed, ds_name, cfg, logger)

    logger.save()
    logger.print_summary()
    print("\n[OK] Done!")


if __name__ == "__main__":
    main()