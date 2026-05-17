import numpy as np
import os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.pipeline.data_loader import load_config, prepare_wadi
from src.models.automata import ProbabilisticAutomata
from src.models.deep_learning import (get_model, make_sequences, train_model,
                                       evaluate_model, find_best_threshold)
from src.utils.metrics import compute_metrics
from src.utils.logger import ExperimentLogger
from src.utils.visualize import (plot_confusion_matrix, plot_roc_curve,
                                  plot_automata_state_diagram,
                                  plot_transition_heatmap, plot_param_sensitivity)


def get_pos_weight(y_train):
    n_normal  = int((y_train == 0).sum())
    n_anomaly = int((y_train == 1).sum())
    if n_anomaly == 0:
        return 1.0
    w = min(float(n_normal) / float(n_anomaly), 10.0)
    return w


# ── K-Fold ────────────────────────────────────────────────────────────────────

def run_kfold_wadi(cfg, logger, seed=42, k=5):
    print("\n" + "=" * 60)
    print("K-FOLD CV -- WADI, k=" + str(k) + ", seed=" + str(seed))
    print("=" * 60)

    from src.pipeline.data_loader import prepare_wadi
    data = prepare_wadi(cfg)

    ws = cfg["automata"]["window_size"]
    ab = cfg["automata"]["alphabet_size"]
    dl_models = ["LSTM", "GRU", "CNN1D"]

    # Reassemble in stratified order (train→val→test)
    X_all = np.vstack([data["X_train"], data["X_val"], data["X_test"]])
    y_all = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])

    tscv = TimeSeriesSplit(n_splits=k)
    fold_scores = {"Automata": []}
    for m in dl_models:
        fold_scores[m] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X_all)):
        print("\n  Fold " + str(fold_idx + 1) + "/" + str(k) +
              "  (train=" + str(len(tr_idx)) + ", test=" + str(len(te_idx)) + ")")

        X_tr_raw = X_all[tr_idx];  y_tr_raw = y_all[tr_idx]
        X_te_raw = X_all[te_idx];  y_te_raw = y_all[te_idx]

        # Hold out last 20% of train as val
        split = int(len(X_tr_raw) * 0.8)
        X_tr2, y_tr2 = X_tr_raw[:split], y_tr_raw[:split]
        X_vl2, y_vl2 = X_tr_raw[split:], y_tr_raw[split:]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr2)
        X_vl_sc = scaler.transform(X_vl2)
        X_te_sc = scaler.transform(X_te_raw)

        pca = PCA(n_components=1)
        X_tr_pca = pca.fit_transform(X_tr_sc)
        X_vl_pca = pca.transform(X_vl_sc)
        X_te_pca = pca.transform(X_te_sc)

        # Automata
        aut = ProbabilisticAutomata(ws, ab)
        aut.fit(X_tr_pca)
        aut.set_threshold_f1(X_vl_pca, y_vl2)
        preds_a, _, _ = aut.predict_with_scores(X_te_pca)
        ml = min(len(preds_a), len(y_te_raw))
        m_a = compute_metrics(y_te_raw[:ml], preds_a[:ml])
        fold_scores["Automata"].append(m_a["f1"])
        logger.log("Automata", "WADI", seed, "kfold_f" + str(fold_idx + 1),
                   m_a, aut.train_time, aut.infer_time,
                   {"fold": fold_idx + 1, "n_test_anomaly": int(y_te_raw.sum())})
        print("    Automata F1=" + str(m_a["f1"]))

        # DL models
        for model_name in dl_models:
            np.random.seed(seed)
            Xs_tr, ys_tr = make_sequences(X_tr_sc, y_tr2)
            Xs_vl, ys_vl = make_sequences(X_vl_sc, y_vl2)
            Xs_te, ys_te = make_sequences(X_te_sc, y_te_raw)

            if len(Xs_tr) == 0 or len(Xs_te) == 0:
                fold_scores[model_name].append(0.0)
                continue

            pos_w = get_pos_weight(y_tr2)
            mdl = get_model(model_name, X_tr_sc.shape[1])
            mdl, tr_t = train_model(mdl, (Xs_tr, ys_tr), (Xs_vl, ys_vl), cfg, seed, pos_w)
            best_t = find_best_threshold(mdl, (Xs_vl, ys_vl), y_vl2[-len(ys_vl):])
            preds_dl, y_true_dl, inf_t = evaluate_model(mdl, (Xs_te, ys_te))
            m_dl = compute_metrics(y_true_dl, preds_dl, threshold=best_t)
            fold_scores[model_name].append(m_dl["f1"])
            logger.log(model_name, "WADI", seed, "kfold_f" + str(fold_idx + 1),
                       m_dl, tr_t, inf_t,
                       {"fold": fold_idx + 1, "n_test_anomaly": int(y_te_raw.sum())})
            print("    " + model_name + " F1=" + str(m_dl["f1"]))

    print("\n  K-Fold summary (mean +/- std across " + str(k) + " folds):")
    for model_name, scores in fold_scores.items():
        if scores:
            print("    " + model_name + ": " +
                  str(round(np.mean(scores), 4)) + " +/- " +
                  str(round(np.std(scores), 4)) +
                  "  " + str([round(s, 3) for s in scores]))

    return fold_scores


# ── Wilcoxon ──────────────────────────────────────────────────────────────────

def run_wilcoxon(fold_scores):
    from scipy.stats import wilcoxon

    print("\n" + "=" * 60)
    print("WILCOXON SIGNED-RANK TEST (pairwise, alpha=0.05)")
    print("=" * 60)

    models = list(fold_scores.keys())
    any_printed = False
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            s1 = np.array(fold_scores[m1])
            s2 = np.array(fold_scores[m2])
            ml = min(len(s1), len(s2))
            if ml < 2:
                continue
            diff = s1[:ml] - s2[:ml]
            if np.all(diff == 0):
                print("  " + m1 + " vs " + m2 + ": all equal -- skip")
                continue
            try:
                stat, p = wilcoxon(s1[:ml], s2[:ml])
                sig = "  ** SIGNIFICANT **" if p < 0.05 else ""
                print("  " + m1 + " vs " + m2 +
                      ":  stat=" + str(round(stat, 3)) +
                      "  p=" + str(round(p, 4)) + sig)
                any_printed = True
            except Exception as e:
                print("  " + m1 + " vs " + m2 + ": " + str(e))
    if not any_printed:
        print("  (no valid pairs to compare)")


# ── Visualizations ────────────────────────────────────────────────────────────

def run_visualizations(cfg, out_dir, seed=42):
    print("\n" + "=" * 60)
    print("VISUALIZATIONS -- WADI original, seed=" + str(seed))
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)
    dl_models = ["LSTM", "GRU", "CNN1D"]

    print("\nLoading WADI...")
    data = prepare_wadi(cfg)

    ws = cfg["automata"]["window_size"]
    ab = cfg["automata"]["alphabet_size"]
    ws_list = cfg["automata"]["experiment_window_sizes"]
    ab_list = cfg["automata"]["experiment_alphabet_sizes"]

    # ── DL: CM + ROC ──────────────────────────────────────────────────────────
    for model_name in dl_models:
        print("\n  [" + model_name + "] fitting for viz...")
        np.random.seed(seed)
        Xs_tr, ys_tr = make_sequences(data["X_train"], data["y_train"])
        Xs_vl, ys_vl = make_sequences(data["X_val"],   data["y_val"])
        Xs_te, ys_te = make_sequences(data["X_test"],  data["y_test"])

        pos_w = get_pos_weight(data["y_train"])
        mdl   = get_model(model_name, data["X_train"].shape[1])
        mdl, _ = train_model(mdl, (Xs_tr, ys_tr), (Xs_vl, ys_vl), cfg, seed, pos_w)
        best_t = find_best_threshold(mdl, (Xs_vl, ys_vl),
                                     data["y_val"][-len(ys_vl):])
        scores, y_true, _ = evaluate_model(mdl, (Xs_te, ys_te))

        y_bin = (scores > best_t).astype(int)
        plot_confusion_matrix(y_true, y_bin, model_name, "WADI", out_dir)
        plot_roc_curve(y_true, scores, model_name, "WADI", out_dir)

    # ── Automata: CM + ROC + state diagram + heatmap ──────────────────────────
    print("\n  [Automata] fitting for viz...")
    np.random.seed(seed)
    aut = ProbabilisticAutomata(ws, ab)
    aut.fit(data["X_train_pca"])
    aut.set_threshold_f1(data["X_val_pca"], data["y_val"])

    preds_a, scores_a, _ = aut.predict_with_scores(data["X_test_pca"])
    y_te = data["y_test"]
    ml   = min(len(preds_a), len(y_te))

    plot_confusion_matrix(y_te[:ml], preds_a[:ml], "Automata", "WADI", out_dir)
    # Negate scores: lower log-prob = more anomalous → negate so higher = anomaly score
    plot_roc_curve(y_te[:ml], -scores_a[:ml], "Automata", "WADI", out_dir)
    plot_automata_state_diagram(aut.transition_probs, out_dir, "WADI")
    plot_transition_heatmap(aut.transition_probs, out_dir, "WADI")

    # ── Parameter sensitivity ─────────────────────────────────────────────────
    print("\n  [Param sweep] building sensitivity data for WADI...")
    sweep_records = []
    for ws_i in ws_list:
        for ab_i in ab_list:
            np.random.seed(seed)
            m = ProbabilisticAutomata(ws_i, ab_i)
            m.fit(data["X_train_pca"])
            m.set_threshold_f1(data["X_val_pca"], data["y_val"])
            p, _, _ = m.predict_with_scores(data["X_test_pca"])
            ml2 = min(len(p), len(data["y_test"]))
            met = compute_metrics(data["y_test"][:ml2], p[:ml2])
            sweep_records.append({
                "dataset":      "WADI",
                "window_size":  ws_i,
                "alphabet_size": ab_i,
                "f1":           met["f1"],
            })
    plot_param_sensitivity(sweep_records, save_dir=out_dir)

    print("\n  All plots saved to: " + out_dir)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg     = load_config("configs/default_config.yaml")
    out_dir = cfg["paths"]["outputs_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # K-Fold
    logger_kf = ExperimentLogger(cfg["paths"]["logs_dir"])
    logger_kf.run_id = logger_kf.run_id + "_kfold"
    fold_scores = run_kfold_wadi(cfg, logger_kf, seed=42, k=5)
    logger_kf.save()

    # Wilcoxon
    run_wilcoxon(fold_scores)

    # Visualizations
    run_visualizations(cfg, out_dir, seed=42)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
