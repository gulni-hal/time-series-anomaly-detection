import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os, yaml, torch

torch.set_num_threads(1)

from src.pipeline.data_loader import load_dataset, get_train_val_test_splits
from src.preprocessing.pipeline import DataPreprocessor
from src.pipeline.evaluate import set_seed
from src.models.deep_learning import LSTMClassifier
from src.models.automata import ProbabilisticAutomata
from src.pipeline.train import train_deep_learning
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

os.makedirs("outputs", exist_ok=True)

NAVY    = "#003366"
PALETTE = ["#003366", "#1a5276", "#2980b9", "#85c1e9"]
sns.set_style("whitegrid")
DPI = 150

df = pd.read_csv("logs/results_20260601_113958.csv")
models    = ["LSTM", "GRU", "CNN1D", "Automata"]
scenarios = ["original", "noisy", "unseen"]

with open("configs/default_config.yaml") as f:
    config = yaml.safe_load(f)

# ── 1. F1 Grouped Bar Charts ──────────────────────────────────────
def f1_bar(dataset):
    sub = df[
        (df.dataset == dataset) &
        (df.scenario.isin(scenarios)) &
        (df.model.isin(models))
    ]
    pivot_mean = sub.groupby(["model","scenario"])["f1"].mean().unstack()[scenarios].reindex(models)
    pivot_std  = sub.groupby(["model","scenario"])["f1"].std().fillna(0).unstack()[scenarios].reindex(models)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    x = np.arange(len(models))
    width  = 0.22
    colors = ["#003366", "#2980b9", "#85c1e9"]
    labels = ["Original", "Noisy", "Unseen"]
    for i, (scen, col, lab) in enumerate(zip(scenarios, colors, labels)):
        ax.bar(x + (i-1)*width, pivot_mean[scen], width,
               label=lab, color=col,
               yerr=pivot_std[scen], capsize=4,
               error_kw=dict(ecolor="#555", lw=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Model F1 Comparison — {dataset}", fontsize=13,
                 fontweight="bold", color=NAVY)
    ax.legend(title="Scenario", fontsize=9)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    path = f"outputs/f1_comparison_{dataset}.png"
    plt.savefig(path, dpi=DPI, facecolor="white")
    plt.close()
    print(f"  saved {path}")

f1_bar("SKAB")
f1_bar("BATADAL")

# ── 2. Noise Robustness Line Charts ──────────────────────────────
def robustness_line(dataset):
    sub = df[
        (df.dataset == dataset) &
        (df.scenario.isin(scenarios)) &
        (df.model.isin(models))
    ]
    pivot = sub.groupby(["model","scenario"])["f1"].mean().unstack()[scenarios].reindex(models)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    x_labels = ["Original", "Noisy", "Unseen"]
    for model, color in zip(models, PALETTE):
        ax.plot(x_labels, pivot.loc[model].values,
                marker="o", linewidth=2.2, markersize=7,
                label=model, color=color)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Robustness to Data Degradation — {dataset}", fontsize=13,
                 fontweight="bold", color=NAVY)
    ax.legend(title="Model", fontsize=9)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    path = f"outputs/noise_robustness_{dataset}.png"
    plt.savefig(path, dpi=DPI, facecolor="white")
    plt.close()
    print(f"  saved {path}")

robustness_line("SKAB")
robustness_line("BATADAL")

# ── 3. Confusion Matrices (LSTM, seed=42) ────────────────────────
def make_seqs(X, y, w):
    Xs, ys = [], []
    for i in range(len(X) - w):
        Xs.append(X[i:i+w])
        ys.append(y[i+w])
    return np.array(Xs), np.array(ys)

def make_loader(X, y, bs, shuffle):
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32)),
        batch_size=bs, shuffle=shuffle)

def cm_plot(dataset, label_col):
    threshold = config['training']['prediction_thresholds'].get(dataset, 0.1)
    data_df = load_dataset(dataset, config['paths'])
    X_tr, y_tr, X_v, y_v, X_te, y_te = get_train_val_test_splits(
        dataset, data_df, config['split_ratios'], label_col)
    prep = DataPreprocessor()
    Xtr_s, _ = prep.fit_transform(X_tr)
    Xv_s,  _ = prep.transform(X_v)
    Xte_s, _ = prep.transform(X_te)
    win = config['automata']['window_size']
    set_seed(42)
    Xtr_q, ytr_q = make_seqs(Xtr_s, y_tr.values, win)
    Xv_q,  yv_q  = make_seqs(Xv_s,  y_v.values,  win)
    Xte_q, yte_q = make_seqs(Xte_s, y_te.values, win)
    bs = config['training']['batch_size']
    model = LSTMClassifier(input_dim=Xtr_s.shape[1])
    model = train_deep_learning(
        model,
        make_loader(Xtr_q, ytr_q, bs, True),
        make_loader(Xv_q,  yv_q,  bs, False),
        config['training'], y_train=y_tr.values)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in make_loader(Xte_q, yte_q, bs, False):
            out = torch.sigmoid(model(batch[0])).squeeze(-1)
            if out.ndim == 0:
                preds.append(1 if out.item() > threshold else 0)
            else:
                preds.extend((out > threshold).int().numpy().tolist())
    preds = np.array(preds)
    cm = confusion_matrix(yte_q, preds)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"],
                linewidths=0.5, linecolor="#cccccc", ax=ax,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — LSTM on {dataset}", fontsize=12,
                 fontweight="bold", color=NAVY)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    path = f"outputs/cm_LSTM_{dataset}.png"
    plt.savefig(path, dpi=DPI, facecolor="white")
    plt.close()
    print(f"  saved {path}")

print("Training LSTM seed=42 on SKAB...")
cm_plot("SKAB", "anomaly")
print("Training LSTM seed=42 on BATADAL...")
cm_plot("BATADAL", "ATT_FLAG")

# ── 4. Automata State Diagram + Transition Heatmap for SKAB ──────
def automata_plots(dataset, label_col):
    threshold = config['training']['prediction_thresholds'].get(dataset, 0.1)
    data_df = load_dataset(dataset, config['paths'])
    X_tr, y_tr, X_v, y_v, _, _ = get_train_val_test_splits(
        dataset, data_df, config['split_ratios'], label_col)
    prep = DataPreprocessor()
    _, Xtr_pca = prep.fit_transform(X_tr)
    _, Xv_pca  = prep.transform(X_v)
    win  = config['automata']['window_size']
    alph = config['automata']['alphabet_size']
    auto = ProbabilisticAutomata(window_size=win, alphabet_size=alph)
    auto.fit(Xtr_pca.flatten())
    auto.set_threshold(Xv_pca.flatten(), threshold_override=threshold)
    tm = auto.transition_matrix

    # State Diagram
    G = nx.DiGraph()
    for src, nexts in tm.items():
        for tgt, prob in nexts.items():
            if prob > 0.05:
                G.add_edge(src, tgt, weight=prob)
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    pos = nx.spring_layout(G, k=0.7, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=PALETTE[1],
                           alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="white",
                            font_weight="bold", ax=ax)
    edges = list(G.edges(data=True))
    weights = [d['weight'] * 6 for _, _, d in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights,
                           arrowsize=18, edge_color=NAVY, alpha=0.7, ax=ax)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=7, ax=ax)
    ax.set_title(f"Automata State Transition Diagram — {dataset}",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    path = f"outputs/state_diagram_{dataset}.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved {path}")

    # Transition Heatmap
    states = sorted(tm.keys())
    mat = np.zeros((len(states), len(states)))
    for i, s in enumerate(states):
        for j, t in enumerate(states):
            mat[i, j] = tm[s].get(t, 0.0)
    fs = max(6, len(states))
    fig, ax = plt.subplots(figsize=(fs, fs - 1), facecolor="white")
    annotate = len(states) <= 12
    sns.heatmap(mat, xticklabels=states, yticklabels=states,
                cmap="Blues", annot=annotate, fmt=".2f",
                linewidths=0.3, ax=ax,
                annot_kws={"size": 7})
    ax.set_xlabel("Next State", fontsize=10)
    ax.set_ylabel("Current State", fontsize=10)
    ax.set_title(f"Automata Transition Heatmap — {dataset}",
                 fontsize=12, fontweight="bold", color=NAVY)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    path = f"outputs/transition_heatmap_{dataset}.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved {path}")

print("Training Automata on SKAB for diagrams...")
automata_plots("SKAB", "anomaly")

# ── File check ───────────────────────────────────────────────────
expected = [
    "outputs/f1_comparison_SKAB.png",
    "outputs/f1_comparison_BATADAL.png",
    "outputs/noise_robustness_SKAB.png",
    "outputs/noise_robustness_BATADAL.png",
    "outputs/cm_LSTM_SKAB.png",
    "outputs/cm_LSTM_BATADAL.png",
    "outputs/state_diagram_SKAB.png",
    "outputs/transition_heatmap_SKAB.png",
]
print("\n-- File check --")
all_ok = True
for p in expected:
    exists = os.path.exists(p)
    size_kb = os.path.getsize(p) // 1024 if exists else 0
    status = "OK" if exists else "MISSING"
    if not exists:
        all_ok = False
    print(f"  {status:7}  {p}  ({size_kb} KB)")
print(f"\n{'All 8 files generated successfully.' if all_ok else 'SOME FILES MISSING.'}")
