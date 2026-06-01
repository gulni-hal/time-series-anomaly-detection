import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, yaml
from sklearn.metrics import f1_score

from src.pipeline.data_loader import load_dataset, get_train_val_test_splits
from src.preprocessing.pipeline import DataPreprocessor
from src.models.automata import ProbabilisticAutomata

os.makedirs("outputs", exist_ok=True)

NAVY    = "#003366"
sns.set_style("whitegrid")
DPI = 150

with open("configs/default_config.yaml") as f:
    config = yaml.safe_load(f)

WINDOW_SIZES   = config['automata']['experiment_window_sizes']   # [3,4,5,6]
ALPHABET_SIZES = config['automata']['experiment_alphabet_sizes'] # [3,4,5,6]
THRESHOLDS     = config['training']['prediction_thresholds']

DATASETS = [
    ("SKAB",    "anomaly"),
    ("BATADAL", "ATT_FLAG"),
]

def compute_f1_grid(dataset_name, label_col):
    threshold = THRESHOLDS.get(dataset_name, THRESHOLDS.get('default', 0.1))
    data_df   = load_dataset(dataset_name, config['paths'])
    X_tr, y_tr, X_v, y_v, X_te, y_te = get_train_val_test_splits(
        dataset_name, data_df, config['split_ratios'], label_col)

    prep = DataPreprocessor()
    _, Xtr_pca = prep.fit_transform(X_tr)
    _, Xv_pca  = prep.transform(X_v)
    _, Xte_pca = prep.transform(X_te)

    grid = np.zeros((len(ALPHABET_SIZES), len(WINDOW_SIZES)))

    for i, a_size in enumerate(ALPHABET_SIZES):
        for j, w_size in enumerate(WINDOW_SIZES):
            auto = ProbabilisticAutomata(window_size=w_size, alphabet_size=a_size)
            auto.fit(Xtr_pca.flatten())
            auto.set_threshold(Xv_pca.flatten(), threshold_override=threshold)
            preds = auto.predict(Xte_pca.flatten())

            # align label length with predictions (automata returns len-1 predictions)
            y_aligned = y_te.values[:len(preds)]
            f1 = f1_score(y_aligned, preds, zero_division=0)
            grid[i, j] = round(f1, 4)
            print(f"  {dataset_name} | w={w_size} a={a_size} -> F1={f1:.4f}")

    return grid

def save_heatmap(grid, dataset_name):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    sns.heatmap(
        grid,
        annot=True, fmt=".3f",
        xticklabels=WINDOW_SIZES,
        yticklabels=ALPHABET_SIZES,
        cmap="Blues",
        linewidths=0.4,
        linecolor="#cccccc",
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Window Size", fontsize=11)
    ax.set_ylabel("Alphabet Size", fontsize=11)
    ax.set_title(f"Automata Parameter Sensitivity (F1) -- {dataset_name}",
                 fontsize=13, fontweight="bold", color=NAVY)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    path = f"outputs/param_sensitivity_{dataset_name}.png"
    plt.savefig(path, dpi=DPI, facecolor="white")
    plt.close()
    print(f"  saved {path}")

for ds_name, lbl in DATASETS:
    print(f"\nComputing grid for {ds_name}...")
    grid = compute_f1_grid(ds_name, lbl)
    save_heatmap(grid, ds_name)

# Delete old SWAT/WADI plots
for stale in ["outputs/param_sensitivity_SWAT.png", "outputs/param_sensitivity_WADI.png"]:
    if os.path.exists(stale):
        os.remove(stale)
        print(f"  deleted {stale}")

# Verify
print("\n-- File check --")
for ds_name, _ in DATASETS:
    p = f"outputs/param_sensitivity_{ds_name}.png"
    exists = os.path.exists(p)
    size_kb = os.path.getsize(p) // 1024 if exists else 0
    print(f"  {'OK' if exists else 'MISSING':7}  {p}  ({size_kb} KB)")
