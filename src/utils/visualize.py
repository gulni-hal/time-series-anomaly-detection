import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


def _save(fig, path):
    fig.savefig(path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print('  [VIZ] Saved: ' + path)


def plot_confusion_matrix(y_true, y_pred_bin, model_name, dataset, save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))

    if _HAS_SNS:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
    else:
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Normal', 'Anomaly'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center')
        plt.colorbar(im, ax=ax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix — ' + model_name + ' on ' + dataset)
    _save(fig, os.path.join(save_dir, 'cm_' + model_name + '_' + dataset + '.png'))


def plot_roc_curve(y_true, y_scores, model_name, dataset, save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)
    if len(np.unique(y_true)) < 2:
        print('  [VIZ] Skipping ROC — only one class in y_true')
        return
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label='AUC = ' + str(round(roc_auc, 3)))
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — ' + model_name + ' on ' + dataset)
    ax.legend(loc='lower right')
    _save(fig, os.path.join(save_dir, 'roc_' + model_name + '_' + dataset + '.png'))


def plot_automata_state_diagram(transition_probs, save_dir='outputs', dataset='', max_states=20, min_prob=0.05):
    os.makedirs(save_dir, exist_ok=True)
    if not _HAS_NX:
        print('  [VIZ] networkx not installed — skipping state diagram')
        return
    if not transition_probs:
        return

    all_states = sorted(transition_probs.keys())[:max_states]
    G = nx.DiGraph()
    for src in all_states:
        for dst, prob in transition_probs[src].items():
            if prob >= min_prob:
                G.add_edge(src, dst, weight=round(prob, 3))

    if G.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 9))
    pos = nx.spring_layout(G, seed=42, k=1.5)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600, node_color='lightsteelblue')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    nx.draw_networkx_edges(G, pos, ax=ax, width=[w * 4 for w in weights],
                           edge_color=weights, edge_cmap=plt.cm.Blues,
                           arrows=True, arrowsize=15, connectionstyle='arc3,rad=0.1')
    edge_labels = {(u, v): str(round(G[u][v]['weight'], 2)) for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=6)
    ax.set_title('Automata State Diagram — ' + dataset)
    ax.axis('off')
    _save(fig, os.path.join(save_dir, 'state_diagram_' + dataset + '.png'))


def plot_param_sensitivity(sweep_records, save_dir='outputs'):
    """
    Parameter sensitivity plot: F1 vs window_size and F1 vs alphabet_size.
    sweep_records: list of dicts with keys dataset, window_size, alphabet_size, f1
    """
    os.makedirs(save_dir, exist_ok=True)
    if not sweep_records:
        return

    datasets = sorted(set(r['dataset'] for r in sweep_records))

    for ds in datasets:
        recs = [r for r in sweep_records if r['dataset'] == ds]
        ws_vals = sorted(set(r['window_size'] for r in recs))
        ab_vals = sorted(set(r['alphabet_size'] for r in recs))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle('Parameter Sensitivity — ' + ds)

        # Left: F1 vs window_size, one line per alphabet_size
        ax = axes[0]
        for ab in ab_vals:
            xs = []
            ys = []
            for ws in ws_vals:
                match = [r['f1'] for r in recs if r['window_size'] == ws and r['alphabet_size'] == ab]
                if match:
                    xs.append(ws)
                    ys.append(match[0])
            if xs:
                ax.plot(xs, ys, marker='o', label='alphabet=' + str(ab))
        ax.set_xlabel('Window Size')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 vs Window Size')
        ax.legend(fontsize=8)
        ax.set_xticks(ws_vals)

        # Right: F1 vs alphabet_size, one line per window_size
        ax = axes[1]
        for ws in ws_vals:
            xs = []
            ys = []
            for ab in ab_vals:
                match = [r['f1'] for r in recs if r['window_size'] == ws and r['alphabet_size'] == ab]
                if match:
                    xs.append(ab)
                    ys.append(match[0])
            if xs:
                ax.plot(xs, ys, marker='o', label='window=' + str(ws))
        ax.set_xlabel('Alphabet Size')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 vs Alphabet Size')
        ax.legend(fontsize=8)
        ax.set_xticks(ab_vals)

        plt.tight_layout()
        _save(fig, os.path.join(save_dir, 'param_sensitivity_' + ds + '.png'))


def plot_transition_heatmap(transition_probs, save_dir='outputs', dataset='', max_states=25):
    os.makedirs(save_dir, exist_ok=True)
    if not transition_probs:
        return

    states = sorted(transition_probs.keys())[:max_states]
    n = len(states)
    if n == 0:
        return

    matrix = np.zeros((n, n))
    for i, src in enumerate(states):
        for j, dst in enumerate(states):
            matrix[i, j] = transition_probs[src].get(dst, 0.0)

    fig_size = max(8, n // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    if _HAS_SNS:
        sns.heatmap(matrix, xticklabels=states, yticklabels=states,
                    cmap='YlOrRd', ax=ax, annot=(n <= 15), fmt='.2f', linewidths=0.3)
    else:
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(n)); ax.set_xticklabels(states, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(states, fontsize=7)
        plt.colorbar(im, ax=ax)

    ax.set_xlabel('Destination State')
    ax.set_ylabel('Source State')
    ax.set_title('Transition Probability Heatmap — ' + dataset)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    _save(fig, os.path.join(save_dir, 'transition_heatmap_' + dataset + '.png'))
