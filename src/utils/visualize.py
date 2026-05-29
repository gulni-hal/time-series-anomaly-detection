import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

def plot_confusion_matrix(cm, classes=["Normal", "Anomaly"], title="Confusion Matrix", save_path="outputs/figures/confusion_matrix.png"):
    """Karmaşıklık matrisini (Confusion Matrix) çizer."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.tight_layout()
    
    # Klasör yoksa oluştur
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_automata_transitions(transition_matrix, threshold=0.05, save_path="outputs/figures/automata_graph.png"):
    """
    Otomata modelinin durum geçişlerini çizge (Graph) olarak çizer.
    Sadece threshold'dan büyük olasılıklı geçişleri çizer ki karmaşık görünmesin.
    """
    G = nx.DiGraph()
    
    for current_state, next_states in transition_matrix.items():
        for next_state, prob in next_states.items():
            if prob > threshold:
                G.add_edge(current_state, next_state, weight=prob)
                
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42) # Düğümleri yaylandırarak dağıt
    
    # Düğümleri (States) çiz
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    
    # Kenarları (Geçişler) kalınlıklarına göre çiz
    edges = G.edges(data=True)
    weights = [data['weight'] * 5 for _, _, data in edges] # Kalınlık ayarı
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, arrowsize=20, edge_color='gray')
    
    # Olasılık değerlerini çizgilere yaz
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    
    plt.title("Otomata Durum Geçiş Diyagramı (State Transition Graph)")
    plt.axis('off')
    
    # Klasör yoksa oluştur ve kaydet
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_parameter_sensitivity(window_sizes, alphabet_sizes, data_1d, save_path="outputs/figures/param_sensitivity.png"):
    """Farklı window ve alphabet size'lara göre toplam durum (state) sayısını hesaplayıp çizer."""
    plt.figure(figsize=(10, 6))
    
    # Döngüsel bağımlılığı önlemek için fonksiyon içinde import ediyoruz
    from src.models.automata import ProbabilisticAutomata
    
    for a_size in alphabet_sizes:
        state_counts = []
        for w_size in window_sizes:
            temp_automata = ProbabilisticAutomata(window_size=w_size, alphabet_size=a_size)
            temp_automata.fit(data_1d)
            state_counts.append(len(temp_automata.transition_matrix))
            
        plt.plot(window_sizes, state_counts, marker='o', linewidth=2, markersize=4, label=f"Alphabet Size = {a_size}")

    plt.title("Parametre Duyarlılık Analizi: Durum Sayısı Değişimi")
    plt.xlabel("Pencere Boyutu (Window Size)")
    plt.ylabel("Toplam Durum Sayısı (State Count)")
    plt.legend(title="Alphabet Size")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()