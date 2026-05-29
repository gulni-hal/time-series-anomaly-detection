from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Tahmin sonuçlarını alıp tüm sınıflandırma metriklerini hesaplar."""
    
    # HATA ÇÖZÜMÜ: 'average=weighted' eklendi. Böylece multiclass (çoklu sınıf) 
    # etiketleri gelse bile metrikler ağırlıklı ortalama ile sorunsuz hesaplanır.
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Confusion Matrix'i de döndür (Görselleştirme için lazım olacak)
    cm = confusion_matrix(y_true, y_pred)
    
    return metrics, cm