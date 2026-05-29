import pandas as pd

class AutomataExplainer:
    def __init__(self, automata, unseen_handler, threshold=0.01):
        self.automata = automata
        self.unseen_handler = unseen_handler
        self.threshold = threshold # Anomali tespiti için olasılık eşiği
        self.history = []

    def explain_sequence(self, sequence_states):
        """
        Zaman serisindeki durum geçişlerini adım adım açıklar.
        sequence_states: Otomatanın SAX ile çıkardığı durumlar listesi.
        """
        self.history = []
        path_probability = 1.0 # Başlangıç yol olasılığı
        
        for i in range(len(sequence_states) - 1):
            current_state = str(sequence_states[i])
            next_state = str(sequence_states[i + 1])
            
            status = "Known"
            mapped_to = "N/A"
            
            # Geçiş olasılığını al
            prob = self.automata.get_transition_probability(current_state, next_state)
            
            # Eğer olasılık 1e-6 gibi çok düşük bir değerse (Unseen State veya Transition)
            if prob <= 1e-5:
                status = "Unseen"
                # Unseen handler ile en yakın state'i bul
                mapped_state, dist = self.unseen_handler.handle(next_state)
                mapped_to = mapped_state
                # Eşlenen yeni durum üzerinden olasılığı tekrar hesapla (cezalandırılmış olarak)
                prob = self.automata.get_transition_probability(current_state, mapped_state)
                prob = prob / (dist + 1) # Mesafe arttıkça olasılığı düşür (Güven cezası)
            
            # Path probability hesaplama (Ardışık olasılıkların çarpımı)
            # Not: Underflow olmaması için pratikte log-sum alınır ancak görsellik için çarpıyoruz.
            path_probability *= max(prob, 1e-5) 
            
            # Confidence Score: Mevcut olasılığın eşiğe olan uzaklığına/oranına göre bir güven skoru
            confidence = (1.0 - prob) if prob < self.threshold else prob
            confidence_score = round(confidence * 100, 2)
            
            decision = "Anomaly" if prob < self.threshold else "Normal"
            
            # Zorunlu formatta kaydet
            report_row = {
                "time_step": i,
                "current_state": current_state,
                "next_pattern": next_state,
                "status": status,
                "mapped_to": mapped_to,
                "transition_probability": round(prob, 5),
                "path_probability": f"{path_probability:.2e}",
                "confidence_score_%": confidence_score,
                "decision": decision
            }
            self.history.append(report_row)
            
            # Path probability çok küçülürse sıfırla (Sliding window etkisi yaratmak için)
            if (i + 1) % 10 == 0:
                path_probability = 1.0 
                
        return pd.DataFrame(self.history)