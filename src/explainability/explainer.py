import pandas as pd

class AutomataExplainer:
    def __init__(self, automata, unseen_handler, threshold=0.01):
        self.automata = automata
        self.unseen_handler = unseen_handler
        self.threshold = threshold  # Anomali tespiti için olasılık eşiği
        self.history = []

    def explain_sequence(self, sequence_states):
        """
        Zaman serisindeki durum geçişlerini adım adım açıklar.
        sequence_states: Otomatanın SAX ile çıkardığı durumlar listesi.
        """
        self.history = []
        path_probability = 1.0  # Başlangıç yol olasılığı

        for i in range(len(sequence_states) - 1):
            current_state = str(sequence_states[i])
            next_state = str(sequence_states[i + 1])

            status = "Known"
            mapped_to = "N/A"

            prob = self.automata.get_transition_probability(current_state, next_state)

            if prob <= 1e-5:
                status = "Unseen"
                result = self.unseen_handler.handle(next_state)
                mapped_state = result["mapped_to"]
                dist = result["distance"]
                mapped_to = mapped_state
                prob = self.automata.get_transition_probability(current_state, mapped_state)
                prob = prob / (dist + 1)  # Mesafe arttıkça olasılığı düşür (güven cezası)

            # Path probability: ardışık geçiş olasılıklarının çarpımı
            path_probability *= max(prob, 1e-5)

            # Güven skoru: geçiş olasılığı doğrudan güveni temsil eder.
            # Yüksek olasılık = yüksek güven (Normal), düşük olasılık = düşük güven (Anomali).
            confidence_score = round(prob * 100, 2)

            decision = "Anomaly" if prob < self.threshold else "Normal"

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

        return pd.DataFrame(self.history)
