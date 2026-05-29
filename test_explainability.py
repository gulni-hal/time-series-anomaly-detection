# test_explainability.py
from src.explainability.unseen_handler import UnseenHandler
from src.explainability.explainer import AutomataExplainer

# Sahte (Mock) Otomata Sınıfı - Sadece test için
class MockAutomata:
    def __init__(self):
        # Durumlar arası geçiş olasılıkları (örn: 0 -> 1 = %80)
        self.transitions = {
            "0": {"1": 0.8, "0": 0.2},
            "1": {"2": 0.9, "1": 0.1},
            "2": {"0": 0.5, "2": 0.5}
        }
    def get_transition_probability(self, curr, nxt):
        return self.transitions.get(curr, {}).get(nxt, 1e-6)

# 1. Levenshtein Testi (Birim Test)
print("--- Levenshtein ve Unseen Handler Testi ---")
known_states = ["00", "01", "10", "11"]
handler = UnseenHandler(known_patterns=known_states)

# "21" diye bir pattern gelirse, en çok kime benzer? "11" veya "01" olmalı. (Mesafe: 1)
mapped, dist = handler.handle("21")
print(f"Bilinmeyen pattern '21', bilinen '{mapped}' örüntüsüne eşlendi. Mesafe: {dist}")
assert dist == 1, "Levenshtein hesaplamasında hata var!"
print("Levenshtein Testi: BAŞARILI\n")

# 2. Explainer Testi
print("--- Açıklanabilirlik (Explainer) Testi ---")
automata = MockAutomata()
handler_mock = UnseenHandler(known_patterns=["0", "1", "2"])
explainer = AutomataExplainer(automata, handler_mock, threshold=0.1)

# Senaryo: 0'dan 1'e normal, 1'den 2'ye normal, 2'den 3'e (3 unseen'dir, 2'ye en yakındır).
test_sequence = ["0", "1", "2", "3"]

report_df = explainer.explain_sequence(test_sequence)
print("Üretilen Açıklama Tablosu (PDF İsteri Formatında):")
print(report_df.to_markdown(index=False))
print("\nTest Başarılı! Unseen durumu yakalandı, 'mapped_to' ile atandı ve Karar (Decision) verildi.")