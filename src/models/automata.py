import numpy as np
from collections import defaultdict
from scipy.stats import norm


def levenshtein_distance(s1, s2):
    """İki string arasındaki Levenshtein (edit) mesafesini hesaplar."""
    s1, s2 = str(s1), str(s2)
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def extract_patterns(data, window_size, alphabet_size):
    """Modül seviyesinde sliding-window pattern çıkarma fonksiyonu."""
    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    arr = np.array(data).flatten()
    states = []
    for i in range(len(arr) - window_size + 1):
        window = arr[i : i + window_size]
        pattern = "".join(str(np.digitize(val, breakpoints)) for val in window)
        states.append(pattern)
    return states


class ProbabilisticAutomata:
    def __init__(self, window_size, alphabet_size):
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        self.state_counts = defaultdict(int)
        # SAX için normal dağılıma göre kesme noktaları (breakpoints)
        self.breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
        self.vocabulary = set()
        self.threshold = None

    def _to_1d(self, data):
        """Veriyi 1D numpy dizisine çevirir (2D girişleri de kabul eder)."""
        return np.array(data).flatten()

    def paa(self, window_data):
        """Piecewise Aggregate Approximation: Penceredeki verinin ortalamasını alır."""
        return np.mean(window_data)

    def sax(self, paa_value):
        """SAX: PAA değerini kesme noktalarına göre bir sembole (duruma) çevirir."""
        return np.digitize(paa_value, self.breakpoints)

    def extract_patterns(self, data_1d):
        """Sliding window ile zaman serisi üzerinde gezinip durumları çıkarır."""
        data_1d = self._to_1d(data_1d)
        states = []
        for i in range(len(data_1d) - self.window_size + 1):
            window = data_1d[i : i + self.window_size]
            pattern = "".join(str(np.digitize(val, self.breakpoints)) for val in window)
            states.append(pattern)
        return states

    def fit(self, data_1d):
        """Eğitim verisi üzerinden geçiş olasılıklarını öğrenir ve vocabulary oluşturur."""
        data_1d = self._to_1d(data_1d)
        states = self.extract_patterns(data_1d)
        self.vocabulary = set(states)

        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            self.transition_matrix[current_state][next_state] += 1
            self.state_counts[current_state] += 1

        vocab = list(self.state_counts.keys())
        vocab_size = len(vocab)

        for current_state in self.transition_matrix:
            total_transitions = self.state_counts[current_state]
            for next_state in vocab:
                # Laplace Smoothing: (count + 1) / (total + vocab_size)
                count = self.transition_matrix[current_state].get(next_state, 0)
                prob = (count + 1) / (total_transitions + vocab_size)
                self.transition_matrix[current_state][next_state] = prob

    def _levenshtein_nearest(self, pattern):
        """Vocabulary'deki en yakın örüntüyü Levenshtein mesafesiyle bulur."""
        if not self.vocabulary:
            return pattern
        best_match = None
        min_dist = float('inf')
        for known in self.vocabulary:
            dist = levenshtein_distance(str(pattern), str(known))
            if dist < min_dist:
                min_dist = dist
                best_match = known
        return best_match

    def get_transition_probability(self, current_state, next_state):
        """İki durum arasındaki geçiş olasılığını döndürür."""
        if current_state not in self.transition_matrix:
            return 1e-6
        prob = self.transition_matrix[current_state].get(next_state, 1e-6)
        return prob

    def set_threshold(self, val_data_1d, threshold_override=None):
        """Validation verisi üzerinden anomali eşiğini hesaplar (5. yüzdelik dilim).
        threshold_override verilirse hesaplama atlanır ve doğrudan o değer kullanılır."""
        if threshold_override is not None:
            self.threshold = float(threshold_override)
            return
        val_data_1d = self._to_1d(val_data_1d)
        states = self.extract_patterns(val_data_1d)
        probs = []
        for i in range(len(states) - 1):
            prob = self.get_transition_probability(states[i], states[i + 1])
            probs.append(prob)
        if probs:
            self.threshold = float(np.percentile(probs, 5))
        else:
            self.threshold = 1e-6

    def predict(self, data_1d):
        """Veriyi tahmin ederek 0 (normal) / 1 (anomali) dizisi döndürür."""
        data_1d = self._to_1d(data_1d)
        threshold = self.threshold if self.threshold is not None else 1e-6
        states = self.extract_patterns(data_1d)
        predictions = []
        for i in range(len(states) - 1):
            current = states[i]
            nxt = states[i + 1]
            # Görülmemiş durum (Unseen State) ise en yakınını bul
            if current not in self.transition_matrix and self.vocabulary:
                current = self._levenshtein_nearest(current)
            if nxt not in self.vocabulary and self.vocabulary:
                nxt = self._levenshtein_nearest(nxt)
            prob = self.get_transition_probability(current, nxt)
            predictions.append(1 if prob < threshold else 0)
        return np.array(predictions, dtype=int)

    def explain(self, data_1d, time_step):
        """Bir veri dilimindeki ilk geçişi açıklar ve dict döndürür."""
        data_1d = self._to_1d(data_1d)
        threshold = self.threshold if self.threshold is not None else 1e-6
        states = self.extract_patterns(data_1d)
        if len(states) < 2:
            return {
                "time_step": time_step,
                "state": "",
                "pattern": "",
                "status": "unknown",
                "mapped_to": "",
                "probability": 0.0,
                "decision": "Normal",
                "confidence": 0.0,
            }
        current = states[0]
        nxt = states[1]
        status = "seen"
        mapped_to = nxt
        if nxt not in self.vocabulary and self.vocabulary:
            status = "unseen"
            mapped_to = self._levenshtein_nearest(nxt)
        prob = self.get_transition_probability(
            current, mapped_to if status == "unseen" else nxt
        )
        decision = "Anomaly" if prob < threshold else "Normal"
        return {
            "time_step": time_step,
            "state": current,
            "pattern": nxt,
            "status": status,
            "mapped_to": mapped_to,
            "probability": round(float(prob), 6),
            "decision": decision,
            "confidence": round(float(prob), 4),
        }
