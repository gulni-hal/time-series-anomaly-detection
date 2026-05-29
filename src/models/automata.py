import numpy as np
from collections import defaultdict
from scipy.stats import norm

class ProbabilisticAutomata:
    def __init__(self, window_size, alphabet_size):
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        self.state_counts = defaultdict(int)
        # SAX için normal dağılıma göre kesme noktaları (breakpoints)
        self.breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])

    def paa(self, window_data):
        """Piecewise Aggregate Approximation: Penceredeki verinin ortalamasını alır."""
        return np.mean(window_data)

    def sax(self, paa_value):
        """SAX: PAA değerini kesme noktalarına göre bir sembole (duruma) çevirir."""
        # 0'dan alphabet_size'a kadar bir tam sayı (state) döndürür
        return np.digitize(paa_value, self.breakpoints)

    def extract_patterns(self, data_1d):
        """Sliding window ile zaman serisi üzerinde gezinip durumları çıkarır."""
        states = []
        for i in range(len(data_1d) - self.window_size + 1):
            window = data_1d[i : i + self.window_size]
            paa_val = self.paa(window)
            state = self.sax(paa_val)
            states.append(str(state))
        return states

    def fit(self, data_1d):
        """Eğitim verisi üzerinden (PCA'dan gelen 1D veri) geçiş olasılıklarını öğrenir."""
        states = self.extract_patterns(data_1d)
        
        # Ardışık geçişleri (transitions) say
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            self.transition_matrix[current_state][next_state] += 1
            self.state_counts[current_state] += 1

        # Smoothing (Laplace/Additive Smoothing) uygulayarak olasılıklara çevir
        # Hiç görülmemiş bir geçişe 0 olasılık vermemek için kullanılır (Zero-Frequency problemi)
        vocab = list(self.state_counts.keys())
        vocab_size = len(vocab)
        
        for current_state in self.transition_matrix:
            total_transitions = self.state_counts[current_state]
            for next_state in vocab:
                # Laplace Smoothing: (count + 1) / (total + vocab_size)
                count = self.transition_matrix[current_state].get(next_state, 0)
                prob = (count + 1) / (total_transitions + vocab_size)
                self.transition_matrix[current_state][next_state] = prob

    def get_transition_probability(self, current_state, next_state):
        """İki durum arasındaki geçiş olasılığını döndürür."""
        # Eğer current_state daha önce hiç görülmediyse (Unseen State), çok düşük bir olasılık dön
        if current_state not in self.transition_matrix:
            return 1e-6 
        
        prob = self.transition_matrix[current_state].get(next_state, 1e-6)
        return prob