"""
Olasılıksal Açıklanabilirlik Modülü
Rubrik Bölüm X — State, pattern, transition, path probability, confidence score
"""

import json
import numpy as np
from src.models.automata import extract_patterns, levenshtein_distance


class AutomataExplainer:
    def __init__(self, automata_model):
        self.model = automata_model

    def explain_window(self, series_1d: np.ndarray, time_step: int = 0) -> dict:
        """
        Tek bir pencere için tam açıklama üretir.
        Döndürür: JSON uyumlu dict (rubrik format)
        """
        m = self.model
        patterns = extract_patterns(series_1d.flatten(), m.window_size, m.alphabet_size)

        if len(patterns) < 2:
            return {"error": "Yeterli pattern yok"}

        src, dst  = patterns[0], patterns[1]
        is_unseen = src not in m.transition_probs

        # Unseen → Levenshtein eşleme
        if is_unseen:
            mapped   = m._levenshtein_nearest(src)
            distance = levenshtein_distance(src, mapped)
        else:
            mapped   = src
            distance = 0

        # Geçiş olasılığı
        prob      = m.transition_probs.get(mapped, {}).get(dst, 1e-6)
        path_prob = prob
        log_p     = np.log(path_prob + 1e-10)
        decision  = "anomaly" if log_p < (m.threshold or -10) else "normal"

        result = {
            "time_step":       time_step,
            "previous_state":  src,
            "incoming_pattern": dst,
            "status":          "unseen" if is_unseen else "seen",
            "nearest_pattern": mapped,
            "edit_distance":   distance,
            "transitions": [
                {
                    "from":        mapped,
                    "to":          dst,
                    "probability": round(prob, 6),
                }
            ],
            "path_probability": round(path_prob, 6),
            "decision":         decision,
            "confidence_score": round(path_prob, 6),
            "interpretation":  (
                "Low probability path → Anomaly candidate"
                if decision == "anomaly"
                else "High probability path → Normal behavior"
            ),
        }
        return result

    def explain_batch(self, X_pca: np.ndarray, n_samples: int = 5) -> list:
        """İlk n_samples pencere için açıklama üretir."""
        m       = self.model
        series  = X_pca.flatten()
        w       = m.window_size
        results = []

        for i in range(min(n_samples, len(series) // w)):
            window = series[i*w:(i+2)*w]
            if len(window) >= w:
                exp = self.explain_window(window, time_step=i*w)
                results.append(exp)

        return results

    def print_explanation(self, exp: dict):
        """Dokümandaki örnek formatta terminale yazar."""
        print("\n[SYSTEM DECISION]")
        print(f"Time Step      : t = {exp.get('time_step', '?')}")
        print(f"Previous State : \"{exp.get('previous_state', '?')}\"")
        print(f"Incoming Pattern: \"{exp.get('incoming_pattern', '?')}\"")
        print(f"Status         : {exp.get('status', '?').upper()}")
        if exp.get('status') == 'unseen':
            print(f"Nearest Pattern: \"{exp.get('nearest_pattern')}\" "
                  f"(distance = {exp.get('edit_distance')})")
        print("Transitions    :")
        for t in exp.get("transitions", []):
            print(f"  {t['from']} -> {t['to']} : {t['probability']}")
        print(f"Path Probability: {exp.get('path_probability')}")
        print(f"Decision       : {exp.get('decision', '?').upper()}")
        print(f"Confidence Score: {exp.get('confidence_score')} "
              f"({'Low' if exp.get('decision') == 'anomaly' else 'High'})")
        print(f"Interpretation : {exp.get('interpretation', '')}")
