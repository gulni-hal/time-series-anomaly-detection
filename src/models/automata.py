"""
Otomata Tabanlı Model
PAA → SAX → Sliding Window → Durum Geçiş Olasılıkları
"""

import numpy as np
from collections import defaultdict
import time


# ─── PAA ──────────────────────────────────────────────────────────────────────
def paa_transform(series: np.ndarray, window_size: int) -> np.ndarray:
    n = len(series)
    n_segments = max(1, n // window_size)
    trimmed = series[:n_segments * window_size]
    return trimmed.reshape(n_segments, window_size).mean(axis=1)


# ─── SAX ──────────────────────────────────────────────────────────────────────
def _get_breakpoints(alphabet_size: int) -> np.ndarray:
    from scipy.stats import norm
    return norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])


def sax_transform(paa_series: np.ndarray, alphabet_size: int) -> str:
    breakpoints = _get_breakpoints(alphabet_size)
    return ''.join(chr(ord('a') + np.searchsorted(breakpoints, v)) for v in paa_series)


# ─── Pattern Çıkarma ──────────────────────────────────────────────────────────
def extract_patterns(series_1d: np.ndarray, window_size: int, alphabet_size: int) -> list:
    series_1d = series_1d.flatten()
    patterns = []
    n = len(series_1d)
    for i in range(0, n - window_size + 1, window_size):
        segment = series_1d[i:i + window_size]
        if len(segment) == window_size:
            paa = paa_transform(segment, window_size)
            sax = sax_transform(paa, alphabet_size)
            patterns.append(sax)
    return patterns


# ─── Levenshtein ──────────────────────────────────────────────────────────────
def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n]


# ─── Otomata Modeli ───────────────────────────────────────────────────────────
class ProbabilisticAutomata:
    def __init__(self, window_size: int = 4, alphabet_size: int = 3):
        self.window_size   = window_size
        self.alphabet_size = alphabet_size
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs  = {}
        self.vocabulary        = set()
        self.threshold         = None
        self.train_time        = 0.0
        self.infer_time        = 0.0

    def fit(self, X_train_pca: np.ndarray):
        t0 = time.perf_counter()
        series = X_train_pca.flatten()
        patterns = extract_patterns(series, self.window_size, self.alphabet_size)
        self.vocabulary = set(patterns)

        for i in range(len(patterns) - 1):
            self.transition_counts[patterns[i]][patterns[i+1]] += 1

        for src, dst_counts in self.transition_counts.items():
            total = sum(dst_counts.values())
            self.transition_probs[src] = {d: c/total for d, c in dst_counts.items()}

        self.train_time = time.perf_counter() - t0
        print(f"  [Automata] Eğitim: {len(self.vocabulary)} pattern, "
              f"{len(self.transition_probs)} durum | {self.train_time:.2f}s")
        return self

    def set_threshold(self, X_val_pca: np.ndarray, percentile: float = 10.0):
        scores, _ = self._score_series(X_val_pca)
        self.threshold = np.percentile(scores, percentile)
        print(f"  [Automata] Eşik: {self.threshold:.4f} (p{percentile})")
        return self

    def _nearest(self, pattern: str) -> str:
        if not self.vocabulary:
            return pattern
        return min(self.vocabulary, key=lambda p: levenshtein_distance(pattern, p))

    def _score_series(self, X_pca: np.ndarray, chunk: int = 5):
        series   = X_pca.flatten()
        patterns = extract_patterns(series, self.window_size, self.alphabet_size)
        scores, explanations = [], []

        for i in range(0, len(patterns) - chunk, chunk):
            sub    = patterns[i:i+chunk]
            log_p  = 0.0
            exp    = {"transitions": [], "unseen": []}

            for j in range(len(sub) - 1):
                src, dst = sub[j], sub[j+1]
                mapped   = src
                if src not in self.transition_probs:
                    mapped = self._nearest(src)
                    exp["unseen"].append({
                        "original": src, "mapped_to": mapped,
                        "distance": levenshtein_distance(src, mapped)
                    })
                prob = self.transition_probs.get(mapped, {}).get(dst, 1e-6)
                log_p += np.log(prob + 1e-10)
                exp["transitions"].append({"from": mapped, "to": dst, "prob": round(prob, 6)})

            scores.append(log_p)
            explanations.append(exp)

        return np.array(scores) if scores else np.array([self.threshold or -10]), explanations

    def predict(self, X_pca: np.ndarray):
        t0 = time.perf_counter()
        scores, _ = self._score_series(X_pca)
        self.infer_time = time.perf_counter() - t0
        return (scores < self.threshold).astype(int)

    def predict_with_scores(self, X_pca: np.ndarray):
        t0 = time.perf_counter()
        scores, explanations = self._score_series(X_pca)
        self.infer_time = time.perf_counter() - t0
        preds = (scores < self.threshold).astype(int)
        return preds, scores, explanations

    def explain(self, X_pca: np.ndarray, time_step: int = 0) -> dict:
        """JSON formatında açıklama üretir — Rubrik Bölüm X."""
        series   = X_pca.flatten()
        patterns = extract_patterns(series, self.window_size, self.alphabet_size)
        if len(patterns) < 2:
            return {"error": "Yeterli pattern yok"}

        src, dst   = patterns[0], patterns[1]
        is_unseen  = src not in self.transition_probs
        mapped     = self._nearest(src) if is_unseen else src
        prob       = self.transition_probs.get(mapped, {}).get(dst, 1e-6)
        path_prob  = prob
        log_p      = np.log(path_prob + 1e-10)
        decision   = "anomaly" if log_p < (self.threshold or -10) else "normal"

        return {
            "time_step":   time_step,
            "state":       src,
            "pattern":     dst,
            "status":      "unseen" if is_unseen else "seen",
            "mapped_to":   mapped,
            "probability": round(path_prob, 6),
            "decision":    decision,
            "confidence":  round(path_prob, 6),
        }
