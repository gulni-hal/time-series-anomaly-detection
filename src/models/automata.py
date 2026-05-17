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
            paa = paa_transform(segment, 1)  # segment_size=1 → window_size PAA values → window_size-char SAX
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

    def set_threshold_f1(self, X_val_pca: np.ndarray, y_val: np.ndarray):
        """Find threshold that maximises F1 on the validation set."""
        from sklearn.metrics import precision_recall_curve
        scores, _ = self._score_series(X_val_pca)
        min_len = min(len(scores), len(y_val))
        scores_v = scores[:min_len]
        y_v = y_val[:min_len]

        if y_v.sum() == 0:
            self.threshold = np.percentile(scores_v, 10.0)
            print(f"  [Automata] No val positives — fallback p10: {self.threshold:.4f}")
            return self

        # anomaly = score < threshold  →  negate so higher = more anomalous
        neg_scores = -scores_v
        precision, recall, thresholds = precision_recall_curve(y_v, neg_scores)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = int(np.argmax(f1_scores[:-1]))
        self.threshold = -float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])
        print(f"  [Automata] F1-optimal threshold: {self.threshold:.4f} (val F1={round(best_f1, 4)})")
        return self

    def _levenshtein_nearest(self, pattern: str) -> str:
        if not self.vocabulary:
            return pattern
        return min(self.vocabulary, key=lambda p: levenshtein_distance(pattern, p))

    def _score_series(self, X_pca: np.ndarray):
        """
        Point-level scoring: each input timestep receives a log-probability score.

        Old approach grouped patterns into fixed chunks, producing ~n/20 scores for
        n timesteps. min_len alignment then compared only the first n/20 labels,
        missing WADI attacks that occur later in the series (F1=0).

        New approach: transition score for pattern pair (i, i+1) is broadcast to
        all timesteps covered by pattern i+1, so output length == input length and
        aligns 1-to-1 with y_test.
        """
        series   = X_pca.flatten()
        n_points = len(series)
        patterns = extract_patterns(series, self.window_size, self.alphabet_size)
        explanations = []

        if len(patterns) < 2:
            return np.zeros(n_points), explanations

        # Score 0.0 (neutral) for the first window — no predecessor exists
        point_scores = np.zeros(n_points)

        for i in range(len(patterns) - 1):
            src, dst = patterns[i], patterns[i + 1]
            exp      = {"transitions": [], "unseen": []}

            mapped = src
            if src not in self.transition_probs:
                mapped = self._levenshtein_nearest(src)
                exp["unseen"].append({
                    "original": src, "mapped_to": mapped,
                    "distance": levenshtein_distance(src, mapped)
                })

            prob  = self.transition_probs.get(mapped, {}).get(dst, 1e-6)
            log_p = np.log(prob + 1e-10)
            exp["transitions"].append({"from": mapped, "to": dst, "prob": round(prob, 6)})

            # Assign score to the timestep range of the destination pattern
            t_start = (i + 1) * self.window_size
            t_end   = min(t_start + self.window_size, n_points)
            point_scores[t_start:t_end] = log_p
            explanations.append(exp)

        return point_scores, explanations

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
        mapped     = self._levenshtein_nearest(src) if is_unseen else src
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
