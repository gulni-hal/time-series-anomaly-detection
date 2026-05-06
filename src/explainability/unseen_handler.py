"""Unseen Pattern Yönetimi — Levenshtein tabanlı eşleme."""

from src.models.automata import levenshtein_distance


class UnseenHandler:
    def __init__(self, vocabulary: set):
        self.vocabulary = vocabulary

    def handle(self, pattern: str) -> dict:
        """
        Sözlükte olmayan pattern için en yakın eşlemeyi döndürür.
        """
        if pattern in self.vocabulary:
            return {"pattern": pattern, "status": "seen", "mapped_to": pattern, "distance": 0}

        nearest  = min(self.vocabulary, key=lambda p: levenshtein_distance(pattern, p))
        distance = levenshtein_distance(pattern, nearest)
        return {
            "pattern":  pattern,
            "status":   "unseen",
            "mapped_to": nearest,
            "distance": distance,
        }

    def unseen_rate(self, patterns: list) -> float:
        """Listdeki unseen pattern oranını döndürür."""
        unseen = sum(1 for p in patterns if p not in self.vocabulary)
        return unseen / max(len(patterns), 1)
