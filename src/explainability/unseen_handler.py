class UnseenHandler:
    def __init__(self, known_patterns):
        """
        known_patterns: Eğitim verisinden elde edilen ve otomata tarafından bilinen
        tüm benzersiz durumların (state/pattern) kümesi.
        """
        self.known_patterns = set(str(p) for p in known_patterns)

    @staticmethod
    def levenshtein_distance(s1, s2):
        """İki string (örüntü) arasındaki minimum düzenleme mesafesini hesaplar."""
        s1, s2 = str(s1), str(s2)
        if len(s1) < len(s2):
            return UnseenHandler.levenshtein_distance(s2, s1)
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

    def handle(self, pattern):
        """
        Pattern'ı kontrol eder: bilinen bir pattern ise "seen", değilse "unseen" döner.
        Dönüş: {"status": "seen"/"unseen", "mapped_to": pattern, "distance": int}
        """
        pattern_str = str(pattern)
        if pattern_str in self.known_patterns:
            return {"status": "seen", "mapped_to": pattern_str, "distance": 0}

        if not self.known_patterns:
            return {"status": "unseen", "mapped_to": pattern_str, "distance": 0}

        min_distance = float('inf')
        best_match = None
        for known in self.known_patterns:
            dist = self.levenshtein_distance(pattern_str, known)
            if dist < min_distance:
                min_distance = dist
                best_match = known

        return {"status": "unseen", "mapped_to": best_match, "distance": int(min_distance)}

    def unseen_rate(self, patterns):
        """Patterns listesindeki görülmemiş örüntülerin oranını döndürür."""
        if not patterns:
            return 0.0
        unseen_count = sum(1 for p in patterns if str(p) not in self.known_patterns)
        return unseen_count / len(patterns)
