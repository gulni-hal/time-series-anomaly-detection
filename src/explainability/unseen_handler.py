class UnseenHandler:
    def __init__(self, known_patterns):
        """
        known_patterns: Eğitim verisinden elde edilen ve otomata tarafından bilinen 
        tüm benzersiz durumların (state/pattern) listesi.
        """
        self.known_patterns = known_patterns

    @staticmethod
    def levenshtein_distance(s1, s2):
        """İki string (örüntü) arasındaki minimum düzenleme mesafesini hesaplar."""
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

    def handle(self, unseen_pattern):
        """
        Görülmemiş bir örüntüyü Levenshtein mesafesine göre en yakın bilinen örüntüye eşler.
        """
        if not self.known_patterns:
            return unseen_pattern, 0 # Bilinen örüntü yoksa kendini dön

        min_distance = float('inf')
        best_match = None
        
        for known in self.known_patterns:
            dist = self.levenshtein_distance(str(unseen_pattern), str(known))
            if dist < min_distance:
                min_distance = dist
                best_match = known
                
        return best_match, min_distance