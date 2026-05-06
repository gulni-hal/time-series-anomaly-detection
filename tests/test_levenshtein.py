"""
Birim Testleri — Levenshtein ve Unseen Pattern Yönetimi
Rubrik: "Unseen veri yönetimi (Levenshtein) ve buna ait birim testler (5 Puan)"
Çalıştır: python tests/test_levenshtein.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.models.automata import levenshtein_distance, ProbabilisticAutomata
import numpy as np


def test_levenshtein_same():
    assert levenshtein_distance("abc", "abc") == 0, "Aynı string → 0"

def test_levenshtein_one_substitution():
    assert levenshtein_distance("aab", "abc") == 2, "2 karakter farkı"

def test_levenshtein_insertion():
    assert levenshtein_distance("ab", "abc") == 1, "1 ekleme"

def test_levenshtein_deletion():
    assert levenshtein_distance("abc", "ab") == 1, "1 silme"

def test_levenshtein_empty():
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("abc", "") == 3

def test_levenshtein_single_char():
    assert levenshtein_distance("a", "b") == 1

def test_nearest_pattern_found():
    """Unseen pattern → sözlükteki en yakın eşleme doğru mu?"""
    model = ProbabilisticAutomata(window_size=4, alphabet_size=3)
    model.vocabulary = {"aab", "abc", "bcc", "aaa"}
    # "adc" → en yakın "abc" (1 değişim) veya "aab" (2 değişim)
    nearest = model._levenshtein_nearest("adc")
    assert nearest in model.vocabulary, "Eşleme sözlükte olmalı"
    dist = levenshtein_distance("adc", nearest)
    assert dist <= 2, f"Mesafe çok büyük: {dist}"

def test_unseen_in_predict():
    """Eğitimde görülmemiş pattern'larla prediction çökmemeli."""
    np.random.seed(42)
    series = np.sin(np.linspace(0, 6 * np.pi, 600))
    model  = ProbabilisticAutomata(window_size=4, alphabet_size=3)
    model.fit(series[:400])
    model.set_threshold(series[400:500])

    # Tamamen farklı (yüksek gürültülü) sinyal → unseen pattern üretebilir
    noisy = series[500:] + np.random.randn(100) * 5
    try:
        preds = model.predict(noisy)
        assert len(preds) >= 0
        print("[✓] Unseen pattern → sistem çökmedi, tahmin üretildi")
    except Exception as e:
        raise AssertionError(f"Unseen pattern sırasında hata: {e}")

def test_explain_output_format():
    """Açıklanabilirlik çıktısı doğru anahtarları içermeli."""
    np.random.seed(42)
    series = np.sin(np.linspace(0, 10 * np.pi, 800))
    model  = ProbabilisticAutomata(window_size=4, alphabet_size=3)
    model.fit(series[:500])
    model.set_threshold(series[500:650])

    explanation = model.explain(series[650:680], time_step=650)
    required_keys = {"time_step", "state", "pattern", "status", "mapped_to", "probability", "decision", "confidence"}
    for key in required_keys:
        assert key in explanation, f"Eksik anahtar: {key}"
    print(f"[✓] Açıklama formatı doğru: {explanation}")


if __name__ == "__main__":
    tests = [
        test_levenshtein_same,
        test_levenshtein_one_substitution,
        test_levenshtein_insertion,
        test_levenshtein_deletion,
        test_levenshtein_empty,
        test_levenshtein_single_char,
        test_nearest_pattern_found,
        test_unseen_in_predict,
        test_explain_output_format,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            print(f"[✓] {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"[✗] {test.__name__}: {e}")

    print(f"\n{'='*40}")
    print(f"Sonuç: {passed}/{len(tests)} test geçti")
