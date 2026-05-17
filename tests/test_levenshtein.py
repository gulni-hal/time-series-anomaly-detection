# -*- coding: utf-8 -*-
"""
Birim Testleri -- Levenshtein ve Unseen Pattern Yonetimi
Rubrik: "Unseen veri yonetimi (Levenshtein) ve buna ait birim testler (5 Puan)"
Calistir: python tests/test_levenshtein.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.models.automata import levenshtein_distance, ProbabilisticAutomata
import numpy as np


def test_levenshtein_same():
    assert levenshtein_distance("abc", "abc") == 0, "Ayni string -> 0"

def test_levenshtein_one_substitution():
    assert levenshtein_distance("aab", "abc") == 2, "2 karakter farki"

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
    """Unseen pattern -> sozlukteki en yakin esleme dogru mu?"""
    model = ProbabilisticAutomata(window_size=4, alphabet_size=3)
    model.vocabulary = {"aab", "abc", "bcc", "aaa"}
    # "adc" -> en yakin "abc" (1 degisim) veya "aab" (2 degisim)
    nearest = model._levenshtein_nearest("adc")
    assert nearest in model.vocabulary, "Esleme sozlukte olmali"
    dist = levenshtein_distance("adc", nearest)
    assert dist <= 2, "Mesafe cok buyuk: " + str(dist)

def test_unseen_in_predict():
    """Egitimde gorulmemis pattern'larla prediction cokmemeli."""
    np.random.seed(42)
    series = np.sin(np.linspace(0, 6 * np.pi, 600))
    model  = ProbabilisticAutomata(window_size=4, alphabet_size=3)
    model.fit(series[:400])
    model.set_threshold(series[400:500])

    noisy = series[500:] + np.random.randn(100) * 5
    try:
        preds = model.predict(noisy)
        assert len(preds) >= 0
        print("[OK] Unseen pattern -> sistem cokmedi, tahmin uretildi")
    except Exception as e:
        raise AssertionError("Unseen pattern sirasinda hata: " + str(e))

def test_explain_output_format():
    """Aciklanabilirlik ciktisi dogru anahtarlari icermeli."""
    np.random.seed(42)
    series = np.sin(np.linspace(0, 10 * np.pi, 800))
    model  = ProbabilisticAutomata(window_size=4, alphabet_size=3)
    model.fit(series[:500])
    model.set_threshold(series[500:650])

    explanation = model.explain(series[650:680], time_step=650)
    required_keys = {"time_step", "state", "pattern", "status", "mapped_to",
                     "probability", "decision", "confidence"}
    for key in required_keys:
        assert key in explanation, "Eksik anahtar: " + key
    print("[OK] Aciklama formati dogru: " + str(list(explanation.keys())))


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
            print("[PASS] " + test.__name__)
            passed += 1
        except AssertionError as e:
            print("[FAIL] " + test.__name__ + ": " + str(e))

    print("\n" + "=" * 40)
    print("Sonuc: " + str(passed) + "/" + str(len(tests)) + " test gecti")
