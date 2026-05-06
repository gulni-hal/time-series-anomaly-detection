"""
Birim Testleri — Levenshtein ve Unseen Pattern
Rubrik: 5 Puan
Çalıştır: py tests/test_unseen.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.automata import levenshtein_distance, ProbabilisticAutomata, extract_patterns
from src.explainability.unseen_handler import UnseenHandler
import numpy as np


def test_levenshtein_same():
    assert levenshtein_distance("abc", "abc") == 0

def test_levenshtein_substitution():
    assert levenshtein_distance("aab", "abc") == 2

def test_levenshtein_insertion():
    assert levenshtein_distance("ab", "abc") == 1

def test_levenshtein_deletion():
    assert levenshtein_distance("abc", "ab") == 1

def test_levenshtein_empty():
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("abc", "") == 3

def test_levenshtein_single():
    assert levenshtein_distance("a", "b") == 1

def test_unseen_handler_seen():
    handler = UnseenHandler({"aab", "abc", "bcc"})
    result  = handler.handle("aab")
    assert result["status"] == "seen"
    assert result["mapped_to"] == "aab"
    assert result["distance"] == 0

def test_unseen_handler_unseen():
    handler = UnseenHandler({"aab", "abc", "bcc"})
    result  = handler.handle("adc")
    assert result["status"] == "unseen"
    assert result["mapped_to"] in {"aab", "abc", "bcc"}
    assert result["distance"] >= 1

def test_unseen_rate():
    handler = UnseenHandler({"aab", "abc"})
    patterns = ["aab", "abc", "zzz", "zzz"]
    rate = handler.unseen_rate(patterns)
    assert abs(rate - 0.5) < 0.01

def test_automata_no_crash_on_unseen():
    np.random.seed(42)
    series = np.sin(np.linspace(0, 6*np.pi, 600))
    model  = ProbabilisticAutomata(window_size=4, alphabet_size=3)
    model.fit(series[:400].reshape(-1, 1))
    model.set_threshold(series[400:500].reshape(-1, 1))
    noisy = (series[500:] + np.random.randn(100) * 5).reshape(-1, 1)
    preds = model.predict(noisy)
    assert len(preds) >= 0, "Unseen pattern'larda sistem çökmemeli"

def test_explain_format():
    np.random.seed(42)
    series = np.sin(np.linspace(0, 10*np.pi, 800))
    model  = ProbabilisticAutomata(window_size=4, alphabet_size=3)
    model.fit(series[:500].reshape(-1, 1))
    model.set_threshold(series[500:650].reshape(-1, 1))
    exp = model.explain(series[650:680].reshape(-1, 1), time_step=650)
    for key in ["time_step", "state", "pattern", "status", "mapped_to",
                "probability", "decision", "confidence"]:
        assert key in exp, f"Eksik anahtar: {key}"


if __name__ == "__main__":
    tests = [
        test_levenshtein_same, test_levenshtein_substitution,
        test_levenshtein_insertion, test_levenshtein_deletion,
        test_levenshtein_empty, test_levenshtein_single,
        test_unseen_handler_seen, test_unseen_handler_unseen,
        test_unseen_rate, test_automata_no_crash_on_unseen,
        test_explain_format,
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
