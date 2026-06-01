# Zaman Serisi Anomali Tespiti — Probabilistik Otomata ve Derin Öğrenme

Bu proje, endüstriyel kontrol sistemleri veri setleri (SKAB ve BATADAL) üzerinde zaman serisi anomali tespiti gerçekleştiren, otomata tabanlı açıklanabilir model ile LSTM, GRU ve CNN1D derin öğrenme modellerini karşılaştıran modüler bir pipeline sunar.

---

## Kurulum

```bash
pip install -r requirements.txt
```

**Gereksinimler:** Python >= 3.8, PyTorch, scikit-learn, scipy, pandas, numpy, matplotlib, seaborn, networkx, pyyaml

---

## Calistirma

```bash
python main.py
```

Tüm sonuçlar `logs/` klasörüne CSV ve JSON olarak kaydedilir.

**Birim testleri:**

```bash
python tests/test_unseen.py
python tests/test_levenshtein.py
```

---

## Proje Yapisi

```
yazlab2/
├── configs/
│   ├── config.py                  # Python sabitleri
│   └── default_config.yaml        # Ana konfigurasyon (tüm parametreler burada)
├── data/
│   └── raw/
│       ├── SKAB/                  # valve1/ ve valve2/ alt klasörleri
│       └── BATADAL/               # BATADAL_Training2.csv
├── logs/                          # CSV ve JSON deney sonuçlari
├── outputs/
│   ├── figures/                   # Karisiklik matrisleri, geçis grafikleri
│   └── results/                   # Otomata aciklanabilirlik raporlari
├── src/
│   ├── explainability/
│   │   ├── explainer.py           # AutomataExplainer: adim adim aciklama
│   │   └── unseen_handler.py      # Levenshtein tabanli unseen pattern yönetimi
│   ├── models/
│   │   ├── automata.py            # ProbabilisticAutomata (SAX + gecis matrisi)
│   │   └── deep_learning.py       # LSTMClassifier, GRUClassifier, CNN1DClassifier
│   ├── pipeline/
│   │   ├── data_loader.py         # SKAB ve BATADAL yükleyiciler
│   │   ├── evaluate.py            # Seed, gürültü, unseen senaryolari
│   │   └── train.py               # DL ve Otomata egitim döngüleri
│   ├── preprocessing/
│   │   └── pipeline.py            # StandardScaler + PCA (sadece train'de fit)
│   └── utils/
│       ├── logger.py              # ExperimentLogger (CSV/JSON)
│       ├── metrics.py             # Accuracy, Precision, Recall, F1
│       ├── stats.py               # Wilcoxon ve McNemar testleri
│       └── visualize.py           # Karisiklik matrisi, gecis diyagrami, parametre duyarlilik
├── tests/
│   ├── test_levenshtein.py        # Levenshtein + Automata unit testleri (9 test)
│   └── test_unseen.py             # UnseenHandler + Automata unit testleri (11 test)
└── main.py                        # Uctan uca pipeline
```

---

## Tablo 1 — Model Karsilastirmasi (Orijinal Test Verisi, 5 Seed Ortalamasi)

Sonuclar `logs/results_20260517_171429_table1.csv` dosyasindan alinmistir.

### SWAT Veri Seti

| Model    | F1 (ort ± std)  | Accuracy (ort ± std) | Egitim Süresi (s) |
|----------|-----------------|----------------------|-------------------|
| LSTM     | 0.0000 ± 0.0000 | 0.9528 ± 0.0235      | ~2.6              |
| GRU      | 0.0000 ± 0.0000 | 0.9587 ± 0.0082      | ~1.4              |
| CNN1D    | 0.0000 ± 0.0000 | 0.9803 ± 0.0103      | ~8.0              |
| Automata | 0.0000 ± 0.0000 | 0.9446 ± 0.0000      | ~0.8              |

> SWAT veri setinde tüm modeller F1=0 üretiyor. Sinif dengesizligi (~%5 anomali) nedeniyle
> modeller tümünü "Normal" olarak siniflandiriyor; bu yüksek accuracy ile sifir F1'ye yol aciyor.

### WADI Veri Seti

| Model    | F1 (ort ± std)  | Accuracy (ort ± std) | En Iyi Seed F1 |
|----------|-----------------|----------------------|----------------|
| LSTM     | 0.2117 ± 0.1853 | 0.6852 ± 0.2508      | 0.5455 (s=999) |
| GRU      | 0.3762 ± 0.2692 | 0.7042 ± 0.3160      | 0.7169 (s=7)   |
| CNN1D    | 0.1289 ± 0.0733 | 0.5623 ± 0.2656      | 0.2291 (s=999) |
| Automata | 0.0002 ± 0.0000 | 0.0098 ± 0.0000      | 0.0002         |

> GRU, WADI veri setinde en yüksek ortalama F1 skoruna ulasyor. Yüksek seed varyans (±0.27)
> modelin baslangic agirliklarına duyarli oldugunu gösteriyor.

---

## SKAB ve BATADAL Veri Seti Karsilastirmasi

| Özellik              | SKAB                        | BATADAL                       |
|----------------------|-----------------------------|-------------------------------|
| Bölme stratejisi     | GroupShuffleSplit (dosya)   | Sirali (temporal)             |
| Etiket sütunu        | `anomaly`                   | `ATT_FLAG`                    |
| Anomali orani        | Düsük (~%5)                 | Daha yüksek (~%10)            |
| Boyut                | Çok özellikli               | Çok özellikli                 |
| DL performansi       | F1 ≈ 0 (tüm modeller)       | Degisken (0.0–0.72 arasi)     |
| Otomata performansi  | F1 ≈ 0                      | F1 ≈ 0.0002 (düsük)           |
| Ön isleme            | Scaler+PCA yalnizca train'e | Scaler+PCA yalnizca train'e   |

---

## Tablo 2 — Gurultü Dayanikliligi ve Unseen Pattern Analizi

Sonuclar `logs/results_20260517_171857_table2.csv` dosyasindan alinmistir.

### WADI — Orijinal vs Gurültülü Karsilastirmasi (5 Seed Ortalamasi)

| Model    | Orijinal F1 | Gurültülü F1 | F1 Düsüsü |
|----------|-------------|--------------|------------|
| LSTM     | 0.2117      | 0.2078       | -0.0039    |
| GRU      | 0.3762      | 0.3673       | -0.0089    |
| CNN1D    | 0.1289      | 0.1275       | -0.0014    |
| Automata | 0.0002      | 0.0002       | 0.0000     |

> Gaussian gürültü (sigma=0.1) tüm modellerde minimal performans düsüsüne neden olmustur.

### Unseen Pattern Analizi (Automata, Seed=42)

| Veri Seti | Unseen Rate | F1 (Unseen) | F1 (Orijinal) |
|-----------|-------------|-------------|---------------|
| SWAT      | 0.0028      | 0.0000      | 0.0000        |
| WADI      | 0.0008      | 0.0002      | 0.0002        |

> Test verisindeki örüntülerin %0.1–%0.3'ü eğitimde görülmemiş. Bu görülmemis örüntüler
> Levenshtein mesafesiyle en yakin bilinen örüntüye eslerek sistem çökmüyor.

---

## Tablo 3 — Çapraz Veri Seti Genelleme Analizi

Sonuclar `logs/results_20260517_172751_table3.csv` dosyasindan alinmistir (Seed=42).

| Model    | Egitim -> Test | F1     | Accuracy |
|----------|----------------|--------|----------|
| LSTM     | SWAT -> WADI   | 0.0266 | 0.9707   |
| GRU      | SWAT -> WADI   | 0.0134 | 0.9706   |
| CNN1D    | SWAT -> WADI   | 0.0331 | 0.9708   |
| Automata | SWAT -> WADI   | 0.0049 | 0.9597   |
| LSTM     | WADI -> SWAT   | 0.0000 | 1.0000   |
| GRU      | WADI -> SWAT   | 0.0000 | 1.0000   |
| CNN1D    | WADI -> SWAT   | 0.0000 | 1.0000   |
| Automata | WADI -> SWAT   | 0.0000 | 0.0998   |

> Capraz veri seti genellemesi düsük F1 skorlari üretiyor. WADI->SWAT yönünde DL modelleri
> %100 dogruluk elde ediyor ancak bu "hepsini Normal say" stratejisidir (F1=0). Domain shift
> zorlugunu acikca gösteriyor.

---

## Tablo 4 — Parametre Duyarlilik Analizi

Sonuclar `logs/results_20260517_172822_table4.csv` dosyasindan alinmistir (Automata, Seed=42).

### WADI — Farkli window_size × alphabet_size Kombinasyonlari

| window_size | alphabet_size | F1     | Accuracy |
|-------------|---------------|--------|----------|
| 3           | 3             | 0.0002 | 0.0099   |
| 3           | 5             | 0.0014 | 0.8594   |
| 4           | 3             | 0.0002 | 0.0098   |
| 4           | 5             | 0.0014 | 0.8565   |
| 5           | 5             | 0.0014 | 0.8576   |
| 6           | 5             | 0.0014 | 0.8537   |
| 6           | 6             | 0.0012 | 0.8345   |

> `alphabet_size=5` sabit tutuldugunda farkli `window_size` degerleri benzer F1 üretiyor.
> `alphabet_size=3` ile dogruluk çöküyor (~1%); `alphabet_size=5` optimal görünüyor.

Görsel: [`outputs/param_sensitivity_WADI.png`](outputs/param_sensitivity_WADI.png)

---

## Aciklanabilirlik Modülü

`src/explainability/` modülü, otomatanin her zaman adimindaki kararini açiklayan bir
raporlama altyapisi sunar.

### Nasil Çalisir

1. **SAX Kodlama:** Her pencere (window_size=4) bir karakter dizisine (örn. `"2222"`) dönüstürülür.
2. **Geçis Olasiligi:** Ardisik iki durum arasindaki ögrenilmis olasilik hesaplanir.
3. **Unseen Yönetimi:** Egitimde görülmemis bir durum Levenshtein mesafesiyle en yakin bilinen duruma eslenir.
4. **Karar:** Olasilik < esik → Anomali; >= esik → Normal.
5. **Güven Skoru:** Dogrudan geçis olasiligina esittir (yüksek olasilik = yüksek güven).

### Örnek `explain()` Çiktisi — Normal Durum

```json
{
  "time_step": 650,
  "state": "2222",
  "pattern": "2222",
  "status": "seen",
  "mapped_to": "2222",
  "probability": 0.609260,
  "decision": "Normal",
  "confidence": 0.6093
}
```

### Örnek `explain()` Çiktisi — Unseen Anomali

```json
{
  "time_step": 32,
  "state": "0012",
  "pattern": "0121",
  "status": "unseen",
  "mapped_to": "0021",
  "probability": 0.003970,
  "decision": "Anomaly",
  "confidence": 0.004
}
```

Tam rapor: [`outputs/results/automata_explanation_report.csv`](outputs/results/automata_explanation_report.csv)

---

## Istatistiksel Anlamlilik Testleri

`main.py` çalistirildiginda `src/utils/stats.py` modülündeki testler otomatik uygulanir
ve sonuçlar `logs/` dosyalarina kaydedilir.

### Wilcoxon Isartelik Sira Testi

5 seed'in F1 skorlarini karsilastirir. Uygulanan çiftler:

- LSTM vs GRU
- LSTM vs Automata
- GRU vs Automata
- LSTM vs CNN1D

### McNemar Testi

En son seed'in örnek bazli tahminleri üzerinde çalisir:

- LSTM vs GRU
- LSTM vs Automata

**Ornek çikti formati:**

```
[Istatistiksel Testler] SKAB
  Wilcoxon LSTM vs GRU: stat=1.0000, p=0.1250
  Wilcoxon LSTM vs Automata: stat=0.0000, p=0.0625
  McNemar LSTM vs GRU: stat=12.3456, p=0.0004
```

Sonuclar `logs/results_<timestamp>.csv` içinde `stat_test` senaryosu olarak saklanir.

---

## Görseller Referansi

| Görsel | Aciklama |
|--------|----------|
| [`outputs/cm_LSTM_SWAT.png`](outputs/cm_LSTM_SWAT.png) | LSTM Karisiklik Matrisi — SWAT |
| [`outputs/cm_GRU_SWAT.png`](outputs/cm_GRU_SWAT.png) | GRU Karisiklik Matrisi — SWAT |
| [`outputs/cm_CNN1D_SWAT.png`](outputs/cm_CNN1D_SWAT.png) | CNN1D Karisiklik Matrisi — SWAT |
| [`outputs/cm_Automata_SWAT.png`](outputs/cm_Automata_SWAT.png) | Automata Karisiklik Matrisi — SWAT |
| [`outputs/cm_LSTM_WADI.png`](outputs/cm_LSTM_WADI.png) | LSTM Karisiklik Matrisi — WADI |
| [`outputs/cm_GRU_WADI.png`](outputs/cm_GRU_WADI.png) | GRU Karisiklik Matrisi — WADI |
| [`outputs/cm_CNN1D_WADI.png`](outputs/cm_CNN1D_WADI.png) | CNN1D Karisiklik Matrisi — WADI |
| [`outputs/cm_Automata_WADI.png`](outputs/cm_Automata_WADI.png) | Automata Karisiklik Matrisi — WADI |
| [`outputs/roc_LSTM_WADI.png`](outputs/roc_LSTM_WADI.png) | LSTM ROC Egrisi — WADI |
| [`outputs/roc_GRU_WADI.png`](outputs/roc_GRU_WADI.png) | GRU ROC Egrisi — WADI |
| [`outputs/state_diagram_SWAT.png`](outputs/state_diagram_SWAT.png) | Otomata Durum Gecis Diyagrami — SWAT |
| [`outputs/state_diagram_WADI.png`](outputs/state_diagram_WADI.png) | Otomata Durum Gecis Diyagrami — WADI |
| [`outputs/transition_heatmap_SWAT.png`](outputs/transition_heatmap_SWAT.png) | Gecis Isi Haritasi — SWAT |
| [`outputs/transition_heatmap_WADI.png`](outputs/transition_heatmap_WADI.png) | Gecis Isi Haritasi — WADI |
| [`outputs/param_sensitivity_SWAT.png`](outputs/param_sensitivity_SWAT.png) | Parametre Duyarlilik — SWAT |
| [`outputs/param_sensitivity_WADI.png`](outputs/param_sensitivity_WADI.png) | Parametre Duyarlilik — WADI |
| [`outputs/figures/automata_graph.png`](outputs/figures/automata_graph.png) | Otomata Gecis Grafigi |
| [`outputs/figures/dl_confusion_matrix.png`](outputs/figures/dl_confusion_matrix.png) | DL Karisiklik Matrisi (pipeline) |

---

## Kaynaklar

1. Lin, J., Keogh, E., Wei, L., & Lonardi, S. (2007). Experiencing SAX: a novel symbolic representation of time series. *Data Mining and Knowledge Discovery*, 15(2), 107-144.
2. Schmidhuber, J. (2015). Deep learning in neural networks: An overview. *Neural Networks*, 61, 85-117.
3. Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP 2014*.
4. Braei, M., & Wagner, S. (2020). Anomaly Detection in Univariate Time-series: A Survey on the State-of-the-Art. *arXiv:2004.00433*.
5. Taormina, R., et al. (2018). The Battle Of The Attack Detection Algorithms: Disclosing Cyber Attacks On Water Distribution Networks. *Journal of Water Resources Planning and Management*.
6. Filonov, P., Lavrentyev, A., & Vorontsov, A. (2016). Multivariate Industrial Time Series with Cyber-Attack Simulation: Fault Detection Using an LSTM-based Predictive Data Model. *NIPS 2016 Workshop*.
7. Navarro-Almanza, R., et al. (2017). Towards explainable deep learning for anomaly detection in time series. *IEEE SSCI*.
