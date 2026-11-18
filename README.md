# Kuş Türlerinin Görsel ve İşitsel Verilerin Füzyonuyla Multimodal Derin Öğrenme Tabanlı Sınıflandırılması

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Audio--Visual%20Fusion-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-MultimodalBirds%20v1.0.0-teal.svg)](https://zenodo.org/records/17633387)
[![License](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Özet:** Kuş fotoğrafı (**iNaturalist RGB**) ile 3 saniyelik kuş ötüşü kaydını (**Xeno-Canto 16 kHz Mono WAV**) iki ayrı derin öğrenme kolunda işleyen, öğrenilebilir bir **Multimodal Feature Fusion** katmanında birleştirerek tür düzeyinde sınıflandırma yapan proje deposu.  
> Bu aşama **Veri Toplama, Temizleme, Tür Bazlı Eşleştirme, Ön İşleme, Keşifsel Veri Analizi (EDA)** ve grup bazlı sızıntısız **PyTorch Multimodal Füzyon** altyapısını içermektedir.

---

## Temel Veri Denetimi ve Pipeline Özeti

| Metrik / Aşama | Değer | Açıklama |
|---|---:|---|
| **Ham Veri Havuzu** | `108.280` | 130 kuş türü (`image` + `sound` satırları) |
| **Temizlikte Elenen Kayıt** | `921` (`%0.85`) | 312 format hatası, 265 düşük BirdNET (`<0.25`), 184 aşırı sessiz, 118 kopya hash, 42 bozuk dosya |
| **Temizlenmiş Unbalanced** | `107.359` | 130 tür |
| **Ana Deney Seti (Regular 33)** | **`27.720` kayıt / `23.100` çift** | **33 tür**, her türde `140` RGB görüntü + `700` ses segmenti (`700 çift/sınıf`) |
| **Prototip Alt Kümesi (Tiny)** | `2.800` kayıt / `1.400` çift | 10 tür, hızlı hata ayıklama için |
| **Veri Bölme (Group-Safe)** | `%70 / %15 / %15` | `16.170` Train / `3.465` Val / `3.465` Test (`for_test` + `base_filename` + `md5_hash` korumalı) |
| **Veri Sızıntısı (Data Leakage)** | **`0` paylaşılan kayıt** | Train/Val/Test arasında 0 ortak `base_filename` ve 0 ortak `image_md5` |

---

## Proje Klasör Yapısı

```text
bird-multimodal-classifier/
├── configs/
│   ├── tiny.yaml
│   └── regular.yaml
├── data/
│   ├── raw/
│   │   ├── metadata/                  # multimodal_bird_info.csv, label_mapping.csv
│   │   └── samples/                   # Türlere ait 128x128 RGB .jpg ve 16kHz 3s .wav örnekleri
│   └── splits/
│       └── pair_manifest.csv          # Deterministik, sızıntısız görüntü-ses eşleşme manifesti
├── src/
│   ├── data/
│   │   ├── prepare_data.py            # Zenodo şemasına uygun veri ve örnek medya hazırlığı
│   │   ├── audit_dataset.py           # Veri miktarı, kalite sorunları ve istatistik denetimi
│   │   ├── pairing.py                 # Tür bazlı eşleştirme ve grup bazlı sızıntısız split
│   │   ├── transforms.py              # Görüntü (224x224) ve Ses (128-bin Log-Mel) ön işleme
│   │   └── dataset.py                 # PyTorch MultimodalBirdDataset ve DataLoader doğrulaması
│   ├── models/
│   │   ├── image_encoder.py           # EfficientNet-B0 görsel özellik çıkarıcı (1280 -> 256)
│   │   ├── audio_encoder.py           # 2D CNN Log-Mel akustik özellik çıkarıcı (256)
│   │   └── fusion_model.py            # Feature Concatenation (512 -> 256 -> num_classes)
│   └── eda/
│       ├── generate_charts.py         # 13 adet 300 DPI bilimsel EDA grafiği üreticisi
│       └── build_notebook.py          # Çalıştırılmış Jupyter Notebook oluşturucu
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── outputs/
│   ├── figures/                       # fig01 .. fig13 (300 DPI PNG)
│   └── reports/                       # CSV/JSON veri denetim ve sızıntı doğrulama raporları
└── tests/
    └── test_pipeline.py
```

---

## Kurulum ve Yeniden Üretim (Reproducibility)

```bash
pip install -r requirements.txt

# 1. Veri ve örnek medya dosyalarını hazırla
python -m src.data.prepare_data

# 2. Veri denetimini ve kalite raporlarını üret
python -m src.data.audit_dataset

# 3. Grup bazlı sızıntısız eşleştirme ve split manifestini oluştur
python -m src.data.pairing

# 4. 13 bilimsel EDA grafiğini ve çalıştırılmış Jupyter Notebook'u üret
python -m src.eda.generate_charts
python -m src.eda.build_notebook

# 5. Tüm veri hattını, 0 sızıntıyı ve PyTorch füzyon ileri besleme testlerini doğrula
python -m pytest -v tests/test_pipeline.py
```

---

## Veri Kaynağı ve Atıf
- Courtois, J. vd. (2026). *MultimodalBirds: A Multimodal Image–Audio Dataset for Bird Species Classification*. Zenodo v1.0.0. DOI: [10.5281/zenodo.17633387](https://doi.org/10.5281/zenodo.17633387) (CC BY 4.0).
- Orijinal kaynaklar: iNaturalist 2021 Birds Only & Xeno-Canto Bird Recordings Extended.
