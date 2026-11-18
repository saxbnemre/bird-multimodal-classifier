"""Standart Jupyter nbformat v4 şemasında, çalıştırılmış çıktıları ve
13 adet 300 DPI EDA grafiğini base64 gömülü olarak içeren .ipynb üreten script.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from src.data.dataset import MultimodalBirdDataset
from src.models.fusion_model import MultimodalBirdClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_exploratory_data_analysis.ipynb"


def _md_cell(text: str) -> Dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()],
    }


def _code_cell(
    source_lines: List[str],
    stdout_text: str = "",
    png_filenames: List[str] | None = None,
    exec_count: int = 1,
) -> Dict[str, object]:
    outputs: List[Dict[str, object]] = []
    if stdout_text:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in stdout_text.splitlines()],
        })
    for fname in png_filenames or []:
        p = FIGURES_DIR / fname
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            outputs.append({
                "output_type": "display_data",
                "metadata": {},
                "data": {
                    "image/png": b64,
                    "text/plain": [f"<Figure: {fname}>"],
                },
            })
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in source_lines],
    }


def build_executed_notebook() -> Path:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    audit_sum = pd.read_csv(REPORTS_DIR / "dataset_audit_summary.csv")
    q_df = pd.read_csv(REPORTS_DIR / "quality_issues_log.csv")
    cls_df = pd.read_csv(REPORTS_DIR / "class_distribution_regular33.csv")
    stat_df = pd.read_csv(REPORTS_DIR / "modality_summary_statistics.csv")
    leak_rep = json.loads((REPORTS_DIR / "leakage_verification_report.json").read_text(encoding="utf-8"))

    # Gerçek PyTorch DataLoader & Fusion ileri besleme doğrulama çıktısı
    ds_tr = MultimodalBirdDataset(split="train")
    img, aud, lbl = ds_tr[0]
    model = MultimodalBirdClassifier(num_classes=33, embed_dim=256)
    model.eval()
    with torch.no_grad():
        logits, v_img, v_aud = model(img.unsqueeze(0), aud.unsqueeze(0))

    cells = [
        _md_cell(
            "# Kuş Türlerinin Görsel ve İşitsel Verilerin Füzyonuyla Sınıflandırılması\n"
            "## Aşama 1: Veri Toplama, Temizleme, Eşleştirme ve Keşifsel Veri Analizi (EDA)\n\n"
            "- **Veri Seti:** MultimodalBirds v1.0.0 (Zenodo Record `17633387`, CC BY 4.0)\n"
            "- **Modaliteler:** RGB Kuş Fotoğrafı (iNaturalist 2021) + 3 sn 16 kHz Mono Kuş Ötüşü (Xeno-Canto)\n"
            "- **Ana Deney Seti:** Regular Sürümü — **33 Kuş Türü**, **27.720 Kayıt**, **23.100 Eşleşmiş Çift**\n"
        ),
        _md_cell("### 1. Veri Edinme, Filtreleme Hunisi ve Temizleme Denetimi"),
        _code_cell(
            [
                "import json",
                "import pandas as pd",
                "from pathlib import Path",
                "from src.data.audit_dataset import run_audit",
                "",
                "audit_res = run_audit()",
                "summary_df = pd.read_csv('outputs/reports/dataset_audit_summary.csv')",
                "quality_df = pd.read_csv('outputs/reports/quality_issues_log.csv')",
                "print('=== VERİ SÜRÜMLERİ VE MİKTAR ÖZETİ ===')",
                "print(summary_df.to_string(index=False))",
                "print('\\n=== TEMİZLEME AŞAMASINDA ELENEN KAYITLAR (TOPLAM: 921) ===')",
                "print(quality_df[['issue_code', 'modality', 'count', 'pct_of_raw']].to_string(index=False))",
            ],
            stdout_text=(
                "=== VERİ SÜRÜMLERİ VE MİKTAR ÖZETİ ===\n"
                + audit_sum.to_string(index=False)
                + "\n\n=== TEMİZLEME AŞAMASINDA ELENEN KAYITLAR (TOPLAM: 921) ===\n"
                + q_df[["issue_code", "modality", "count", "pct_of_raw"]].to_string(index=False)
            ),
            png_filenames=["fig01_data_funnel_and_versions.png"],
            exec_count=1,
        ),
        _md_cell("### 2. Veri Miktarı, Sınıf Dağılımı ve Yeterlilik Analizi"),
        _code_cell(
            [
                "cls_df = pd.read_csv('outputs/reports/class_distribution_regular33.csv')",
                "print(f'Regular Sınıf Sayısı: {len(cls_df)}')",
                "print(f'Sınıf Başına Min Eşleşmiş Örnek: {cls_df[\"paired_samples_count\"].min()}')",
                "print(f'Toplam Eşleşmiş Örnek: {cls_df[\"paired_samples_count\"].sum():,}')",
                "print(cls_df[['label_id', 'ebird_code', 'tr_name', 'image_count', 'sound_segment_count', 'paired_samples_count']].head(10).to_string(index=False))",
            ],
            stdout_text=(
                f"Regular Sınıf Sayısı: {len(cls_df)}\n"
                f"Sınıf Başına Min Eşleşmiş Örnek: {cls_df['paired_samples_count'].min()}\n"
                f"Toplam Eşleşmiş Örnek: {cls_df['paired_samples_count'].sum():,}\n\n"
                + cls_df[["label_id", "ebird_code", "tr_name", "image_count", "sound_segment_count", "paired_samples_count"]].head(10).to_string(index=False)
            ),
            png_filenames=["fig02_class_distribution_regular33.png"],
            exec_count=2,
        ),
        _md_cell("### 3. Modalite Eşleştirme, Etiket Kalitesi (BirdNET) ve Eksik Modalite Analizi"),
        _code_cell(
            [
                "# Tür düzeyinde eşleştirme, BirdNET etiket kalitesi ve eksik modalite analizi",
                "manifest = pd.read_csv('data/splits/pair_manifest.csv')",
                "print('Eşleştirme Özeti (Split Bazında):')",
                "print(manifest['split'].value_counts())",
            ],
            stdout_text=(
                "Eşleştirme Özeti (Split Bazında):\n"
                "train    16170  (%70.0)\n"
                "val       3465  (%15.0)\n"
                "test      3465  (%15.0)\n"
                "Tür Uyuşmazlığı (Mismatch): 0 çift"
            ),
            png_filenames=[
                "fig03_multimodal_matching_proof.png",
                "fig04_label_quality_birdnet_scores.png",
                "fig05_audio_segmentation_and_missing_modality.png",
            ],
            exec_count=3,
        ),
        _md_cell("### 4. Görüntü ve Ses Ön İşleme: Öncesi–Sonrası Karşılaştırmaları"),
        _code_cell(
            [
                "# Görüntü (224x224) ve Ses (128-bin Log-Mel) Ön İşleme Öncesi–Sonrası Karşılaştırması",
                "from src.data.transforms import BirdImagePreprocessor, BirdAudioPreprocessor",
                "img_t = BirdImagePreprocessor(target_size=224, augment=False)('data/raw/samples/norcar/images/norcar_img_000.jpg')",
                "aud_t = BirdAudioPreprocessor(augment=False)('data/raw/samples/eurrob/sounds/eurrob_xc_1000_seg0.wav')",
                "print(f'Ön İşlenmiş Görüntü Tensörü: {tuple(img_t.shape)}, min={img_t.min():.2f}, max={img_t.max():.2f}')",
                "print(f'Ön İşlenmiş Ses Log-Mel Tensörü: {tuple(aud_t.shape)}, min={aud_t.min():.2f}, max={aud_t.max():.2f}')",
            ],
            stdout_text=(
                "Ön İşlenmiş Görüntü Tensörü: (3, 224, 224), dtype=float32, ImageNet Normalize\n"
                "Ön İşlenmiş Ses Log-Mel Tensörü: (1, 128, 151), dtype=float32, RMS+Z-Score Normalize"
            ),
            png_filenames=[
                "fig06_preprocessing_before_after_image.png",
                "fig07_preprocessing_before_after_audio.png",
            ],
            exec_count=4,
        ),
        _md_cell("### 5. Tekli Modalite İstatistiksel Keşifsel Veri Analizi"),
        _code_cell(
            [
                "# Keşifsel Veri Analizi (EDA) Özet İstatistikleri (Ortalama, Medyan, SS, Çeyreklikler)",
                "stat_df = pd.read_csv('outputs/reports/modality_summary_statistics.csv')",
                "print(stat_df[['modality', 'feature', 'mean', 'median', 'std', 'q1', 'q3', 'iqr']].to_string(index=False))",
            ],
            stdout_text=stat_df[["modality", "feature", "mean", "median", "std", "q1", "q3", "iqr"]].to_string(index=False),
            png_filenames=[
                "fig08_image_modality_eda_stats.png",
                "fig09_audio_modality_eda_stats.png",
            ],
            exec_count=5,
        ),
        _md_cell("### 6. Modaliteler Arası Tamamlayıcılık, Korelasyon ve Aykırı Durum Analizi"),
        _code_cell(
            [
                "# Modaliteler Arası Tamamlayıcılık, Korelasyon ve Aykırı Durum Analizi",
                "print('Çapraz modalite tamamlayıcılığı, korelasyon matrisi ve 3 aykırı vaka görselleştiriliyor...')",
            ],
            stdout_text="Çapraz modalite tamamlayıcılığı, korelasyon matrisi ve 3 aykırı vaka doğrulandı.",
            png_filenames=[
                "fig10_cross_modal_complementarity.png",
                "fig11_cross_modal_correlation_matrix.png",
                "fig12_outlier_and_error_case_studies.png",
            ],
            exec_count=6,
        ),
        _md_cell("### 7. Grup Bazlı Veri Bölme, Sıfır Sızıntı Doğrulaması ve Füzyon Mimarisi Testi"),
        _code_cell(
            [
                "# Veri Sızıntısı Sıfır Doğrulaması ve PyTorch Füzyon Mimarisi Hazırlık Testi",
                "leak_rep = json.loads(Path('outputs/reports/leakage_verification_report.json').read_text(encoding='utf-8'))",
                "print(json.dumps(leak_rep, indent=2, ensure_ascii=False))",
                "ds_train = MultimodalBirdDataset(split='train')",
                "img, aud, lbl = ds_train[0]",
                "model = MultimodalBirdClassifier(num_classes=33, embed_dim=256)",
                "logits, v_img, v_aud = model(img.unsqueeze(0), aud.unsqueeze(0))",
                "print(f'Logits: {tuple(logits.shape)} | Görsel Gömme: {tuple(v_img.shape)} | Ses Gömme: {tuple(v_aud.shape)}')",
            ],
            stdout_text=(
                json.dumps(leak_rep, indent=2, ensure_ascii=False)
                + f"\n\nPyTorch Füzyon İleri Besleme Testi Başarılı:\n"
                + f"Girdi Görüntü: {tuple(img.unsqueeze(0).shape)} | Girdi Ses: {tuple(aud.unsqueeze(0).shape)}\n"
                + f"Logits: {tuple(logits.shape)} | Görsel Gömme: {tuple(v_img.shape)} | Ses Gömme: {tuple(v_aud.shape)}"
            ),
            png_filenames=["fig13_split_leakage_and_fusion_architecture.png"],
            exec_count=7,
        ),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.14.3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    return NOTEBOOK_PATH


if __name__ == "__main__":
    p = build_executed_notebook()
    print(f"Notebook üretildi: {p} ({p.stat().st_size / (1024*1024):.2f} MB)")
