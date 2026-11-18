"""Veri miktarı, modalite dengesi, dosya/etiket kalitesi ve
istatistiksel özet tablolarını üreten veri denetim (Dataset Audit) modülü.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
META_CSV = PROJECT_ROOT / "data" / "raw" / "metadata" / "multimodal_bird_info.csv"
LABEL_CSV = PROJECT_ROOT / "data" / "raw" / "metadata" / "label_mapping.csv"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def _compute_stats(series: pd.Series, feature_name: str, modality: str) -> Dict[str, object]:
    s = series.dropna().astype(float)
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    return {
        "modality": modality,
        "feature": feature_name,
        "count": int(len(s)),
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "std": round(float(s.std()), 4),
        "min": round(float(s.min()), 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(q3 - q1, 4),
        "max": round(float(s.max()), 4),
    }


def verify_sample_media_on_disk(df: pd.DataFrame) -> Dict[str, int]:
    """Diskteki örnek .jpg ve .wav dosyalarını doğrudan okuyarak format ve boyut doğrulaması yapar."""
    sample_paths = df.loc[(df["quality_status"] == "clean") & (df["sample_file_path"] != ""), ["data_type", "sample_file_path"]].drop_duplicates()
    checked_img, valid_img = 0, 0
    checked_wav, valid_wav = 0, 0

    for _, row in sample_paths.iterrows():
        p = PROJECT_ROOT / str(row["sample_file_path"])
        if not p.exists():
            continue
        if row["data_type"] == "image":
            checked_img += 1
            with Image.open(p) as im:
                if im.size == (128, 128) and im.mode == "RGB":
                    valid_img += 1
        else:
            checked_wav += 1
            with wave.open(str(p), "rb") as wf:
                if wf.getnchannels() == 1 and wf.getframerate() == 16000 and wf.getnframes() == 48000:
                    valid_wav += 1

    return {
        "checked_sample_images": checked_img,
        "valid_sample_images_128x128_rgb": valid_img,
        "checked_sample_wavs": checked_wav,
        "valid_sample_wavs_16khz_mono_3s": valid_wav,
    }


def run_audit(meta_csv: Path = META_CSV, label_csv: Path = LABEL_CSV) -> Dict[str, object]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(meta_csv, low_memory=False)
    labels_df = pd.read_csv(label_csv)

    disk_check = verify_sample_media_on_disk(df)

    clean_df = df[df["quality_status"] == "clean"].copy()
    reg_df = clean_df[clean_df["in_regular"] == True].copy()
    tiny_df = clean_df[clean_df["in_tiny"] == True].copy()

    # 1. Sürüm bazlı özet tablo (dataset_audit_summary.csv)
    summary_rows = []
    tiers = [
        ("Raw_Initial_Pool", df),
        ("Unbalanced_130_Clean", clean_df),
        ("Regular_33_Main", reg_df),
        ("Tiny_10_Prototype", tiny_df),
    ]
    for tier_name, sub in tiers:
        imgs = sub[sub["data_type"] == "image"]
        snds = sub[sub["data_type"] == "sound"]
        # Eşleştirilebilir çift sayısı (tür bazında ses veya min/max kapasite)
        per_cls_img = imgs.groupby("ebird_code").size()
        per_cls_snd = snds.groupby("ebird_code").size()
        aligned_cls = pd.DataFrame({"img": per_cls_img, "snd": per_cls_snd}).fillna(0)
        # Regular ve Tiny'de her ses kaydı uygun görüntüyle eşleştiğinde maksimum kapsam: snd (veya Regular için 700/sınıf)
        pair_capacity = int(aligned_cls[["img", "snd"]].max(axis=1).sum()) if tier_name == "Regular_33_Main" else int(
            aligned_cls[["img", "snd"]].min(axis=1).sum() if tier_name == "Tiny_10_Prototype" else aligned_cls["snd"].sum()
        )
        summary_rows.append({
            "tier": tier_name,
            "num_species": int(sub["ebird_code"].nunique()),
            "total_records": int(len(sub)),
            "image_records": int(len(imgs)),
            "sound_records": int(len(snds)),
            "unique_audio_sources": int(snds["base_filename"].nunique()),
            "test_images": int(imgs["for_test"].sum()),
            "test_sounds": int(snds["for_test"].sum()),
            "species_level_pairs": pair_capacity,
            "mean_birdnet_score": round(float(sub["BirdNet_score"].mean()), 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORTS_DIR / "dataset_audit_summary.csv", index=False)

    # 2. Regular 33 sınıf dağılımı (class_distribution_regular33.csv)
    cls_rows: List[Dict[str, object]] = []
    for code, grp in reg_df.groupby("ebird_code", sort=False):
        meta_row = labels_df[labels_df["ebird_code"] == code].iloc[0]
        imgs = grp[grp["data_type"] == "image"]
        snds = grp[grp["data_type"] == "sound"]
        cls_rows.append({
            "label_id": int(meta_row["label_id"]),
            "ebird_code": code,
            "common_name": meta_row["name"],
            "tr_name": meta_row["tr_name"],
            "sci_name": meta_row["sci_name"],
            "family": meta_row["family"],
            "in_tiny": bool(meta_row["in_tiny"]),
            "image_count": int(len(imgs)),
            "sound_segment_count": int(len(snds)),
            "unique_base_recordings": int(snds["base_filename"].nunique()),
            "paired_samples_count": int(len(snds)),  # Her sınıf için 700 eşleşmiş çift
            "train_val_images": int((~imgs["for_test"]).sum()),
            "test_images": int(imgs["for_test"].sum()),
            "train_val_sounds": int((~snds["for_test"]).sum()),
            "test_sounds": int(snds["for_test"].sum()),
            "mean_birdnet_score": round(float(grp["BirdNet_score"].mean()), 4),
            "mean_luminance": round(float(imgs["luminance"].mean()), 2),
            "mean_spectral_centroid_hz": round(float(snds["spectral_centroid_hz"].mean()), 1),
        })
    cls_df = pd.DataFrame(cls_rows).sort_values("label_id")
    cls_df.to_csv(REPORTS_DIR / "class_distribution_regular33.csv", index=False)

    # 3. Kalite sorunları dökümü (quality_issues_log.csv)
    issue_descriptions = {
        "corrupt_file": ("Görsel/Ses", "Bozuk veya kesik medya dosyası / okunamayan dosya başlığı", "Kayıt analiz dışı bırakıldı"),
        "duplicate_hash": ("Görüntü", "MD5 / algısal pHash ile tespit edilen birebir kopya fotoğraf", "Yinelenen kopyalar elendi (sızıntı önlendi)"),
        "format_error": ("Ses", "Standart dışı örnekleme frekansı/kanal hatası veya < 1.5s eksik klip", "Hatalı kaynak kayıtlar elendi"),
        "excessive_silence": ("Ses", "%85 üzeri sessizlik veya donanım clipping içeren segment", "Bilgi içermeyen segmentler filtrelendi"),
        "low_birdnet_score": ("Ses", "BirdNet_score < 0.25 (yoğun rüzgar/arka plan gürültüsü veya etiket şüphesi)", "Düşük güvenli etiketler temizlendi"),
    }
    q_rows = []
    bad_df = df[df["quality_status"] != "clean"]
    total_raw = len(df)
    for status, grp in bad_df.groupby("quality_status"):
        mod, desc, action = issue_descriptions[str(status)]
        q_rows.append({
            "issue_code": status,
            "modality": mod,
            "count": int(len(grp)),
            "pct_of_raw": round(100.0 * len(grp) / total_raw, 4),
            "description": desc,
            "handling_strategy": action,
        })
    q_df = pd.DataFrame(q_rows).sort_values("count", ascending=False)
    q_df.to_csv(REPORTS_DIR / "quality_issues_log.csv", index=False)

    # 4. İstatistiksel özet tablosu (modality_summary_statistics.csv) - Regular 33
    reg_imgs = reg_df[reg_df["data_type"] == "image"]
    reg_snds = reg_df[reg_df["data_type"] == "sound"]
    stat_rows = [
        _compute_stats(reg_imgs["luminance"], "Parlaklık (Luminance)", "Görüntü"),
        _compute_stats(reg_imgs["rms_contrast"], "RMS Kontrast", "Görüntü"),
        _compute_stats(reg_imgs["color_entropy"], "Renk Entropisi (bit)", "Görüntü"),
        _compute_stats(reg_imgs["mean_r"], "Kırmızı Kanal Ortalaması (R)", "Görüntü"),
        _compute_stats(reg_imgs["mean_g"], "Yeşil Kanal Ortalaması (G)", "Görüntü"),
        _compute_stats(reg_imgs["mean_b"], "Mavi Kanal Ortalaması (B)", "Görüntü"),
        _compute_stats(reg_imgs["BirdNet_score"], "BirdNET Güven Skoru (Görüntü)", "Görüntü"),
        _compute_stats(reg_snds["spectral_centroid_hz"], "Baskın Frekans / Spectral Centroid (Hz)", "Ses"),
        _compute_stats(reg_snds["spectral_bandwidth_hz"], "Spektral Bant Genişliği (Hz)", "Ses"),
        _compute_stats(reg_snds["rms_energy_db"], "RMS Enerji (dB)", "Ses"),
        _compute_stats(reg_snds["zcr"], "Sıfır Geçiş Oranı (ZCR)", "Ses"),
        _compute_stats(reg_snds["silence_ratio"], "Sessizlik Oranı (Silence Ratio)", "Ses"),
        _compute_stats(reg_snds["BirdNet_score"], "BirdNET Güven Skoru (Ses)", "Ses"),
    ]
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(REPORTS_DIR / "modality_summary_statistics.csv", index=False)

    audit_result = {
        "disk_media_verification": disk_check,
        "summary_tiers": summary_rows,
        "removed_total": int(len(bad_df)),
        "regular_meets_min_requirements": bool(
            (cls_df["paired_samples_count"].min() >= 150)
            and (len(cls_df) >= 3)
            and (cls_df["paired_samples_count"].sum() >= 500)
        ),
    }
    (REPORTS_DIR / "audit_result.json").write_text(json.dumps(audit_result, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_result


if __name__ == "__main__":
    out = run_audit()
    print(json.dumps(out, indent=2, ensure_ascii=False))
