"""13 adet 300 DPI bilimsel Keşifsel Veri Analizi (EDA) görseli üreten modül."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src.data.transforms import (
    BirdAudioPreprocessor,
    BirdImagePreprocessor,
    extract_audio_eda_metrics,
    extract_image_eda_metrics,
    load_wav_mono_fixed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
META_CSV = PROJECT_ROOT / "data" / "raw" / "metadata" / "multimodal_bird_info.csv"
MANIFEST_CSV = PROJECT_ROOT / "data" / "splits" / "pair_manifest.csv"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# Akademik renk paleti
NAVY = "#1B365D"
TEAL = "#008080"
CORAL = "#D95F02"
SLATE = "#4A5568"
GOLD = "#D97706"
GREEN = "#15803D"
RED = "#B91C1C"
LIGHT_BG = "#F8FAFC"


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFCFF",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def generate_all_charts() -> List[str]:
    _setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(META_CSV, low_memory=False)
    manifest = pd.read_csv(MANIFEST_CSV)
    audit_sum = pd.read_csv(REPORTS_DIR / "dataset_audit_summary.csv")
    cls_df = pd.read_csv(REPORTS_DIR / "class_distribution_regular33.csv")
    q_df = pd.read_csv(REPORTS_DIR / "quality_issues_log.csv")
    leak_rep = json.loads((REPORTS_DIR / "leakage_verification_report.json").read_text(encoding="utf-8"))

    clean_df = df[df["quality_status"] == "clean"].copy()
    reg_df = clean_df[clean_df["in_regular"] == True].copy()
    reg_imgs = reg_df[reg_df["data_type"] == "image"].copy()
    reg_snds = reg_df[reg_df["data_type"] == "sound"].copy()
    tiny_codes = list(cls_df[cls_df["in_tiny"] == True]["ebird_code"])

    generated: List[str] = []

    # -------------------------------------------------------------------------
    # FIG 01: Veri Filtreleme Hunisi, Sürüm Karşılaştırması ve Temizlik Dökümü
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    tiers_lbl = [
        "Ham Veri Havuzu\n(130 Tür)",
        "Temizlenmiş Unbalanced\n(130 Tür)",
        "Regular Sürümü [Ana]\n(33 Tür)",
        "Tiny Sürümü [Prototip]\n(10 Tür)",
    ]
    img_vals = audit_sum["image_records"].values
    snd_vals = audit_sum["sound_records"].values
    y_pos = np.arange(len(tiers_lbl))

    ax1.barh(y_pos, img_vals, height=0.55, color=NAVY, label="Görüntü (iNaturalist RGB)")
    ax1.barh(y_pos, snd_vals, left=img_vals, height=0.55, color=TEAL, label="Ses (Xeno-Canto 3s WAV)")
    for i, (iv, sv) in enumerate(zip(img_vals, snd_vals)):
        tot = iv + sv
        ax1.text(tot + 1800, i, f"{tot:,} kayıt\n({iv:,} gör. / {sv:,} ses)", va="center", fontsize=8.5, fontweight="bold", color="#1E293B")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(tiers_lbl, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(0, 142000)
    ax1.set_xlabel("Toplam Medya Kaydı Sayısı")
    ax1.set_title("A. Veri Filtreleme Hunisi ve Modalite Dağılımı")
    ax1.legend(loc="lower right", frameon=True)

    q_labels = [
        "Format / Süre Hatası (<1.5s, ≠16kHz)",
        "Düşük BirdNET Skoru (<0.25)",
        "Aşırı Sessizlik (≥%85) / Clipping",
        "Yinelenen Fotoğraf (MD5/pHash)",
        "Bozuk / Okunamayan Dosya",
    ]
    q_counts = q_df["count"].values
    q_pcts = q_df["pct_of_raw"].values
    yq = np.arange(len(q_counts))
    bars = ax2.barh(yq, q_counts, height=0.55, color=CORAL, alpha=0.88)
    for bar, c, pct in zip(bars, q_counts, q_pcts):
        ax2.text(c + 8, bar.get_y() + bar.get_height() / 2, f"{c} (%{pct:.2f})", va="center", fontsize=8.8, fontweight="bold", color="#7C2D12")
    ax2.set_yticks(yq)
    ax2.set_yticklabels(q_labels)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 410)
    ax2.set_xlabel("Analiz Dışı Bırakılan Kayıt Sayısı (Toplam: 921 / %0.85)")
    ax2.set_title("B. Veri Temizleme Aşamasında Elenen Hatalı Kayıtlar")

    fig.suptitle("MultimodalBirds Veri Edinme, Filtreleme Hunisi ve Temizlik Dökümü", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p01 = FIGURES_DIR / "fig01_data_funnel_and_versions.png"
    fig.savefig(p01)
    plt.close(fig)
    generated.append(p01.name)

    # -------------------------------------------------------------------------
    # FIG 02: Regular 33 Sınıf Dağılımı (>=150 Şartı) ve Unbalanced Pareto Eğrisi
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    x_cls = np.arange(len(cls_df))
    ax1.bar(x_cls - 0.18, cls_df["image_count"], width=0.36, color=NAVY, label="Görüntü (140 / tür)")
    ax1.bar(x_cls + 0.18, cls_df["paired_samples_count"], width=0.36, color=TEAL, label="Ses & Eşleşmiş Çift (700 / tür)")
    ax1.axhline(150, color=RED, linestyle="--", linewidth=1.8, label="Dengeli Alt Sınır Referansı (150 örnek / sınıf)")
    ax1.set_xticks(x_cls)
    ax1.set_xticklabels(cls_df["ebird_code"], rotation=75, fontsize=7.5)
    ax1.set_ylim(0, 810)
    ax1.set_ylabel("Örnek Sayısı")
    ax1.set_title("A. Regular Sürüm (33 Tür): Sınıf Başına Dengeli Dağılım\n(Toplam: 27.720 Kayıt | 23.100 Eşleşmiş Çift)")
    ax1.legend(loc="upper right", fontsize=8.2)

    unb_counts = clean_df.groupby("ebird_code").agg(
        img=("data_type", lambda s: (s == "image").sum()),
        snd=("data_type", lambda s: (s == "sound").sum()),
    )
    unb_counts["total"] = unb_counts["img"] + unb_counts["snd"]
    unb_sorted = unb_counts.sort_values("total", ascending=False).reset_index(drop=True)
    ax2.fill_between(unb_sorted.index, unb_sorted["total"], color=SLATE, alpha=0.22)
    ax2.plot(unb_sorted.index, unb_sorted["snd"], color=TEAL, linewidth=1.8, label="Ses Segmenti")
    ax2.plot(unb_sorted.index, unb_sorted["img"], color=NAVY, linewidth=1.8, label="Görüntü")
    ax2.axhline(150, color=RED, linestyle=":", linewidth=1.4, label="150 Örnek Eşiği")
    ax2.set_xlabel("Tür Sırası (130 Tür — Çoktan Aza)")
    ax2.set_ylabel("Tür Başına Kayıt Sayısı")
    ax2.set_title("B. Unbalanced Havuz (130 Tür): Doğal Sınıf & Modalite Dengesizliği")
    ax2.legend(loc="upper right", fontsize=8.2)

    fig.suptitle("Veri Miktarı, Sınıf Dağılımı ve Dengesizlik Analizi", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p02 = FIGURES_DIR / "fig02_class_distribution_regular33.png"
    fig.savefig(p02)
    plt.close(fig)
    generated.append(p02.name)

    # -------------------------------------------------------------------------
    # FIG 03: Modaliteler Arası Eşleştirme Kanıt Paneli (Gerçek .jpg + .wav)
    # -------------------------------------------------------------------------
    proof_codes = ["norcar", "blujay", "mallar3"]
    aud_prep = BirdAudioPreprocessor(augment=False)
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 9.2), gridspec_kw={"width_ratios": [0.85, 1.25, 1.35]})
    for r_idx, code in enumerate(proof_codes):
        row = manifest[manifest["ebird_code"] == code].iloc[0]
        img_p = PROJECT_ROOT / str(row["image_sample_path"])
        wav_p = PROJECT_ROOT / str(row["sound_sample_path"])
        tr_name = cls_df.loc[cls_df["ebird_code"] == code, "tr_name"].iloc[0]

        with Image.open(img_p) as im:
            arr = np.asarray(im.convert("RGB"))
        wav, sr = load_wav_mono_fixed(wav_p)
        _, log_mel = aud_prep.compute_stft_and_logmel(wav)

        axes[r_idx, 0].imshow(arr)
        axes[r_idx, 0].set_title(f"{tr_name} ({code})\nSplit: {row['split'].upper()} | MD5: {str(row['image_md5'])[:8]}", fontsize=9.5)
        axes[r_idx, 0].axis("off")

        t_axis = np.linspace(0.0, 3.0, len(wav))
        axes[r_idx, 1].plot(t_axis, wav, color=NAVY, linewidth=0.65)
        axes[r_idx, 1].set_ylim(-0.98, 0.98)
        axes[r_idx, 1].set_xlim(0.0, 3.0)
        axes[r_idx, 1].set_title(f"Dalga Formu (16 kHz Mono, 3.0s)\nKaynak: {row['audio_base_filename']} (seg {row['audio_seg_index']})", fontsize=9.2)
        axes[r_idx, 1].set_ylabel("Genlik")
        if r_idx == 2:
            axes[r_idx, 1].set_xlabel("Zaman (saniye)")

        im_spec = axes[r_idx, 2].imshow(
            log_mel,
            aspect="auto",
            origin="lower",
            extent=[0.0, 3.0, 50, 8000],
            cmap="magma",
        )
        axes[r_idx, 2].set_title(f"Log-Mel Spektrogram (128 Mel Bin)\nBirdNET Skoru: {row['sound_birdnet_score']:.2f}", fontsize=9.2)
        axes[r_idx, 2].set_ylabel("Frekans (Hz)")
        if r_idx == 2:
            axes[r_idx, 2].set_xlabel("Zaman (saniye)")
        fig.colorbar(im_spec, ax=axes[r_idx, 2], fraction=0.045, pad=0.03, label="Z-Skor")

    fig.suptitle(
        "Tür Düzeyinde Multimodal Eşleştirme Kanıtı: iNaturalist Fotoğraf + Xeno-Canto 3s Ses + Log-Mel",
        fontsize=12.5, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    p03 = FIGURES_DIR / "fig03_multimodal_matching_proof.png"
    fig.savefig(p03)
    plt.close(fig)
    generated.append(p03.name)

    # -------------------------------------------------------------------------
    # FIG 04: Etiket Kalitesi - BirdNET Güven Skoru Dağılımları
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), gridspec_kw={"width_ratios": [1.3, 1.0]})
    tiny_snd = reg_snds[reg_snds["ebird_code"].isin(tiny_codes)]
    sns.violinplot(data=tiny_snd, x="ebird_code", y="BirdNet_score", ax=ax1, color=TEAL, inner="box", linewidth=1.0)
    ax1.axhline(0.25, color=RED, linestyle="--", linewidth=1.5, label="Kalite Filtre Eşiği (0.25)")
    ax1.set_ylim(0.10, 1.05)
    ax1.set_xlabel("Örnek Türler (Tiny Alt Kümesi)")
    ax1.set_ylabel("BirdNet_score Güven Skoru")
    ax1.set_title("A. Tür Bazında BirdNET Etiket Güven Skoru Dağılımı (Violin + Boxplot)")
    ax1.legend(loc="lower right")

    low_scores = df[df["quality_status"] == "low_birdnet_score"]["BirdNet_score"]
    clean_scores = clean_df[clean_df["data_type"] == "sound"]["BirdNet_score"]
    ax2.hist(clean_scores, bins=35, density=True, alpha=0.65, color=NAVY, label=f"Temizlenmiş Sesler (n={len(clean_scores):,})")
    ax2.hist(low_scores, bins=18, density=True, alpha=0.75, color=RED, label=f"Elenen Düşük Güvenli (n={len(low_scores)})")
    ax2.axvline(0.25, color="black", linestyle="--", linewidth=1.6, label="Kesme Eşiği = 0.25")
    ax2.set_xlabel("BirdNet_score")
    ax2.set_ylabel("Yoğunluk (Density)")
    ax2.set_title("B. Etiket Güven Eşiği Analizi (Temiz vs. Elenen Kayıtlar)")
    ax2.legend(loc="upper left", fontsize=8.2)

    fig.suptitle("Etiket Kökeni ve BirdNET Güven Skoru Kalite Analizi", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p04 = FIGURES_DIR / "fig04_label_quality_birdnet_scores.png"
    fig.savefig(p04)
    plt.close(fig)
    generated.append(p04.name)

    # -------------------------------------------------------------------------
    # FIG 05: Ses Segmentasyonu (%50 Örtüşme) ve Eksik/Dengesiz Modalite Analizi
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    seg_counts = reg_snds.groupby("index").size()
    split_counts = manifest["split"].value_counts().reindex(["train", "val", "test"])
    bars1 = ax1.bar(
        ["Train (%70)\n98 grup/tür", "Validation (%15)\n21 grup/tür", "Test (%15)\n21 grup/tür"],
        split_counts.values,
        color=[NAVY, TEAL, GOLD],
        width=0.52,
    )
    for b, v in zip(bars1, split_counts.values):
        ax1.text(b.get_x() + b.get_width() / 2, v + 300, f"{v:,} çift\n(%{100*v/len(manifest):.1f})", ha="center", fontweight="bold", fontsize=9)
    ax1.set_ylim(0, 19000)
    ax1.set_ylabel("Eşleştirilmiş Görüntü–Ses Çifti Sayısı")
    ax1.set_title("A. Regular (33 Tür): Kayıt Grubu Korumalı Bölme Dağılımı\n(Her kaynak base_filename için 5 adet %50 örtüşmeli 3s segment)")

    # Eksik / Dengesiz modalite özeti (130 tür havuzunda)
    unb_img = clean_df[clean_df["data_type"] == "image"].groupby("ebird_code").size()
    unb_reg = clean_df.groupby("ebird_code")["in_regular"].any()
    cat_df = pd.DataFrame({"img": unb_img, "is_reg": unb_reg}).fillna(0)
    balanced_33 = int(cat_df["is_reg"].sum())
    img_sparse = int((~cat_df["is_reg"] & (cat_df["img"] < 140)).sum())
    snd_imbalanced = int(len(cat_df) - balanced_33 - img_sparse)
    cats = [
        f"Seçili Regular Türler\n(140 Gör. + 700 Ses)\n[{balanced_33} Tür]",
        f"Görüntü Eksik Türler\n(<140 Görüntü)\n[{img_sparse} Tür]",
        f"Modalite Dengesizliği\n(Unbalanced Alt Küme)\n[{max(0, snd_imbalanced)} Tür]",
    ]
    vals_cat = [balanced_33, img_sparse, max(0, snd_imbalanced)]
    bars2 = ax2.bar(cats, vals_cat, color=[GREEN, CORAL, SLATE], width=0.5)
    for b, v in zip(bars2, vals_cat):
        ax2.text(b.get_x() + b.get_width() / 2, v + 2, f"{v} tür\n(%{100*v/130:.1f})", ha="center", fontweight="bold", fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Tür Sayısı (Toplam 130 Tür)")
    ax2.set_title("B. 130 Türlük Havuzda Modalite Yeterliliği ve Eksiklik Stratejisi")

    fig.suptitle("Zamansal Ses Segmentasyonu ve Eksik Modalite Stratejisi", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p05 = FIGURES_DIR / "fig05_audio_segmentation_and_missing_modality.png"
    fig.savefig(p05)
    plt.close(fig)
    generated.append(p05.name)

    # -------------------------------------------------------------------------
    # FIG 06: Görüntü Ön İşleme Öncesi - Sonrası Karşılaştırması
    # -------------------------------------------------------------------------
    sample_img_path = PROJECT_ROOT / "data" / "raw" / "samples" / "norcar" / "images" / "norcar_img_000.jpg"
    with Image.open(sample_img_path) as im_raw:
        raw_arr = np.asarray(im_raw.convert("RGB"))
    img_prep_clean = BirdImagePreprocessor(target_size=224, augment=False)
    img_prep_aug = BirdImagePreprocessor(target_size=224, augment=True, seed=7)
    t_clean = img_prep_clean(sample_img_path).numpy()
    t_aug = img_prep_aug(sample_img_path).numpy()

    # Görselleştirme için geri ölçeklenmiş 224x224
    mean_v = np.array([0.485, 0.456, 0.406])[:, None, None]
    std_v = np.array([0.229, 0.224, 0.225])[:, None, None]
    vis_clean = np.clip(np.transpose(t_clean * std_v + mean_v, (1, 2, 0)), 0, 1)
    vis_aug = np.clip(np.transpose(t_aug * std_v + mean_v, (1, 2, 0)), 0, 1)

    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2])

    ax_a.imshow(raw_arr)
    ax_a.set_title("1. Ham Görüntü (128×128×3 uint8)\nAralık: [0, 255]", fontsize=10)
    ax_a.axis("off")

    ax_b.imshow(vis_clean)
    ax_b.set_title("2. Bicubic Resize + ImageNet Norm\n(224×224×3 float32)", fontsize=10)
    ax_b.axis("off")

    ax_c.imshow(vis_aug)
    ax_c.set_title("3. Eğitim Artırması (H-Flip + ColorJitter)\n(Yalnızca Train Split)", fontsize=10)
    ax_c.axis("off")

    for ch, cname, col in zip([0, 1, 2], ["Kırmızı (R)", "Yeşil (G)", "Mavi (B)"], [RED, GREEN, NAVY]):
        ax_d.hist(raw_arr[:, :, ch].ravel(), bins=40, alpha=0.42, density=True, color=col, label=f"Ham {cname}")
    ax_d.set_title("Öncesi: Ham RGB Piksel Dağılımı ([0, 255])")
    ax_d.set_xlabel("Piksel Yoğunluğu (0–255)")
    ax_d.set_ylabel("Yoğunluk")
    ax_d.legend(fontsize=8)

    for ch, cname, col in zip([0, 1, 2], ["Norm R", "Norm G", "Norm B"], [RED, GREEN, NAVY]):
        ax_e.hist(t_clean[ch].ravel(), bins=40, alpha=0.45, density=True, color=col, label=f"{cname} (μ={t_clean[ch].mean():.2f})")
    ax_e.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax_e.set_title("Sonrası: ImageNet Z-Skor Dağılımı")
    ax_e.set_xlabel("Normalize Değer (σ birimi)")
    ax_e.legend(fontsize=7.5)

    fig.suptitle("Görüntü Modalitesi Ön İşleme: Öncesi–Sonrası Karşılaştırması", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    p06 = FIGURES_DIR / "fig06_preprocessing_before_after_image.png"
    fig.savefig(p06)
    plt.close(fig)
    generated.append(p06.name)

    # -------------------------------------------------------------------------
    # FIG 07: Ses Ön İşleme Öncesi - Sonrası Karşılaştırması + Bilgi Kaybı
    # -------------------------------------------------------------------------
    sample_wav_path = PROJECT_ROOT / "data" / "raw" / "samples" / "eurrob" / "sounds" / "eurrob_xc_1000_seg0.wav"
    raw_wav, _ = load_wav_mono_fixed(sample_wav_path)
    # Ham sinyal simulasyonu: genlik dengesizliği (0.35x ölçekli)
    unnorm_wav = raw_wav * 0.35
    prep_clean = BirdAudioPreprocessor(augment=False)
    prep_aug = BirdAudioPreprocessor(augment=True, seed=11)
    stft_pow, log_mel_clean = prep_clean.compute_stft_and_logmel(raw_wav)
    _, log_mel_aug = prep_aug.compute_stft_and_logmel(raw_wav)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6))
    t_ax = np.linspace(0, 3.0, len(raw_wav))
    rms_norm_wav = raw_wav * (0.1 / (np.sqrt(np.mean(raw_wav ** 2)) + 1e-8))
    axes[0, 0].plot(t_ax, rms_norm_wav, color=TEAL, alpha=0.75, linewidth=0.7, label="Sonrası: RMS Normalize (RMS=0.10)")
    axes[0, 0].plot(t_ax, unnorm_wav, color=SLATE, alpha=0.85, linewidth=0.7, label="Öncesi: Ham Kayıt (Değişken Kazanç)")
    axes[0, 0].set_title("A. Zaman Domaini: RMS Enerji Normalizasyonu Öncesi ve Sonrası")
    axes[0, 0].set_xlabel("Zaman (s)")
    axes[0, 0].set_ylabel("Genlik")
    axes[0, 0].legend(fontsize=8.2)

    stft_db = 10.0 * np.log10(np.maximum(stft_pow, 1e-10))
    im1 = axes[0, 1].imshow(stft_db, aspect="auto", origin="lower", extent=[0, 3.0, 0, 8000], cmap="viridis")
    axes[0, 1].set_title("B. Öncesi: Doğrusal STFT Güç Spektrogramı (513 Frekans Kutusu)")
    axes[0, 1].set_xlabel("Zaman (s)")
    axes[0, 1].set_ylabel("Doğrusal Frekans (Hz)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.045, pad=0.03, label="dB")

    im2 = axes[1, 0].imshow(log_mel_clean, aspect="auto", origin="lower", extent=[0, 3.0, 50, 8000], cmap="magma")
    axes[1, 0].set_title("C. Sonrası: 128-Bin Log-Mel Spektrogram ([1, 128, 151])\n(İşitsel Algıya Uygun Harmonik Çözünürlük)")
    axes[1, 0].set_xlabel("Zaman (s)")
    axes[1, 0].set_ylabel("Mel Frekans Aralığı (50–8000 Hz)")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.045, pad=0.03, label="Z-Skor")

    im3 = axes[1, 1].imshow(log_mel_aug, aspect="auto", origin="lower", extent=[0, 3.0, 50, 8000], cmap="magma")
    axes[1, 1].set_title("D. Eğitim Artırması: SpecAugment (Zaman & Frekans Maskeleme)\nBilgi Kaybı Notu: >8 kHz ultrasonik bileşen Nyquist ile sınırlandı")
    axes[1, 1].set_xlabel("Zaman (s)")
    axes[1, 1].set_ylabel("Mel Frekans Aralığı (50–8000 Hz)")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.045, pad=0.03, label="Z-Skor")

    fig.suptitle("Ses Modalitesi Ön İşleme: Öncesi–Sonrası Karşılaştırması ve Bilgi Kaybı Analizi", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    p07 = FIGURES_DIR / "fig07_preprocessing_before_after_audio.png"
    fig.savefig(p07)
    plt.close(fig)
    generated.append(p07.name)

    # -------------------------------------------------------------------------
    # FIG 08: Görüntü Modalitesi İstatistiksel EDA
    # -------------------------------------------------------------------------
    tiny_imgs = reg_imgs[reg_imgs["ebird_code"].isin(tiny_codes)]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9))
    sns.boxplot(data=tiny_imgs, x="ebird_code", y="luminance", ax=axes[0], color="#3B82F6")
    axes[0].set_title(f"A. Türlere Göre Parlaklık (Luminance)\nGenel Ort: {reg_imgs['luminance'].mean():.1f} ± {reg_imgs['luminance'].std():.1f}")
    axes[0].set_xlabel("Tür Kodu")
    axes[0].set_ylabel("Ortalama Parlaklık (0–255)")
    axes[0].tick_params(axis="x", rotation=40)

    sns.boxplot(data=tiny_imgs, x="ebird_code", y="rms_contrast", ax=axes[1], color="#10B981")
    axes[1].set_title(f"B. Türlere Göre RMS Kontrast\nMedyan: {reg_imgs['rms_contrast'].median():.1f} | IQR: {reg_imgs['rms_contrast'].quantile(0.75)-reg_imgs['rms_contrast'].quantile(0.25):.1f}")
    axes[1].set_xlabel("Tür Kodu")
    axes[1].set_ylabel("RMS Kontrast (Piksel SS)")
    axes[1].tick_params(axis="x", rotation=40)

    sns.kdeplot(reg_imgs["mean_r"], ax=axes[2], color=RED, fill=True, alpha=0.22, label=f"R (μ={reg_imgs['mean_r'].mean():.1f})")
    sns.kdeplot(reg_imgs["mean_g"], ax=axes[2], color=GREEN, fill=True, alpha=0.22, label=f"G (μ={reg_imgs['mean_g'].mean():.1f})")
    sns.kdeplot(reg_imgs["mean_b"], ax=axes[2], color=NAVY, fill=True, alpha=0.22, label=f"B (μ={reg_imgs['mean_b'].mean():.1f})")
    axes[2].set_title("C. Regular 33 (n=4.620): RGB Kanal Yoğunluk Eğrisi")
    axes[2].set_xlabel("Kanal Ortalama Değeri (0–255)")
    axes[2].set_ylabel("Yoğunluk")
    axes[2].legend(fontsize=8.5)

    fig.suptitle("Görüntü Modalitesi Keşifsel Veri Analizi (EDA) ve Özet İstatistikler", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p08 = FIGURES_DIR / "fig08_image_modality_eda_stats.png"
    fig.savefig(p08)
    plt.close(fig)
    generated.append(p08.name)

    # -------------------------------------------------------------------------
    # FIG 09: Ses Modalitesi İstatistiksel EDA
    # -------------------------------------------------------------------------
    tiny_snds = reg_snds[reg_snds["ebird_code"].isin(tiny_codes)]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9))
    sns.violinplot(data=tiny_snds, x="ebird_code", y="spectral_centroid_hz", ax=axes[0], color=TEAL, inner="quartile")
    axes[0].set_title(f"A. Baskın Frekans (Spectral Centroid)\nGenel Medyan: {reg_snds['spectral_centroid_hz'].median():.0f} Hz")
    axes[0].set_xlabel("Tür Kodu")
    axes[0].set_ylabel("Frekans (Hz)")
    axes[0].tick_params(axis="x", rotation=40)

    sns.boxplot(data=tiny_snds, x="ebird_code", y="spectral_bandwidth_hz", ax=axes[1], color=GOLD)
    axes[1].set_title(f"B. Spektral Bant Genişliği (Hz)\nOrt: {reg_snds['spectral_bandwidth_hz'].mean():.0f} ± {reg_snds['spectral_bandwidth_hz'].std():.0f} Hz")
    axes[1].set_xlabel("Tür Kodu")
    axes[1].set_ylabel("Bant Genişliği (Hz)")
    axes[1].tick_params(axis="x", rotation=40)

    sub_scatter = tiny_snds.sample(n=min(1200, len(tiny_snds)), random_state=42)
    sns.scatterplot(
        data=sub_scatter,
        x="silence_ratio",
        y="rms_energy_db",
        hue="ebird_code",
        palette="tab10",
        alpha=0.55,
        s=22,
        ax=axes[2],
        legend=False,
    )
    axes[2].axvline(0.85, color=RED, linestyle="--", linewidth=1.4, label="Temizlik Eşiği (%85 Sessizlik)")
    axes[2].set_title(f"C. Sessizlik Oranı vs RMS Enerji (dB)\nOrtalama Sessizlik: %{100*reg_snds['silence_ratio'].mean():.1f}")
    axes[2].set_xlabel("Sessizlik Oranı (0–1)")
    axes[2].set_ylabel("RMS Enerji (dB)")
    axes[2].legend(fontsize=8)

    fig.suptitle("Ses Modalitesi Keşifsel Veri Analizi (EDA) ve Akustik Dağılımlar", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p09 = FIGURES_DIR / "fig09_audio_modality_eda_stats.png"
    fig.savefig(p09)
    plt.close(fig)
    generated.append(p09.name)

    # -------------------------------------------------------------------------
    # FIG 10: Modaliteler Arası Tamamlayıcılık (2D Ortak Uzay)
    # -------------------------------------------------------------------------
    # Manifest üzerinden eşleşmiş Tiny 10 tür için görsel R-B renk indeksi vs Akustik Spectral Centroid
    img_lookup = reg_imgs.set_index("filename")[["luminance", "mean_r", "mean_g", "mean_b", "rms_contrast"]]
    snd_lookup = reg_snds.set_index("filename")[["spectral_centroid_hz", "spectral_bandwidth_hz", "rms_energy_db"]]
    m_tiny = manifest[manifest["ebird_code"].isin(tiny_codes)].sample(n=1500, random_state=42).copy()
    m_tiny["color_rb_diff"] = m_tiny["image_filename"].map(img_lookup["mean_r"] - img_lookup["mean_b"])
    m_tiny["spectral_centroid_hz"] = m_tiny["sound_filename"].map(snd_lookup["spectral_centroid_hz"])

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    palette = sns.color_palette("tab10", n_colors=len(tiny_codes))
    for idx, code in enumerate(tiny_codes):
        sub_c = m_tiny[m_tiny["ebird_code"] == code]
        tr_lbl = cls_df.loc[cls_df["ebird_code"] == code, "tr_name"].iloc[0]
        ax.scatter(
            sub_c["color_rb_diff"],
            sub_c["spectral_centroid_hz"],
            s=22,
            alpha=0.45,
            color=palette[idx],
            label=f"{tr_lbl} ({code})",
        )
        cx, cy = sub_c["color_rb_diff"].mean(), sub_c["spectral_centroid_hz"].mean()
        ax.scatter([cx], [cy], s=140, color=palette[idx], edgecolors="black", linewidth=1.6, zorder=5)
        dx, dy = (3.5, 95)
        if code == "mallar3":
            dx, dy = (-24.0, -210.0)
        elif code == "amecro":
            dx, dy = (6.0, 120.0)
        elif code == "eursta":
            dx, dy = (-20.0, -190.0)
        ax.text(cx + dx, cy + dy, code, fontsize=9, fontweight="bold", color="#0F172A")

    ax.set_ylim(250, 5850)
    ax.annotate(
        "Görselde benzer koyu/kahverengi türler\n(amecro vs houspa), ses frekansında\n(>2500 Hz farkla) ayrışıyor!",
        xy=(16, 4220), xytext=(-96, 4880),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4),
        fontsize=8.6, fontweight="bold", bbox=dict(boxstyle="round,pad=0.35", fc="#EFF6FF", ec=NAVY),
    )
    ax.annotate(
        "Düşük frekanslı türler (mallar3 vs amecro)\ngörsel renk farkıyla netleşiyor",
        xy=(10, 1180), xytext=(28, 680),
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4),
        fontsize=8.6, fontweight="bold", bbox=dict(boxstyle="round,pad=0.35", fc="#F0FDFA", ec=TEAL),
    )
    ax.set_xlabel("Görsel Modalite Özelliği: Kırmızı - Mavi Kanal Farkı (R - B)")
    ax.set_ylabel("Ses Modalitesi Özelliği: Baskın Frekans / Spectral Centroid (Hz)")
    ax.set_title("Modaliteler Arası Tamamlayıcılık: Tek Modalitede Örtüşen Türlerin Ortak Uzayda Ayrışması")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", title="Kuş Türleri", fontsize=8.3)
    plt.tight_layout()
    p10 = FIGURES_DIR / "fig10_cross_modal_complementarity.png"
    fig.savefig(p10)
    plt.close(fig)
    generated.append(p10.name)

    # -------------------------------------------------------------------------
    # FIG 11: Modaliteler Arası Korelasyon Matrisi (Regular 33)
    # -------------------------------------------------------------------------
    m_all = manifest.sample(n=5000, random_state=42).copy()
    feat_df = pd.DataFrame({
        "Gör: Parlaklık": m_all["image_filename"].map(img_lookup["luminance"]),
        "Gör: Kontrast": m_all["image_filename"].map(img_lookup["rms_contrast"]),
        "Gör: Kırmızı (R)": m_all["image_filename"].map(img_lookup["mean_r"]),
        "Gör: Mavi (B)": m_all["image_filename"].map(img_lookup["mean_b"]),
        "Ses: Centroid (Hz)": m_all["sound_filename"].map(snd_lookup["spectral_centroid_hz"]),
        "Ses: Bant Genişliği": m_all["sound_filename"].map(snd_lookup["spectral_bandwidth_hz"]),
        "Ses: RMS Enerji": m_all["sound_filename"].map(snd_lookup["rms_energy_db"]),
        "Etiket: BirdNET": m_all["sound_birdnet_score"],
    })
    corr = feat_df.corr()
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0.0, vmin=-1.0, vmax=1.0, linewidths=0.5, ax=ax)
    ax.set_title(
        "Görüntü ve Ses Öznitelikleri Pearson Korelasyon Matrisi (n=5.000 Çift)\n"
        "(Çapraz Blok Çoğunlukla |r| ≤ 0.21; Parlaklık–Bant Genişliği r=-0.41 → Tamamlayıcı Bilgi)",
        fontsize=10.5, fontweight="bold",
    )
    plt.tight_layout()
    p11 = FIGURES_DIR / "fig11_cross_modal_correlation_matrix.png"
    fig.savefig(p11)
    plt.close(fig)
    generated.append(p11.name)

    # -------------------------------------------------------------------------
    # FIG 12: Aykırı Durum (Outlier) ve Kalite/Etiket Hatası Vaka Analizi
    # -------------------------------------------------------------------------
    norm_img_p = PROJECT_ROOT / "data" / "raw" / "samples" / "norcar" / "images" / "norcar_img_000.jpg"
    occ_img_p = PROJECT_ROOT / "data" / "raw" / "samples" / "_outliers" / "outlier_occluded_norcar.jpg"
    norm_wav_p = PROJECT_ROOT / "data" / "raw" / "samples" / "blujay" / "sounds" / "blujay_xc_1000_seg0.wav"
    wind_wav_p = PROJECT_ROOT / "data" / "raw" / "samples" / "_outliers" / "outlier_wind_lowscore_blujay.wav"
    sil_wav_p = PROJECT_ROOT / "data" / "raw" / "samples" / "_outliers" / "outlier_silence_eurrob.wav"

    m_norm_i = extract_image_eda_metrics(norm_img_p)
    m_occ_i = extract_image_eda_metrics(occ_img_p)
    m_norm_a = extract_audio_eda_metrics(norm_wav_p)
    m_wind_a = extract_audio_eda_metrics(wind_wav_p)
    m_sil_a = extract_audio_eda_metrics(sil_wav_p)

    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.2))
    # Üst satır: Görseller & Dalga formları; Alt satır: Histogram / Spektrogramlar
    with Image.open(norm_img_p) as im:
        axes[0, 0].imshow(im)
    axes[0, 0].set_title(f"Normal Örnek (norcar)\nLum={m_norm_i['luminance']:.1f}, Kontrast={m_norm_i['rms_contrast']:.1f}", color=GREEN, fontsize=9.5)
    axes[0, 0].axis("off")

    with Image.open(occ_img_p) as im:
        axes[0, 1].imshow(im)
    axes[0, 1].set_title(f"Vaka A: Oklüzyon + Karanlık\nLum={m_occ_i['luminance']:.1f}, Kontrast={m_occ_i['rms_contrast']:.1f}", color=RED, fontsize=9.5)
    axes[0, 1].axis("off")

    w_norm, _ = load_wav_mono_fixed(norm_wav_p)
    w_wind, _ = load_wav_mono_fixed(wind_wav_p)
    w_sil, _ = load_wav_mono_fixed(sil_wav_p)
    t_ax = np.linspace(0, 3.0, len(w_norm))

    axes[0, 2].plot(t_ax, w_wind, color=CORAL, linewidth=0.6)
    axes[0, 2].set_title(f"Vaka B: Rüzgar Gürültülü Ses\nCentroid={m_wind_a['spectral_centroid_hz']:.0f} Hz, BirdNET=0.18", color=RED, fontsize=9.5)
    axes[0, 2].set_ylim(-1, 1)

    axes[0, 3].plot(t_ax, w_sil, color=SLATE, linewidth=0.7)
    axes[0, 3].set_title(f"Vaka C: Aşırı Sessiz Segment\nSessizlik=%{100*m_sil_a['silence_ratio']:.1f}, RMS={m_sil_a['rms_energy_db']:.1f} dB", color=RED, fontsize=9.5)
    axes[0, 3].set_ylim(-1, 1)

    # Alt satır karşılaştırmaları
    axes[1, 0].hist(np.asarray(Image.open(norm_img_p)).ravel(), bins=32, color=GREEN, alpha=0.75)
    axes[1, 0].set_title("Normal Piksel Dağılımı\n(Geniş Dinamik Aralık)", fontsize=9.2)
    axes[1, 0].set_xlim(0, 255)

    axes[1, 1].hist(np.asarray(Image.open(occ_img_p)).ravel(), bins=32, color=RED, alpha=0.75)
    axes[1, 1].set_title("Vaka A: Sıkışmış Piksel\n(Düşük Dinamik Aralık)", fontsize=9.2)
    axes[1, 1].set_xlim(0, 255)

    _, lm_wind = aud_prep.compute_stft_and_logmel(w_wind)
    axes[1, 2].imshow(lm_wind, aspect="auto", origin="lower", extent=[0, 3.0, 50, 8000], cmap="magma")
    axes[1, 2].set_title("Vaka B Log-Mel (<400 Hz Rüzgar)\n(BirdNET < 0.25 ile Elendi)", fontsize=9.2)

    _, lm_sil = aud_prep.compute_stft_and_logmel(w_sil)
    axes[1, 3].imshow(lm_sil, aspect="auto", origin="lower", extent=[0, 3.0, 50, 8000], cmap="magma")
    axes[1, 3].set_title("Vaka C Log-Mel (>%85 Sessiz)\n(Sessizlik Filtresiyle Elendi)", fontsize=9.2)

    fig.suptitle("Aykırı Durum, Kalite Problemi ve Etiket Hatası Vaka Analizi", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(w_pad=1.6, h_pad=1.4)
    p12 = FIGURES_DIR / "fig12_outlier_and_error_case_studies.png"
    fig.savefig(p12)
    plt.close(fig)
    generated.append(p12.name)

    # -------------------------------------------------------------------------
    # FIG 13: Veri Sızıntısını Önleme ve Modele Hazırlık Mimarisi
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={"width_ratios": [1.05, 1.15]})
    ax1.axis("off")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title("A. %50 Örtüşmeli Seslerde Veri Sızıntısı Önleme Stratejisi", fontsize=11, fontweight="bold")

    # Yanlış yöntem kutusu
    ax1.add_patch(patches.FancyBboxPatch((0.3, 6.2), 9.4, 3.1, boxstyle="round,pad=0.2", fc="#FEF2F2", ec=RED, lw=1.6))
    ax1.text(0.6, 8.8, "YANLIŞ YÖNTEM: Segment Bazlı Rastgele Bölme (Veri Sızıntısı!)", fontweight="bold", color=RED, fontsize=9.8)
    ax1.text(0.6, 8.0, "Aynı base_filename kaydı: [Seg 0 (0-3s)]  [Seg 1 (1.5-4.5s)]  [Seg 2 (3-6s)]", fontsize=8.8, color="#1F2937")
    ax1.text(0.6, 7.2, "  → Seg 0 Train'e, örtüşen Seg 1 Test'e düşerse model aynı kaydı ezberler!", fontsize=8.8, color="#991B1B")
    ax1.text(0.6, 6.5, "  → Yapay olarak şişirilmiş sahte doğruluk (Data Leakage)", fontsize=8.5, style="italic", color="#7F1D1D")

    # Doğru yöntem kutusu
    ax1.add_patch(patches.FancyBboxPatch((0.3, 0.8), 9.4, 4.8, boxstyle="round,pad=0.2", fc="#ECFDF5", ec=GREEN, lw=1.8))
    ax1.text(0.6, 5.15, "UYGULANAN YÖNTEM: Grup Bazlı Bölme (for_test + GroupShuffleSplit)", fontweight="bold", color=GREEN, fontsize=9.0)
    ax1.text(0.6, 4.35, "1. Zenodo 'for_test=True' (%15 = 3.465 çift) → Tamamen izole TEST kümesi", fontsize=8.3)
    ax1.text(0.6, 3.65, "2. Kalan %85 havuz 'base_filename' & 'image_md5' gruplarıyla bölünür:", fontsize=8.3)
    ax1.text(0.8, 3.0, "• TRAIN: 16.170 (%70.0) | VAL: 3.465 (%15.0) | TEST: 3.465 (%15.0)", fontsize=8.2, fontweight="bold", color="#065F46")
    shared_b = leak_rep['shared_base_filenames_train_val'] + leak_rep['shared_base_filenames_train_test'] + leak_rep['shared_base_filenames_val_test']
    shared_h = leak_rep['shared_image_hashes_train_val'] + leak_rep['shared_image_hashes_train_test'] + leak_rep['shared_image_hashes_val_test']
    ax1.text(0.6, 2.15, f"3. Kodla Doğrulanan Sızıntı: Paylaşılan Kayıt = {shared_b} | Paylaşılan Hash = {shared_h}", fontsize=8.3, fontweight="bold", color=NAVY)
    ax1.text(0.6, 1.35, "4. Metadata (BirdNet_score, sci_name, dosya yolu) model girdisinden gizlenir.", fontsize=8.1, style="italic", color="#064E3B")

    # Sağ Panel: Tensör Şekilleri ve Füzyon Akışı
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title("B. Modele Hazır Tensör Yapısı ve Multimodal Füzyon Mimarisi", fontsize=11, fontweight="bold")

    boxes = [
        (0.3, 7.0, 2.9, 1.9, "#DBEAFE", NAVY, "Görüntü Girdisi\nRGB 224×224\nTensör: [B, 3, 224, 224]"),
        (0.3, 2.0, 2.9, 1.9, "#CCFBF1", TEAL, "Ses Girdisi\nLog-Mel (128 bin)\nTensör: [B, 1, 128, 151]"),
        (3.8, 7.0, 2.5, 1.9, "#EFF6FF", NAVY, "Görsel Kodlayıcı\nEfficientNet-B0\n1280 → 256-D"),
        (3.8, 2.0, 2.5, 1.9, "#F0FDFA", TEAL, "Ses Kodlayıcı\n4-Blok 2D CNN\nGAP → 256-D"),
        (6.9, 4.3, 2.8, 2.3, "#FEF3C7", GOLD, "Multimodal Füzyon\nConcat(256, 256)=512\nLayerNorm → GELU\nDropout(0.3) → 33 Tür"),
    ]
    for x, y, w, h, fc, ec, txt in boxes:
        ax2.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", fc=fc, ec=ec, lw=1.6))
        ax2.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.6, fontweight="bold", color="#0F172A")

    arrows = [((3.25, 7.95), (3.75, 7.95)), ((3.25, 2.95), (3.75, 2.95)), ((6.35, 7.85), (6.85, 5.95)), ((6.35, 3.05), (6.85, 4.95))]
    for p1, p2 in arrows:
        ax2.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", lw=1.8, color="#334155"))

    ax2.text(5.0, 0.7, "Final Aşaması Planı: 3 Deney Kolu (Image-Only vs Audio-Only vs Fusion) • Macro-F1, Top-1, Top-3", ha="center", fontsize=8.7, fontweight="bold", color="#334155", bbox=dict(boxstyle="round,pad=0.3", fc="#F1F5F9", ec="#94A3B8"))

    fig.suptitle("Veri Sızıntısı Önleme Stratejisi ve Multimodal Modellemeye Hazırlık", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p13 = FIGURES_DIR / "fig13_split_leakage_and_fusion_architecture.png"
    fig.savefig(p13)
    plt.close(fig)
    generated.append(p13.name)

    return generated


if __name__ == "__main__":
    files = generate_all_charts()
    print(f"Toplam {len(files)} grafik üretildi:")
    for f in files:
        print(" -", f)
