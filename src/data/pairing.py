"""Tür düzeyinde görüntü-ses eşleştirme ve veri sızıntısını önleyen
grup bazlı (base_filename & MD5 hash korumalı) train/val/test bölme modülü.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
META_CSV = PROJECT_ROOT / "data" / "raw" / "metadata" / "multimodal_bird_info.csv"
LABEL_CSV = PROJECT_ROOT / "data" / "raw" / "metadata" / "label_mapping.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def build_group_safe_pairs(
    tier: str = "regular",
    seed: int = 42,
    meta_csv: Path = META_CSV,
    label_csv: Path = LABEL_CSV,
) -> Dict[str, object]:
    """Regular (33 tür) veya Tiny (10 tür) için sızıntısız pair_manifest.csv üretir."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_csv, low_memory=False)
    labels_df = pd.read_csv(label_csv)
    code_to_id = dict(zip(labels_df["ebird_code"], labels_df["label_id"]))

    clean = df[df["quality_status"] == "clean"].copy()
    if tier == "regular":
        sub = clean[clean["in_regular"] == True].copy()
    elif tier == "tiny":
        sub = clean[clean["in_tiny"] == True].copy()
    else:
        raise ValueError(f"Desteklenmeyen tier: {tier}")

    rng = np.random.default_rng(seed)
    manifest_rows: List[Dict[str, object]] = []

    for code, cls_df in sub.groupby("ebird_code", sort=False):
        label_id = int(code_to_id[code])
        sci_name = str(cls_df["sci_name"].iloc[0])
        common_name = str(cls_df["name"].iloc[0])

        imgs = cls_df[cls_df["data_type"] == "image"].copy()
        snds = cls_df[cls_df["data_type"] == "sound"].copy()

        # 1) Zenodo'nun for_test atamasını koru (%15 Test)
        imgs_test = imgs[imgs["for_test"] == True].reset_index(drop=True)
        imgs_tv = imgs[imgs["for_test"] == False].reset_index(drop=True)

        snds_test = snds[snds["for_test"] == True].reset_index(drop=True)
        snds_tv = snds[snds["for_test"] == False].reset_index(drop=True)

        # 2) Train+Val içinden Validation'ı ses kaynak grubu (base_filename) bazında ayır
        # 119 train+val grubundan 21 grup val (21/119 = 0.17647 -> toplamın tam %15'i), 98 grup train (%70)
        gss_snd = GroupShuffleSplit(n_splits=1, test_size=0.17647, random_state=seed)
        tr_idx, val_idx = next(gss_snd.split(snds_tv, groups=snds_tv["base_filename"]))
        snds_train = snds_tv.iloc[tr_idx].reset_index(drop=True)
        snds_val = snds_tv.iloc[val_idx].reset_index(drop=True)

        # Görüntüler için md5_hash grubunu koruyarak %70 Train / %15 Val bölmesi
        gss_img = GroupShuffleSplit(n_splits=1, test_size=0.17647, random_state=seed)
        itr_idx, ival_idx = next(gss_img.split(imgs_tv, groups=imgs_tv["md5_hash"]))
        imgs_train = imgs_tv.iloc[itr_idx].reset_index(drop=True)
        imgs_val = imgs_tv.iloc[ival_idx].reset_index(drop=True)

        # 3) Her split içinde yalnızca aynı tür (ebird_code) ve aynı split dahilinde eşleştirme yap
        split_buckets = [
            ("train", imgs_train, snds_train),
            ("val", imgs_val, snds_val),
            ("test", imgs_test, snds_test),
        ]
        for split_name, sp_imgs, sp_snds in split_buckets:
            n_pairs = len(sp_snds)
            img_perm = rng.permutation(len(sp_imgs))
            for k in range(n_pairs):
                s_row = sp_snds.iloc[k]
                i_row = sp_imgs.iloc[int(img_perm[k % len(sp_imgs)])]
                manifest_rows.append({
                    "pair_id": f"{code}_{split_name}_{k:04d}",
                    "split": split_name,
                    "label_id": label_id,
                    "ebird_code": code,
                    "common_name": common_name,
                    "sci_name": sci_name,
                    "image_filename": i_row["filename"],
                    "image_path": i_row["data_path"],
                    "image_sample_path": i_row["sample_file_path"],
                    "image_md5": i_row["md5_hash"],
                    "sound_filename": s_row["filename"],
                    "sound_path": s_row["data_path"],
                    "sound_sample_path": s_row["sample_file_path"],
                    "audio_base_filename": s_row["base_filename"],
                    "audio_subgroup_id": s_row["subgroup_id"],
                    "audio_seg_index": int(s_row["index"]),
                    "image_birdnet_score": float(i_row["BirdNet_score"]),
                    "sound_birdnet_score": float(s_row["BirdNet_score"]),
                })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = SPLITS_DIR / "pair_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    # 4) Sıfır veri sızıntısı (Zero Leakage) matematiksel doğrulaması
    splits = {s: manifest_df[manifest_df["split"] == s] for s in ("train", "val", "test")}
    base_sets = {s: set(df_s["audio_base_filename"].unique()) for s, df_s in splits.items()}
    hash_sets = {s: set(df_s["image_md5"].unique()) for s, df_s in splits.items()}

    leakage_report = {
        "tier": tier,
        "seed": seed,
        "total_pairs": int(len(manifest_df)),
        "split_counts": {s: int(len(df_s)) for s, df_s in splits.items()},
        "split_percentages": {
            s: round(100.0 * len(df_s) / len(manifest_df), 2) for s, df_s in splits.items()
        },
        "unique_base_recordings": {s: int(len(base_sets[s])) for s in splits},
        "shared_base_filenames_train_val": len(base_sets["train"] & base_sets["val"]),
        "shared_base_filenames_train_test": len(base_sets["train"] & base_sets["test"]),
        "shared_base_filenames_val_test": len(base_sets["val"] & base_sets["test"]),
        "shared_image_hashes_train_val": len(hash_sets["train"] & hash_sets["val"]),
        "shared_image_hashes_train_test": len(hash_sets["train"] & hash_sets["test"]),
        "shared_image_hashes_val_test": len(hash_sets["val"] & hash_sets["test"]),
        "species_mismatch_pairs": int(
            sum(
                not (
                    str(r.image_path).startswith(f"data/{r.ebird_code}/")
                    and str(r.sound_path).startswith(f"data/{r.ebird_code}/")
                )
                for r in manifest_df.itertuples(index=False)
            )
        ),
        "zero_leakage_verified": (
            len(base_sets["train"] & base_sets["val"]) == 0
            and len(base_sets["train"] & base_sets["test"]) == 0
            and len(base_sets["val"] & base_sets["test"]) == 0
            and len(hash_sets["train"] & hash_sets["val"]) == 0
            and len(hash_sets["train"] & hash_sets["test"]) == 0
            and len(hash_sets["val"] & hash_sets["test"]) == 0
        ),
    }
    (REPORTS_DIR / "leakage_verification_report.json").write_text(
        json.dumps(leakage_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return leakage_report


if __name__ == "__main__":
    rep = build_group_safe_pairs(tier="regular", seed=42)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
