"""Veri hattı, denetim, grup bazlı sızıntısız eşleştirme, ön işleme,
model ileri besleme, 13 EDA görseli ve Jupyter Notebook doğrulama testleri.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from src.data.dataset import MultimodalBirdDataset
from src.models.fusion_model import MultimodalBirdClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_audit_reports_exist_and_meet_criteria() -> None:
    summary = pd.read_csv(PROJECT_ROOT / "outputs" / "reports" / "dataset_audit_summary.csv")
    reg = summary[summary["tier"] == "Regular_33_Main"].iloc[0]
    assert int(reg["num_species"]) == 33
    assert int(reg["total_records"]) == 27720
    assert int(reg["species_level_pairs"]) == 23100

    cls_df = pd.read_csv(PROJECT_ROOT / "outputs" / "reports" / "class_distribution_regular33.csv")
    assert len(cls_df) == 33
    assert int(cls_df["paired_samples_count"].min()) >= 150
    assert int(cls_df["paired_samples_count"].sum()) >= 500


def test_zero_data_leakage_across_splits() -> None:
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "pair_manifest.csv")
    assert len(manifest) == 23100

    splits = {s: manifest[manifest["split"] == s] for s in ("train", "val", "test")}
    assert len(splits["train"]) == 16170
    assert len(splits["val"]) == 3465
    assert len(splits["test"]) == 3465

    tr_base = set(splits["train"]["audio_base_filename"])
    va_base = set(splits["val"]["audio_base_filename"])
    te_base = set(splits["test"]["audio_base_filename"])
    assert len(tr_base & va_base) == 0
    assert len(tr_base & te_base) == 0
    assert len(va_base & te_base) == 0

    tr_hash = set(splits["train"]["image_md5"])
    va_hash = set(splits["val"]["image_md5"])
    te_hash = set(splits["test"]["image_md5"])
    assert len(tr_hash & va_hash) == 0
    assert len(tr_hash & te_hash) == 0
    assert len(va_hash & te_hash) == 0


def test_pytorch_dataset_and_fusion_forward_pass() -> None:
    ds_train = MultimodalBirdDataset(split="train")
    img, aud, lbl = ds_train[0]
    assert img.shape == (3, 224, 224)
    assert aud.shape == (1, 128, 151)
    assert not torch.isnan(img).any()
    assert not torch.isnan(aud).any()
    assert 0 <= int(lbl.item()) < 33

    model = MultimodalBirdClassifier(num_classes=33, embed_dim=256)
    model.eval()
    with torch.no_grad():
        logits, v_img, v_aud = model(img.unsqueeze(0), aud.unsqueeze(0))
    assert logits.shape == (1, 33)
    assert v_img.shape == (1, 256)
    assert v_aud.shape == (1, 256)


def test_all_13_eda_figures_exist() -> None:
    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    pngs = sorted(fig_dir.glob("fig*.png"))
    assert len(pngs) == 13
    for p in pngs:
        assert p.stat().st_size > 15_000


def test_notebook_exists_and_valid() -> None:
    nb_path = PROJECT_ROOT / "notebooks" / "01_exploratory_data_analysis.ipynb"
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 12
