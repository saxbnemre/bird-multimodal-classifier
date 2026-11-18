"""PyTorch MultimodalBirdDataset ve DataLoader modülü.
Metadata sızıntısına yol açacak BirdNet_score, sci_name, dosya yolu gibi alanları kesinlikle gizler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import BirdAudioPreprocessor, BirdImagePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "pair_manifest.csv"


class MultimodalBirdDataset(Dataset):
    """Tür düzeyinde eşleştirilmiş (Görüntü, Ses Log-Mel, Etiket) veri yükleyicisi."""

    def __init__(
        self,
        manifest_path: Path | str = DEFAULT_MANIFEST,
        split: str = "train",
        augment: bool | None = None,
        seed: int = 42,
    ) -> None:
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"Manifest içinde '{split}' bölmesi boş!")

        use_aug = (split == "train") if augment is None else augment
        self.image_transform = BirdImagePreprocessor(target_size=224, augment=use_aug, seed=seed)
        self.audio_transform = BirdAudioPreprocessor(augment=use_aug, seed=seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = PROJECT_ROOT / str(row["image_sample_path"])
        wav_path = PROJECT_ROOT / str(row["sound_sample_path"])

        img_tensor = self.image_transform(img_path)      # [3, 224, 224]
        audio_tensor = self.audio_transform(wav_path)    # [1, 128, 151]
        label_tensor = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return img_tensor, audio_tensor, label_tensor


def verify_dataloader_shapes(batch_size: int = 8) -> dict:
    """Train/Val/Test DataLoader'larından birer mini-batch çekip tensör şekillerini doğrular."""
    results = {}
    for split in ("train", "val", "test"):
        ds = MultimodalBirdDataset(split=split)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
        imgs, auds, labels = next(iter(dl))
        results[split] = {
            "num_samples": len(ds),
            "image_batch_shape": list(imgs.shape),
            "audio_batch_shape": list(auds.shape),
            "label_batch_shape": list(labels.shape),
            "image_dtype": str(imgs.dtype),
            "audio_dtype": str(auds.dtype),
            "image_nan_count": int(torch.isnan(imgs).sum().item()),
            "audio_nan_count": int(torch.isnan(auds).sum().item()),
        }
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(verify_dataloader_shapes(), indent=2))
