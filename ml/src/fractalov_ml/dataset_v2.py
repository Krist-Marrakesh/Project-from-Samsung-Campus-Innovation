"""PyTorch Dataset for the Slice 2 / build-dataset schema.

Differences from the legacy :class:`fractalov_ml.dataset.FractalDataset`:

* Reads ``split`` from a column on ``labels.parquet`` itself — no separate
  ``splits.parquet`` join. Required because Slice 2 datasets bake the split
  decision into generation, not as an afterthought.
* Image path is namespaced by split: ``images/<split>/<family>/<id>.png``.
* Yields a structured :class:`Sample` dataclass instead of an ad-hoc dict so
  downstream training code does not pluck strings.
* Adds ``has_julia`` / ``has_multibrot`` masks for the v2 multi-task heads.

The legacy class is left untouched so existing checkpoints + eval scripts
continue to work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset import FamilyEncoder


SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLIT_NEAR_OOD = "near_ood"
SPLIT_HARD_OVERLAP = "hard_overlap"
ALL_SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_NEAR_OOD, SPLIT_HARD_OVERLAP)


@dataclass(frozen=True)
class Sample:
    """One row served by :class:`FractalDatasetV2`."""

    image: torch.Tensor                      # (3, H, W) float32 in [0, 1]
    family_idx: int                          # alphabetic index from FamilyEncoder
    family: str
    c_re: float                              # NaN when not Julia
    c_im: float                              # NaN when not Julia
    exponent: float                          # NaN when not Multibrot
    max_iter: int
    viewport: tuple[float, float, float, float]   # xMin, xMax, yMin, yMax
    has_julia: bool
    has_multibrot: bool


def collate(batch: list[Sample]) -> dict:
    """Custom collate that batches a list of :class:`Sample` into tensor dict.

    DataLoader's default collate would happily concatenate the dataclass via
    its constructor; we keep an explicit collate so the on-the-wire shape
    is part of the public API and easy to inspect.
    """
    images = torch.stack([s.image for s in batch], dim=0)
    family_idx = torch.tensor([s.family_idx for s in batch], dtype=torch.long)
    c_re = torch.tensor([s.c_re for s in batch], dtype=torch.float32)
    c_im = torch.tensor([s.c_im for s in batch], dtype=torch.float32)
    exponent = torch.tensor([s.exponent for s in batch], dtype=torch.float32)
    max_iter = torch.tensor([s.max_iter for s in batch], dtype=torch.long)
    viewport = torch.tensor([list(s.viewport) for s in batch], dtype=torch.float32)
    has_julia = torch.tensor([s.has_julia for s in batch], dtype=torch.bool)
    has_multibrot = torch.tensor([s.has_multibrot for s in batch], dtype=torch.bool)
    return {
        "image": images,
        "family_idx": family_idx,
        "family_strs": [s.family for s in batch],
        "c": torch.stack([c_re, c_im], dim=1),     # (B, 2)
        "exponent": exponent,                       # (B,)
        "max_iter": max_iter,
        "viewport": viewport,                       # (B, 4)
        "has_julia": has_julia,
        "has_multibrot": has_multibrot,
    }


class FractalDatasetV2(Dataset):
    """Reads the Slice 2 dataset format. Set ``split`` to one of
    :data:`ALL_SPLITS` (or any custom name in the parquet) to scope to that
    split.

    Parameters
    ----------
    root:
        Dataset directory produced by ``fractalov-ml build-dataset``.
    split:
        Required string; we don't default because the parquet now carries
        all splits in one file and "all" is rarely the right answer for
        training.
    transform:
        Optional callable mapping ``PIL.Image -> torch.Tensor``. Defaults to
        a deterministic ``[0, 1]`` float32 conversion.
    encoder:
        Stable family encoder. The default's class order is alphabetic
        across :data:`fractalov_ml.recipes.FAMILIES` so checkpoints are
        portable across datasets with different family ratios.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        transform: Optional[Callable] = None,
        encoder: Optional[FamilyEncoder] = None,
    ) -> None:
        self.root = Path(root)
        labels_path = self.root / "labels.parquet"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"labels.parquet not found under {self.root}. Did build-dataset run?"
            )
        df = pd.read_parquet(labels_path)
        if "split" not in df.columns:
            raise ValueError(
                f"{labels_path} has no 'split' column — this dataset was built by "
                "the legacy `generate` command. Use FractalDataset (v1) for it."
            )
        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"split '{split}' is empty in {labels_path}. "
                f"Available: {sorted(set(pd.read_parquet(labels_path)['split']))}"
            )
        self.labels = df
        self.transform = transform
        self.encoder = encoder or FamilyEncoder.default()
        self.split = split

        unknown = set(self.labels["family"].unique()) - set(self.encoder.classes)
        if unknown:
            raise ValueError(f"unknown family in labels.parquet: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Sample:
        row = self.labels.iloc[idx]
        path = self.root / row["image_path"]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            arr = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        family = str(row["family"])
        return Sample(
            image=tensor,
            family_idx=self.encoder.encode(family),
            family=family,
            c_re=_float_or_nan(row["c_re"]),
            c_im=_float_or_nan(row["c_im"]),
            exponent=_float_or_nan(row["exponent"]),
            max_iter=int(row["max_iter"]),
            viewport=(
                float(row["vp_x_min"]),
                float(row["vp_x_max"]),
                float(row["vp_y_min"]),
                float(row["vp_y_max"]),
            ),
            has_julia=bool(not pd.isna(row["c_re"])),
            has_multibrot=bool(not pd.isna(row["exponent"])),
        )


def _float_or_nan(v) -> float:
    return math.nan if pd.isna(v) else float(v)
