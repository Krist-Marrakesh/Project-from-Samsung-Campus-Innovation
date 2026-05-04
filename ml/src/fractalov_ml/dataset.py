"""PyTorch Dataset for fractalov.

Joins ``labels.parquet`` and ``splits.parquet`` on ``id`` and yields
``(image_tensor, family_label_int, params_dict)`` tuples.

Two label flavours coexist:
  * ``family_label`` — int in ``[0..len(FAMILIES)-1]`` from a stable
    ``LabelEncoder``. Stable means that `mandelbrot=0, julia=1, ...` is fixed by
    the alphabetic order of :data:`fractalov_ml.recipes.FAMILIES`, not by the
    order they appear in the parquet — so a model trained on one dataset
    transfers cleanly to a re-generated dataset with a different family ratio.
  * ``params`` — dict of family-specific numeric labels (cRe, cIm, exponent,
    maxIter, vp_*, smoothing). Stage 6 will pick the subset relevant to its task.

Image tensors are float32 in ``[0, 1]`` with shape ``(C, H, W)``.

Family-specific params (``c_re``, ``c_im``, ``exponent``) are filled with
sentinels — ``NaN`` for floats and ``-1`` for ints — when the row does not have
them (e.g. ``c_re`` for Mandelbrot). This keeps the default ``DataLoader``
collate happy (it cannot batch ``None``) and gives Stage 6 a single masking
predicate per field via ``has_c`` / ``has_exponent``.
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

from .recipes import FAMILIES


@dataclass(frozen=True)
class FamilyEncoder:
    """Stable encoding ``family_str <-> family_int``."""

    classes: tuple[str, ...]

    @classmethod
    def default(cls) -> FamilyEncoder:
        return cls(classes=tuple(sorted(FAMILIES)))

    def encode(self, family: str) -> int:
        return self.classes.index(family)

    def decode(self, idx: int) -> str:
        return self.classes[idx]


class FractalDataset(Dataset):
    """Lazy-loading PyTorch dataset for a generated fractalov split.

    Parameters
    ----------
    root:
        Directory containing ``labels.parquet`` and (optionally) ``splits.parquet``,
        with sub-directory ``images/<family>/...``.
    split:
        ``"train" | "val" | "test"`` — uses ``splits.parquet``. If ``None``, the
        whole dataset is used (handy for EDA / re-splitting).
    transform:
        Optional torchvision-style transform applied to the loaded ``PIL.Image``;
        when ``None``, the image is converted to a float32 ``(C, H, W)`` tensor in
        ``[0, 1]`` via :func:`numpy.asarray`.
    encoder:
        Override the default :class:`FamilyEncoder` if you trained a model with
        a different class ordering.
    """

    def __init__(
        self,
        root: Path,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        encoder: Optional[FamilyEncoder] = None,
    ) -> None:
        self.root = Path(root)
        labels = pd.read_parquet(self.root / "labels.parquet")
        if split is not None:
            splits_df = pd.read_parquet(self.root / "splits.parquet")
            requested = splits_df[splits_df["split"] == split]["id"]
            labels = labels.merge(requested.to_frame(), on="id", how="inner")
            if labels.empty:
                raise ValueError(
                    f"split '{split}' is empty — generate or regenerate splits.parquet"
                )
        self.labels = labels.reset_index(drop=True)
        self.transform = transform
        self.encoder = encoder or FamilyEncoder.default()

        # Ensure the data on disk matches the encoder's class set: a label seen
        # in parquet but missing from FAMILIES would silently break .encode().
        seen = set(self.labels["family"].unique())
        unknown = seen - set(self.encoder.classes)
        if unknown:
            raise ValueError(f"unknown family in labels.parquet: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict]:
        row = self.labels.iloc[idx]
        path = self.root / row["image_path"]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            arr = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        family_label = self.encoder.encode(str(row["family"]))
        params = {
            "fractal_type": str(row["fractal_type"]),
            "max_iter": int(row["max_iter"]),
            "escape_radius": float(row["escape_radius"]),
            "smoothing": bool(row["smoothing"]),
            "c_re": _float_or_nan(row["c_re"]),
            "c_im": _float_or_nan(row["c_im"]),
            "exponent": _int_or_minus_one(row["exponent"]),
            "vp_x_min": float(row["vp_x_min"]),
            "vp_x_max": float(row["vp_x_max"]),
            "vp_y_min": float(row["vp_y_min"]),
            "vp_y_max": float(row["vp_y_max"]),
            "has_c": not pd.isna(row["c_re"]),
            "has_exponent": not pd.isna(row["exponent"]),
        }
        return tensor, family_label, params


def _float_or_nan(v) -> float:
    return math.nan if pd.isna(v) else float(v)


def _int_or_minus_one(v) -> int:
    return -1 if pd.isna(v) else int(v)
