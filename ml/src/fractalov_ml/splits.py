"""Stratified train/val/test split.

Why stratify? The classifier's prior over the four families should match the
training prior — if 80/10/10 cuts off all `multibrot` examples into the val set
by chance, the train head will never see them. ``sklearn.train_test_split`` with
``stratify=family`` gives us a per-family proportional split.

Output is a parquet with a single ``id, split`` mapping; the original
``labels.parquet`` is untouched. The ``FractalDataset`` joins the two on ``id``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split(
    labels_path: Path,
    out_path: Path,
    val_size: float = 0.1,
    test_size: float = 0.1,
    stratify_by: str = "family",
    seed: int = 42,
) -> Path:
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must leave room for train")
    df = pd.read_parquet(labels_path)
    if stratify_by not in df.columns:
        raise ValueError(f"stratify column '{stratify_by}' not in labels parquet")

    # First carve out the test set, then split the rest into train/val.
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_by],
        random_state=seed,
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[stratify_by],
        random_state=seed,
    )

    parts = [
        train.assign(split="train"),
        val.assign(split="val"),
        test.assign(split="test"),
    ]
    out = pd.concat(parts)[["id", "split"]].reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path
