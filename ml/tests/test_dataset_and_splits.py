"""Offline coverage of dataset / split tooling using synthetic PNGs.

Doesn't talk to the backend: we hand-build a fake `labels.parquet` + image tree
and exercise the parquet → tensor path end-to-end.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from fractalov_ml.dataset import FamilyEncoder, FractalDataset
from fractalov_ml.recipes import FAMILIES
from fractalov_ml.splits import split as split_dataset


def _png(width: int = 8, height: int = 8, color=(80, 100, 120)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_synthetic_dataset(root: Path, per_family: int = 12) -> Path:
    rows = []
    for fam in FAMILIES:
        (root / "images" / fam).mkdir(parents=True, exist_ok=True)
        for i in range(per_family):
            example_id = f"{fam}_{i:03d}"
            rel = f"images/{fam}/{example_id}.png"
            (root / rel).write_bytes(_png())
            rows.append(
                {
                    "id": example_id,
                    "family": fam,
                    "fractal_type": fam,
                    "c_re": -0.7 if fam == "julia" else None,
                    "c_im": 0.27 if fam == "julia" else None,
                    "exponent": 3 if fam == "multibrot" else None,
                    "max_iter": 200,
                    "escape_radius": 2.0,
                    "smoothing": True,
                    "vp_x_min": -2.0,
                    "vp_x_max": 1.0,
                    "vp_y_min": -1.2,
                    "vp_y_max": 1.2,
                    "palette": "fire",
                    "color_mode": "linear",
                    "samples_per_axis": 1,
                    "image_path": rel,
                    "perf_render_ms": 5,
                    "perf_colorize_ms": 1,
                    "perf_encode_ms": 1,
                    "perf_total_ms": 7,
                    "file_size_bytes": 100,
                    "request_id": f"req-{example_id}",
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(root / "labels.parquet", index=False)
    return root / "labels.parquet"


def test_dataset_loads_image_and_label(tmp_path: Path) -> None:
    _make_synthetic_dataset(tmp_path, per_family=4)
    ds = FractalDataset(tmp_path)
    assert len(ds) == 4 * len(FAMILIES)

    tensor, family_int, params = ds[0]
    assert tensor.shape == (3, 8, 8)
    assert tensor.dtype.is_floating_point
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
    assert 0 <= family_int < len(FAMILIES)
    assert "fractal_type" in params


def test_family_encoder_is_alphabetical_and_stable() -> None:
    enc = FamilyEncoder.default()
    assert enc.classes == ("burning_ship", "julia", "mandelbrot", "multibrot")
    assert enc.encode("julia") == 1
    assert enc.decode(0) == "burning_ship"


def test_split_is_stratified(tmp_path: Path) -> None:
    labels_path = _make_synthetic_dataset(tmp_path, per_family=20)
    splits_path = split_dataset(
        labels_path=labels_path,
        out_path=tmp_path / "splits.parquet",
        val_size=0.2,
        test_size=0.2,
        seed=7,
    )
    splits = pd.read_parquet(splits_path)
    labels = pd.read_parquet(labels_path)
    merged = labels.merge(splits, on="id", how="inner")

    # Every example landed in exactly one split.
    assert len(merged) == len(labels)
    assert set(merged["split"]) == {"train", "val", "test"}

    # Stratification: each split contains every family (each family has 20 rows
    # and the split is 60/20/20, so the smallest bucket has at least 4).
    for split_name in ("train", "val", "test"):
        sub = merged[merged["split"] == split_name]
        assert set(sub["family"]) == set(FAMILIES), (
            f"split {split_name} missing a family: {sub['family'].unique()}"
        )


def test_dataset_split_filter(tmp_path: Path) -> None:
    _make_synthetic_dataset(tmp_path, per_family=20)
    split_dataset(
        labels_path=tmp_path / "labels.parquet",
        out_path=tmp_path / "splits.parquet",
        val_size=0.2,
        test_size=0.2,
        seed=7,
    )
    train = FractalDataset(tmp_path, split="train")
    val = FractalDataset(tmp_path, split="val")
    test = FractalDataset(tmp_path, split="test")
    total = 20 * len(FAMILIES)
    assert len(train) + len(val) + len(test) == total
    assert len(val) > 0
    assert len(test) > 0


def test_dataset_rejects_unknown_family(tmp_path: Path) -> None:
    _make_synthetic_dataset(tmp_path, per_family=2)
    df = pd.read_parquet(tmp_path / "labels.parquet")
    df.loc[0, "family"] = "phantom_family"
    df.to_parquet(tmp_path / "labels.parquet", index=False)
    with pytest.raises(ValueError, match="unknown family"):
        FractalDataset(tmp_path)
