"""End-to-end integration test for the v2 training pipeline.

This is the only test that exercises the *combination* of:
  * dataset_v2 (parquet + image loader)
  * cnn_v2 (multi-task model)
  * loss_v2 (MVE + supervised contrastive)
  * train_v2 (training loop + epoch_callback hook)

Each unit test in :mod:`test_v2_pipeline` covers one of these in
isolation; this module runs them together against a tiny synthetic
dataset and asserts that **training actually reduces the validation
classification loss**.

Synthetic-data design notes:
  * Images are 16×16 RGB patches with a per-family base colour. The
    model can't learn fractal geometry from these, but it can learn
    "this colour ⇒ that family", which is enough to drive cls loss
    down.
  * Each family gets a deterministic centre-of-cluster + pixel
    Gaussian noise so every example is unique. Without per-row noise
    the model would memorise four constants and the test would not
    actually exercise gradient descent.
  * Resolution 16×16 matches the model's :class:`AdaptiveAvgPool2d` —
    no resizing path needed inside the test.

Marked ``slow`` so the default ``pytest`` invocation skips it. Run via
``pytest -m slow`` (or simply ``pytest tests/test_train_v2_integration.py``).

Wall-clock on a 2024-era Mac: ~6–8 s. We deliberately do not use the
backend client — the test must be self-contained so it can run in CI
without docker-compose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from fractalov_ml.training.config_v2 import TrainingConfigV2
from fractalov_ml.training.train_v2 import train_v2


# Per-family RGB centroid. Reasonably distant in colour space so the
# tiny CNN can separate them in two epochs at low base_channels.
_FAMILY_COLORS: dict[str, tuple[int, int, int]] = {
    "mandelbrot":  (200,  60,  40),   # warm red
    "julia":       ( 40, 100, 200),   # cool blue
    "burning_ship":(180, 180,  40),   # mustard
    "multibrot":   ( 80, 200, 120),   # green
}


def _make_image(rng: np.random.Generator, base: tuple[int, int, int]) -> Image.Image:
    """16×16 RGB tile centred at ``base`` with per-pixel Gaussian noise.

    σ=20 is enough to make every image unique without obscuring the
    base colour — the network still has plenty of signal at
    base_channels=4.
    """
    arr = np.empty((16, 16, 3), dtype=np.uint8)
    for ch in range(3):
        noise = rng.normal(0, 20, size=(16, 16))
        arr[..., ch] = np.clip(base[ch] + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _build_synthetic_dataset(
    root: Path,
    *,
    train_per_family: int,
    val_per_family: int,
    seed: int,
) -> Path:
    """Write a Slice-2-shaped dataset to ``root``: PNGs in
    ``images/<split>/<family>/`` plus a single ``labels.parquet`` with
    the columns :class:`FractalDatasetV2` expects."""
    rng = np.random.default_rng(seed)
    images_root = root / "images"
    rows: list[dict] = []

    splits = (("train", train_per_family), ("val", val_per_family))
    for split_name, n in splits:
        for family, base_color in _FAMILY_COLORS.items():
            family_dir = images_root / split_name / family
            family_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                example_id = f"{split_name}_{family}_{i:04d}"
                rel_path = f"images/{split_name}/{family}/{example_id}.png"
                _make_image(rng, base_color).save(root / rel_path)
                rows.append(
                    _row_for(
                        example_id=example_id,
                        split=split_name,
                        family=family,
                        rel_path=rel_path,
                    )
                )

    labels_path = root / "labels.parquet"
    pd.DataFrame(rows).to_parquet(labels_path, index=False)
    return labels_path


def _row_for(*, example_id: str, split: str, family: str, rel_path: str) -> dict:
    """Build a parquet row. The family-specific param fields use the
    same NaN / sentinel conventions as the real generator, so masking
    in :func:`compute_loss` exercises the same code paths."""
    is_julia = family == "julia"
    is_multibrot = family == "multibrot"
    return {
        "id": example_id,
        "split": split,
        "family": family,
        "fractal_type": family,
        "c_re": 0.123 if is_julia else None,
        "c_im": -0.456 if is_julia else None,
        "exponent": 3.0 if is_multibrot else None,
        "max_iter": 100,
        "escape_radius": 2.0,
        "smoothing": True,
        "vp_x_min": -2.0,
        "vp_x_max": 1.0,
        "vp_y_min": -1.2,
        "vp_y_max": 1.2,
        "palette": "fire",
        "color_mode": "linear",
        "samples_per_axis": 1,
        "image_path": rel_path,
        "perf_render_ms": 1,
        "perf_colorize_ms": 1,
        "perf_encode_ms": 1,
        "perf_total_ms": 3,
        "file_size_bytes": 100,
        "request_id": f"req-{example_id}",
    }


@pytest.mark.slow
def test_train_v2_reduces_val_cls_loss_on_synthetic_data(tmp_path: Path) -> None:
    """The headline contract: real training, on a tiny self-contained
    dataset, must drive validation classification loss DOWN over two
    epochs.

    If this test fails, something is fundamentally broken in the
    training loop — either the gradient is not flowing (most likely a
    .detach() in the wrong place), or the loss is being computed
    against the wrong target, or the optimiser isn't stepping. The
    synthetic data is engineered so a tiny network at low capacity
    can solve it; a *flat* val loss curve here points at a code bug,
    not a hard task.
    """
    torch.manual_seed(0)

    data_root = tmp_path / "ds"
    _build_synthetic_dataset(
        data_root,
        train_per_family=12,
        val_per_family=8,
        seed=1234,
    )

    out_dir = tmp_path / "run"
    cfg = TrainingConfigV2(
        data_root=data_root,
        out_dir=out_dir,
        epochs=2,
        batch_size=8,
        learning_rate=5e-3,
        weight_decay=0.0,
        cls_weight=1.0,
        c_weight=0.0,
        exp_weight=0.0,
        vp_weight=0.0,
        contrastive_weight=0.0,
        base_channels=4,
        patience=10,
        device_override="cpu",
        enable_augmentation=False,
        seed=0,
        # Per-split eval at end of training expects test/near_ood/etc;
        # synthetic dataset only has train+val. Skip.
        eval_splits=(),
    )

    epoch_metrics: list[dict] = []
    train_v2(
        cfg=cfg,
        epoch_callback=lambda epoch, _train_m, val_m: epoch_metrics.append(val_m),
    )

    assert len(epoch_metrics) == 2, (
        f"expected 2 epochs of metrics, got {len(epoch_metrics)} — "
        "early stop fired unexpectedly"
    )
    epoch1_cls = epoch_metrics[0]["cls"]
    epoch2_cls = epoch_metrics[1]["cls"]
    assert epoch2_cls < epoch1_cls, (
        f"val cls_loss did not decrease: epoch1={epoch1_cls:.4f}, "
        f"epoch2={epoch2_cls:.4f}. Either the optimiser isn't stepping, "
        "the gradient path is broken, or the synthetic task got harder "
        "(check colour separation)."
    )
    assert (out_dir / "best.pt").exists(), "checkpoint was not written"
    assert (out_dir / "metrics_v2.json").exists(), "metrics_v2.json was not written"


@pytest.mark.slow
def test_train_v2_with_multitask_losses_does_not_explode(tmp_path: Path) -> None:
    """Same setup as the headline test but with all four loss heads
    weighted positive, plus contrastive on. This exercises the masked
    NLL paths and the SupCon log-softmax path simultaneously — the
    failure mode it guards against is a NaN loss or a runaway σ
    exploding the joint loss."""
    torch.manual_seed(0)

    data_root = tmp_path / "ds"
    _build_synthetic_dataset(
        data_root,
        train_per_family=10,
        val_per_family=6,
        seed=4321,
    )

    cfg = TrainingConfigV2(
        data_root=data_root,
        out_dir=tmp_path / "run",
        epochs=2,
        batch_size=8,
        learning_rate=2e-3,
        weight_decay=0.0,
        cls_weight=1.0,
        c_weight=0.5,
        exp_weight=0.5,
        vp_weight=0.25,
        contrastive_weight=0.3,
        base_channels=4,
        patience=10,
        device_override="cpu",
        enable_augmentation=False,
        seed=0,
        eval_splits=(),
    )

    losses: list[float] = []
    train_v2(
        cfg=cfg,
        epoch_callback=lambda _e, _tm, val_m: losses.append(val_m["loss"]),
    )

    assert len(losses) == 2
    for loss in losses:
        assert loss == loss, f"NaN val loss observed: {losses}"
        # Joint loss should stay bounded; ~50 is a generous ceiling for
        # a 4-channel model on 16×16 images with all heads engaged.
        assert loss < 50.0, f"runaway joint loss: {losses}"
