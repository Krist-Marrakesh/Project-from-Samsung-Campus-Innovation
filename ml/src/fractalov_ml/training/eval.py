"""Test-set evaluation + reconstruction quality.

Two phases:

* :func:`evaluate` — runs the trained model over the test split, computes
  family classification accuracy + confusion matrix and Julia c MSE. No
  backend interaction.

* :func:`reconstruct` — for every Julia row in the test split, takes the
  predicted ``(c_re, c_im)``, rebuilds the original recipe with the new c,
  asks the backend to re-render, and compares the new PNG to the original.
  Pixel MSE in [0, 1] space is the primary metric — it directly answers
  "how close does the model get to the actual fractal it should have
  produced?", which is the headline figure for the Stage 6 paper section.
"""

from __future__ import annotations

import io
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

from ..client import BackendClient
from ..dataset import FamilyEncoder, FractalDataset
from ..models import FractalCNN
from .config import pick_device


@dataclass
class TestMetrics:
    family_accuracy: float
    family_confusion: list[list[int]]
    family_classes: list[str]
    julia_c_mse: Optional[float]
    julia_c_re_mae: Optional[float]
    julia_c_im_mae: Optional[float]
    julia_count: int
    total: int


def _load_model(checkpoint: Path, device: torch.device) -> FractalCNN:
    """Load a checkpoint and rebuild a matching FractalCNN.

    `base_channels` is inferred from the checkpoint's first conv weight: stem
    is ``Conv2d(3 → c1, 3x3)``, so ``stem.0.weight`` has shape ``(c1, 3, 3, 3)``
    and ``c1 == base_channels``. This avoids storing the value separately and
    keeps existing checkpoints loadable.
    """
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    base_channels = state["stem.0.weight"].shape[0]
    model = FractalCNN(base_channels=base_channels).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(
    data_root: Path,
    checkpoint: Path,
    out_dir: Path | None = None,
    batch_size: int = 64,
    device_override: str | None = None,
    console: Console | None = None,
) -> TestMetrics:
    console = console or Console()
    device = pick_device(device_override)
    encoder = FamilyEncoder.default()

    ds = FractalDataset(data_root, split="test")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = _load_model(checkpoint, device)

    confusion = np.zeros((len(encoder.classes), len(encoder.classes)), dtype=np.int64)
    correct = 0
    julia_squared = 0.0
    julia_re_abs = 0.0
    julia_im_abs = 0.0
    julia_n = 0

    with torch.no_grad():
        for images, family_int, params in loader:
            images = images.to(device)
            family_int = family_int.to(device).long()
            # Cast on host first — MPS does not support float64.
            c_re = params["c_re"].float().to(device)
            c_im = params["c_im"].float().to(device)
            has_c = params["has_c"].to(device).bool()

            out = model(images)
            preds = out.family_logits.argmax(dim=1)
            correct += int((preds == family_int).sum().item())
            for t, p in zip(family_int.cpu().tolist(), preds.cpu().tolist()):
                confusion[t][p] += 1

            if has_c.any():
                idx = has_c.nonzero(as_tuple=False).squeeze(1)
                pred_c = out.c_pred[idx]
                target_c = torch.stack([c_re[idx], c_im[idx]], dim=1)
                diff = pred_c - target_c
                julia_squared += float(diff.pow(2).sum().item())
                julia_re_abs += float(diff[:, 0].abs().sum().item())
                julia_im_abs += float(diff[:, 1].abs().sum().item())
                julia_n += int(idx.numel())

    metrics = TestMetrics(
        family_accuracy=correct / max(len(ds), 1),
        family_confusion=confusion.tolist(),
        family_classes=list(encoder.classes),
        julia_c_mse=(julia_squared / (2 * julia_n)) if julia_n > 0 else None,
        julia_c_re_mae=(julia_re_abs / julia_n) if julia_n > 0 else None,
        julia_c_im_mae=(julia_im_abs / julia_n) if julia_n > 0 else None,
        julia_count=julia_n,
        total=len(ds),
    )

    _print_metrics(console, metrics)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "test_metrics.json").write_text(json.dumps(asdict(metrics), indent=2))
        console.print(f"[green]wrote {out_dir / 'test_metrics.json'}")
    return metrics


def _print_metrics(console: Console, m: TestMetrics) -> None:
    console.rule("test metrics")
    console.print(f"family accuracy = {m.family_accuracy:.4f}  ({m.total} examples)")
    if m.julia_c_mse is not None:
        console.print(
            f"julia c MSE    = {m.julia_c_mse:.4f}  "
            f"|  c_re MAE = {m.julia_c_re_mae:.4f}  "
            f"|  c_im MAE = {m.julia_c_im_mae:.4f}  "
            f"({m.julia_count} julia samples)"
        )

    table = Table(title="confusion (rows=true, cols=pred)")
    table.add_column("")
    for c in m.family_classes:
        table.add_column(c)
    for i, true_label in enumerate(m.family_classes):
        row = [true_label]
        for j in range(len(m.family_classes)):
            row.append(str(m.family_confusion[i][j]))
        table.add_row(*row)
    console.print(table)


@dataclass
class ReconstructionMetrics:
    n_attempted: int
    n_succeeded: int
    pixel_mse_mean: float
    pixel_mse_median: float
    pixel_mse_p90: float


def reconstruct(
    data_root: Path,
    checkpoint: Path,
    out_dir: Path | None = None,
    backend_url: str = "http://localhost:8080",
    max_samples: int = 100,
    device_override: str | None = None,
    console: Console | None = None,
) -> ReconstructionMetrics:
    """Re-render Julia test images using predicted c and compare to the original.

    The 'reconstruction quality' angle is what makes the c-regression head
    interesting beyond a numeric MSE: a model that gets c_re to two decimals
    can still produce a visually-different fractal because the Julia set is
    non-linear in c. Pixel MSE catches this directly.
    """
    console = console or Console()
    device = pick_device(device_override)
    model = _load_model(checkpoint, device)

    labels = pd.read_parquet(data_root / "labels.parquet")
    splits = pd.read_parquet(data_root / "splits.parquet")
    test_ids = splits[splits["split"] == "test"]["id"]
    test_julia = labels.merge(test_ids.to_frame(), on="id").query("family == 'julia'")
    if max_samples is not None and max_samples > 0:
        test_julia = test_julia.head(max_samples)
    console.print(f"[bold]reconstructing {len(test_julia)} julia test samples")

    pixel_mses: list[float] = []
    succeeded = 0
    failed = 0

    with BackendClient(base_url=backend_url) as client:
        with torch.no_grad():
            for _, row in test_julia.iterrows():
                # Predict c from the original image.
                img = Image.open(data_root / row["image_path"]).convert("RGB")
                width_px, height_px = img.size  # PIL: (W, H)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                out = model(tensor)
                c_re_pred, c_im_pred = out.c_pred[0].cpu().tolist()

                # Rebuild the original recipe but swap c.
                recipe = _row_to_recipe(row, c_re_pred, c_im_pred, width_px, height_px)
                try:
                    rendered = client.render(recipe)
                except Exception as ex:
                    console.log(f"[yellow]reconstruction backend call failed: {ex}")
                    failed += 1
                    continue

                pred_arr = (
                    np.asarray(Image.open(io.BytesIO(rendered.image_bytes)).convert("RGB"))
                    .astype(np.float32)
                    / 255.0
                )
                if pred_arr.shape != arr.shape:
                    console.log(
                        f"[yellow]shape mismatch (orig={arr.shape}, pred={pred_arr.shape})"
                    )
                    failed += 1
                    continue

                pixel_mses.append(float(np.mean((arr - pred_arr) ** 2)))
                succeeded += 1

    if not pixel_mses:
        raise RuntimeError("no reconstructions succeeded — is the backend up?")

    arr = np.array(pixel_mses)
    metrics = ReconstructionMetrics(
        n_attempted=succeeded + failed,
        n_succeeded=succeeded,
        pixel_mse_mean=float(arr.mean()),
        pixel_mse_median=float(np.median(arr)),
        pixel_mse_p90=float(np.quantile(arr, 0.9)),
    )

    table = Table(title="reconstruction quality (pixel MSE in [0,1])")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("n_succeeded", str(metrics.n_succeeded))
    table.add_row("mean", f"{metrics.pixel_mse_mean:.5f}")
    table.add_row("median", f"{metrics.pixel_mse_median:.5f}")
    table.add_row("p90", f"{metrics.pixel_mse_p90:.5f}")
    console.print(table)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "reconstruction_metrics.json").write_text(
            json.dumps(asdict(metrics), indent=2)
        )
        console.print(f"[green]wrote {out_dir / 'reconstruction_metrics.json'}")
    return metrics


def _row_to_recipe(
    row: pd.Series, c_re: float, c_im: float, width_px: int, height_px: int
) -> dict:
    """Rebuild the wire-shape FractalRecipe used to generate the original image,
    swapping in predicted (c_re, c_im). Image size is taken from the on-disk PNG
    because the parquet schema doesn't store per-row dimensions.
    """
    return {
        "viewport": {
            "xMin": float(row["vp_x_min"]),
            "xMax": float(row["vp_x_max"]),
            "yMin": float(row["vp_y_min"]),
            "yMax": float(row["vp_y_max"]),
        },
        "renderSettings": {
            "widthPx": int(width_px),
            "heightPx": int(height_px),
            "samplesPerAxis": int(row["samples_per_axis"]),
        },
        "colorSettings": {
            "paletteName": str(row["palette"]),
            "mode": str(row["color_mode"]),
        },
        "fractalType": "julia",
        "params": {
            "cRe": float(c_re),
            "cIm": float(c_im),
            "maxIter": int(row["max_iter"]),
            "escapeRadius": float(row["escape_radius"]),
            "smoothing": bool(row["smoothing"]),
        },
    }
