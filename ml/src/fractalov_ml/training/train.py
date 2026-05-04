"""Train loop for :class:`fractalov_ml.models.FractalCNN`.

Multi-task setup:
  * Family classification — cross-entropy over all examples
  * Julia c-regression — masked MSE; only julia rows contribute, identified
    by the ``has_c`` boolean from :class:`FractalDataset`

Loss is the weighted sum ``α·cls + β·reg``. Gradient flow into the c-head from
non-julia rows is killed by zeroing the masked term, so the regressor never
sees pressure to push c toward 0 just because Mandelbrot dominates the batch.

Outputs in ``out_dir``::

    metrics.json     per-epoch list of {epoch, train_*, val_*, lr, time_s}
    best.pt          checkpoint with lowest val loss seen so far
    last.pt          latest checkpoint (use it to resume)
    config.json      a copy of the TrainingConfig that produced this run
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from rich.console import Console
from rich.table import Table
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..dataset import FractalDataset
from ..models import FractalCNN
from .config import TrainingConfig, pick_device


@dataclass
class TrainResult:
    best_val_loss: float
    best_checkpoint: Path
    metrics: list[dict]
    epochs_run: int


class TrainingAborted(RuntimeError):
    """Raised when training cannot continue (e.g. NaN loss, callback abort)."""


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _augment() -> T.Compose:
    """Conservative augmentations.

    Horizontal flip is *not* meaningful for c-regression — the Julia set for
    c is the mirror image of the set for conjugate(c) — so the regression
    target would have to be conjugated alongside the image. We skip it
    entirely; the dataset is already quite diverse via viewport jitter on
    the generation side.
    """
    return T.Compose(
        [
            T.ToTensor(),  # PIL → (C, H, W) float in [0, 1]
            T.ColorJitter(brightness=0.05, contrast=0.05),
        ]
    )


def _no_augment() -> T.Compose:
    return T.Compose([T.ToTensor()])


def _device_str(d: torch.device) -> str:
    return f"{d.type}:{d.index}" if d.index is not None else d.type


def _epoch_table(
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    lr: float,
    epoch_seconds: float,
) -> Table:
    table = Table(title=f"epoch {epoch:03d}", show_header=True, expand=False)
    table.add_column("split")
    table.add_column("loss")
    table.add_column("cls_loss")
    table.add_column("reg_loss")
    table.add_column("acc")
    table.add_column("c_mse")
    for name, m in (("train", train_metrics), ("val", val_metrics)):
        table.add_row(
            name,
            f"{m['loss']:.4f}",
            f"{m['cls_loss']:.4f}",
            f"{m['reg_loss']:.4f}",
            f"{m['accuracy']:.4f}",
            f"{m['c_mse']:.4f}" if m["c_mse"] is not None else "-",
        )
    table.caption = f"lr={lr:.2e}  time={epoch_seconds:.1f}s"
    return table


def _move_targets(family_int: torch.Tensor, params: dict, device: torch.device):
    # MPS does not support float64; pandas Python floats collate into float64
    # tensors. Cast on host first, then move — moving as float64 to MPS errors
    # before the .float() ever runs.
    family_int = family_int.to(device, non_blocking=True).long()
    c_re = params["c_re"].float().to(device, non_blocking=True)
    c_im = params["c_im"].float().to(device, non_blocking=True)
    has_c = params["has_c"].to(device, non_blocking=True).bool()
    c_target = torch.stack([c_re, c_im], dim=1)
    # Zero out NaN targets just in case — they're masked away by has_c, but
    # NaNs can poison the loss tensor before masking takes effect.
    c_target = torch.where(torch.isnan(c_target), torch.zeros_like(c_target), c_target)
    return family_int, c_target, has_c


def _step(
    model: FractalCNN,
    images: torch.Tensor,
    family_int: torch.Tensor,
    c_target: torch.Tensor,
    has_c: torch.Tensor,
    cfg: TrainingConfig,
):
    out = model(images)
    cls_loss = F.cross_entropy(out.family_logits, family_int)

    # Masked MSE on Julia rows.
    # We INDEX-SELECT julia rows before subtracting from c_target. The
    # alternative — compute (pred - target).pow(2) on every row then multiply
    # by `valid` — fails on IEEE: if c_pred for a non-julia row goes NaN
    # (which happens any time the c-head's Linear/ReLU stack lands in a bad
    # spot), `nan * 0 = nan` propagates through the sum. Indexing first
    # cleanly excludes those rows from the autograd graph.
    if has_c.any():
        idx = has_c.nonzero(as_tuple=False).squeeze(1)
        julia_pred = out.c_pred[idx]
        julia_target = c_target[idx]
        per_julia = (julia_pred - julia_target).pow(2).mean(dim=1)  # (n_julia,)
        reg_loss = per_julia.mean()
        n_julia = idx.numel()
        c_sum_for_log = per_julia.sum().detach()
    else:
        # Zero-julia batch: skip the regression head entirely. We multiply by
        # zero rather than calling .detach() so the autograd graph stays
        # consistent across batches (Optuna step counters were getting upset
        # otherwise).
        reg_loss = (out.c_pred.sum() * 0.0).mean()
        n_julia = 0
        c_sum_for_log = torch.zeros((), device=out.c_pred.device)

    loss = cfg.cls_weight * cls_loss + cfg.reg_weight * reg_loss
    pred = out.family_logits.argmax(dim=1)
    correct = (pred == family_int).sum().item()
    return loss, cls_loss.detach(), reg_loss.detach(), correct, c_sum_for_log, n_julia


def _summarise(parts: dict, total: int) -> dict:
    n_valid = parts["n_valid"]
    return {
        "loss": parts["loss_sum"] / max(parts["batches"], 1),
        "cls_loss": parts["cls_sum"] / max(parts["batches"], 1),
        "reg_loss": parts["reg_sum"] / max(parts["batches"], 1),
        "accuracy": parts["correct"] / max(total, 1),
        # n_valid here counts julia rows (not pixel-pairs); per_julia.mean
        # already averaged over the (cRe, cIm) axis so dividing by n_valid
        # alone gives the per-julia c-MSE.
        "c_mse": (parts["c_sum"] / n_valid) if n_valid > 0 else None,
    }


def train(
    cfg: TrainingConfig,
    console: Console | None = None,
    epoch_callback: Optional[Callable[[int, dict, dict], None]] = None,
) -> TrainResult:
    """Run the train loop.

    Parameters
    ----------
    epoch_callback:
        Optional ``(epoch, train_metrics, val_metrics) -> None`` hook fired at
        the end of every epoch. Used by the Optuna driver to report intermediate
        values and prune unpromising trials. The callback may raise to abort
        training; the raised exception bubbles up unchanged so callers can
        distinguish prune signals from real errors.
    """
    console = console or Console()
    _seed_everything(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(cfg.device_override)
    console.print(
        f"[bold]device[/bold] = {_device_str(device)}, "
        f"torch={torch.__version__}, "
        f"data_root={cfg.data_root}"
    )

    train_tf = _augment() if cfg.enable_augmentation else _no_augment()
    eval_tf = _no_augment()
    train_ds = FractalDataset(cfg.data_root, split="train", transform=train_tf)
    val_ds = FractalDataset(cfg.data_root, split="val", transform=eval_tf)
    console.print(
        f"[dim]train={len(train_ds)} val={len(val_ds)} "
        f"batch={cfg.batch_size} workers={cfg.num_workers}[/dim]"
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    model = FractalCNN(base_channels=cfg.base_channels).to(device)
    console.print(
        f"[dim]params={model.num_parameters:,} base_channels={cfg.base_channels}[/dim]"
    )
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optim, T_max=cfg.epochs)

    best_val_loss = math.inf
    best_path = cfg.out_dir / "best.pt"
    last_path = cfg.out_dir / "last.pt"
    metrics_log: list[dict] = []
    epochs_no_improve = 0

    epochs_run = 0
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        # ----- train -----
        model.train()
        tparts = _zero_parts()
        for images, family_int, params in train_loader:
            images = images.to(device, non_blocking=True)
            family_int, c_target, has_c = _move_targets(family_int, params, device)

            loss, cls, reg, correct, per_elem, valid = _step(
                model, images, family_int, c_target, has_c, cfg
            )
            # NaN/Inf guard. On MPS, occasional batches produce non-finite c-head
            # activations even with gradient clipping — silently dropping those
            # batches keeps training going (the next batch resets the optimiser
            # state's contribution from the bad step).
            if not torch.isfinite(loss):
                tparts["dropped_batches"] = tparts.get("dropped_batches", 0) + 1
                if tparts["dropped_batches"] > 50:
                    # Too many drops in a single epoch → unrecoverable; let
                    # Optuna mark the trial as pruned and the CLI surface the
                    # underlying instability.
                    raise TrainingAborted(
                        f"too many non-finite losses at epoch {epoch} "
                        f"(>{tparts['dropped_batches']} batches)"
                    )
                continue
            optim.zero_grad(set_to_none=True)
            loss.backward()
            # Clip gradients to keep early-training instabilities under control
            # without hand-tuning lr per architecture; harmless when grads are
            # small.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            _accumulate(tparts, loss, cls, reg, correct, per_elem, valid)

        # ----- val -----
        model.eval()
        vparts = _zero_parts()
        with torch.no_grad():
            for images, family_int, params in val_loader:
                images = images.to(device, non_blocking=True)
                family_int, c_target, has_c = _move_targets(family_int, params, device)
                loss, cls, reg, correct, per_elem, valid = _step(
                    model, images, family_int, c_target, has_c, cfg
                )
                _accumulate(vparts, loss, cls, reg, correct, per_elem, valid)

        scheduler.step()
        epoch_seconds = time.time() - epoch_start

        train_metrics = _summarise(tparts, len(train_ds))
        val_metrics = _summarise(vparts, len(val_ds))
        lr_now = optim.param_groups[0]["lr"]

        if epoch % cfg.log_every_epochs == 0 or epoch == cfg.epochs:
            console.print(_epoch_table(epoch, train_metrics, val_metrics, lr_now, epoch_seconds))

        metrics_log.append(
            {
                "epoch": epoch,
                "lr": lr_now,
                "time_s": round(epoch_seconds, 3),
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        # Move state to CPU before saving. On MPS, ``torch.save`` of an
        # MPS-resident state_dict can write tensors in a state where the
        # running BatchNorm statistics aren't synced, leading to a checkpoint
        # that loads back with effectively random BN statistics. The CPU
        # roundtrip costs <50 ms per epoch and produces deterministic
        # checkpoints regardless of train device.
        cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(
            {
                "epoch": epoch,
                "model_state": cpu_state,
                "val_loss": val_metrics["loss"],
            },
            last_path,
        )

        # Best-checkpoint criterion: classification loss, not the joint loss.
        # The joint loss is dominated by the regression head (which collapses
        # to ~0 quickly because non-julia rows contribute zero); a model that
        # got reg=0 by predicting nothing useful for classification would beat
        # a genuinely good model on joint loss. Tracking val_cls_loss aligns
        # the checkpoint with the head we actually care about for the
        # downstream family classifier.
        score = val_metrics["cls_loss"]
        if score < best_val_loss - cfg.min_delta:
            best_val_loss = score
            torch.save(cpu_state, best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epochs_run = epoch

        # Hook for tuning / external monitoring. Called *after* checkpoints are
        # written, so a callback that aborts still leaves a valid best.pt.
        if epoch_callback is not None:
            epoch_callback(epoch, train_metrics, val_metrics)

        if epochs_no_improve >= cfg.patience:
            console.print(
                f"[yellow]early stop after {epoch} epochs "
                f"(no improvement for {cfg.patience})"
            )
            break

    (cfg.out_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2))
    (cfg.out_dir / "config.json").write_text(
        json.dumps(
            {
                **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
                "device": _device_str(device),
            },
            indent=2,
        )
    )
    console.print(f"[green]best val loss = {best_val_loss:.4f}  →  {best_path}")
    return TrainResult(
        best_val_loss=best_val_loss,
        best_checkpoint=best_path,
        metrics=metrics_log,
        epochs_run=epochs_run,
    )


def _zero_parts() -> dict:
    return {
        "loss_sum": 0.0,
        "cls_sum": 0.0,
        "reg_sum": 0.0,
        "correct": 0,
        "c_sum": 0.0,
        "n_valid": 0,
        "batches": 0,
    }


def _accumulate(parts, loss, cls, reg, correct, c_sum_for_log, n_julia) -> None:
    parts["loss_sum"] += float(loss.item())
    parts["cls_sum"] += float(cls.item())
    parts["reg_sum"] += float(reg.item())
    parts["correct"] += correct
    parts["c_sum"] += float(c_sum_for_log.item())
    parts["n_valid"] += int(n_julia)
    parts["batches"] += 1
