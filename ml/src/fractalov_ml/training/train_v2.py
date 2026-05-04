"""Multi-task training loop for FractalCNNv2.

Three-headed model (family + Julia c MVE + Multibrot exponent MVE) trained
on the Slice 2 dataset format. The loop is structurally similar to v1
``train.py`` but pivots on the new loss + new dataset shape.

Outputs in ``out_dir``:

    metrics_v2.json        per-epoch + final per-split summary
    best.pt                checkpoint with the lowest val cls_loss
    last.pt                most recent epoch (resume-friendly)
    config_v2.json         the TrainingConfigV2 that produced this run
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
import torchvision.transforms as T
from rich.console import Console
from rich.table import Table
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..dataset_v2 import FractalDatasetV2, collate
from ..models.cnn_v2 import FractalCNNv2
from .config import pick_device
from .config_v2 import TrainingConfigV2
from .loss_v2 import compute_loss


@dataclass
class TrainResultV2:
    best_val_score: float
    best_checkpoint: Path
    metrics: list[dict]
    epochs_run: int
    final_per_split: dict[str, dict]


class TrainingAborted(RuntimeError):
    pass


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _augment() -> T.Compose:
    # Same rationale as v1: do NOT horizontal-flip — it would require c-conjugation.
    return T.Compose([T.ToTensor(), T.ColorJitter(brightness=0.05, contrast=0.05)])


def _no_augment() -> T.Compose:
    return T.Compose([T.ToTensor()])


def _device_str(d: torch.device) -> str:
    return f"{d.type}:{d.index}" if d.index is not None else d.type


def _to_device_batch(batch: dict, device: torch.device) -> dict:
    return {
        "image": batch["image"].to(device, non_blocking=True),
        "family_idx": batch["family_idx"].to(device, non_blocking=True).long(),
        "c": batch["c"].to(device, non_blocking=True).float(),
        "exponent": batch["exponent"].to(device, non_blocking=True).float(),
        "viewport": batch["viewport"].to(device, non_blocking=True).float(),
        "has_julia": batch["has_julia"].to(device, non_blocking=True).bool(),
        "has_multibrot": batch["has_multibrot"].to(device, non_blocking=True).bool(),
    }


def _eval_loop(model: FractalCNNv2, loader: DataLoader, device: torch.device, cfg: TrainingConfigV2) -> dict:
    model.eval()
    parts = _zero_parts()
    n_correct = 0
    n_total = 0
    c_abs_err_sum = 0.0
    c_n = 0
    exp_abs_err_sum = 0.0
    exp_n = 0
    c_within_sigma = 0
    vp_abs_err_sum = 0.0
    vp_n = 0
    with torch.no_grad():
        for batch in loader:
            b = _to_device_batch(batch, device)
            out = model(b["image"])
            # NaN-substitute targets for index-select safety; index already masks.
            c_target = torch.where(torch.isnan(b["c"]), torch.zeros_like(b["c"]), b["c"])
            exp_target = torch.where(torch.isnan(b["exponent"]), torch.zeros_like(b["exponent"]), b["exponent"])
            pieces = compute_loss(
                out,
                family_idx=b["family_idx"],
                c_target=c_target,
                exp_target=exp_target,
                vp_target=b["viewport"],
                has_julia=b["has_julia"],
                has_multibrot=b["has_multibrot"],
                cls_weight=cfg.cls_weight,
                c_weight=cfg.c_weight,
                exp_weight=cfg.exp_weight,
                vp_weight=cfg.vp_weight,
                contrastive_weight=cfg.contrastive_weight,
                contrastive_temperature=cfg.contrastive_temperature,
            )
            _accumulate(parts, pieces)
            preds = out.family_logits.argmax(dim=1)
            n_correct += int((preds == b["family_idx"]).sum().item())
            n_total += int(b["family_idx"].numel())

            if b["has_julia"].any():
                jidx = b["has_julia"].nonzero(as_tuple=False).squeeze(1)
                cm = out.c_mean.index_select(0, jidx)
                ct = c_target.index_select(0, jidx)
                err = (cm - ct).abs()
                c_abs_err_sum += float(err.sum().item())
                c_n += int(err.numel())
                # Calibration: fraction of (target - μ) inside ±σ predicted.
                clv = out.c_log_var.index_select(0, jidx).clamp(-7.0, 7.0)
                sigma = torch.exp(0.5 * clv)
                inside = ((cm - ct).abs() <= sigma).float()
                c_within_sigma += int(inside.sum().item())

            if b["has_multibrot"].any():
                midx = b["has_multibrot"].nonzero(as_tuple=False).squeeze(1)
                em = out.exp_mean.index_select(0, midx)
                et = exp_target.index_select(0, midx)
                err = (em - et).abs()
                exp_abs_err_sum += float(err.sum().item())
                exp_n += int(err.numel())

            # Viewport — unmasked, every row contributes.
            vp_err = (out.viewport_mean - b["viewport"]).abs()
            vp_abs_err_sum += float(vp_err.sum().item())
            vp_n += int(vp_err.numel())

    summary = _summarise(parts)
    summary["accuracy"] = (n_correct / n_total) if n_total else 0.0
    summary["c_mae"] = (c_abs_err_sum / c_n) if c_n else None
    summary["exp_mae"] = (exp_abs_err_sum / exp_n) if exp_n else None
    summary["vp_mae"] = (vp_abs_err_sum / vp_n) if vp_n else None
    summary["c_calibration_within_sigma"] = (c_within_sigma / c_n) if c_n else None
    summary["n_total"] = n_total
    summary["n_julia"] = c_n // 2 if c_n else 0   # c is 2-D
    summary["n_multibrot"] = exp_n
    return summary


def _epoch_table(
    epoch: int,
    train_m: dict,
    val_m: dict,
    lr: float,
    epoch_seconds: float,
) -> Table:
    table = Table(title=f"epoch {epoch:03d}", show_header=True, expand=False)
    table.add_column("split")
    table.add_column("loss")
    table.add_column("cls")
    table.add_column("c_nll")
    table.add_column("exp_nll")
    table.add_column("vp_nll")
    table.add_column("acc")
    table.add_column("c_mae")
    table.add_column("exp_mae")
    table.add_column("vp_mae")
    table.add_column("σ-cov")
    for name, m in (("train", train_m), ("val", val_m)):
        table.add_row(
            name,
            f"{m['loss']:.4f}",
            f"{m['cls']:.4f}",
            f"{m['c_nll']:.4f}",
            f"{m['exp_nll']:.4f}",
            f"{m.get('vp_nll', 0.0):.4f}",
            f"{m.get('accuracy', float('nan')):.4f}" if "accuracy" in m else "-",
            _fmt(m.get("c_mae")),
            _fmt(m.get("exp_mae")),
            _fmt(m.get("vp_mae")),
            _fmt(m.get("c_calibration_within_sigma")),
        )
    table.caption = f"lr={lr:.2e}  time={epoch_seconds:.1f}s"
    return table


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4f}"


def train_v2(
    cfg: TrainingConfigV2,
    console: Optional[Console] = None,
    epoch_callback: Optional[Callable[[int, dict, dict], None]] = None,
) -> TrainResultV2:
    """Run the v2 training loop.

    Parameters
    ----------
    epoch_callback:
        Optional ``(epoch, train_metrics, val_metrics) -> None`` hook fired
        after each epoch's checkpoints have been written. The Optuna driver
        uses this to report intermediate values and prune unpromising trials.
        Raising from the callback aborts the loop with the exception
        propagating to the caller; the v1 train loop does the same and the
        :mod:`tune_v2` driver relies on it for ``optuna.TrialPruned``.
    """
    console = console or Console()
    _seed_everything(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device_override)
    console.print(
        f"[bold]device[/bold] = {_device_str(device)}, "
        f"torch={torch.__version__}, data_root={cfg.data_root}"
    )

    train_tf = _augment() if cfg.enable_augmentation else _no_augment()
    eval_tf = _no_augment()
    train_ds = FractalDatasetV2(cfg.data_root, split="train", transform=train_tf)
    val_ds = FractalDatasetV2(cfg.data_root, split="val", transform=eval_tf)
    console.print(f"[dim]train={len(train_ds)} val={len(val_ds)}[/dim]")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_memory, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_memory, collate_fn=collate,
    )

    model = FractalCNNv2(base_channels=cfg.base_channels).to(device)
    console.print(
        f"[dim]params={model.num_parameters:,} base_channels={cfg.base_channels}[/dim]"
    )
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optim, T_max=cfg.epochs)

    best_score = math.inf
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
        dropped = 0
        n_train_correct = 0
        n_train_total = 0
        for batch in train_loader:
            b = _to_device_batch(batch, device)
            out = model(b["image"])
            c_target = torch.where(torch.isnan(b["c"]), torch.zeros_like(b["c"]), b["c"])
            exp_target = torch.where(torch.isnan(b["exponent"]), torch.zeros_like(b["exponent"]), b["exponent"])
            pieces = compute_loss(
                out,
                family_idx=b["family_idx"],
                c_target=c_target,
                exp_target=exp_target,
                vp_target=b["viewport"],
                has_julia=b["has_julia"],
                has_multibrot=b["has_multibrot"],
                cls_weight=cfg.cls_weight,
                c_weight=cfg.c_weight,
                exp_weight=cfg.exp_weight,
                vp_weight=cfg.vp_weight,
                contrastive_weight=cfg.contrastive_weight,
                contrastive_temperature=cfg.contrastive_temperature,
            )
            if not torch.isfinite(pieces.total):
                dropped += 1
                if dropped > 50:
                    raise TrainingAborted(
                        f"too many non-finite losses at epoch {epoch} (>{dropped})"
                    )
                continue
            optim.zero_grad(set_to_none=True)
            pieces.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            _accumulate(tparts, pieces)
            preds = out.family_logits.argmax(dim=1)
            n_train_correct += int((preds == b["family_idx"]).sum().item())
            n_train_total += int(b["family_idx"].numel())

        train_metrics = _summarise(tparts)
        train_metrics["accuracy"] = (n_train_correct / n_train_total) if n_train_total else 0.0

        # ----- val -----
        val_metrics = _eval_loop(model, val_loader, device, cfg)

        scheduler.step()
        epoch_seconds = time.time() - epoch_start
        lr_now = optim.param_groups[0]["lr"]

        if epoch % cfg.log_every_epochs == 0 or epoch == cfg.epochs:
            console.print(_epoch_table(epoch, train_metrics, val_metrics, lr_now, epoch_seconds))

        metrics_log.append({
            "epoch": epoch,
            "lr": lr_now,
            "time_s": round(epoch_seconds, 3),
            "dropped_batches": dropped,
            "train": train_metrics,
            "val": val_metrics,
        })

        cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(
            {"epoch": epoch, "model_state": cpu_state, "val_metrics": val_metrics, "config": _config_to_jsonable(cfg)},
            last_path,
        )

        # Best criterion: val cls_loss (same rationale as v1) — regression
        # NLL alone could be gamed by the variance head exploding to high σ.
        score = float(val_metrics["cls"])
        if score < best_score - cfg.min_delta:
            best_score = score
            torch.save({"model_state": cpu_state, "config": _config_to_jsonable(cfg)}, best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epochs_run = epoch
        # Hook fires after the checkpoint write, so a callback that raises
        # leaves the on-disk best.pt valid.
        if epoch_callback is not None:
            epoch_callback(epoch, train_metrics, val_metrics)

        if epochs_no_improve >= cfg.patience:
            console.print(f"[yellow]early stop after {epoch} epochs (no improvement for {cfg.patience})")
            break

    # ----- per-split eval at end of training -----
    final_per_split: dict[str, dict] = {}
    if cfg.eval_splits:
        # Reload best checkpoint so per-split numbers reflect the saved model,
        # not the last (post-overfit) state.
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        for split_name in cfg.eval_splits:
            try:
                ds = FractalDatasetV2(cfg.data_root, split=split_name, transform=eval_tf)
            except (FileNotFoundError, ValueError) as ex:
                console.log(f"[yellow]skip eval split {split_name}: {ex}")
                continue
            loader = DataLoader(
                ds, batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=pin_memory, collate_fn=collate,
            )
            split_metrics = _eval_loop(model, loader, device, cfg)
            final_per_split[split_name] = split_metrics
            console.print(
                f"[cyan]split={split_name}[/cyan] acc={_fmt(split_metrics.get('accuracy'))} "
                f"c_mae={_fmt(split_metrics.get('c_mae'))} "
                f"exp_mae={_fmt(split_metrics.get('exp_mae'))} "
                f"vp_mae={_fmt(split_metrics.get('vp_mae'))} "
                f"σ-cov={_fmt(split_metrics.get('c_calibration_within_sigma'))} "
                f"(n={split_metrics.get('n_total')})"
            )

    (cfg.out_dir / "metrics_v2.json").write_text(
        json.dumps({"epochs": metrics_log, "per_split": final_per_split}, indent=2)
    )
    (cfg.out_dir / "config_v2.json").write_text(
        json.dumps({**_config_to_jsonable(cfg), "device": _device_str(device)}, indent=2)
    )
    console.print(f"[green]best val cls_loss = {best_score:.4f}  →  {best_path}")
    return TrainResultV2(
        best_val_score=best_score,
        best_checkpoint=best_path,
        metrics=metrics_log,
        epochs_run=epochs_run,
        final_per_split=final_per_split,
    )


def _zero_parts() -> dict:
    return {
        "loss_sum": 0.0,
        "cls_sum": 0.0,
        "c_nll_sum": 0.0,
        "exp_nll_sum": 0.0,
        "vp_nll_sum": 0.0,
        "con_sum": 0.0,
        "batches": 0,
    }


def _accumulate(parts: dict, pieces) -> None:
    parts["loss_sum"] += float(pieces.total.item())
    parts["cls_sum"] += float(pieces.cls.item())
    parts["c_nll_sum"] += float(pieces.c_nll.item())
    parts["exp_nll_sum"] += float(pieces.exp_nll.item())
    parts["vp_nll_sum"] += float(pieces.vp_nll.item())
    parts["con_sum"] += float(pieces.contrastive.item())
    parts["batches"] += 1


def _summarise(parts: dict) -> dict:
    n = max(parts["batches"], 1)
    return {
        "loss": parts["loss_sum"] / n,
        "cls": parts["cls_sum"] / n,
        "c_nll": parts["c_nll_sum"] / n,
        "exp_nll": parts["exp_nll_sum"] / n,
        "vp_nll": parts["vp_nll_sum"] / n,
        "contrastive": parts["con_sum"] / n,
    }


def _config_to_jsonable(cfg: TrainingConfigV2) -> dict:
    d = asdict(cfg)
    d["data_root"] = str(d["data_root"])
    d["out_dir"] = str(d["out_dir"])
    d["eval_splits"] = list(d["eval_splits"])
    return d
