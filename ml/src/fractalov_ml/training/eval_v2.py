"""Evaluation pipeline for the v2 multi-task / hybrid-refinement story.

Produces ``eval_v2.json`` with three layers of metrics for every requested
split:

* ``classification`` — family acc + confusion against the family head.
* ``regression``     — Julia c MAE/MSE, Multibrot exponent MAE,
                       calibration coverage (fraction of targets inside
                       μ ± σ).
* ``reconstruction`` — pixel-MSE against the target image, computed with
                       both the pure-NN warm-start params and the
                       L-BFGS-refined params. Only Julia + Multibrot
                       contribute (Mandelbrot / Burning-Ship have no
                       continuous family-internal knob to refine).

The point of carrying both pure-NN and hybrid numbers in the same file is
the central research result of Slice 3: the hybrid approach should
strictly improve median reconstruction MSE on Julia and either improve or
match it on Multibrot. If it doesn't, that is itself a useful number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

from ..dataset import FamilyEncoder
from ..dataset_v2 import FractalDatasetV2, collate
from ..models.cnn_v2 import FractalCNNv2
from .config import pick_device
from .refine import RefineRequest, refine


@dataclass
class EvalV2Result:
    per_split: dict[str, dict]


def evaluate_v2(
    *,
    data_root: Path,
    checkpoint: Path,
    out_dir: Optional[Path],
    splits: Iterable[str] = ("test", "near_ood", "hard_overlap"),
    base_channels: Optional[int] = None,
    refine_n_iters: int = 20,
    refine_max_samples_per_family: int = 50,
    device_override: Optional[str] = None,
    console: Optional[Console] = None,
) -> EvalV2Result:
    console = console or Console()
    device = pick_device(device_override)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config") or {}
    bc = base_channels or int(ckpt_cfg.get("base_channels", 32))
    model = FractalCNNv2(base_channels=bc).to(device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    encoder = FamilyEncoder.default()
    per_split: dict[str, dict] = {}
    for split_name in splits:
        try:
            ds = FractalDatasetV2(data_root, split=split_name)
        except (FileNotFoundError, ValueError) as ex:
            console.log(f"[yellow]skip {split_name}: {ex}")
            continue
        loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
        cls_metrics, regr_metrics, sample_records = _classification_and_regression(model, loader, device, encoder)
        recon_metrics = _reconstruction(
            ds=ds,
            sample_records=sample_records,
            device=device,
            n_iters=refine_n_iters,
            max_per_family=refine_max_samples_per_family,
            console=console,
            split_name=split_name,
        )
        per_split[split_name] = {
            "n": len(ds),
            "classification": cls_metrics,
            "regression": regr_metrics,
            "reconstruction": recon_metrics,
        }
        _print_split_summary(console, split_name, per_split[split_name])

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_v2.json").write_text(json.dumps(per_split, indent=2))
        console.print(f"[green]wrote {out_dir / 'eval_v2.json'}")
    return EvalV2Result(per_split=per_split)


# ---------------- internals ----------------


def _classification_and_regression(
    model: FractalCNNv2,
    loader: DataLoader,
    device: torch.device,
    encoder: FamilyEncoder,
):
    n_classes = len(encoder.classes)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    n_total = 0
    n_correct = 0

    c_abs_err = []
    c_within_sigma = 0
    c_n = 0
    exp_abs_err = []
    exp_within_sigma = 0
    exp_n = 0

    sample_records: list[dict] = []
    sample_idx = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            family_idx = batch["family_idx"].to(device).long()
            c_target = batch["c"].to(device).float()
            exp_target = batch["exponent"].to(device).float()
            has_julia = batch["has_julia"].to(device).bool()
            has_multibrot = batch["has_multibrot"].to(device).bool()

            out = model(images)
            preds = out.family_logits.argmax(dim=1)
            for t, p in zip(family_idx.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1
            n_correct += int((preds == family_idx).sum().item())
            n_total += int(family_idx.numel())

            # Per-sample bookkeeping for reconstruction phase.
            mu_c = out.c_mean.cpu().numpy()
            sigma_c = torch.exp(0.5 * out.c_log_var.clamp(-7.0, 7.0)).cpu().numpy()
            mu_exp = out.exp_mean.cpu().numpy()
            sigma_exp = torch.exp(0.5 * out.exp_log_var.clamp(-7.0, 7.0)).cpu().numpy()
            mu_vp = out.viewport_mean.cpu().numpy()
            preds_np = preds.cpu().numpy()
            for i in range(family_idx.numel()):
                sample_records.append({
                    "ds_idx": sample_idx + i,
                    "true_family_idx": int(family_idx[i].item()),
                    "pred_family_idx": int(preds_np[i]),
                    "pred_c_re": float(mu_c[i, 0]),
                    "pred_c_im": float(mu_c[i, 1]),
                    "pred_c_sigma_re": float(sigma_c[i, 0]),
                    "pred_c_sigma_im": float(sigma_c[i, 1]),
                    "pred_exp": float(mu_exp[i]),
                    "pred_exp_sigma": float(sigma_exp[i]),
                    "pred_viewport": (
                        float(mu_vp[i, 0]), float(mu_vp[i, 1]),
                        float(mu_vp[i, 2]), float(mu_vp[i, 3]),
                    ),
                })
            sample_idx += family_idx.numel()

            if has_julia.any():
                jidx = has_julia.nonzero(as_tuple=False).squeeze(1)
                pred = out.c_mean.index_select(0, jidx)
                tgt = c_target.index_select(0, jidx)
                err = (pred - tgt).abs()
                c_abs_err.extend(err.cpu().numpy().reshape(-1).tolist())
                c_n += int(err.numel())
                sigma = torch.exp(0.5 * out.c_log_var.index_select(0, jidx).clamp(-7.0, 7.0))
                inside = ((pred - tgt).abs() <= sigma).float().sum()
                c_within_sigma += int(inside.item())

            if has_multibrot.any():
                midx = has_multibrot.nonzero(as_tuple=False).squeeze(1)
                pred = out.exp_mean.index_select(0, midx)
                tgt = exp_target.index_select(0, midx)
                err = (pred - tgt).abs()
                exp_abs_err.extend(err.cpu().numpy().tolist())
                exp_n += int(err.numel())
                sigma = torch.exp(0.5 * out.exp_log_var.index_select(0, midx).clamp(-7.0, 7.0))
                inside = ((pred - tgt).abs() <= sigma).float().sum()
                exp_within_sigma += int(inside.item())

    cls_metrics = {
        "accuracy": n_correct / n_total if n_total else 0.0,
        "confusion_matrix": confusion.tolist(),
        "classes": list(encoder.classes),
    }
    regr_metrics = {
        "julia_c_mae": float(np.mean(c_abs_err)) if c_abs_err else None,
        "julia_c_mse": float(np.mean(np.square(c_abs_err))) if c_abs_err else None,
        "julia_c_n": c_n // 2 if c_n else 0,
        "julia_c_calibration_within_sigma": (c_within_sigma / c_n) if c_n else None,
        "multibrot_exp_mae": float(np.mean(exp_abs_err)) if exp_abs_err else None,
        "multibrot_exp_n": exp_n,
        "multibrot_exp_calibration_within_sigma": (
            exp_within_sigma / exp_n if exp_n else None
        ),
    }
    return cls_metrics, regr_metrics, sample_records


def _reconstruction(
    *,
    ds: FractalDatasetV2,
    sample_records: list[dict],
    device: torch.device,
    n_iters: int,
    max_per_family: int,
    console: Console,
    split_name: str,
    refine_viewport_for_julia_multibrot: bool = False,
) -> dict:
    """Run reconstruction + hybrid refinement on a subset of every family.

    We cap per-family because each refine() call does ~20 differentiable
    renders. On a 200-row split × 4 families × 50 caps that bounds the
    refiner work to 4000 mini-renders per split — about a minute on CPU.
    ``max_per_family=50`` (default) gives stable median statistics.

    Mandelbrot and Burning Ship use the model's viewport-head prediction
    as the warm start; for them, viewport is the only continuous knob the
    refiner can exploit. Julia and Multibrot keep their family-internal
    parameter refinement; ``refine_viewport_for_julia_multibrot`` toggles
    whether the viewport gets a degree of freedom there too (default off
    so numbers stay comparable to Slice 3 baseline).
    """
    counters = {"julia": 0, "multibrot": 0, "mandelbrot": 0, "burning_ship": 0}
    results: dict[str, list[tuple[float, float]]] = {fam: [] for fam in counters}

    for rec in sample_records:
        sample = ds[rec["ds_idx"]]
        family = sample.family
        if family not in counters:
            continue
        if counters[family] >= max_per_family:
            continue

        # Warm-start viewport: prefer the model's prediction when present;
        # fall back to the dataset viewport (this is the original training
        # viewport, not a leak — it documents where the user wanted to look).
        warm_vp = rec.get("pred_viewport") or tuple(sample.viewport)
        warm_vp = _sanitize_viewport(warm_vp, fallback=tuple(sample.viewport))

        if family == "julia":
            req = RefineRequest(
                family="julia",
                target_image=sample.image,
                viewport=warm_vp,
                max_iter=sample.max_iter,
                init_c_re=rec["pred_c_re"],
                init_c_im=rec["pred_c_im"],
                refine_viewport=refine_viewport_for_julia_multibrot,
            )
        elif family == "multibrot":
            init_n = float(np.clip(rec["pred_exp"], 2.0, 10.0))
            req = RefineRequest(
                family="multibrot",
                target_image=sample.image,
                viewport=warm_vp,
                max_iter=sample.max_iter,
                init_exponent=init_n,
                refine_viewport=refine_viewport_for_julia_multibrot,
            )
        else:
            # mandelbrot / burning_ship — viewport-only refinement.
            req = RefineRequest(
                family=family,
                target_image=sample.image,
                viewport=warm_vp,
                max_iter=sample.max_iter,
            )

        res = refine(req, n_iters=n_iters, device=device)
        results[family].append((res.pre_mse, res.post_mse))
        counters[family] += 1

    return {fam: _summarise_pre_post(rows) for fam, rows in results.items()}


def _sanitize_viewport(
    vp: tuple[float, float, float, float],
    *,
    fallback: tuple[float, float, float, float],
    min_span: float = 1e-3,
) -> tuple[float, float, float, float]:
    """Guard against pathological NN viewport predictions.

    Early in training the viewport head can emit ``x_max < x_min`` or
    spans of effectively zero. The refiner's softplus parameterisation
    accepts arbitrary inputs but the resulting render is uninformative.
    We fall back to the dataset viewport in those cases — it's the
    "what the user asked to render" anchor, and is a strictly better
    warm start than a model prediction that hasn't converged yet.
    """
    x_min, x_max, y_min, y_max = vp
    if not all(map(_is_finite, vp)):
        return fallback
    if (x_max - x_min) < min_span or (y_max - y_min) < min_span:
        return fallback
    return (x_min, x_max, y_min, y_max)


def _is_finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def _summarise_pre_post(rows: list[tuple[float, float]]) -> dict:
    if not rows:
        return {"n": 0}
    pre = np.array([r[0] for r in rows])
    post = np.array([r[1] for r in rows])
    return {
        "n": len(rows),
        "pure_nn_mse_median": float(np.median(pre)),
        "pure_nn_mse_mean": float(np.mean(pre)),
        "pure_nn_mse_p90": float(np.quantile(pre, 0.9)),
        "refined_mse_median": float(np.median(post)),
        "refined_mse_mean": float(np.mean(post)),
        "refined_mse_p90": float(np.quantile(post, 0.9)),
        "improvement_median": float(np.median(pre - post)),
        "improvement_fraction": float(np.mean(post < pre)),
    }


def _print_split_summary(console: Console, name: str, m: dict) -> None:
    table = Table(title=f"split: {name} (n={m['n']})", show_header=True, expand=False)
    table.add_column("metric")
    table.add_column("value")
    table.add_row("family_accuracy", f"{m['classification']['accuracy']:.4f}")
    r = m["regression"]
    table.add_row("julia_c_mae", _fmt(r.get("julia_c_mae")))
    table.add_row("julia_c_calibration_within_sigma", _fmt(r.get("julia_c_calibration_within_sigma")))
    table.add_row("multibrot_exp_mae", _fmt(r.get("multibrot_exp_mae")))
    table.add_row("multibrot_exp_calibration_within_sigma", _fmt(r.get("multibrot_exp_calibration_within_sigma")))
    rec = m["reconstruction"]
    for fam in ("mandelbrot", "julia", "burning_ship", "multibrot"):
        block = rec.get(fam, {})
        if block.get("n"):
            table.add_row(
                f"{fam} recon pre→post p50",
                f"{block['pure_nn_mse_median']:.4f} → {block['refined_mse_median']:.4f}",
            )
            table.add_row(f"{fam} improved fraction", _fmt(block.get("improvement_fraction")))
    console.print(table)


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4f}"
