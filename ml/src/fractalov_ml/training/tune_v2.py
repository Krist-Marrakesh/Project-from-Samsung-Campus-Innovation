"""Optuna hyperparameter tuning for FractalCNNv2 (multi-task / MVE).

Search space differences vs v1 ``tune.py``:

* The single ``reg_weight`` knob is split into two: ``c_weight`` (Julia
  Gaussian-NLL term) and ``exp_weight`` (Multibrot Gaussian-NLL term). The
  two heads have different gradient scales — Julia c is bounded inside a
  small disk, Multibrot exponent ranges over [2, 8] — so a single shared
  weight ends up either starving Julia or oversaturating Multibrot.
* ``cls_weight`` is held at 1.0. Tuning all three weights simultaneously
  is over-parameterised; we pin the classifier as the reference and let
  Optuna scale the regression heads against it.
* Augmentation toggle and ``base_channels`` carry over.

Objective:

* The objective value Optuna minimises is **val cls_loss**, the same metric
  ``train_v2`` uses for best.pt selection. Aligning tune objective ↔
  checkpoint criterion keeps the best trial and the best checkpoint
  pointing at the same model. Reporting the joint loss instead would
  reward trials that drove ``c_nll`` to zero by predicting unbounded σ —
  exactly the failure mode the MVE clamp guards against.

Outputs in ``out_dir``:

* ``best_params_v2.json`` — pass to ``train-v2 --from-best``
* ``study_summary_v2.json`` — ranked list of trials with params + state
* per-trial subdirs with the standard ``metrics_v2.json`` + ``best.pt``,
  so a successful trial doubles as a usable checkpoint

This module deliberately does not share code with v1 :mod:`tune` even
though the structure is similar. The v1 module pins on the v1
:class:`TrainingConfig`; merging would require generic plumbing that
makes both harder to read for no real win — they diverge only in the
search space.
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from rich.console import Console

from .config_v2 import TrainingConfigV2
from .train_v2 import TrainingAborted, train_v2


@dataclass
class TuneConfigV2:
    data_root: Path
    out_dir: Path
    n_trials: int = 20
    epochs_per_trial: int = 10
    study_name: str = "fractalov_v2"
    seed: int = 42
    device_override: str | None = None


def _build_trial_cfg(
    trial: optuna.Trial,
    tcfg: TuneConfigV2,
    trial_dir: Path,
) -> TrainingConfigV2:
    """Materialise a TrainingConfigV2 from the trial's suggested params."""
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    c_weight = trial.suggest_float("c_weight", 0.05, 5.0, log=True)
    exp_weight = trial.suggest_float("exp_weight", 0.05, 5.0, log=True)
    vp_weight = trial.suggest_float("vp_weight", 0.05, 2.0, log=True)
    # Linear (not log) range with a 0.0 anchor — Optuna's TPE handles
    # mixed log/linear search spaces fine, and a flat 0.0 sample at the
    # bottom is the "no SupCon" baseline that we genuinely want the
    # study to consider, not just a small positive value approximating it.
    contrastive_weight = trial.suggest_float("contrastive_weight", 0.0, 2.0)
    base_channels = trial.suggest_categorical("base_channels", [16, 32, 48])
    enable_augmentation = trial.suggest_categorical("enable_augmentation", [False, True])

    return TrainingConfigV2(
        data_root=tcfg.data_root,
        out_dir=trial_dir,
        epochs=tcfg.epochs_per_trial,
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        cls_weight=1.0,
        c_weight=c_weight,
        exp_weight=exp_weight,
        vp_weight=vp_weight,
        contrastive_weight=contrastive_weight,
        base_channels=base_channels,
        enable_augmentation=enable_augmentation,
        # Generous patience inside short trials — let the scheduler matter; the
        # MedianPruner is the real early-stop.
        patience=tcfg.epochs_per_trial,
        device_override=tcfg.device_override,
        seed=tcfg.seed,
        log_every_epochs=tcfg.epochs_per_trial,  # silence inside trials
        # Skip the per-split eval rerun inside trials — costly and the
        # objective is val cls_loss anyway. The full per-split breakdown is
        # produced once at the end via the standalone train-v2 command on
        # the winning hyperparameters.
        eval_splits=(),
    )


def _free_device_memory(device: str) -> None:
    """Release accumulated MPS/CUDA allocations between trials.

    Same rationale as v1 :mod:`tune`: without this, multi-trial runs
    gradually bloat the wired memory pool. Empty-cache + a forced GC pass
    is cheap and keeps trials independent.
    """
    gc.collect()
    if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _objective(trial: optuna.Trial, tcfg: TuneConfigV2, console: Console) -> float:
    trial_dir = tcfg.out_dir / f"trial_{trial.number:03d}"
    cfg = _build_trial_cfg(trial, tcfg, trial_dir)

    def epoch_callback(epoch: int, _train_metrics: dict, val_metrics: dict) -> None:
        # We report val cls_loss (not joint loss) so MedianPruner ranks trials
        # the same way best.pt selection does. See module docstring.
        trial.report(val_metrics["cls"], step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    try:
        result = train_v2(cfg, console=console, epoch_callback=epoch_callback)
    except optuna.TrialPruned:
        raise
    except TrainingAborted as ex:
        # Numerical instability → mark the trial as pruned. The parameter
        # combo is unviable; we don't want it to fail the whole study.
        trial.set_user_attr("aborted_reason", str(ex))
        raise optuna.TrialPruned()
    finally:
        _free_device_memory(cfg.device_override or "mps")

    trial.set_user_attr("epochs_run", result.epochs_run)
    trial.set_user_attr("best_checkpoint", str(result.best_checkpoint))
    return result.best_val_score


def run_study_v2(tcfg: TuneConfigV2, console: Optional[Console] = None) -> dict:
    console = console or Console()
    tcfg.out_dir.mkdir(parents=True, exist_ok=True)

    sampler = TPESampler(seed=tcfg.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=tcfg.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    console.print(
        f"[bold]optuna study[/bold] (v2) n_trials={tcfg.n_trials} "
        f"epochs_per_trial={tcfg.epochs_per_trial} "
        f"data_root={tcfg.data_root}"
    )

    def _opt_callback(trial_obj: optuna.Trial) -> float:
        return _objective(trial_obj, tcfg, console)

    study.optimize(_opt_callback, n_trials=tcfg.n_trials, show_progress_bar=False)

    best = study.best_trial
    summary = {
        "study_name": tcfg.study_name,
        "n_trials": len(study.trials),
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": dict(best.params),
        "trials": [
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": dict(t.params),
                "user_attrs": dict(t.user_attrs),
            }
            for t in study.trials
        ],
        "tune_config": {**asdict(tcfg), "data_root": str(tcfg.data_root), "out_dir": str(tcfg.out_dir)},
    }

    (tcfg.out_dir / "best_params_v2.json").write_text(json.dumps(best.params, indent=2))
    (tcfg.out_dir / "study_summary_v2.json").write_text(json.dumps(summary, indent=2))

    console.rule("optuna best (v2)")
    console.print(f"[bold green]best val cls_loss = {best.value:.4f}  (trial {best.number})")
    for k, v in best.params.items():
        console.print(f"  {k} = {v}")
    return summary


def load_best_params_v2(study_dir: Path) -> dict:
    """Load the JSON best-params from a v2 study, for ``train-v2 --from-best``."""
    return json.loads((study_dir / "best_params_v2.json").read_text())
