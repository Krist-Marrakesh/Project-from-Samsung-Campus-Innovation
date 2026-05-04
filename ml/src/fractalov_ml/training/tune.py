"""Optuna hyperparameter tuning for the FractalCNN baseline.

The search space targets the parameters most likely to misbehave on MPS for
this model: optimiser scale (`learning_rate`, `weight_decay`), the relative
weight of the regression head (`reg_weight` — too high blows gradients on
randomly-initialised c-head, too low starves the regressor), `batch_size`,
augmentation toggle, and model capacity (`base_channels`).

We use Optuna's TPESampler (the default) and MedianPruner, which terminates
trials whose intermediate val loss looks below median after a warmup. With
~20 trials and ~10 epochs per trial we hit a useful baseline in under ten
minutes on M-class MPS. The cls + reg loss weights are kept relatively close
together (β ∈ [0.1, 50]) — Optuna can pick whichever balance the dataset
prefers without us having to guess.

Outputs in ``out_dir``:

* ``best_params.json``  – the best-trial hyperparameters, suitable for
  ``train --from-best``
* ``study_summary.json`` – ranked list of all trials with their parameters,
  best val loss, and prune/complete state
* per-trial subdirectories with the usual ``metrics.json`` + ``best.pt``,
  so a successful trial doubles as a usable checkpoint
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

from .config import TrainingConfig
from .train import TrainingAborted, train


@dataclass
class TuneConfig:
    data_root: Path
    out_dir: Path
    n_trials: int = 20
    epochs_per_trial: int = 10
    study_name: str = "fractalov_baseline"
    seed: int = 42
    device_override: str | None = None


def _build_trial_cfg(
    trial: optuna.Trial,
    tcfg: TuneConfig,
    trial_dir: Path,
) -> TrainingConfig:
    """Materialise a TrainingConfig from the trial's suggested params."""
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    reg_weight = trial.suggest_float("reg_weight", 0.1, 20.0, log=True)
    base_channels = trial.suggest_categorical("base_channels", [16, 32, 48])
    enable_augmentation = trial.suggest_categorical("enable_augmentation", [False, True])

    return TrainingConfig(
        data_root=tcfg.data_root,
        out_dir=trial_dir,
        epochs=tcfg.epochs_per_trial,
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        cls_weight=1.0,
        reg_weight=reg_weight,
        base_channels=base_channels,
        enable_augmentation=enable_augmentation,
        # Generous patience inside short trials — let the scheduler matter; the
        # MedianPruner is the real early-stop.
        patience=tcfg.epochs_per_trial,
        device_override=tcfg.device_override,
        seed=tcfg.seed,
        log_every_epochs=tcfg.epochs_per_trial,  # silence inside trials
    )


def _free_device_memory(device: str) -> None:
    """Release accumulated MPS allocations between trials.

    Without this, multi-trial runs gradually bloat the wired memory pool: each
    trial's optimiser state hangs around until garbage-collected, and on MPS
    the wired pool isn't trimmed automatically. Empty-cache + a forced GC pass
    is cheap and keeps trials independent.
    """
    gc.collect()
    if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _objective(trial: optuna.Trial, tcfg: TuneConfig, console: Console) -> float:
    trial_dir = tcfg.out_dir / f"trial_{trial.number:03d}"
    cfg = _build_trial_cfg(trial, tcfg, trial_dir)

    def epoch_callback(epoch: int, _train_metrics: dict, val_metrics: dict) -> None:
        # Optuna reads `should_prune` after the report; we report val loss as
        # the intermediate value so MedianPruner can compare across trials.
        trial.report(val_metrics["loss"], step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    try:
        result = train(cfg, console=console, epoch_callback=epoch_callback)
    except optuna.TrialPruned:
        raise
    except TrainingAborted as ex:
        # Treat numerical instability as a pruned trial — the parameter combo
        # is unviable, but we don't want it to fail the whole study.
        trial.set_user_attr("aborted_reason", str(ex))
        raise optuna.TrialPruned()
    finally:
        _free_device_memory(cfg.device_override or "mps")

    trial.set_user_attr("epochs_run", result.epochs_run)
    trial.set_user_attr("best_checkpoint", str(result.best_checkpoint))
    return result.best_val_loss


def run_study(tcfg: TuneConfig, console: Optional[Console] = None) -> dict:
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
        f"[bold]optuna study[/bold] n_trials={tcfg.n_trials} "
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

    (tcfg.out_dir / "best_params.json").write_text(json.dumps(best.params, indent=2))
    (tcfg.out_dir / "study_summary.json").write_text(json.dumps(summary, indent=2))

    console.rule("optuna best")
    console.print(f"[bold green]best val loss = {best.value:.4f}  (trial {best.number})")
    for k, v in best.params.items():
        console.print(f"  {k} = {v}")
    return summary


def load_best_params(study_dir: Path) -> dict:
    """Convenience for ``train --from-best``: loads the JSON dict."""
    return json.loads((study_dir / "best_params.json").read_text())
