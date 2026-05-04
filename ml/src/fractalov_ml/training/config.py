"""Training configuration + device selection.

The device picker prefers Apple Metal (MPS) when available, falls back to CUDA
if a discrete GPU is present, and finally CPU. The ``override`` argument lets
callers force a specific backend for benchmarking — useful when the goal is a
fair "MPS vs CPU on the same data" comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class TrainingConfig:
    # Where the dataset lives (output of `fractalov-ml generate`).
    data_root: Path
    # Where to write checkpoints + metrics.
    out_dir: Path

    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Loss weights. c-regression operates in [-1.5, 1.5] so its raw MSE numbers
    # are similar in magnitude to cross-entropy at random init; pushing the
    # weight on regression slightly higher accelerates convergence on the
    # geometrically-richer signal.
    cls_weight: float = 1.0
    reg_weight: float = 5.0

    # DataLoader workers. macOS + MPS does not benefit from >0 workers because
    # the inference path already lives off the main thread; >0 also breaks
    # forking semantics with PIL on some Python builds.
    num_workers: int = 0

    # Early stopping. Patience counts epochs without val loss improvement.
    patience: int = 8
    min_delta: float = 1e-4

    # Optional explicit device override ("mps", "cuda", "cpu", or None=auto).
    device_override: str | None = None

    # Reproducibility.
    seed: int = 42

    # Logging cadence (epochs between Rich-table prints; per-batch progress is
    # always shown).
    log_every_epochs: int = 1

    # Augmentation toggle. Conservative augmentations are baked into the train
    # loop; flip this off for ablation runs.
    enable_augmentation: bool = True

    # Model capacity: FractalCNN's first conv channels (the rest of the body
    # scales as 2x/4x/8x). Tuned by Optuna; defaults match the hand-picked
    # 32-channel baseline.
    base_channels: int = 32

    # Tags written into metrics.json so a run is self-describing.
    tags: dict = field(default_factory=dict)


def pick_device(override: str | None = None) -> torch.device:
    if override is not None:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
