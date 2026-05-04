"""TrainingConfigV2 for the multi-task / uncertainty-aware pipeline.

Distinct from :class:`TrainingConfig` because the loss weight knobs are
different (three terms, not two) and the head set is different. Defaults
mirror the v1 baseline so a dataset that worked there should work here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfigV2:
    data_root: Path
    out_dir: Path

    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    cls_weight: float = 1.0
    c_weight: float = 0.5
    exp_weight: float = 0.5
    # Viewport MVE term. Always-on (every recipe has a viewport) so unlike
    # c/exp it is not masked. Default is moderate because raw viewport NLL
    # has wider scale than per-family knobs and tends to dominate at parity.
    vp_weight: float = 0.25
    # Supervised-contrastive weight. 0 (default) leaves the embedding head
    # unsupervised — useful when you want a baseline without representation
    # learning. >0 turns on the SupCon term using family labels.
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.1

    num_workers: int = 0
    patience: int = 8
    min_delta: float = 1e-4

    device_override: str | None = None
    seed: int = 42

    log_every_epochs: int = 1
    enable_augmentation: bool = True
    base_channels: int = 32

    # Splits beyond train/val are reported on at end-of-training. Empty list
    # means "skip per-split eval"; default covers the Slice 2 research config.
    eval_splits: tuple[str, ...] = ("test", "near_ood", "hard_overlap")

    tags: dict = field(default_factory=dict)
