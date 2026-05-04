"""Search-space sanity for the Optuna tuning module.

We mock the trial object to confirm that suggested hyperparameters land
inside the documented bounds without actually running Optuna's solver. A
hidden bug here would silently train good models on the wrong distribution
and we'd never notice.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fractalov_ml.training.tune import TuneConfig, _build_trial_cfg


def _fake_trial(values: dict) -> MagicMock:
    trial = MagicMock()

    def suggest_float(name: str, low: float, high: float, log: bool = False) -> float:
        # Capture and validate bounds; return a deterministic value for shape checks.
        recorded.append(("float", name, low, high, log))
        return values.get(name, low)

    def suggest_categorical(name: str, choices):
        recorded.append(("categorical", name, tuple(choices), None, False))
        return values.get(name, choices[0])

    recorded: list[tuple] = []
    trial.suggest_float.side_effect = suggest_float
    trial.suggest_categorical.side_effect = suggest_categorical
    trial._recorded = recorded
    return trial


def test_search_space_bounds_are_documented(tmp_path: Path) -> None:
    tcfg = TuneConfig(data_root=tmp_path, out_dir=tmp_path, epochs_per_trial=3)
    trial = _fake_trial({})
    cfg = _build_trial_cfg(trial, tcfg, tmp_path / "trial_0")

    space = {row[1]: row for row in trial._recorded}
    # learning rate
    _, _, lo, hi, log = space["learning_rate"]
    assert (lo, hi, log) == (1e-5, 5e-3, True)
    # weight decay
    _, _, lo, hi, log = space["weight_decay"]
    assert (lo, hi, log) == (1e-6, 1e-2, True)
    # reg_weight
    _, _, lo, hi, log = space["reg_weight"]
    assert (lo, hi, log) == (0.1, 20.0, True)
    # categoricals
    assert space["batch_size"][2] == (32, 64, 128)
    assert space["base_channels"][2] == (16, 32, 48)
    assert space["enable_augmentation"][2] == (False, True)

    # Smoke: cfg comes back populated.
    assert cfg.epochs == 3
    assert cfg.cls_weight == 1.0  # fixed; only relative β matters


def test_built_config_inherits_data_root(tmp_path: Path) -> None:
    tcfg = TuneConfig(data_root=tmp_path / "data", out_dir=tmp_path, epochs_per_trial=5)
    trial = _fake_trial({"batch_size": 64, "base_channels": 32, "enable_augmentation": True})
    cfg = _build_trial_cfg(trial, tcfg, tmp_path / "trial_0")
    assert cfg.data_root == tmp_path / "data"
    assert cfg.out_dir == tmp_path / "trial_0"
    assert cfg.batch_size == 64
    assert cfg.base_channels == 32
    assert cfg.enable_augmentation is True
