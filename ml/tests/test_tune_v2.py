"""Lightweight tests for tune_v2.

Strategy: monkeypatch ``train_v2`` so the Optuna study runs without actually
training. Each fake trial returns a deterministic loss derived from the
suggested params, which is enough to verify (a) the search space is wired
up correctly, (b) the best_params_v2.json round-trips through
load_best_params_v2, and (c) the val cls_loss objective ranking lines up
with train-v2's best.pt criterion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fractalov_ml.training.train_v2 import TrainResultV2
from fractalov_ml.training.tune_v2 import (
    TuneConfigV2,
    load_best_params_v2,
    run_study_v2,
)


def _fake_train_factory(captured: list):
    """Fake train_v2 that records the configs it sees and returns a
    deterministic-but-non-trivial best_val_score so Optuna has something to
    rank trials by."""
    def fake_train_v2(cfg, console=None, epoch_callback=None):
        captured.append(cfg)
        # Score derived from the params: smaller lr → larger loss; small
        # c_weight → larger loss. Ensures TPESampler has a real signal to
        # converge against.
        score = 1.0 / (cfg.learning_rate * 1e3) + 1.0 / max(cfg.c_weight, 1e-6)
        # Drive an intermediate report so the pruner path gets exercised.
        if epoch_callback is not None:
            epoch_callback(1, {"loss": score, "cls": score, "c_nll": 0.0, "exp_nll": 0.0},
                           {"loss": score, "cls": score, "c_nll": 0.0, "exp_nll": 0.0})
        return TrainResultV2(
            best_val_score=score,
            best_checkpoint=cfg.out_dir / "best.pt",
            metrics=[],
            epochs_run=1,
            final_per_split={},
        )
    return fake_train_v2


def test_run_study_v2_minimises_val_cls_loss(tmp_path: Path, monkeypatch) -> None:
    captured: list = []
    monkeypatch.setattr(
        "fractalov_ml.training.tune_v2.train_v2",
        _fake_train_factory(captured),
    )

    tcfg = TuneConfigV2(
        data_root=tmp_path / "ds",   # never read in this test
        out_dir=tmp_path / "study",
        n_trials=4,
        epochs_per_trial=1,
        seed=0,
    )
    summary = run_study_v2(tcfg)
    assert summary["n_trials"] == 4
    assert "best_params" in summary

    # All trials produced configs.
    assert len(captured) == 4
    # Best trial should have the lowest val_score per the objective shape.
    values = [(t["number"], t["value"]) for t in summary["trials"]]
    finite = [(n, v) for n, v in values if v is not None]
    assert finite, "expected at least one finite trial value"
    best_number = min(finite, key=lambda nv: nv[1])[0]
    assert summary["best_trial"] == best_number


def test_best_params_v2_round_trips(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "fractalov_ml.training.tune_v2.train_v2",
        _fake_train_factory([]),
    )
    tcfg = TuneConfigV2(
        data_root=tmp_path / "ds",
        out_dir=tmp_path / "study",
        n_trials=3,
        epochs_per_trial=1,
        seed=0,
    )
    summary = run_study_v2(tcfg)

    loaded = load_best_params_v2(tcfg.out_dir)
    assert loaded == summary["best_params"]
    # Search-space coverage: every tunable knob must appear in the dict.
    expected_keys = {
        "learning_rate",
        "weight_decay",
        "batch_size",
        "c_weight",
        "exp_weight",
        "base_channels",
        "enable_augmentation",
    }
    assert expected_keys <= set(loaded.keys()), \
        f"missing tunable knobs: {expected_keys - set(loaded.keys())}"


def test_search_space_includes_v2_specific_weights(tmp_path: Path, monkeypatch) -> None:
    """v2 splits the v1 single ``reg_weight`` into ``c_weight`` and
    ``exp_weight`` — the test guards against a refactor that silently
    collapses them back to one knob."""
    captured: list = []
    monkeypatch.setattr(
        "fractalov_ml.training.tune_v2.train_v2",
        _fake_train_factory(captured),
    )
    tcfg = TuneConfigV2(
        data_root=tmp_path / "ds",
        out_dir=tmp_path / "study",
        n_trials=2,
        epochs_per_trial=1,
        seed=0,
    )
    run_study_v2(tcfg)

    # Each captured config carries the suggested c_weight / exp_weight; they
    # are independently varied across trials.
    c_weights = {cfg.c_weight for cfg in captured}
    exp_weights = {cfg.exp_weight for cfg in captured}
    # With n_trials=2 they SHOULD differ; if Optuna ever pinned both to the
    # default we'd see a singleton set here.
    assert len(c_weights) == 2
    assert len(exp_weights) == 2


def test_load_best_params_v2_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_best_params_v2(tmp_path)


def test_summary_json_has_per_trial_breakdown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "fractalov_ml.training.tune_v2.train_v2",
        _fake_train_factory([]),
    )
    tcfg = TuneConfigV2(
        data_root=tmp_path / "ds",
        out_dir=tmp_path / "study",
        n_trials=3,
        epochs_per_trial=1,
        seed=0,
    )
    run_study_v2(tcfg)
    summary = json.loads((tcfg.out_dir / "study_summary_v2.json").read_text())
    assert summary["n_trials"] == 3
    assert all("state" in t and "params" in t for t in summary["trials"])
