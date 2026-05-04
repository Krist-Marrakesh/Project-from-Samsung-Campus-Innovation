"""Offline tests for the model + loss masking semantics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fractalov_ml.models import FractalCNN
from fractalov_ml.recipes import FAMILIES


def test_forward_shapes_for_varied_input_sizes() -> None:
    model = FractalCNN()
    model.eval()
    for hw in (96, 128, 192):
        x = torch.randn(2, 3, hw, hw)
        out = model(x)
        assert out.family_logits.shape == (2, len(FAMILIES))
        assert out.c_pred.shape == (2, 2)
        # Tanh head scaled by 1.5 ⇒ output strictly in [-1.5, 1.5].
        assert out.c_pred.min().item() >= -1.5 - 1e-6
        assert out.c_pred.max().item() <= 1.5 + 1e-6


def test_param_count_in_target_band() -> None:
    """Sanity guard: any architecture change that explodes capacity should at
    least be intentional. 1.2M ± 30% leaves a comfortable working range."""
    model = FractalCNN()
    n = model.num_parameters
    assert 700_000 < n < 2_000_000, f"unexpected param count: {n:,}"


def _index_select_reg(pred: torch.Tensor, target: torch.Tensor, has_c: torch.Tensor) -> torch.Tensor:
    """Mirrors the index-select logic used inside _step()."""
    if has_c.any():
        idx = has_c.nonzero(as_tuple=False).squeeze(1)
        return (pred[idx] - target[idx]).pow(2).mean(dim=1).mean()
    return (pred.sum() * 0.0).mean()


def test_index_select_regression_zero_when_no_julia() -> None:
    """When the batch has no julia rows the regression contribution must be
    zero (and finite). The classification head still learns from the batch."""
    pred = torch.randn(4, 2, requires_grad=True)
    target = torch.zeros(4, 2)
    has_c = torch.zeros(4, dtype=torch.bool)
    reg = _index_select_reg(pred, target, has_c)
    assert torch.isfinite(reg)
    assert reg.item() == 0.0


def test_index_select_skips_non_julia_garbage() -> None:
    """Non-julia rows can carry NaN/garbage in the target after where-zeroing
    or even in pred (if the c-head had a bad gradient step). Index-selecting
    julia rows up front means those rows never touch the loss tensor."""
    pred = torch.tensor([[0.5, -0.3], [float("nan"), float("nan")], [-0.7, 0.2]])
    target = torch.tensor([[0.5, -0.3], [0.0, 0.0], [-0.7, 0.2]])
    has_c = torch.tensor([True, False, True])
    reg = _index_select_reg(pred, target, has_c)
    # Both julia rows are perfect → exactly zero, despite the NaN in row 1.
    assert torch.isfinite(reg)
    assert reg.item() == 0.0


def test_index_select_uses_only_julia_rows() -> None:
    pred = torch.tensor([[0.0, 0.0], [9.9, 9.9], [1.0, 1.0]])
    target = torch.tensor([[0.5, -0.3], [0.0, 0.0], [0.0, 0.0]])
    has_c = torch.tensor([True, False, True])
    reg = _index_select_reg(pred, target, has_c)
    # Row 0 MSE = mean(0.25, 0.09) = 0.17; row 2 MSE = mean(1, 1) = 1.0.
    # Their mean = (0.17 + 1.0) / 2 = 0.585.
    assert torch.allclose(reg, torch.tensor(0.585), atol=1e-5)
