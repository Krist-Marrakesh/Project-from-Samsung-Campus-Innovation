"""Slice 4b coverage: Burning Ship renderer + viewport head + per-family
reconstruction refinement."""

from __future__ import annotations

import pytest
import torch

from fractalov_ml import torch_renderer as tr
from fractalov_ml.models.cnn_v2 import FractalCNNv2
from fractalov_ml.training.refine import RefineRequest, refine


# ---------------- Burning Ship renderer ----------------


def test_burning_ship_origin_in_set() -> None:
    """c=0 produces a fixed point at z=0 → in-set sentinel."""
    vp = tr.ViewportT(-0.05, 0.05, -0.05, 0.05)
    escape = tr.render_burning_ship(vp, width=3, height=3, max_iter=80, smoothing=False)
    assert escape[1, 1].item() == pytest.approx(-1.0)


def test_burning_ship_far_point_escapes() -> None:
    vp = tr.ViewportT(5.0, 5.001, 5.0, 5.001)
    escape = tr.render_burning_ship(vp, 1, 1, max_iter=50, smoothing=False)
    assert 0.0 <= escape[0, 0].item() < 5.0


def test_burning_ship_classic_window_has_mixed_pixels() -> None:
    vp = tr.ViewportT(-2.0, 1.5, -2.0, 1.0)
    escape = tr.render_burning_ship(vp, 24, 24, max_iter=80, smoothing=False)
    assert (escape < 0.0).any().item(), "expected some in-set pixels"
    assert (escape >= 0.0).any().item(), "expected some escape pixels"


def test_burning_ship_subgradient_flows_to_viewport() -> None:
    """Burning Ship's |·| step provides a subgradient via torch.abs.
    Gradients must flow back to viewport tensors despite the
    non-holomorphic kernel."""
    x_min = torch.tensor(-2.0, requires_grad=True)
    x_max = torch.tensor(1.5, requires_grad=True)
    y_min = torch.tensor(-2.0, requires_grad=True)
    y_max = torch.tensor(1.0, requires_grad=True)
    escape = tr.render_burning_ship(
        (x_min, x_max, y_min, y_max), 8, 8, max_iter=20, smoothing=True,
    )
    image = tr.escape_to_image(escape, max_iter=20, rgb=False).squeeze(0)
    image.sum().backward()
    for t, name in [(x_min, "x_min"), (x_max, "x_max"), (y_min, "y_min"), (y_max, "y_max")]:
        assert t.grad is not None
        assert torch.isfinite(t.grad), f"non-finite grad on {name}: {t.grad}"


# ---------------- Differentiable Mandelbrot via tensor viewport ----------------


def test_mandelbrot_viewport_gradients() -> None:
    """The new ``make_grid_from_tensors`` path must propagate gradients
    back to viewport bounds for Mandelbrot too."""
    x_min = torch.tensor(-2.0, requires_grad=True)
    x_max = torch.tensor(1.0, requires_grad=True)
    y_min = torch.tensor(-1.2, requires_grad=True)
    y_max = torch.tensor(1.2, requires_grad=True)
    escape = tr.render_mandelbrot(
        (x_min, x_max, y_min, y_max), 8, 8, max_iter=30, smoothing=True,
    )
    image = tr.escape_to_image(escape, max_iter=30, rgb=False).squeeze(0)
    image.sum().backward()
    for t in (x_min, x_max, y_min, y_max):
        assert torch.isfinite(t.grad)


def test_grid_from_tensors_matches_static_grid() -> None:
    """When the same numeric viewport is passed both as floats and as
    detached tensors, the resulting grid must be identical."""
    static = tr.make_grid(tr.ViewportT(-1.0, 1.0, -0.5, 0.5), width=4, height=3)
    tensor_vp = tr.make_grid_from_tensors(
        torch.tensor(-1.0), torch.tensor(1.0),
        torch.tensor(-0.5), torch.tensor(0.5),
        width=4, height=3,
    )
    assert torch.allclose(static[0], tensor_vp[0])
    assert torch.allclose(static[1], tensor_vp[1])


# ---------------- viewport head ----------------


def test_viewport_head_shapes() -> None:
    model = FractalCNNv2(base_channels=8)
    out = model(torch.zeros(4, 3, 32, 32))
    assert out.viewport_mean.shape == (4, 4)
    assert out.viewport_log_var.shape == (4, 4)


# ---------------- refiner: viewport-only families ----------------


def test_refiner_mandelbrot_reduces_mse_with_perturbed_viewport() -> None:
    """Render the Mandelbrot at a known viewport, perturb, then refine.
    The refined post-MSE should be <= the pre-MSE (synthetic target,
    convex enough basin)."""
    true_vp = (-2.0, 1.0, -1.2, 1.2)
    vp_t = tr.ViewportT(*true_vp)
    target_escape = tr.render_mandelbrot(vp_t, 32, 32, max_iter=80, smoothing=True)
    target_image = tr.escape_to_image(target_escape, max_iter=80, rgb=True).detach()

    # Slightly off viewport — keep the perturbation small to stay in basin.
    perturbed = (true_vp[0] + 0.05, true_vp[1] - 0.05,
                 true_vp[2] + 0.05, true_vp[3] - 0.05)
    req = RefineRequest(
        family="mandelbrot",
        target_image=target_image,
        viewport=perturbed,
        max_iter=80,
    )
    res = refine(req, n_iters=10, device="cpu")
    assert res.refined
    assert res.post_mse <= res.pre_mse + 1e-6
    assert res.final_viewport is not None


def test_refiner_burning_ship_reduces_mse_with_perturbed_viewport() -> None:
    true_vp = (-2.0, 1.5, -2.0, 1.0)
    vp_t = tr.ViewportT(*true_vp)
    target_escape = tr.render_burning_ship(vp_t, 32, 32, max_iter=60, smoothing=True)
    target_image = tr.escape_to_image(target_escape, max_iter=60, rgb=True).detach()

    perturbed = (true_vp[0] + 0.03, true_vp[1] - 0.03,
                 true_vp[2] + 0.03, true_vp[3] - 0.03)
    req = RefineRequest(
        family="burning_ship",
        target_image=target_image,
        viewport=perturbed,
        max_iter=60,
    )
    res = refine(req, n_iters=10, device="cpu")
    assert res.refined
    assert res.post_mse <= res.pre_mse + 1e-6
    assert res.final_viewport is not None


def test_refiner_julia_with_viewport_refinement_flag() -> None:
    """Julia refinement with ``refine_viewport=True`` should still reduce
    MSE (or match) on a synthetic target where only c is perturbed."""
    vp_t = tr.ViewportT(-1.5, 1.5, -1.5, 1.5)
    true_c_re, true_c_im = -0.7, 0.27015
    target_escape = tr.render_julia(vp_t, 32, 32,
                                    c_re=true_c_re, c_im=true_c_im,
                                    max_iter=80, smoothing=True)
    target_image = tr.escape_to_image(target_escape, max_iter=80, rgb=True).detach()
    req = RefineRequest(
        family="julia",
        target_image=target_image,
        viewport=(-1.5, 1.5, -1.5, 1.5),
        max_iter=80,
        init_c_re=true_c_re + 0.04,
        init_c_im=true_c_im - 0.04,
        refine_viewport=True,
    )
    res = refine(req, n_iters=15, device="cpu")
    assert res.refined
    assert res.post_mse <= res.pre_mse + 1e-6
    assert res.final_viewport is not None


def test_refiner_unknown_family_returns_unrefined() -> None:
    """Genuinely unknown family must surface as ``refined=False`` rather
    than crashing — the old behaviour for burning_ship before this slice."""
    vp_t = tr.ViewportT(-1.0, 1.0, -1.0, 1.0)
    image = torch.zeros(3, 8, 8)
    req = RefineRequest(
        family="not-a-real-family",
        target_image=image,
        viewport=(-1.0, 1.0, -1.0, 1.0),
        max_iter=10,
    )
    res = refine(req, n_iters=5, device="cpu")
    assert not res.refined
    assert res.pre_mse == res.post_mse


# ---------------- viewport softplus parameterisation ----------------


def test_refined_viewport_preserves_monotonicity() -> None:
    """After refinement, ``x_max > x_min`` and ``y_max > y_min`` must hold —
    the softplus parameterisation guarantees this regardless of optimiser
    trajectory."""
    vp_t = tr.ViewportT(-2.0, 1.0, -1.2, 1.2)
    target_escape = tr.render_mandelbrot(vp_t, 24, 24, max_iter=60, smoothing=True)
    target_image = tr.escape_to_image(target_escape, max_iter=60, rgb=True).detach()
    req = RefineRequest(
        family="mandelbrot",
        target_image=target_image,
        viewport=(-2.5, 0.5, -1.5, 1.5),  # noticeable perturbation
        max_iter=60,
    )
    res = refine(req, n_iters=15, device="cpu")
    assert res.final_viewport is not None
    x_min, x_max, y_min, y_max = res.final_viewport
    assert x_max > x_min
    assert y_max > y_min
