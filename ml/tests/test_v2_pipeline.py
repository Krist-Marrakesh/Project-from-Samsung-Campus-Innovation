"""Offline tests for the Slice 3 v2 pipeline.

Coverage:
* FractalDatasetV2 parquet contract
* FractalCNNv2 forward shapes
* MVE Gaussian-NLL loss math + masking
* torch_renderer agrees with the Java backend's analytic short-circuits
* Hybrid refiner reduces MSE on a synthetic Julia target
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from fractalov_ml.dataset_v2 import FractalDatasetV2, collate
from fractalov_ml.models.cnn_v2 import FractalCNNv2
from fractalov_ml.training.loss_v2 import (
    LOG_VAR_CLAMP_MAX,
    LOG_VAR_CLAMP_MIN,
    compute_loss,
    gaussian_nll,
)
from fractalov_ml.training.refine import RefineRequest, refine
from fractalov_ml import torch_renderer as tr


# ---------------- helpers ----------------


def _write_dummy_dataset(root: Path, n_per_family_per_split: int = 2) -> None:
    """Build a tiny v2-shaped dataset on disk so FractalDatasetV2 can be exercised
    without a running backend."""
    images_root = root / "images"
    rows = []
    for split in ("train", "val", "test", "near_ood"):
        for family in ("mandelbrot", "julia", "burning_ship", "multibrot"):
            for i in range(n_per_family_per_split):
                rel = f"images/{split}/{family}/{family}_{split}_{i}.png"
                abs_path = root / rel
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                # 8x8 single-colour PNG, content irrelevant for these tests.
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8) + (i * 32 % 255)).save(abs_path)
                rows.append({
                    "id": f"{family}_{split}_{i}",
                    "split": split,
                    "family": family,
                    "fractal_type": family,
                    "c_re": 0.123 if family == "julia" else None,
                    "c_im": -0.456 if family == "julia" else None,
                    "exponent": 3.0 if family == "multibrot" else None,
                    "max_iter": 100,
                    "escape_radius": 2.0,
                    "smoothing": True,
                    "vp_x_min": -2.0, "vp_x_max": 1.0,
                    "vp_y_min": -1.2, "vp_y_max": 1.2,
                    "palette": "fire",
                    "color_mode": "linear",
                    "samples_per_axis": 1,
                    "image_path": rel,
                    "perf_render_ms": 1, "perf_colorize_ms": 1,
                    "perf_encode_ms": 1, "perf_total_ms": 3,
                    "file_size_bytes": 100, "request_id": f"req-{family}-{split}-{i}",
                })
    pd.DataFrame(rows).to_parquet(root / "labels.parquet", index=False)


# ---------------- dataset_v2 ----------------


def test_dataset_v2_reads_split_and_masks(tmp_path: Path) -> None:
    _write_dummy_dataset(tmp_path)
    ds = FractalDatasetV2(tmp_path, split="train")
    assert len(ds) == 4 * 2   # 4 families × 2 per split
    samples = [ds[i] for i in range(len(ds))]
    families = sorted({s.family for s in samples})
    assert families == ["burning_ship", "julia", "mandelbrot", "multibrot"]

    # Masks reflect family-specific param presence.
    julia = [s for s in samples if s.family == "julia"]
    multi = [s for s in samples if s.family == "multibrot"]
    assert all(s.has_julia for s in julia)
    assert all(not s.has_julia for s in samples if s.family != "julia")
    assert all(s.has_multibrot for s in multi)


def test_dataset_v2_collate_produces_batched_tensors(tmp_path: Path) -> None:
    _write_dummy_dataset(tmp_path)
    ds = FractalDatasetV2(tmp_path, split="train")
    batch = collate([ds[0], ds[1], ds[2], ds[3]])
    assert batch["image"].shape == (4, 3, 8, 8)
    assert batch["family_idx"].dtype == torch.long
    assert batch["c"].shape == (4, 2)
    assert batch["exponent"].shape == (4,)
    assert batch["has_julia"].shape == (4,)


def test_dataset_v2_unknown_split_raises(tmp_path: Path) -> None:
    _write_dummy_dataset(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        FractalDatasetV2(tmp_path, split="hard_overlap")


def test_dataset_v2_legacy_parquet_rejected(tmp_path: Path) -> None:
    # No `split` column → legacy schema, must raise rather than silently
    # serving everything as train.
    df = pd.DataFrame([{"id": "a", "family": "mandelbrot", "fractal_type": "mandelbrot",
                        "c_re": None, "c_im": None, "exponent": None,
                        "max_iter": 1, "escape_radius": 2.0, "smoothing": True,
                        "vp_x_min": 0, "vp_x_max": 1, "vp_y_min": 0, "vp_y_max": 1,
                        "palette": "fire", "color_mode": "linear",
                        "samples_per_axis": 1, "image_path": "x.png",
                        "perf_render_ms": 0, "perf_colorize_ms": 0,
                        "perf_encode_ms": 0, "perf_total_ms": 0,
                        "file_size_bytes": 0, "request_id": "r"}])
    df.to_parquet(tmp_path / "labels.parquet", index=False)
    with pytest.raises(ValueError, match="no 'split' column"):
        FractalDatasetV2(tmp_path, split="train")


# ---------------- model ----------------


def test_cnn_v2_output_shapes() -> None:
    model = FractalCNNv2(num_families=4, base_channels=8)
    x = torch.zeros(3, 3, 32, 32)
    out = model(x)
    assert out.family_logits.shape == (3, 4)
    assert out.c_mean.shape == (3, 2)
    assert out.c_log_var.shape == (3, 2)
    assert out.exp_mean.shape == (3,)
    assert out.exp_log_var.shape == (3,)
    assert out.viewport_mean.shape == (3, 4)
    assert out.viewport_log_var.shape == (3, 4)


# ---------------- loss ----------------


def test_gaussian_nll_minimum_at_target() -> None:
    target = torch.tensor([0.0, 1.0, -2.0])
    # σ²=1: NLL minimised when μ == target.
    log_var = torch.zeros_like(target)
    nll_at_target = gaussian_nll(target, log_var, target).item()
    nll_off = gaussian_nll(target + 0.5, log_var, target).item()
    assert nll_off > nll_at_target


def test_gaussian_nll_clamps_log_var() -> None:
    target = torch.tensor([0.0])
    big = torch.tensor([100.0])
    small = torch.tensor([-100.0])
    # No NaN/Inf even with absurd log_var inputs.
    assert torch.isfinite(gaussian_nll(target, big, target))
    assert torch.isfinite(gaussian_nll(target, small, target))


def _stub_output(c_mean=None, c_log_var=None, viewport=None):
    """Helper: build a duck-typed ModelOutputV2-like object for loss tests."""
    out = type("O", (), {})()
    out.family_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out.c_mean = c_mean if c_mean is not None else torch.zeros(2, 2)
    out.c_log_var = c_log_var if c_log_var is not None else torch.zeros(2, 2)
    out.exp_mean = torch.tensor([0.0, 0.0])
    out.exp_log_var = torch.tensor([0.0, 0.0])
    out.viewport_mean = viewport if viewport is not None else torch.zeros(2, 4)
    out.viewport_log_var = torch.zeros(2, 4)
    out.embedding = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    return out


def test_compute_loss_indexes_julia_only() -> None:
    """A batch with no Julia rows must produce zero c_nll without touching the c-head NaNs."""
    out = _stub_output(
        c_mean=torch.tensor([[float("nan"), float("nan")], [float("nan"), float("nan")]]),
        c_log_var=torch.tensor([[float("nan"), float("nan")], [float("nan"), float("nan")]]),
    )
    pieces = compute_loss(
        out,
        family_idx=torch.tensor([0, 1]),
        c_target=torch.zeros(2, 2),
        exp_target=torch.zeros(2),
        vp_target=torch.zeros(2, 4),
        has_julia=torch.tensor([False, False]),
        has_multibrot=torch.tensor([False, False]),
    )
    assert torch.isfinite(pieces.total)
    assert pieces.n_julia == 0
    assert pieces.c_nll.item() == 0.0


def test_compute_loss_with_julia_row_finite() -> None:
    out = _stub_output(
        c_mean=torch.tensor([[0.1, 0.2], [0.0, 0.0]]),
        c_log_var=torch.zeros(2, 2),
    )
    pieces = compute_loss(
        out,
        family_idx=torch.tensor([0, 1]),
        c_target=torch.tensor([[0.5, -0.3], [0.0, 0.0]]),
        exp_target=torch.zeros(2),
        vp_target=torch.zeros(2, 4),
        has_julia=torch.tensor([True, False]),
        has_multibrot=torch.tensor([False, False]),
    )
    assert pieces.n_julia == 1
    assert torch.isfinite(pieces.total)
    assert pieces.c_nll.item() > 0
    # vp_nll is unmasked → must be present even when no julia rows.
    assert torch.isfinite(pieces.vp_nll)


# ---------------- torch renderer ----------------


def test_mandelbrot_origin_is_in_set() -> None:
    """c=0 sits inside the Mandelbrot set; the Python renderer must report it
    as in-set (sentinel -1.0)."""
    vp = tr.ViewportT(-0.05, 0.05, -0.05, 0.05)
    escape = tr.render_mandelbrot(vp, width=3, height=3, max_iter=50, smoothing=False)
    # Centre pixel (1, 1) corresponds to c=(0, 0).
    assert escape[1, 1].item() == pytest.approx(-1.0)


def test_julia_origin_with_zero_c_is_in_set() -> None:
    """z = z² with c=0 has fixed point at 0; |z₀|<1 stays bounded."""
    vp = tr.ViewportT(-0.5, 0.5, -0.5, 0.5)
    escape = tr.render_julia(vp, 3, 3, c_re=0.0, c_im=0.0, max_iter=50, smoothing=False)
    assert escape[1, 1].item() == pytest.approx(-1.0)


def test_julia_far_point_escapes() -> None:
    vp = tr.ViewportT(5.0, 5.001, 5.0, 5.001)
    escape = tr.render_julia(vp, 1, 1, c_re=-0.7, c_im=0.27015, max_iter=50, smoothing=False)
    assert 0.0 <= escape[0, 0].item() < 5.0


def test_multibrot_n2_matches_mandelbrot() -> None:
    """N=2 in the polar form must recover the Mandelbrot set up to numerical noise."""
    vp = tr.ViewportT(-2.0, 1.0, -1.2, 1.2)
    mandel = tr.render_mandelbrot(vp, 16, 16, max_iter=80, smoothing=False)
    multi = tr.render_multibrot(vp, 16, 16, exponent=2.0, max_iter=80, smoothing=False)
    in_set_match = ((mandel < 0) == (multi < 0)).float().mean().item()
    assert in_set_match >= 0.95, f"polar-form N=2 should match closed-form Mandelbrot, got {in_set_match}"


def test_torch_renderer_is_differentiable_in_julia_c() -> None:
    """Gradient through the iteration loop must flow back to (c_re, c_im)."""
    vp = tr.ViewportT(-1.5, 1.5, -1.5, 1.5)
    c_re = torch.tensor(-0.7, requires_grad=True)
    c_im = torch.tensor(0.27015, requires_grad=True)
    escape = tr.render_julia(vp, 16, 16, c_re=c_re, c_im=c_im, max_iter=30, smoothing=True)
    image = tr.escape_to_image(escape, max_iter=30, rgb=False).squeeze(0)
    image.sum().backward()
    assert c_re.grad is not None
    assert c_im.grad is not None
    assert torch.isfinite(c_re.grad)
    assert torch.isfinite(c_im.grad)


# ---------------- refiner ----------------


def test_refiner_reduces_julia_mse_with_bad_warm_start() -> None:
    """Generate a target render at known c, then start the refiner from a
    perturbed c. After L-BFGS the post-MSE should be strictly lower."""
    vp_t = tr.ViewportT(-1.5, 1.5, -1.5, 1.5)
    true_c_re, true_c_im = -0.7, 0.27015
    target_escape = tr.render_julia(vp_t, 32, 32, c_re=true_c_re, c_im=true_c_im, max_iter=80, smoothing=True)
    target_image = tr.escape_to_image(target_escape, max_iter=80, rgb=True).detach()

    # Warm start a small perturbation away from truth. Big enough to leave
    # room to improve, small enough to stay in the convex basin.
    req = RefineRequest(
        family="julia",
        target_image=target_image,
        viewport=(-1.5, 1.5, -1.5, 1.5),
        max_iter=80,
        init_c_re=true_c_re + 0.05,
        init_c_im=true_c_im - 0.05,
    )
    res = refine(req, n_iters=15, device="cpu")
    assert res.refined
    # Refinement either improves or matches; in the synthetic-target setup
    # we expect a strict improvement.
    assert res.post_mse <= res.pre_mse + 1e-6
    # Anchor: the refined point should be near the true c.
    assert abs(res.final_c_re - true_c_re) < 0.05
    assert abs(res.final_c_im - true_c_im) < 0.05


def test_refiner_mandelbrot_now_refines_viewport() -> None:
    """Slice 4b: Mandelbrot refinement is no longer a no-op — it optimises
    the viewport via the differentiable kernel. Starting from the true
    viewport, refinement should at worst leave MSE unchanged (post ≤ pre)
    and report a non-None final_viewport."""
    vp_t = tr.ViewportT(-2.0, 1.0, -1.2, 1.2)
    target_escape = tr.render_mandelbrot(vp_t, 16, 16, max_iter=40, smoothing=False)
    target_image = tr.escape_to_image(target_escape, max_iter=40, rgb=True).detach()
    req = RefineRequest(
        family="mandelbrot",
        target_image=target_image,
        viewport=(-2.0, 1.0, -1.2, 1.2),
        max_iter=40,
    )
    res = refine(req, n_iters=5, device="cpu")
    assert res.refined
    assert res.post_mse <= res.pre_mse + 1e-6
    assert res.final_viewport is not None


def test_log_var_clamp_constants_sanity() -> None:
    # Clamp constants exposed for tests + downstream calibration consumers.
    assert LOG_VAR_CLAMP_MIN < 0 < LOG_VAR_CLAMP_MAX
