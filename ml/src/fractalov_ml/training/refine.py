"""Hybrid inverse-inference refiner: NN warm start → L-BFGS reconstruction.

Pure neural-network prediction has a hard floor on reconstruction quality —
it commits to a single point estimate from one forward pass. Pure pixel-MSE
optimisation has a hard floor on tractability — fractal loss surfaces are
highly non-convex and L-BFGS without a good initial point lands in a random
local minimum. The hybrid approach sidesteps both:

  1. The NN provides a warm-start parameter estimate (μ from the MVE heads).
  2. L-BFGS, run against the differentiable :mod:`torch_renderer`,
     refines the parameters against pixel-MSE versus the target image.

What gets refined depends on the family:

* ``julia``        — refine ``(c_re, c_im)``. The dominant continuous DoF.
                     Optionally also refine the viewport.
* ``multibrot``    — refine ``exponent`` continuously (snapped to int at
                     the end). Optionally also refine the viewport.
* ``mandelbrot``   — refine the four viewport bounds. No recipe-internal
                     continuous parameter, so viewport is the only knob.
* ``burning_ship`` — same as Mandelbrot. Burning Ship is non-holomorphic
                     but ``torch.abs`` provides a subgradient that L-BFGS
                     can follow; see :mod:`torch_renderer` for the note.

Viewport parameterisation: rather than optimising over the four raw bounds
directly, we parameterise as ``(cx, cy, half_w, half_h)`` with
``half_w/half_h = softplus(raw_w/raw_h)``. This guarantees positive widths
(monotonic ``x_max > x_min``, ``y_max > y_min``) without needing barrier
constraints, and produces stable gradients across the unconstrained
``raw`` parameters that L-BFGS is happiest with. The mapping is bijective
on the constraint set.

The MSE used here is between *grayscale-linear* renders, computed by
averaging the RGB channels of the target image and comparing to
:func:`torch_renderer.escape_to_image` at ``rgb=False``. This deliberately
ignores the recipe's palette — comparing across palettes would require
shipping the Java backend's palette LUTs into Python, which adds friction
without helping the inverse-inference signal. The same mapping is applied
to both pure-NN and refined renders, so the comparison is fair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .. import torch_renderer as tr


@dataclass(frozen=True)
class RefineRequest:
    family: str
    target_image: torch.Tensor       # (3, H, W) in [0, 1]
    viewport: tuple[float, float, float, float]    # warm-start viewport
    max_iter: int
    # Warm-start parameters. Unused fields can stay at default — the family
    # dispatch ignores them.
    init_c_re: float = 0.0
    init_c_im: float = 0.0
    init_exponent: float = 2.0
    # When True, also refine the viewport for julia / multibrot. Default
    # False keeps Slice 3 numbers comparable; turn on for full reconstruction.
    refine_viewport: bool = False


@dataclass(frozen=True)
class RefineResult:
    family: str
    refined: bool                          # False only on unsupported families
    pre_mse: float
    post_mse: float
    final_c_re: float | None
    final_c_im: float | None
    final_exponent: float | None
    final_viewport: tuple[float, float, float, float] | None
    n_iters: int


def refine(
    req: RefineRequest,
    *,
    n_iters: int = 20,
    lr: float = 0.05,
    device: torch.device | str = "cpu",
) -> RefineResult:
    """Run hybrid refinement on a single target. Returns metrics + refined params.

    L-BFGS is used because (a) it converges in ~20 steps without manual lr
    tuning across families, and (b) the search space is at most 6-D
    (viewport + 2D Julia c, with viewport refinement on) — well within the
    regime where quasi-Newton beats SGD.
    """
    target_gray = req.target_image.to(device=device, dtype=torch.float32).mean(dim=0)
    height, width = target_gray.shape

    if req.family == "julia":
        return _refine_julia(req, target_gray, width, height, n_iters, lr, device)
    if req.family == "multibrot":
        return _refine_multibrot(req, target_gray, width, height, n_iters, lr, device)
    if req.family in ("mandelbrot", "burning_ship"):
        return _refine_viewport_only(req, target_gray, width, height, n_iters, lr, device)

    # Genuinely unknown family — surface as not refined, MSE 0.
    return RefineResult(
        family=req.family,
        refined=False,
        pre_mse=0.0,
        post_mse=0.0,
        final_c_re=None, final_c_im=None,
        final_exponent=None, final_viewport=None,
        n_iters=0,
    )


# --------------------------- viewport math --------------------------


@dataclass(frozen=True)
class _ViewportParams:
    """Unconstrained ``(cx, cy, raw_w, raw_h)`` parameterisation.

    Width / height are recovered as ``softplus(raw_w/h)`` which keeps them
    positive and differentiable. Initial values are computed from a
    desired ``(half_w, half_h)`` so the warm start sits at the model's
    predicted viewport without surprises.
    """
    cx: torch.Tensor
    cy: torch.Tensor
    raw_w: torch.Tensor
    raw_h: torch.Tensor

    @classmethod
    def from_bounds(cls, viewport: tuple[float, float, float, float], device, dtype=torch.float32) -> "_ViewportParams":
        x_min, x_max, y_min, y_max = viewport
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        half_w = max(0.5 * (x_max - x_min), 1e-6)
        half_h = max(0.5 * (y_max - y_min), 1e-6)
        raw_w = _softplus_inv(half_w)
        raw_h = _softplus_inv(half_h)
        return cls(
            cx=torch.tensor(cx, device=device, dtype=dtype, requires_grad=True),
            cy=torch.tensor(cy, device=device, dtype=dtype, requires_grad=True),
            raw_w=torch.tensor(raw_w, device=device, dtype=dtype, requires_grad=True),
            raw_h=torch.tensor(raw_h, device=device, dtype=dtype, requires_grad=True),
        )

    def parameters(self) -> list[torch.Tensor]:
        return [self.cx, self.cy, self.raw_w, self.raw_h]

    def bounds_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        half_w = F.softplus(self.raw_w)
        half_h = F.softplus(self.raw_h)
        x_min = self.cx - half_w
        x_max = self.cx + half_w
        y_min = self.cy - half_h
        y_max = self.cy + half_h
        return x_min, x_max, y_min, y_max

    def bounds_floats(self) -> tuple[float, float, float, float]:
        x_min, x_max, y_min, y_max = self.bounds_tensors()
        return (
            float(x_min.detach().item()),
            float(x_max.detach().item()),
            float(y_min.detach().item()),
            float(y_max.detach().item()),
        )


def _softplus_inv(y: float) -> float:
    """Inverse of ``softplus`` for warm-starting the unconstrained params.

    ``softplus(x) = log(1 + exp(x))``, so ``softplus⁻¹(y) = log(exp(y) − 1)``
    for ``y > 0``. For large ``y`` the ``exp(y) − 1`` overflows, so we
    fall back to ``y`` (asymptotically ``softplus(x) ≈ x`` for large x).
    """
    if y > 20.0:
        return y
    return math.log(math.expm1(y))


# ----------------------- per-family refiners ------------------------


def _refine_julia(req, target_gray, width, height, n_iters, lr, device) -> RefineResult:
    c_re = torch.tensor(float(req.init_c_re), device=device, dtype=torch.float32, requires_grad=True)
    c_im = torch.tensor(float(req.init_c_im), device=device, dtype=torch.float32, requires_grad=True)
    vp_params = _ViewportParams.from_bounds(req.viewport, device) if req.refine_viewport else None

    def make_viewport():
        return vp_params.bounds_tensors() if vp_params else _viewport_static(req)

    def loss_fn():
        vp = make_viewport()
        escape = tr.render_julia(vp, width, height, c_re=c_re, c_im=c_im,
                                 max_iter=req.max_iter, device=device)
        rendered = tr.escape_to_image(escape, max_iter=req.max_iter, rgb=False).squeeze(0)
        return ((rendered - target_gray) ** 2).mean()

    pre_mse = loss_fn().detach().item()
    params = [c_re, c_im] + (vp_params.parameters() if vp_params else [])
    iters_run = _run_lbfgs(params, loss_fn, n_iters, lr)
    post_mse = loss_fn().detach().item()

    return RefineResult(
        family="julia",
        refined=True,
        pre_mse=pre_mse,
        post_mse=post_mse,
        final_c_re=c_re.detach().item(),
        final_c_im=c_im.detach().item(),
        final_exponent=None,
        final_viewport=vp_params.bounds_floats() if vp_params else None,
        n_iters=iters_run,
    )


def _refine_multibrot(req, target_gray, width, height, n_iters, lr, device) -> RefineResult:
    n = torch.tensor(float(req.init_exponent), device=device, dtype=torch.float32, requires_grad=True)
    vp_params = _ViewportParams.from_bounds(req.viewport, device) if req.refine_viewport else None

    def make_viewport():
        return vp_params.bounds_tensors() if vp_params else _viewport_static(req)

    def loss_fn():
        vp = make_viewport()
        escape = tr.render_multibrot(vp, width, height, exponent=n,
                                     max_iter=req.max_iter, device=device)
        rendered = tr.escape_to_image(escape, max_iter=req.max_iter, rgb=False).squeeze(0)
        return ((rendered - target_gray) ** 2).mean()

    pre_mse = loss_fn().detach().item()
    params = [n] + (vp_params.parameters() if vp_params else [])
    iters_run = _run_lbfgs(params, loss_fn, n_iters, lr)
    post_mse = loss_fn().detach().item()

    return RefineResult(
        family="multibrot",
        refined=True,
        pre_mse=pre_mse,
        post_mse=post_mse,
        final_c_re=None,
        final_c_im=None,
        final_exponent=float(n.detach().item()),
        final_viewport=vp_params.bounds_floats() if vp_params else None,
        n_iters=iters_run,
    )


def _refine_viewport_only(req, target_gray, width, height, n_iters, lr, device) -> RefineResult:
    """Mandelbrot / Burning Ship refinement: viewport is the only continuous
    knob the dataset varies for these families."""
    vp_params = _ViewportParams.from_bounds(req.viewport, device)
    is_burning = req.family == "burning_ship"

    def loss_fn():
        vp = vp_params.bounds_tensors()
        if is_burning:
            escape = tr.render_burning_ship(vp, width, height,
                                            max_iter=req.max_iter, device=device)
        else:
            escape = tr.render_mandelbrot(vp, width, height,
                                          max_iter=req.max_iter, device=device)
        rendered = tr.escape_to_image(escape, max_iter=req.max_iter, rgb=False).squeeze(0)
        return ((rendered - target_gray) ** 2).mean()

    pre_mse = loss_fn().detach().item()
    iters_run = _run_lbfgs(vp_params.parameters(), loss_fn, n_iters, lr)
    post_mse = loss_fn().detach().item()

    return RefineResult(
        family=req.family,
        refined=True,
        pre_mse=pre_mse,
        post_mse=post_mse,
        final_c_re=None,
        final_c_im=None,
        final_exponent=None,
        final_viewport=vp_params.bounds_floats(),
        n_iters=iters_run,
    )


def _run_lbfgs(params, loss_fn, n_iters: int, lr: float) -> int:
    """Common L-BFGS driver. Returns the number of closure invocations."""
    optimizer = torch.optim.LBFGS(
        params,
        lr=lr,
        max_iter=n_iters,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )
    iters_run = {"n": 0}

    def closure():
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        iters_run["n"] += 1
        return loss

    optimizer.step(closure)
    return iters_run["n"]


def _viewport_static(req: RefineRequest) -> tr.ViewportT:
    x_min, x_max, y_min, y_max = req.viewport
    return tr.ViewportT(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
