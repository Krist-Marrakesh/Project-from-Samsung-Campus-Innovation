"""Differentiable fractal renderer for inverse-inference research.

This is **not** a replacement for the Java backend. The backend renders
high-fidelity coloured PNGs with palettes, histogram coloring, SSAA, and
distance-estimate shading; this module renders single-channel escape-time
maps in pure PyTorch. The point is *differentiability*: gradients flow back
through every iteration step, so L-BFGS can refine recipe parameters by
minimising MSE against a target image.

Four kernels are supported, mirroring the Java backend:

* ``mandelbrot``:    ``z_{n+1} = z_n² + c``, ``z_0 = 0``
* ``julia``:         ``z_{n+1} = z_n² + c``, ``z_0 = pixel``
* ``multibrot``:     ``z_{n+1} = z_n^N + c``, ``z_0 = 0``  (continuous N)
* ``burning_ship``:  ``z_{n+1} = (|Re z_n| + i|Im z_n|)² + c``, ``z_0 = 0``

Burning Ship is non-holomorphic — the ``|·|`` step's derivative is undefined
on the real and imaginary axes — but ``torch.abs`` provides a subgradient of
``0`` at ``x = 0`` (and ``±1`` elsewhere), which is exactly the analytic case
where the formal derivative is undefined. This gives finite gradients almost
everywhere, sufficient for L-BFGS refinement of the viewport against a
target image. We document this honestly: the kernel does not pretend to be a
gradient of a smooth function; it provides a subgradient that an
optimiser can follow.

Differentiable viewport: every render takes a :class:`ViewportT` (Python
floats) OR a tuple of four real tensors. When tensors are passed, the
spatial grid is constructed via ``torch.linspace(0, 1, W)``-rescaling so
gradients flow back to the four viewport bounds — this is what powers
viewport refinement for Mandelbrot and Burning Ship, which have no
recipe-internal continuous parameter to refine.

Output normalisation matches the Java backend's ``LINEAR`` colour mode at
grayscale: continuous escape values via the smoothing trick, divided by
``max_iter``, clamped to ``[0, 1]``. In-set points are exactly 0. The
single-channel result is broadcast to RGB at the very end so we can compare
against three-channel input images directly.

Performance: a 64×64 render with max_iter=200 is ~30ms on CPU. Not fast
enough for real-time UI, fast enough to put inside an L-BFGS loop with 20
steps — three seconds per refinement, comparable to the backend round-trip
path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ViewportT:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


ViewportLike = "ViewportT | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]"


def make_grid(
    viewport: ViewportT,
    width: int,
    height: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the per-pixel complex coordinates as two real tensors of shape
    ``(H, W)``. The y-axis runs ``y_max`` at row 0 to ``y_min`` at row
    H-1 — same convention as the Java :class:`GridSweep`.

    Non-differentiable: viewport is plain Python floats. Used by callers that
    don't need gradients on the bounds (eg the rendering tests). For
    viewport-refinement see :func:`make_grid_from_tensors`.
    """
    xs = torch.linspace(viewport.x_min, viewport.x_max, width, device=device, dtype=dtype)
    ys = torch.linspace(viewport.y_max, viewport.y_min, height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def make_grid_from_tensors(
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    y_min: torch.Tensor,
    y_max: torch.Tensor,
    width: int,
    height: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the per-pixel complex coordinates with autograd flowing back to
    the four viewport bounds.

    Implementation: a fixed unit-interval ``linspace`` rescaled by
    ``(x_max - x_min) + x_min``. ``torch.linspace`` itself doesn't accept
    tensor endpoints, so we lift the affine transform into real tensor
    arithmetic. This produces a sensible Jacobian
    ``∂cx/∂x_min = 1 - t``, ``∂cx/∂x_max = t`` exactly as the analytic
    grid mapping prescribes.
    """
    if width < 1 or height < 1:
        raise ValueError(f"width/height must be ≥ 1; got {width}x{height}")
    base_x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)   # (W,)
    base_y = torch.linspace(1.0, 0.0, height, device=device, dtype=dtype)  # (H,)
    xs = x_min + (x_max - x_min) * base_x       # (W,)
    ys = y_min + (y_max - y_min) * base_y       # (H,)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def _grid_for(viewport: ViewportLike, width: int, height: int, device, dtype=torch.float32):
    """Dispatch grid construction by viewport flavour: plain ViewportT or
    a 4-tuple of tensors."""
    if isinstance(viewport, ViewportT):
        return make_grid(viewport, width, height, device=device, dtype=dtype)
    if isinstance(viewport, tuple) and len(viewport) == 4:
        x_min, x_max, y_min, y_max = viewport
        return make_grid_from_tensors(
            _as_tensor(x_min, device=device, dtype=dtype),
            _as_tensor(x_max, device=device, dtype=dtype),
            _as_tensor(y_min, device=device, dtype=dtype),
            _as_tensor(y_max, device=device, dtype=dtype),
            width, height, device=device, dtype=dtype,
        )
    raise TypeError(
        f"viewport must be ViewportT or 4-tuple of tensors/floats; got {type(viewport).__name__}"
    )


# ----------------------------- kernels -----------------------------


def render_mandelbrot(
    viewport,
    width: int,
    height: int,
    *,
    max_iter: int = 200,
    escape_radius: float = 2.0,
    smoothing: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Render the Mandelbrot escape-time field.

    ``viewport`` accepts either a :class:`ViewportT` (no gradients on bounds)
    or a 4-tuple ``(x_min, x_max, y_min, y_max)`` of tensors with
    ``requires_grad=True`` — the second form is what powers viewport
    refinement, the only continuous knob the Mandelbrot family exposes.
    """
    cx, cy = _grid_for(viewport, width, height, device=device)
    return _mandelbrot_like(cx, cy, max_iter=max_iter, escape_radius=escape_radius, smoothing=smoothing)


def render_julia(
    viewport,
    width: int,
    height: int,
    *,
    c_re: torch.Tensor | float,
    c_im: torch.Tensor | float,
    max_iter: int = 200,
    escape_radius: float = 2.0,
    smoothing: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Render a Julia set at parameter ``c = c_re + i c_im``.

    ``c_re`` and ``c_im`` may be Python floats or 0-d tensors with
    ``requires_grad=True``. When tensors, autograd flows through the entire
    iteration chain — that is the one feature that makes hybrid refinement
    possible at all. ``viewport`` accepts the same flavours as
    :func:`render_mandelbrot`."""
    zx, zy = _grid_for(viewport, width, height, device=device)
    c_re_t = _as_tensor(c_re, device=device, dtype=zx.dtype)
    c_im_t = _as_tensor(c_im, device=device, dtype=zx.dtype)
    return _julia_iterate(
        zx, zy, c_re_t, c_im_t,
        max_iter=max_iter, escape_radius=escape_radius, smoothing=smoothing,
    )


def render_multibrot(
    viewport,
    width: int,
    height: int,
    *,
    exponent: torch.Tensor | float,
    max_iter: int = 200,
    escape_radius: float = 2.0,
    smoothing: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Render Multibrot at exponent N (continuous, dispatch via polar form).

    Continuous exponents are accepted because the gradient path needs them —
    the L-BFGS refiner sees ``exponent`` as a real-valued knob and only
    snaps it to an integer at the end."""
    cx, cy = _grid_for(viewport, width, height, device=device)
    n_t = _as_tensor(exponent, device=device, dtype=cx.dtype)
    return _multibrot_iterate(
        cx, cy, n_t,
        max_iter=max_iter, escape_radius=escape_radius, smoothing=smoothing,
    )


def render_burning_ship(
    viewport,
    width: int,
    height: int,
    *,
    max_iter: int = 200,
    escape_radius: float = 2.0,
    smoothing: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Render the Burning Ship escape-time field.

    Same viewport-tensor support as :func:`render_mandelbrot`. The kernel is
    non-holomorphic — see the module docstring for the subgradient note —
    but the implementation is otherwise structurally identical to
    Mandelbrot, with the key difference being the ``|·|`` step on each axis
    inside the iteration body."""
    cx, cy = _grid_for(viewport, width, height, device=device)
    return _burning_ship_iterate(
        cx, cy,
        max_iter=max_iter, escape_radius=escape_radius, smoothing=smoothing,
    )


# ------------------------ normalised image -------------------------


def escape_to_image(
    escape: torch.Tensor,
    *,
    max_iter: int,
    rgb: bool = True,
) -> torch.Tensor:
    """Convert escape-time field to ``[0, 1]`` image tensor.

    The escape field has ``-1.0`` for in-set points and ``[0, max_iter)``
    for escape points. We map in-set to 0 and escape to ``escape/max_iter``
    clamped, mirroring the Java backend's ``LINEAR`` mode at grayscale.

    When ``rgb=True`` (default) the single channel is broadcast to three so
    the result lines up with 3-channel input images.
    """
    in_set = escape < 0.0
    t = torch.where(in_set, torch.zeros_like(escape), escape / float(max_iter))
    t = t.clamp(0.0, 1.0)
    if rgb:
        return t.unsqueeze(0).expand(3, -1, -1)
    return t.unsqueeze(0)


# ------------------------- iteration cores -------------------------


def _julia_iterate(
    zx: torch.Tensor,
    zy: torch.Tensor,
    c_re: torch.Tensor,
    c_im: torch.Tensor,
    *,
    max_iter: int,
    escape_radius: float,
    smoothing: bool,
) -> torch.Tensor:
    """z_{n+1} = z_n² + c, with z_0 from the spatial grid. Vectorised across
    the (H, W) grid; iteration count is the only python-level loop."""
    esc_r2 = float(escape_radius) ** 2
    iters = torch.zeros_like(zx)
    final_zx = zx.clone()
    final_zy = zy.clone()
    finished = torch.zeros_like(zx, dtype=torch.bool)

    for it in range(max_iter):
        z2 = zx * zx
        z2_im = zy * zy
        mag2 = z2 + z2_im
        # Lock pixels that escaped this step.
        newly_escaped = (mag2 > esc_r2) & ~finished
        finished = finished | newly_escaped
        iters = torch.where(newly_escaped, torch.full_like(iters, float(it)), iters)
        # Snapshot z at escape (used for smoothing).
        final_zx = torch.where(newly_escaped, zx, final_zx)
        final_zy = torch.where(newly_escaped, zy, final_zy)

        if bool(finished.all()):
            break

        zx_new = z2 - z2_im + c_re
        zy_new = 2.0 * zx * zy + c_im
        # Inactive pixels keep their values; the where() below freezes them.
        zx = torch.where(finished, zx, zx_new)
        zy = torch.where(finished, zy, zy_new)

    in_set = ~finished
    if smoothing:
        mag2 = final_zx * final_zx + final_zy * final_zy
        mag2 = mag2.clamp(min=1e-30)
        log_zn = 0.5 * torch.log(mag2)
        # log2 via natural log avoids special-case torch ops.
        ln2 = math.log(2.0)
        smooth = iters + 1.0 - torch.log(log_zn / ln2 + 1e-30) / ln2
        escape_field = torch.where(in_set, torch.full_like(iters, -1.0), smooth)
    else:
        escape_field = torch.where(in_set, torch.full_like(iters, -1.0), iters)
    return escape_field


def _mandelbrot_like(
    cx: torch.Tensor,
    cy: torch.Tensor,
    *,
    max_iter: int,
    escape_radius: float,
    smoothing: bool,
) -> torch.Tensor:
    """z_{n+1} = z_n² + c, z_0 = 0. Reuses the Julia core with z_0 = 0
    and c bound to the spatial grid."""
    zero_x = torch.zeros_like(cx)
    zero_y = torch.zeros_like(cy)
    return _julia_iterate(
        zero_x, zero_y, cx, cy,
        max_iter=max_iter, escape_radius=escape_radius, smoothing=smoothing,
    )


def _multibrot_iterate(
    cx: torch.Tensor,
    cy: torch.Tensor,
    n: torch.Tensor,
    *,
    max_iter: int,
    escape_radius: float,
    smoothing: bool,
) -> torch.Tensor:
    """z_{n+1} = z^N + c via polar form. ``n`` is a 0-d or 1-d tensor."""
    esc_r2 = float(escape_radius) ** 2
    zx = torch.zeros_like(cx)
    zy = torch.zeros_like(cy)
    iters = torch.zeros_like(cx)
    final_zx = zx.clone()
    final_zy = zy.clone()
    finished = torch.zeros_like(cx, dtype=torch.bool)
    for it in range(max_iter):
        mag2 = zx * zx + zy * zy
        newly_escaped = (mag2 > esc_r2) & ~finished
        finished = finished | newly_escaped
        iters = torch.where(newly_escaped, torch.full_like(iters, float(it)), iters)
        final_zx = torch.where(newly_escaped, zx, final_zx)
        final_zy = torch.where(newly_escaped, zy, final_zy)
        if bool(finished.all()):
            break

        # Polar form: z^N = r^N · (cos(N·θ) + i·sin(N·θ)).
        # mag2 floored away from 0 so atan2's gradient remains finite at z=0;
        # at z=0 the derivative N·z^{N-1} vanishes for any N > 1, so the
        # exact value of theta there does not feed gradients.
        safe_mag2 = mag2.clamp(min=1e-30)
        r = torch.sqrt(safe_mag2)
        theta = torch.atan2(zy, zx)
        rN = torch.pow(r, n)
        nTheta = n * theta
        zx_new = rN * torch.cos(nTheta) + cx
        zy_new = rN * torch.sin(nTheta) + cy

        zx = torch.where(finished, zx, zx_new)
        zy = torch.where(finished, zy, zy_new)

    in_set = ~finished
    if smoothing:
        mag2 = final_zx * final_zx + final_zy * final_zy
        mag2 = mag2.clamp(min=1e-30)
        log_zn = 0.5 * torch.log(mag2)
        log_n = torch.log(n.clamp(min=1.001))
        smooth = iters + 1.0 - torch.log(log_zn / log_n + 1e-30) / log_n
        escape_field = torch.where(in_set, torch.full_like(iters, -1.0), smooth)
    else:
        escape_field = torch.where(in_set, torch.full_like(iters, -1.0), iters)
    return escape_field


def _burning_ship_iterate(
    cx: torch.Tensor,
    cy: torch.Tensor,
    *,
    max_iter: int,
    escape_radius: float,
    smoothing: bool,
) -> torch.Tensor:
    """z_{n+1} = (|Re z| + i|Im z|)² + c, z_0 = 0. Vectorised over the grid.

    ``torch.abs`` is differentiable everywhere except at the origin where
    it provides a subgradient of 0; this matches the analytic case where
    the formal derivative is undefined and lets autograd flow through the
    iteration chain.
    """
    esc_r2 = float(escape_radius) ** 2
    zx = torch.zeros_like(cx)
    zy = torch.zeros_like(cy)
    iters = torch.zeros_like(cx)
    final_zx = zx.clone()
    final_zy = zy.clone()
    finished = torch.zeros_like(cx, dtype=torch.bool)

    for it in range(max_iter):
        z2 = zx * zx
        z2_im = zy * zy
        mag2 = z2 + z2_im
        newly_escaped = (mag2 > esc_r2) & ~finished
        finished = finished | newly_escaped
        iters = torch.where(newly_escaped, torch.full_like(iters, float(it)), iters)
        final_zx = torch.where(newly_escaped, zx, final_zx)
        final_zy = torch.where(newly_escaped, zy, final_zy)

        if bool(finished.all()):
            break

        # (|zRe| + i|zIm|)² = zRe² − zIm² + i · 2|zRe·zIm|
        zx_new = z2 - z2_im + cx
        zy_new = 2.0 * torch.abs(zx * zy) + cy

        zx = torch.where(finished, zx, zx_new)
        zy = torch.where(finished, zy, zy_new)

    in_set = ~finished
    if smoothing:
        mag2 = final_zx * final_zx + final_zy * final_zy
        mag2 = mag2.clamp(min=1e-30)
        log_zn = 0.5 * torch.log(mag2)
        ln2 = math.log(2.0)
        smooth = iters + 1.0 - torch.log(log_zn / ln2 + 1e-30) / ln2
        escape_field = torch.where(in_set, torch.full_like(iters, -1.0), smooth)
    else:
        escape_field = torch.where(in_set, torch.full_like(iters, -1.0), iters)
    return escape_field


def _as_tensor(v, *, device, dtype) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.to(device=device, dtype=dtype)
    return torch.tensor(float(v), device=device, dtype=dtype)
