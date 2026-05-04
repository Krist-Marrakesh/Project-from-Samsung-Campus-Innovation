"""Random recipe sampling for dataset generation.

Each `sample_*` function returns a dict matching the backend's `FractalRecipe`
JSON wire shape exactly — same field names, same `fractalType` discriminator,
same nested structure. The sampler is fed a `numpy.random.Generator` for
reproducibility; identical seeds across runs produce identical recipe streams.

Sampling strategy notes:
  * Mandelbrot/Burning-Ship: viewport perturbations around their classic windows
    so the resulting image is still recognisably the family. Random pure noise
    over (-2, 2) × (-2, 2) would put the camera in trivial all-escape regions for
    most samples and starve the dataset of in-set pixels.
  * Julia: c sampled uniformly inside the disk |c| <= 1.5 — outside this disk
    the set is generally trivial. Viewport stays close to the classic [-1.5, 1.5]
    square because Julia is full-plane, not localised like Mandelbrot.
  * Multibrot: exponent ∈ {2..6}. N=2 recovers Mandelbrot but is intentionally
    kept as a multibrot example so downstream classifiers can't trivially learn
    "exponent=2 ⇒ multibrot, but mandelbrot is special" — both labels coexist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PALETTES = ("grayscale", "fire", "ocean", "rainbow_cyclic")
COLOR_MODES = ("linear", "histogram")
FAMILIES = ("mandelbrot", "julia", "burning_ship", "multibrot")


@dataclass(frozen=True)
class SampleConfig:
    width_px: int = 128
    height_px: int = 128
    samples_per_axis: int = 1
    max_iter_min: int = 100
    max_iter_max: int = 400
    escape_radius: float = 2.0
    smoothing_prob: float = 1.0  # always smooth for richer escape signal


def _viewport_around(
    rng: np.random.Generator,
    cx: float,
    cy: float,
    span_min: float,
    span_max: float,
    aspect_jitter: float = 0.15,
) -> dict:
    """Square-ish viewport centred near (cx, cy)."""
    span = float(rng.uniform(span_min, span_max))
    aspect = float(rng.uniform(1.0 - aspect_jitter, 1.0 + aspect_jitter))
    cx_jitter = float(rng.uniform(-span * 0.25, span * 0.25))
    cy_jitter = float(rng.uniform(-span * 0.25, span * 0.25))
    half_w = span / 2.0
    half_h = span / (2.0 * aspect)
    return {
        "xMin": cx + cx_jitter - half_w,
        "xMax": cx + cx_jitter + half_w,
        "yMin": cy + cy_jitter - half_h,
        "yMax": cy + cy_jitter + half_h,
    }


def _color_settings(rng: np.random.Generator) -> dict:
    return {
        "paletteName": str(rng.choice(PALETTES)),
        "mode": str(rng.choice(COLOR_MODES)),
    }


def _render_settings(cfg: SampleConfig) -> dict:
    return {
        "widthPx": cfg.width_px,
        "heightPx": cfg.height_px,
        "samplesPerAxis": cfg.samples_per_axis,
    }


def _max_iter(rng: np.random.Generator, cfg: SampleConfig) -> int:
    return int(rng.integers(cfg.max_iter_min, cfg.max_iter_max + 1))


def sample_mandelbrot(rng: np.random.Generator, cfg: SampleConfig) -> dict:
    return {
        "viewport": _viewport_around(rng, cx=-0.5, cy=0.0, span_min=2.0, span_max=3.0),
        "renderSettings": _render_settings(cfg),
        "colorSettings": _color_settings(rng),
        "fractalType": "mandelbrot",
        "params": {
            "maxIter": _max_iter(rng, cfg),
            "escapeRadius": cfg.escape_radius,
            "smoothing": bool(rng.random() < cfg.smoothing_prob),
        },
    }


def sample_julia(rng: np.random.Generator, cfg: SampleConfig) -> dict:
    # Sample c uniformly inside the disk |c| <= 1.5 via rejection sampling.
    while True:
        c_re = float(rng.uniform(-1.5, 1.5))
        c_im = float(rng.uniform(-1.5, 1.5))
        if c_re * c_re + c_im * c_im <= 1.5 * 1.5:
            break
    return {
        "viewport": _viewport_around(rng, cx=0.0, cy=0.0, span_min=2.5, span_max=3.5),
        "renderSettings": _render_settings(cfg),
        "colorSettings": _color_settings(rng),
        "fractalType": "julia",
        "params": {
            "cRe": c_re,
            "cIm": c_im,
            "maxIter": _max_iter(rng, cfg),
            "escapeRadius": cfg.escape_radius,
            "smoothing": bool(rng.random() < cfg.smoothing_prob),
        },
    }


def sample_burning_ship(rng: np.random.Generator, cfg: SampleConfig) -> dict:
    return {
        # Burning-Ship's recognisable hull sits roughly at (-0.5, -0.5).
        "viewport": _viewport_around(rng, cx=-0.5, cy=-0.5, span_min=2.5, span_max=3.5),
        "renderSettings": _render_settings(cfg),
        "colorSettings": _color_settings(rng),
        "fractalType": "burning_ship",
        "params": {
            "maxIter": _max_iter(rng, cfg),
            "escapeRadius": cfg.escape_radius,
            "smoothing": bool(rng.random() < cfg.smoothing_prob),
        },
    }


def sample_multibrot(rng: np.random.Generator, cfg: SampleConfig) -> dict:
    exponent = int(rng.integers(2, 7))  # {2..6}
    return {
        "viewport": _viewport_around(rng, cx=0.0, cy=0.0, span_min=2.5, span_max=3.5),
        "renderSettings": _render_settings(cfg),
        "colorSettings": _color_settings(rng),
        "fractalType": "multibrot",
        "params": {
            "exponent": exponent,
            "maxIter": _max_iter(rng, cfg),
            "escapeRadius": cfg.escape_radius,
            "smoothing": bool(rng.random() < cfg.smoothing_prob),
        },
    }


SAMPLERS = {
    "mandelbrot": sample_mandelbrot,
    "julia": sample_julia,
    "burning_ship": sample_burning_ship,
    "multibrot": sample_multibrot,
}


def sample(family: str, rng: np.random.Generator, cfg: SampleConfig) -> dict:
    if family not in SAMPLERS:
        raise ValueError(f"unknown family: {family}")
    return SAMPLERS[family](rng, cfg)
