"""Config-driven recipe sampler.

Replaces the older ``recipes.sample(family, rng, cfg)`` API with one that
takes the merged per-(split, family) :class:`FamilyRanges` and a
:class:`RenderSpec` produced by :mod:`dataset_config`. The legacy module is
kept untouched for back-compat with existing tests.

Sampling guarantees:

* Identical ``(seed, family_name, split_name, family_ranges, render)``
  inputs produce byte-identical recipe streams. The split seed offset is
  computed in :func:`dataset_config.split_seed`, so each split is an
  independent reproducible sub-stream of the dataset.
* Recipes are wire-compatible with the Java backend's ``FractalRecipe`` —
  same field names, same external ``fractalType`` discriminator, same
  nested layout. The single test of this is round-tripping a sampled
  recipe through ``client.BackendClient.render``.
"""

from __future__ import annotations

import numpy as np

from .dataset_config import (
    DEFAULT_COLOR_MODES,
    DEFAULT_PALETTES,
    DatasetConfig,
    FamilyRanges,
    FamilySpec,
    Range,
    RenderSpec,
    merged_ranges,
    split_seed,
)

# Default ranges per family — what the old hardcoded sampler used. These keep
# old behaviour available when a config supplies no explicit ranges.
DEFAULT_RANGES: dict[str, FamilyRanges] = {
    "mandelbrot": FamilyRanges(
        max_iter=Range(100, 400),
        span=Range(2.0, 3.0),
    ),
    "julia": FamilyRanges(
        max_iter=Range(100, 400),
        span=Range(2.5, 3.5),
        c_re=Range(-1.5, 1.5),
        c_im=Range(-1.5, 1.5),
    ),
    "burning_ship": FamilyRanges(
        max_iter=Range(100, 400),
        span=Range(2.5, 3.5),
    ),
    "multibrot": FamilyRanges(
        max_iter=Range(100, 400),
        span=Range(2.5, 3.5),
        exponent=Range(2, 6),
    ),
}

# Centres of the recognisable hull for each family. Viewport sampling jitters
# around these so a randomly drawn recipe still shows the family's character.
FAMILY_CENTERS: dict[str, tuple[float, float]] = {
    "mandelbrot": (-0.5, 0.0),
    "julia": (0.0, 0.0),
    "burning_ship": (-0.5, -0.5),
    "multibrot": (0.0, 0.0),
}


def effective_ranges(family: str, ranges: FamilyRanges) -> FamilyRanges:
    """Fill in missing ranges from per-family defaults."""
    base = DEFAULT_RANGES.get(family, FamilyRanges())
    return base.merged_with(ranges)


def sample_recipe(
    family: str,
    rng: np.random.Generator,
    family_spec: FamilySpec,
    family_ranges: FamilyRanges,
    render: RenderSpec,
    palettes: tuple[str, ...] = DEFAULT_PALETTES,
    color_modes: tuple[str, ...] = DEFAULT_COLOR_MODES,
) -> dict:
    """Sample one recipe. Returns the wire-shape dict expected by the
    Java backend."""
    rng_ranges = effective_ranges(family, family_ranges)
    cx, cy = FAMILY_CENTERS.get(family, (0.0, 0.0))
    span_lo, span_hi = _range_or_default(rng_ranges.span, (2.5, 3.5))
    span = float(rng.uniform(span_lo, span_hi))
    cx_jitter = float(rng.uniform(-span * 0.25, span * 0.25))
    cy_jitter = float(rng.uniform(-span * 0.25, span * 0.25))
    aspect = float(rng.uniform(0.85, 1.15))
    half_w = span / 2.0
    half_h = span / (2.0 * aspect)
    viewport = {
        "xMin": cx + cx_jitter - half_w,
        "xMax": cx + cx_jitter + half_w,
        "yMin": cy + cy_jitter - half_h,
        "yMax": cy + cy_jitter + half_h,
    }

    palette = str(rng.choice(palettes))
    color_mode = str(rng.choice(color_modes))
    # Burning Ship has no holomorphic distance estimator; force fallback so
    # the resulting recipe is renderable.
    if family == "burning_ship" and color_mode == "distance_estimate":
        color_mode = "linear"

    iter_lo, iter_hi = _range_or_default(rng_ranges.max_iter, (100, 400))
    max_iter = int(rng.integers(int(iter_lo), int(iter_hi) + 1))

    params: dict = {
        "maxIter": max_iter,
        "escapeRadius": float(family_spec.escape_radius),
        "smoothing": bool(family_spec.smoothing),
    }

    if family == "julia":
        c_re_lo, c_re_hi = _range_or_default(rng_ranges.c_re, (-1.5, 1.5))
        c_im_lo, c_im_hi = _range_or_default(rng_ranges.c_im, (-1.5, 1.5))
        # Rejection-sample inside the unit-1.5 disk so most c values land in
        # the visually rich region. If the requested rectangle does not
        # intersect the disk we fall back to a uniform draw — this is the
        # near_ood / hard_overlap case where overlap with the disk is
        # intentionally limited.
        c_re, c_im = _sample_julia_c(rng, c_re_lo, c_re_hi, c_im_lo, c_im_hi)
        params["cRe"] = c_re
        params["cIm"] = c_im
    elif family == "multibrot":
        exp_lo, exp_hi = _range_or_default(rng_ranges.exponent, (2, 6))
        params["exponent"] = int(rng.integers(int(exp_lo), int(exp_hi) + 1))

    return {
        "viewport": viewport,
        "renderSettings": {
            "widthPx": render.width_px,
            "heightPx": render.height_px,
            "samplesPerAxis": render.samples_per_axis,
        },
        "colorSettings": {
            "paletteName": palette,
            "mode": color_mode,
        },
        "fractalType": family,
        "params": params,
    }


def split_rng(config: DatasetConfig, split_name: str) -> np.random.Generator:
    """Per-split RNG built from ``config.seed`` and the split name. The two
    contracts: (a) reordering splits in the YAML does not perturb other
    splits' streams, and (b) running the same config twice produces
    byte-identical streams."""
    return np.random.default_rng(split_seed(config.seed, split_name))


def iterate_split(config: DatasetConfig, split_name: str):
    """Yield ``(family, recipe)`` tuples for every example in a split, in a
    deterministic family-major order. The generator runs all examples of
    family ``F0`` before moving on to ``F1`` so the disk image layout
    (``images/<split>/<family>/...``) lines up with the recipe stream — and
    so a generation crash leaves a clean prefix on disk rather than
    sparse holes."""
    spec = config.split(split_name)
    rng = split_rng(config, split_name)
    family_names = config.family_names()
    if spec.restrict_to_overrides:
        family_names = tuple(f for f in family_names if f in spec.family_overrides)
    for family in family_names:
        family_spec = config.family(family)
        ranges = merged_ranges(config, split_name, family)
        for _ in range(spec.examples_per_family):
            yield family, sample_recipe(
                family,
                rng,
                family_spec,
                ranges,
                config.render,
                palettes=config.palettes,
                color_modes=config.color_modes,
            )


def _range_or_default(r: Range | None, default: tuple[float, float]) -> tuple[float, float]:
    if r is None:
        return default
    return (r.lo, r.hi)


def _sample_julia_c(
    rng: np.random.Generator,
    c_re_lo: float,
    c_re_hi: float,
    c_im_lo: float,
    c_im_hi: float,
    *,
    max_attempts: int = 64,
) -> tuple[float, float]:
    """Rejection-sample c inside the disk |c| ≤ 1.5 and inside the requested
    rectangle. Falls back to a uniform draw on the rectangle alone after
    ``max_attempts`` tries — this happens deliberately for ``near_ood`` /
    ``hard_overlap`` overrides that constrain c to a region with little or no
    disk overlap."""
    for _ in range(max_attempts):
        c_re = float(rng.uniform(c_re_lo, c_re_hi))
        c_im = float(rng.uniform(c_im_lo, c_im_hi))
        if c_re * c_re + c_im * c_im <= 1.5 * 1.5 + 1e-9:
            return c_re, c_im
    # Fallback path — preserves determinism by consuming the same number of
    # rng calls as one rectangle draw.
    c_re = float(rng.uniform(c_re_lo, c_re_hi))
    c_im = float(rng.uniform(c_im_lo, c_im_hi))
    return c_re, c_im
