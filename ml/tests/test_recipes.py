"""Recipe sampler is the only generation-side code that's reproducible without a
running backend, so it gets the most thorough offline coverage."""

from __future__ import annotations

import numpy as np
import pytest

from fractalov_ml.recipes import FAMILIES, SampleConfig, sample


@pytest.fixture
def cfg() -> SampleConfig:
    return SampleConfig(width_px=128, height_px=128, samples_per_axis=1)


@pytest.mark.parametrize("family", FAMILIES)
def test_sampler_returns_backend_compatible_shape(cfg: SampleConfig, family: str) -> None:
    rng = np.random.default_rng(0)
    recipe = sample(family, rng, cfg)

    assert recipe["fractalType"] == family
    for key in ("viewport", "renderSettings", "colorSettings", "params"):
        assert key in recipe
    vp = recipe["viewport"]
    assert vp["xMax"] > vp["xMin"]
    assert vp["yMax"] > vp["yMin"]

    rs = recipe["renderSettings"]
    assert rs["widthPx"] == cfg.width_px
    assert rs["heightPx"] == cfg.height_px
    assert 1 <= rs["samplesPerAxis"] <= 3

    params = recipe["params"]
    assert cfg.max_iter_min <= params["maxIter"] <= cfg.max_iter_max
    assert params["escapeRadius"] > 0
    assert isinstance(params["smoothing"], bool)


def test_julia_c_lies_in_disk(cfg: SampleConfig) -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        recipe = sample("julia", rng, cfg)
        c_re = recipe["params"]["cRe"]
        c_im = recipe["params"]["cIm"]
        assert c_re * c_re + c_im * c_im <= 1.5 * 1.5 + 1e-9


def test_multibrot_exponent_in_range(cfg: SampleConfig) -> None:
    rng = np.random.default_rng(2)
    for _ in range(100):
        recipe = sample("multibrot", rng, cfg)
        assert 2 <= recipe["params"]["exponent"] <= 6


def test_seeded_streams_are_reproducible(cfg: SampleConfig) -> None:
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    for fam in FAMILIES:
        assert sample(fam, rng_a, cfg) == sample(fam, rng_b, cfg)


def test_unknown_family_rejected(cfg: SampleConfig) -> None:
    with pytest.raises(ValueError, match="unknown family"):
        sample("not-a-family", np.random.default_rng(0), cfg)
