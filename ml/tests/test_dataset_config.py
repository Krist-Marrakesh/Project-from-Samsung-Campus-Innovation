"""Offline tests for the YAML-driven dataset config + per-split sampler.

These tests do not need a running backend — they cover the parts of the
Slice 2 pipeline that decide *what* to generate, not the act of generation
itself. The contract under test is the reproducibility / determinism
guarantee that the manifest documents.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from fractalov_ml.dataset_config import (
    DatasetConfig,
    FamilyRanges,
    Range,
    load_dataset_config,
    parse_dataset_config,
    split_seed,
)
from fractalov_ml.recipes_v2 import iterate_split, split_rng


def _minimal_config_dict() -> dict:
    return {
        "name": "unit",
        "seed": 7,
        "render": {"width_px": 32, "height_px": 32, "samples_per_axis": 1},
        "palettes": ["fire"],
        "color_modes": ["linear"],
        "families": [
            {"name": "mandelbrot", "ranges": {"max_iter": [50, 60], "span": [2.0, 2.0]}},
            {"name": "julia", "ranges": {
                "max_iter": [50, 60], "span": [2.5, 2.5],
                "c_re": [-0.5, 0.5], "c_im": [-0.5, 0.5],
            }},
        ],
        "splits": [
            {"name": "train", "examples_per_family": 3},
            {"name": "val", "examples_per_family": 2},
            {
                "name": "hard_overlap",
                "examples_per_family": 1,
                "restrict_to_overrides": True,
                "family_overrides": {
                    "julia": {"c_re": [-0.05, 0.05], "c_im": [-0.05, 0.05]},
                },
            },
        ],
    }


def test_yaml_round_trip(tmp_path: Path) -> None:
    yaml = textwrap.dedent(
        """
        name: round-trip
        seed: 1
        render: {width_px: 16, height_px: 16, samples_per_axis: 1}
        palettes: [fire]
        color_modes: [linear]
        families:
          - {name: mandelbrot, ranges: {max_iter: [50, 60], span: [2.0, 2.0]}}
        splits:
          - {name: train, examples_per_family: 1}
        """
    ).strip()
    p = tmp_path / "c.yaml"
    p.write_text(yaml)
    cfg = load_dataset_config(p)
    assert cfg.name == "round-trip"
    assert cfg.family_names() == ("mandelbrot",)
    assert cfg.split_names() == ("train",)


def test_missing_required_fields_rejected() -> None:
    with pytest.raises(ValueError, match="missing required field: name"):
        parse_dataset_config({"seed": 1, "families": [], "splits": []})
    with pytest.raises(ValueError, match="missing required field: seed"):
        parse_dataset_config({"name": "x", "families": [], "splits": []})
    with pytest.raises(ValueError, match="at least one family"):
        parse_dataset_config({"name": "x", "seed": 1, "splits": [{"name": "train", "examples_per_family": 1}]})
    with pytest.raises(ValueError, match="at least one split"):
        parse_dataset_config({"name": "x", "seed": 1, "families": [{"name": "mandelbrot"}]})


def test_duplicate_family_or_split_rejected() -> None:
    raw = _minimal_config_dict()
    raw["families"].append({"name": "mandelbrot"})
    with pytest.raises(ValueError, match="duplicate family"):
        parse_dataset_config(raw)


def test_range_hi_must_be_geq_lo() -> None:
    with pytest.raises(ValueError, match="hi < lo"):
        Range(lo=1.0, hi=0.0)


def test_split_seed_is_deterministic_and_split_dependent() -> None:
    a1 = split_seed(42, "train")
    a2 = split_seed(42, "train")
    b1 = split_seed(42, "val")
    assert a1 == a2
    assert a1 != b1
    # Different base seed → different stream.
    assert split_seed(42, "train") != split_seed(43, "train")


def test_iterate_split_is_reproducible() -> None:
    cfg = parse_dataset_config(_minimal_config_dict())
    a = list(iterate_split(cfg, "train"))
    b = list(iterate_split(cfg, "train"))
    assert a == b


def test_reordering_splits_does_not_perturb_other_splits() -> None:
    raw1 = _minimal_config_dict()
    cfg1 = parse_dataset_config(raw1)
    train1 = list(iterate_split(cfg1, "train"))

    raw2 = _minimal_config_dict()
    raw2["splits"] = list(reversed(raw2["splits"]))   # swap split order
    cfg2 = parse_dataset_config(raw2)
    train2 = list(iterate_split(cfg2, "train"))

    assert train1 == train2


def test_restrict_to_overrides_emits_only_listed_families() -> None:
    cfg = parse_dataset_config(_minimal_config_dict())
    rows = list(iterate_split(cfg, "hard_overlap"))
    assert len(rows) == 1
    fam, _ = rows[0]
    assert fam == "julia"


def test_split_overrides_constrain_julia_c() -> None:
    cfg = parse_dataset_config(_minimal_config_dict())
    rows = list(iterate_split(cfg, "hard_overlap"))
    for fam, recipe in rows:
        if fam == "julia":
            c_re = recipe["params"]["cRe"]
            c_im = recipe["params"]["cIm"]
            # Override is [-0.05, 0.05]; sampler may fall back outside the
            # disk but stays inside the rectangle.
            assert -0.05 <= c_re <= 0.05
            assert -0.05 <= c_im <= 0.05


def test_train_max_iter_inside_declared_range() -> None:
    cfg = parse_dataset_config(_minimal_config_dict())
    for fam, recipe in iterate_split(cfg, "train"):
        max_iter = recipe["params"]["maxIter"]
        assert 50 <= max_iter <= 60, f"{fam} max_iter out of range: {max_iter}"


def test_recipes_match_backend_wire_shape() -> None:
    cfg = parse_dataset_config(_minimal_config_dict())
    expected_keys = {"viewport", "renderSettings", "colorSettings", "fractalType", "params"}
    for _, recipe in iterate_split(cfg, "train"):
        assert set(recipe.keys()) == expected_keys
        assert recipe["fractalType"] in {"mandelbrot", "julia"}
        vp = recipe["viewport"]
        assert vp["xMax"] > vp["xMin"]
        assert vp["yMax"] > vp["yMin"]


def test_config_sha256_stable_for_byte_identical_input() -> None:
    cfg_a = parse_dataset_config(_minimal_config_dict())
    cfg_b = parse_dataset_config(_minimal_config_dict())
    assert cfg_a.config_sha256() == cfg_b.config_sha256()


def test_config_sha256_changes_when_seed_changes() -> None:
    cfg_a = parse_dataset_config(_minimal_config_dict())
    raw = _minimal_config_dict()
    raw["seed"] = 999
    cfg_b = parse_dataset_config(raw)
    assert cfg_a.config_sha256() != cfg_b.config_sha256()


def test_split_rng_reproducible() -> None:
    cfg = parse_dataset_config(_minimal_config_dict())
    rng1 = split_rng(cfg, "train")
    rng2 = split_rng(cfg, "train")
    assert rng1.uniform() == rng2.uniform()


def test_burning_ship_de_falls_back_to_linear() -> None:
    raw = _minimal_config_dict()
    raw["families"].append({"name": "burning_ship", "ranges": {"max_iter": [50, 60], "span": [2.5, 2.5]}})
    raw["color_modes"] = ["distance_estimate"]
    raw["splits"] = [{"name": "train", "examples_per_family": 5}]
    cfg = parse_dataset_config(raw)
    saw_burning_ship = False
    for fam, recipe in iterate_split(cfg, "train"):
        if fam == "burning_ship":
            saw_burning_ship = True
            assert recipe["colorSettings"]["mode"] == "linear", \
                "burning_ship+DE must downgrade because DE is not defined for the family"
    assert saw_burning_ship


def test_research_v1_yaml_is_parseable() -> None:
    """The shipped config in configs/dataset/research-v1.yaml must always
    parse — this is what the build-dataset command points at by default."""
    path = Path(__file__).resolve().parents[1] / "configs" / "dataset" / "research-v1.yaml"
    cfg = load_dataset_config(path)
    assert cfg.name == "research-v1"
    assert set(cfg.split_names()) == {"train", "val", "test", "near_ood", "hard_overlap"}
    # Anchor: hash should be stable. We don't assert a specific hex (the
    # config evolves), but the hash must be 64 hex chars.
    h = cfg.config_sha256()
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
