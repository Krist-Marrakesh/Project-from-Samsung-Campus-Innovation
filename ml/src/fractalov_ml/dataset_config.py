"""Config-driven dataset specification.

Replaces the older "fixed defaults + CLI flags" generation model with a single
YAML file that fully describes a dataset run. The YAML, plus the
``manifest.py``-emitted manifest, are what makes a dataset a *frozen artifact*
rather than a side effect of a script invocation.

A config defines:

* the sampling families and their per-family parameter ranges,
* one or more named *splits*, each with its own example count and (optional)
  per-family range overrides,
* render settings (resolution, SSAA),
* palette / colour-mode pools,
* the master seed.

The five canonical split names —
``train``, ``val``, ``test``, ``near_ood``, ``hard_overlap`` — are not
hardcoded; any string is accepted. The interpretation of ``near_ood`` and
``hard_overlap`` is encoded in the config (parameter range overrides), not in
this module. That keeps the config file the *complete* specification: a
reviewer reading the YAML knows exactly what each split contains.

Reproducibility contract:

* The base ``seed`` plus the split name fully determines the per-split RNG
  stream. Reordering splits in the YAML or skipping one does not perturb the
  others.
* The full config dict (after defaults are filled in) is hashed into the
  manifest. Two configs that hash identically must produce byte-identical
  recipe streams.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

DEFAULT_FAMILIES = ("mandelbrot", "julia", "burning_ship", "multibrot")
DEFAULT_PALETTES = ("grayscale", "fire", "ocean", "rainbow_cyclic")
DEFAULT_COLOR_MODES = ("linear", "histogram", "distance_estimate")


@dataclass(frozen=True)
class Range:
    """Inclusive numeric range. ``int`` ranges keep their type; floats stay floats."""
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            raise ValueError(f"range hi < lo: lo={self.lo}, hi={self.hi}")

    @classmethod
    def parse(cls, raw: Any) -> "Range":
        if isinstance(raw, Range):
            return raw
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return cls(lo=raw[0], hi=raw[1])
        if isinstance(raw, dict) and {"lo", "hi"} <= raw.keys():
            return cls(lo=raw["lo"], hi=raw["hi"])
        raise ValueError(f"cannot parse Range from {raw!r}")


@dataclass(frozen=True)
class FamilyRanges:
    """Per-family parameter ranges. Fields not relevant to the family are ignored
    by the sampler — e.g. ``c_re_range`` only matters for julia.

    The fields are *all* optional at this level so a split can override only
    what it needs. Final ranges are produced by merging defaults → family
    section → split override (split wins).
    """
    max_iter: Range | None = None
    span: Range | None = None
    # Julia-specific
    c_re: Range | None = None
    c_im: Range | None = None
    # Multibrot-specific
    exponent: Range | None = None

    def merged_with(self, other: "FamilyRanges") -> "FamilyRanges":
        """Override fields with non-None values from ``other``."""
        return FamilyRanges(
            max_iter=other.max_iter or self.max_iter,
            span=other.span or self.span,
            c_re=other.c_re or self.c_re,
            c_im=other.c_im or self.c_im,
            exponent=other.exponent or self.exponent,
        )

    @classmethod
    def parse(cls, raw: dict[str, Any] | None) -> "FamilyRanges":
        if not raw:
            return cls()
        return cls(
            max_iter=Range.parse(raw["max_iter"]) if "max_iter" in raw else None,
            span=Range.parse(raw["span"]) if "span" in raw else None,
            c_re=Range.parse(raw["c_re"]) if "c_re" in raw else None,
            c_im=Range.parse(raw["c_im"]) if "c_im" in raw else None,
            exponent=Range.parse(raw["exponent"]) if "exponent" in raw else None,
        )


@dataclass(frozen=True)
class FamilySpec:
    """One family's defaults. Used as the merge base for per-split overrides."""
    name: str
    ranges: FamilyRanges
    # Render-time toggles. Constants for now; promote to ranges if a split
    # ever needs to vary them.
    escape_radius: float = 2.0
    smoothing: bool = True


@dataclass(frozen=True)
class SplitSpec:
    """One split. ``examples_per_family`` resolves an integer count for every
    family in the dataset; ``family_overrides`` lets a split widen / shift
    ranges for selected families."""
    name: str
    examples_per_family: int
    family_overrides: dict[str, FamilyRanges] = field(default_factory=dict)
    # If True, only the families listed in family_overrides are emitted
    # (others get zero examples). Used by hard_overlap/near_ood to focus on
    # the families where the override is meaningful.
    restrict_to_overrides: bool = False

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "SplitSpec":
        overrides_raw = raw.get("family_overrides") or {}
        return cls(
            name=raw["name"],
            examples_per_family=int(raw["examples_per_family"]),
            family_overrides={
                fam: FamilyRanges.parse(rng) for fam, rng in overrides_raw.items()
            },
            restrict_to_overrides=bool(raw.get("restrict_to_overrides", False)),
        )


@dataclass(frozen=True)
class RenderSpec:
    width_px: int = 128
    height_px: int = 128
    samples_per_axis: int = 1


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    seed: int
    families: tuple[FamilySpec, ...]
    splits: tuple[SplitSpec, ...]
    render: RenderSpec
    palettes: tuple[str, ...]
    color_modes: tuple[str, ...]

    def family(self, name: str) -> FamilySpec:
        for f in self.families:
            if f.name == name:
                return f
        raise KeyError(f"unknown family: {name}")

    def split(self, name: str) -> SplitSpec:
        for s in self.splits:
            if s.name == name:
                return s
        raise KeyError(f"unknown split: {name}")

    def family_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.families)

    def split_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.splits)

    def to_dict(self) -> dict[str, Any]:
        """Canonicalised dict — used by config_sha256 and JSON round-trip."""
        return {
            "name": self.name,
            "seed": self.seed,
            "families": [
                {
                    "name": f.name,
                    "escape_radius": f.escape_radius,
                    "smoothing": f.smoothing,
                    "ranges": _ranges_to_dict(f.ranges),
                }
                for f in self.families
            ],
            "splits": [
                {
                    "name": s.name,
                    "examples_per_family": s.examples_per_family,
                    "restrict_to_overrides": s.restrict_to_overrides,
                    "family_overrides": {
                        fam: _ranges_to_dict(rng)
                        for fam, rng in s.family_overrides.items()
                    },
                }
                for s in self.splits
            ],
            "render": asdict(self.render),
            "palettes": list(self.palettes),
            "color_modes": list(self.color_modes),
        }

    def config_sha256(self) -> str:
        """Stable hash of the canonicalised config. Two configs with the same
        hash are guaranteed to yield byte-identical recipe streams (given the
        same generator code version)."""
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _ranges_to_dict(r: FamilyRanges) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for k, v in (
        ("max_iter", r.max_iter),
        ("span", r.span),
        ("c_re", r.c_re),
        ("c_im", r.c_im),
        ("exponent", r.exponent),
    ):
        if v is not None:
            out[k] = [v.lo, v.hi]
    return out


def load_dataset_config(path: Path | str) -> DatasetConfig:
    """Load and validate a YAML dataset config from disk."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    return parse_dataset_config(raw)


def parse_dataset_config(raw: dict[str, Any]) -> DatasetConfig:
    """Build a DatasetConfig from an in-memory dict (parsed YAML / JSON / test
    fixture). Centralised so the validation logic does not split between
    on-disk and in-memory paths."""
    if "name" not in raw:
        raise ValueError("dataset config missing required field: name")
    if "seed" not in raw:
        raise ValueError("dataset config missing required field: seed")

    families_raw = raw.get("families")
    if not families_raw:
        raise ValueError("dataset config must declare at least one family")
    families: list[FamilySpec] = []
    seen: set[str] = set()
    for f in families_raw:
        name = f["name"]
        if name in seen:
            raise ValueError(f"duplicate family in config: {name}")
        seen.add(name)
        families.append(
            FamilySpec(
                name=name,
                ranges=FamilyRanges.parse(f.get("ranges")),
                escape_radius=float(f.get("escape_radius", 2.0)),
                smoothing=bool(f.get("smoothing", True)),
            )
        )

    splits_raw = raw.get("splits")
    if not splits_raw:
        raise ValueError("dataset config must declare at least one split")
    splits: list[SplitSpec] = []
    seen_split: set[str] = set()
    for s in splits_raw:
        if s["name"] in seen_split:
            raise ValueError(f"duplicate split in config: {s['name']}")
        seen_split.add(s["name"])
        splits.append(SplitSpec.parse(s))

    render_raw = raw.get("render") or {}
    render = RenderSpec(
        width_px=int(render_raw.get("width_px", 128)),
        height_px=int(render_raw.get("height_px", 128)),
        samples_per_axis=int(render_raw.get("samples_per_axis", 1)),
    )

    palettes = tuple(raw.get("palettes") or DEFAULT_PALETTES)
    color_modes = tuple(raw.get("color_modes") or DEFAULT_COLOR_MODES)

    return DatasetConfig(
        name=raw["name"],
        seed=int(raw["seed"]),
        families=tuple(families),
        splits=tuple(splits),
        render=render,
        palettes=palettes,
        color_modes=color_modes,
    )


def split_seed(base_seed: int, split_name: str) -> int:
    """Deterministic per-split seed. Built from the base seed and a stable
    32-bit hash of the split name. Reordering splits in the YAML cannot
    perturb other splits' streams.

    Implementation note: we use SHA-256 of the split name rather than
    Python's ``hash()`` because Python's hash is randomised per interpreter
    process unless ``PYTHONHASHSEED=0`` is set."""
    digest = hashlib.sha256(split_name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big")
    # numpy's default_rng accepts arbitrary uint64; we mask to keep it
    # deterministic across architectures.
    return (base_seed + offset) & 0xFFFFFFFF


def merged_ranges(config: DatasetConfig, split_name: str, family_name: str) -> FamilyRanges:
    """Final per-(split, family) ranges after applying the override chain:
    family defaults → split override (override wins on present fields)."""
    base = config.family(family_name).ranges
    sp = config.split(split_name)
    override = sp.family_overrides.get(family_name)
    return base.merged_with(override) if override else base
