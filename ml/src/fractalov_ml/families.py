"""Single source of truth for fractal family identity in the Python side.

Two contracts that have to stay aligned across three runtimes:

* **Wire string.** The lowercase value that appears in every JSON envelope —
  ``"mandelbrot"``, ``"julia"``, ``"burning_ship"``, ``"multibrot"``.
  The Java backend uses ``@JsonTypeInfo(EXTERNAL_PROPERTY, property="fractalType")``
  on these exact strings; the Android client mirrors them via
  ``@SerialName(...)`` on its sealed ``FractalParams`` hierarchy.
* **Encoder index.** Alphabetic order on the wire string, used by every
  ML head that emits family probabilities. The order is fixed by
  :class:`fractalov_ml.dataset.FamilyEncoder` so a model trained on one
  dataset can be loaded against another with a different family ratio.

Until this module existed the Python side carried bare strings tuples in
:mod:`recipes`, :mod:`recipes_v2`, :mod:`build_dataset`, and the dataset
config — easy to drift one of them and not notice. Now everywhere we need
"the four family identifiers" we go through :class:`FractalFamily`, and
the matching wire strings stay generated from a single ``str``-valued
enum.
"""

from __future__ import annotations

from enum import Enum


class FractalFamily(str, Enum):
    """The four families this project understands. ``str`` mixin so an
    instance compares equal to its wire string and JSON-serialises
    naturally as the bare string, matching the cross-language contract."""

    MANDELBROT = "mandelbrot"
    JULIA = "julia"
    BURNING_SHIP = "burning_ship"
    MULTIBROT = "multibrot"

    @classmethod
    def all(cls) -> tuple["FractalFamily", ...]:
        """Stable alphabetic ordering on the wire string. Matches the
        :class:`fractalov_ml.dataset.FamilyEncoder` ``classes`` attribute."""
        return tuple(sorted(cls, key=lambda f: f.value))

    @classmethod
    def wire_strings(cls) -> tuple[str, ...]:
        """The four wire strings in alphabetic order. Replacement for the
        old loose ``FAMILIES`` tuple that lived in :mod:`recipes`."""
        return tuple(f.value for f in cls.all())

    @classmethod
    def from_wire(cls, value: str) -> "FractalFamily":
        try:
            return cls(value)
        except ValueError as ex:
            allowed = ", ".join(f.value for f in cls.all())
            raise ValueError(f"unknown family wire string: {value!r} (allowed: {allowed})") from ex

    @property
    def supports_distance_estimate(self) -> bool:
        """Burning Ship is non-holomorphic; the standard distance
        estimator does not apply. Used by samplers to force-fallback to
        ``linear`` when colour mode is ``distance_estimate``."""
        return self is not FractalFamily.BURNING_SHIP


# Convenience alias for callers that just want the strings (most common
# import target). New callers should prefer FractalFamily directly.
FAMILY_NAMES: tuple[str, ...] = FractalFamily.wire_strings()
