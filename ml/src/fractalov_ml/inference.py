"""Stage 7 inference: turn an image into a recipe.

The trained :class:`fractalov_ml.models.FractalCNN` predicts (a) the fractal
family and (b) for Julia, the value of `c`. This module wraps that model into
a single :func:`predict_recipe` call that returns a dict matching the Java
backend's ``FractalRecipe`` wire shape — exactly what the backend's ``/render``
endpoint accepts.

Why a *recipe* instead of raw model output:
  * The mobile UI / Stage 8 workflow wants something it can immediately re-render
    or hand to the user as a starting point — predicting "this is julia and
    c≈(-0.7, 0.27)" is half the answer; the other half is "...so render it like
    this", which means filling in viewport, palette, max_iter, escape_radius,
    smoothing, color mode.
  * The non-julia families don't have a regressed parameter, so we synthesise
    sensible defaults (classic viewport per family) and let the user tweak.

The default heuristics here are intentionally simple — Stage 7 is about wiring
up inference, not nailing the perfect recipe. Stage 8 / 10 may add image-based
viewport regression to the model itself.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .dataset import FamilyEncoder
from .models import FractalCNN
from .training.config import pick_device


@dataclass
class Suggestion:
    family: str
    family_confidence: float
    family_distribution: dict[str, float]
    c_re: Optional[float]
    c_im: Optional[float]
    recipe: dict


# Per-family default viewport / param scaffolding. Matches the dataset
# generator's "around the classic window" sampling distribution so the
# returned recipe always renders something recognisable, even when the
# model only contributes the family label.
_DEFAULTS = {
    "mandelbrot": {
        "viewport": {"xMin": -2.0, "xMax": 1.0, "yMin": -1.2, "yMax": 1.2},
        "params": {"maxIter": 200, "escapeRadius": 2.0, "smoothing": True},
    },
    "julia": {
        "viewport": {"xMin": -1.5, "xMax": 1.5, "yMin": -1.5, "yMax": 1.5},
        # cRe / cIm are filled in from the model output.
        "params": {"maxIter": 200, "escapeRadius": 2.0, "smoothing": True},
    },
    "burning_ship": {
        "viewport": {"xMin": -2.0, "xMax": 1.5, "yMin": -2.0, "yMax": 1.0},
        "params": {"maxIter": 200, "escapeRadius": 2.0, "smoothing": True},
    },
    "multibrot": {
        "viewport": {"xMin": -1.5, "xMax": 1.5, "yMin": -1.5, "yMax": 1.5},
        # exponent is not regressed yet — pick a default that's visually
        # interesting (3 produces three-fold symmetry).
        "params": {"exponent": 3, "maxIter": 200, "escapeRadius": 2.0, "smoothing": True},
    },
}


class InferenceService:
    """Loads a checkpoint once, holds it on the chosen device, exposes
    ``predict_recipe`` and a few convenience accessors."""

    def __init__(
        self,
        checkpoint: Path,
        device_override: Optional[str] = None,
        default_palette: str = "fire",
        default_color_mode: str = "linear",
        default_width_px: int = 512,
        default_height_px: int = 512,
    ) -> None:
        self.device = pick_device(device_override)
        self.encoder = FamilyEncoder.default()
        self.default_palette = default_palette
        self.default_color_mode = default_color_mode
        self.default_width_px = default_width_px
        self.default_height_px = default_height_px

        # Same checkpoint-aware loader as eval.py: infer base_channels from
        # the stem conv weight so the run-time service rebuilds the right
        # capacity.
        state = torch.load(checkpoint, map_location=self.device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        base_channels = state["stem.0.weight"].shape[0]
        self.model = FractalCNN(base_channels=base_channels).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.checkpoint = Path(checkpoint).resolve()
        self.base_channels = base_channels

    @torch.no_grad()
    def predict_recipe(
        self,
        image_bytes: bytes,
        target_width_px: Optional[int] = None,
        target_height_px: Optional[int] = None,
    ) -> Suggestion:
        """Run inference on a single PNG (or any PIL-readable image)."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self.model(tensor)

        family_idx = int(out.family_logits.argmax(dim=1).item())
        family = self.encoder.decode(family_idx)
        probs = torch.softmax(out.family_logits[0], dim=0).tolist()
        family_dist = {self.encoder.decode(i): float(p) for i, p in enumerate(probs)}

        c_re_val, c_im_val = (float(x) for x in out.c_pred[0].tolist())
        # Only surface c when the model actually thinks it's julia — for other
        # families the c head's output is meaningless.
        emit_c = family == "julia"

        recipe = self._compose_recipe(
            family,
            c_re_val if emit_c else None,
            c_im_val if emit_c else None,
            target_width_px,
            target_height_px,
        )

        return Suggestion(
            family=family,
            family_confidence=float(probs[family_idx]),
            family_distribution=family_dist,
            c_re=c_re_val if emit_c else None,
            c_im=c_im_val if emit_c else None,
            recipe=recipe,
        )

    def _compose_recipe(
        self,
        family: str,
        c_re: Optional[float],
        c_im: Optional[float],
        width_px: Optional[int],
        height_px: Optional[int],
    ) -> dict:
        defaults = _DEFAULTS[family]
        params = dict(defaults["params"])
        if family == "julia":
            params["cRe"] = float(c_re if c_re is not None else 0.0)
            params["cIm"] = float(c_im if c_im is not None else 0.0)
        return {
            "viewport": dict(defaults["viewport"]),
            "renderSettings": {
                "widthPx": int(width_px or self.default_width_px),
                "heightPx": int(height_px or self.default_height_px),
                "samplesPerAxis": 1,
            },
            "colorSettings": {
                "paletteName": self.default_palette,
                "mode": self.default_color_mode,
            },
            "fractalType": family,
            "params": params,
        }


def variations_around(recipe: dict, count: int, seed: int = 42) -> list[dict]:
    """Return ``count`` recipes near ``recipe`` by jittering safe-to-perturb
    fields. This is intentionally rule-based rather than generative —
    Stage 7 explicitly scopes 'AI-assisted exploration' to the small,
    deterministic surface that doesn't require a second model.

    Perturbations:
      * viewport: zoom factor in [0.6, 1.4] and centre offset in [-15%, 15%]
      * params.maxIter: ±25%
      * params.cRe / cIm (julia only): ±0.05 each
      * colorSettings.paletteName: random pick from the four registered
        palettes (back-end will reject an unknown one anyway)
    """
    rng = np.random.default_rng(seed)
    palettes = ("grayscale", "fire", "ocean", "rainbow_cyclic")
    out = []
    for _ in range(count):
        cp = _deep_copy_recipe(recipe)
        _jitter_viewport(cp["viewport"], rng)
        _jitter_max_iter(cp["params"], rng)
        if cp["fractalType"] == "julia":
            _jitter_julia_c(cp["params"], rng)
        cp["colorSettings"]["paletteName"] = str(rng.choice(palettes))
        out.append(cp)
    return out


def _deep_copy_recipe(recipe: dict) -> dict:
    return {
        "viewport": dict(recipe["viewport"]),
        "renderSettings": dict(recipe["renderSettings"]),
        "colorSettings": dict(recipe["colorSettings"]),
        "fractalType": recipe["fractalType"],
        "params": dict(recipe["params"]),
    }


def _jitter_viewport(vp: dict, rng: np.random.Generator) -> None:
    cx = (vp["xMin"] + vp["xMax"]) / 2.0
    cy = (vp["yMin"] + vp["yMax"]) / 2.0
    span_x = vp["xMax"] - vp["xMin"]
    span_y = vp["yMax"] - vp["yMin"]

    zoom = float(rng.uniform(0.6, 1.4))
    dx = float(rng.uniform(-0.15, 0.15) * span_x)
    dy = float(rng.uniform(-0.15, 0.15) * span_y)

    new_span_x = span_x * zoom
    new_span_y = span_y * zoom
    new_cx = cx + dx
    new_cy = cy + dy
    vp["xMin"] = new_cx - new_span_x / 2.0
    vp["xMax"] = new_cx + new_span_x / 2.0
    vp["yMin"] = new_cy - new_span_y / 2.0
    vp["yMax"] = new_cy + new_span_y / 2.0


def _jitter_max_iter(params: dict, rng: np.random.Generator) -> None:
    factor = float(rng.uniform(0.75, 1.25))
    params["maxIter"] = max(50, int(params["maxIter"] * factor))


def _jitter_julia_c(params: dict, rng: np.random.Generator) -> None:
    params["cRe"] = float(params["cRe"] + rng.uniform(-0.05, 0.05))
    params["cIm"] = float(params["cIm"] + rng.uniform(-0.05, 0.05))
