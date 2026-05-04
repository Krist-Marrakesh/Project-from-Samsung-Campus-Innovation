"""Dataset generation driver.

Layout produced under ``--output-dir``::

    output/
      images/<family>/<id>.png      one PNG per example
      labels.parquet                one row per example (see Row spec below)
      generation_meta.json          run-level metadata: seed, backend, sample_config

Row spec (parquet columns):
  id, family, fractal_type, c_re, c_im, exponent, max_iter, escape_radius,
  smoothing, vp_x_min, vp_x_max, vp_y_min, vp_y_max, palette, color_mode,
  samples_per_axis, image_path (relative to output dir),
  perf_render_ms, perf_colorize_ms, perf_encode_ms, perf_total_ms,
  file_size_bytes, request_id

Family-specific params live in nullable columns: c_re/c_im are populated only
for julia, exponent only for multibrot. This keeps the table flat and easy to
filter/groupby in pandas without schema gymnastics.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .client import BackendClient, BackendError
from .recipes import FAMILIES, SampleConfig, sample


def generate(
    output_dir: Path,
    per_family: int,
    seed: int = 42,
    backend_url: str = "http://localhost:8080",
    width_px: int = 128,
    height_px: int = 128,
    samples_per_axis: int = 1,
    families: Optional[list[str]] = None,
    console: Optional[Console] = None,
) -> Path:
    families = list(families) if families else list(FAMILIES)
    cfg = SampleConfig(
        width_px=width_px,
        height_px=height_px,
        samples_per_axis=samples_per_axis,
    )
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_root = output_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    console = console or Console()
    rows: list[dict] = []
    failed = 0
    total = per_family * len(families)

    with BackendClient(base_url=backend_url) as client, Progress(
        TextColumn("[bold]{task.fields[fam]:<13}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("dataset", total=total, fam="starting")
        for family in families:
            (images_root / family).mkdir(parents=True, exist_ok=True)
            progress.update(task, fam=family)
            for _ in range(per_family):
                recipe = sample(family, rng, cfg)
                example_id = uuid.uuid4().hex
                try:
                    rendered = client.render(recipe)
                except BackendError as ex:
                    failed += 1
                    console.log(f"[yellow]skip {family}/{example_id}: {ex}")
                    progress.advance(task)
                    continue

                rel_path = f"images/{family}/{example_id}.png"
                abs_path = output_dir / rel_path
                abs_path.write_bytes(rendered.image_bytes)

                rows.append(_row(example_id, family, recipe, rendered, rel_path))
                progress.advance(task)

    if not rows:
        raise RuntimeError("no successful renders — check backend is reachable and accepting recipes")

    df = pd.DataFrame(rows)
    labels_path = output_dir / "labels.parquet"
    df.to_parquet(labels_path, index=False)

    meta = {
        "seed": seed,
        "backend_url": backend_url,
        "per_family": per_family,
        "families": families,
        "sample_config": asdict(cfg),
        "successful": int(len(rows)),
        "failed": failed,
    }
    (output_dir / "generation_meta.json").write_text(json.dumps(meta, indent=2))

    console.print(
        f"[green]wrote {len(rows)} labels → {labels_path} "
        f"(failed={failed}, families={families})"
    )
    return labels_path


def _row(example_id: str, family: str, recipe: dict, rendered, rel_path: str) -> dict:
    params = recipe["params"]
    vp = recipe["viewport"]
    color = recipe["colorSettings"]
    rs = recipe["renderSettings"]
    perf = rendered.perf
    return {
        "id": example_id,
        "family": family,
        "fractal_type": recipe["fractalType"],
        "c_re": params.get("cRe"),
        "c_im": params.get("cIm"),
        "exponent": params.get("exponent"),
        "max_iter": params["maxIter"],
        "escape_radius": params["escapeRadius"],
        "smoothing": params["smoothing"],
        "vp_x_min": vp["xMin"],
        "vp_x_max": vp["xMax"],
        "vp_y_min": vp["yMin"],
        "vp_y_max": vp["yMax"],
        "palette": color["paletteName"],
        "color_mode": color.get("mode") or "linear",
        "samples_per_axis": rs.get("samplesPerAxis", 1),
        "image_path": rel_path,
        "perf_render_ms": int(perf.get("renderMs", 0)),
        "perf_colorize_ms": int(perf.get("colorizeMs", 0)),
        "perf_encode_ms": int(perf.get("encodeMs", 0)),
        "perf_total_ms": int(perf.get("totalMs", 0)),
        "file_size_bytes": len(rendered.image_bytes),
        "request_id": rendered.request_id,
    }
