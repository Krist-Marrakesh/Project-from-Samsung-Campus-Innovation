"""Config-driven dataset builder.

Replaces the older ``generate.py`` + ``splits.py`` two-step pipeline with a
single command driven by :mod:`dataset_config`. Output layout::

    output_dir/
      images/<split>/<family>/<id>.png      one PNG per example
      labels.parquet                        one row per example, with `split`
      dataset-manifest.json                 frozen-artifact manifest
      config-resolved.json                  the full config that ran (after defaults)

The ``split`` column lives in ``labels.parquet`` itself rather than in a
separate ``splits.parquet`` — there is no scenario in this pipeline where
the same labels would be re-split multiple ways, and folding them removes a
join from the dataloader path.

Failure model: per-example backend errors are logged and skipped, recorded
in ``manifest.counts.failed_total``. The build does not abort on transient
errors so a 5000-example run does not lose 4999 of them to one network
hiccup. Hard errors (config parse failure, IO error on the output directory,
no successful renders at all) raise.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

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
from .dataset_config import DatasetConfig, load_dataset_config
from .manifest import write_manifest
from .recipes_v2 import iterate_split


def build_dataset(
    config_path: Path,
    output_dir: Path,
    backend_url: str = "http://localhost:8080",
    console: Console | None = None,
) -> Path:
    config = load_dataset_config(config_path)
    return build_from_config(
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        backend_url=backend_url,
        console=console,
    )


def build_from_config(
    *,
    config: DatasetConfig,
    config_path: Path | None,
    output_dir: Path,
    backend_url: str = "http://localhost:8080",
    console: Console | None = None,
) -> Path:
    console = console or Console()
    output_dir.mkdir(parents=True, exist_ok=True)
    images_root = output_dir / "images"

    # Persist the resolved config alongside the manifest so a reviewer can
    # diff easily without parsing the manifest itself.
    resolved_dict = config.to_dict()
    (output_dir / "config-resolved.json").write_text(
        json.dumps(resolved_dict, indent=2, sort_keys=False)
    )

    rows: list[dict] = []
    per_split_counts: dict[str, dict[str, int]] = {}
    failed = 0

    backend_health = _try_health(backend_url, console)

    total = sum(
        s.examples_per_family * _family_count(config, s.name)
        for s in config.splits
    )

    with BackendClient(base_url=backend_url) as client, Progress(
        TextColumn("[bold]{task.fields[split]:<14}"),
        TextColumn("{task.fields[family]:<13}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("dataset", total=total, split="starting", family="")
        for split_name in config.split_names():
            split_counts = per_split_counts.setdefault(split_name, {})
            for family, recipe in iterate_split(config, split_name):
                progress.update(task, split=split_name, family=family)
                example_id = uuid.uuid4().hex
                try:
                    rendered = client.render(recipe)
                except BackendError as ex:
                    failed += 1
                    console.log(f"[yellow]skip {split_name}/{family}/{example_id}: {ex}")
                    progress.advance(task)
                    continue

                rel_path = f"images/{split_name}/{family}/{example_id}.png"
                abs_path = output_dir / rel_path
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_bytes(rendered.image_bytes)

                rows.append(_row(example_id, split_name, family, recipe, rendered, rel_path))
                split_counts[family] = split_counts.get(family, 0) + 1
                progress.advance(task)

    if not rows:
        raise RuntimeError(
            "no successful renders — check backend is reachable and accepting recipes"
        )

    df = pd.DataFrame(rows)
    labels_path = output_dir / "labels.parquet"
    df.to_parquet(labels_path, index=False)

    manifest_path = write_manifest(
        output_dir,
        dataset_name=config.name,
        config_dict=resolved_dict,
        config_sha256=config.config_sha256(),
        rng_seed=config.seed,
        backend_url=backend_url,
        backend_health=backend_health,
        per_split_counts=per_split_counts,
        failed=failed,
        parquet_path=labels_path,
        images_root=images_root,
        extra={
            "config_path": str(config_path) if config_path else None,
        },
    )

    console.print(
        f"[green]wrote {len(rows)} labels → {labels_path} "
        f"(failed={failed}, manifest={manifest_path.name})"
    )
    return labels_path


def _family_count(config: DatasetConfig, split_name: str) -> int:
    sp = config.split(split_name)
    if sp.restrict_to_overrides:
        # Only families with explicit overrides actually emit examples.
        return len(sp.family_overrides)
    return len(config.families)


def _row(
    example_id: str,
    split: str,
    family: str,
    recipe: dict,
    rendered,
    rel_path: str,
) -> dict:
    params = recipe["params"]
    vp = recipe["viewport"]
    color = recipe["colorSettings"]
    rs = recipe["renderSettings"]
    perf = rendered.perf
    return {
        "id": example_id,
        "split": split,
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


def _try_health(backend_url: str, console: Console) -> dict[str, Any] | None:
    """Best-effort GET /health. Logged into the manifest for traceability —
    if the backend was misconfigured during the run we want it on record."""
    import httpx  # local import keeps module import-time cheap
    try:
        resp = httpx.get(f"{backend_url}/health", timeout=5.0)
        if resp.status_code == 200:
            return resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}
    except httpx.HTTPError as ex:
        console.log(f"[yellow]health probe failed: {ex}")
    return None
