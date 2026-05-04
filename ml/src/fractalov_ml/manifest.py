"""Frozen-artifact manifest.

The manifest is what makes a generated dataset *identifiable* — a hash chain
that pins down the exact configuration, generator code version, backend
target, and on-disk byte content. Two datasets with matching
``dataset-manifest.json`` files are guaranteed to be the same data, modulo
the trivial renaming of the directory they live in.

Manifest fields:

* ``dataset_name`` / ``timestamp_utc`` — human-readable identity.
* ``config_sha256`` — hash of the canonicalised YAML config. If this matches,
  the recipe stream is identical (assuming generator code version matches).
* ``generator_version`` — pinned to the package version. Bump the package
  version when changing recipe-shape semantics so old manifests don't
  silently match new (incompatible) datasets.
* ``backend_url`` / ``backend_health`` — what backend produced the renders.
* ``per_split_counts`` — actual rendered example counts per split / family
  (after any failed renders).
* ``parquet_sha256`` — hash of ``labels.parquet``.
* ``images_root_sha256`` — hash of the sorted (relative_path, sha256)
  list of every PNG. Catches missing / extra / corrupted images that a
  parquet hash alone would not.
* ``rng_seed`` — repeated for convenience; also lives inside the embedded
  config block.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__ as _PKG_VERSION

MANIFEST_FILENAME = "dataset-manifest.json"


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a single file. Used for individual PNGs and the
    labels parquet."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_image_tree(images_root: Path) -> tuple[str, list[dict[str, str]]]:
    """Hash the entire image tree. Returns (root_hash, manifest_rows)
    where each row is ``{relative_path, sha256, size_bytes}``. The root hash
    is SHA-256 of the deterministic byte representation of the rows so it is
    invariant under filesystem listing order but changes if any individual
    file changes."""
    if not images_root.exists():
        return hashlib.sha256(b"").hexdigest(), []
    rows: list[dict[str, str]] = []
    for path in sorted(images_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(images_root).as_posix()
        rows.append({
            "relative_path": rel,
            "sha256": sha256_of_file(path),
            "size_bytes": str(path.stat().st_size),
        })
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    root_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return root_hash, rows


def write_manifest(
    output_dir: Path,
    *,
    dataset_name: str,
    config_dict: dict[str, Any],
    config_sha256: str,
    rng_seed: int,
    backend_url: str,
    backend_health: dict[str, Any] | None,
    per_split_counts: dict[str, dict[str, int]],
    failed: int,
    parquet_path: Path | None,
    images_root: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``dataset-manifest.json`` into ``output_dir``. Returns the
    manifest path. Idempotent — overwriting is allowed, and the new manifest
    fully captures the new state of the directory."""
    parquet_hash = sha256_of_file(parquet_path) if parquet_path and parquet_path.exists() else None
    images_root_hash, _ = sha256_of_image_tree(images_root)

    manifest: dict[str, Any] = {
        "dataset_name": dataset_name,
        "generator_version": _PKG_VERSION,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "rng_seed": rng_seed,
        "config_sha256": config_sha256,
        "config": config_dict,
        "backend": {
            "url": backend_url,
            "health": backend_health,
        },
        "counts": {
            "successful_total": sum(
                sum(by_family.values()) for by_family in per_split_counts.values()
            ),
            "failed_total": failed,
            "per_split": per_split_counts,
        },
        "artifacts": {
            "labels_parquet_sha256": parquet_hash,
            "images_root_sha256": images_root_hash,
        },
    }
    if extra:
        manifest["extra"] = extra

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    return path


def read_manifest(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / MANIFEST_FILENAME).read_text())
