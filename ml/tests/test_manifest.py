"""Manifest hashing + write/read round-trip."""

from __future__ import annotations

import json
from pathlib import Path

from fractalov_ml.dataset_config import parse_dataset_config
from fractalov_ml.manifest import (
    MANIFEST_FILENAME,
    read_manifest,
    sha256_of_file,
    sha256_of_image_tree,
    write_manifest,
)


def _minimal_dict() -> dict:
    return {
        "name": "manifest-test",
        "seed": 1,
        "render": {"width_px": 16, "height_px": 16, "samples_per_axis": 1},
        "palettes": ["fire"],
        "color_modes": ["linear"],
        "families": [{"name": "mandelbrot", "ranges": {"max_iter": [50, 60], "span": [2.0, 2.0]}}],
        "splits": [{"name": "train", "examples_per_family": 1}],
    }


def test_sha256_of_image_tree_invariant_under_rerun(tmp_path: Path) -> None:
    images = tmp_path / "images"
    (images / "a").mkdir(parents=True)
    (images / "a" / "1.png").write_bytes(b"\x89PNGabc")
    (images / "a" / "2.png").write_bytes(b"\x89PNGdef")

    h1, rows1 = sha256_of_image_tree(images)
    h2, rows2 = sha256_of_image_tree(images)
    assert h1 == h2
    assert rows1 == rows2
    # Sorted, with 2 entries.
    assert [r["relative_path"] for r in rows1] == ["a/1.png", "a/2.png"]


def test_sha256_of_image_tree_changes_when_file_changes(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "x.png").write_bytes(b"v1")
    h1, _ = sha256_of_image_tree(images)
    (images / "x.png").write_bytes(b"v2")
    h2, _ = sha256_of_image_tree(images)
    assert h1 != h2


def test_write_manifest_round_trip(tmp_path: Path) -> None:
    cfg = parse_dataset_config(_minimal_dict())
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.png").write_bytes(b"PNGSTUB")

    parquet = tmp_path / "labels.parquet"
    parquet.write_bytes(b"PARQUETSTUB")

    written = write_manifest(
        tmp_path,
        dataset_name=cfg.name,
        config_dict=cfg.to_dict(),
        config_sha256=cfg.config_sha256(),
        rng_seed=cfg.seed,
        backend_url="http://localhost:8080",
        backend_health={"status": "OK"},
        per_split_counts={"train": {"mandelbrot": 1}},
        failed=0,
        parquet_path=parquet,
        images_root=images,
    )
    assert written.name == MANIFEST_FILENAME
    loaded = read_manifest(tmp_path)

    assert loaded["dataset_name"] == "manifest-test"
    assert loaded["rng_seed"] == 1
    assert loaded["config_sha256"] == cfg.config_sha256()
    assert loaded["counts"]["successful_total"] == 1
    assert loaded["counts"]["failed_total"] == 0
    assert loaded["artifacts"]["labels_parquet_sha256"] == sha256_of_file(parquet)
    # Hashes are non-empty hex strings.
    assert len(loaded["artifacts"]["images_root_sha256"]) == 64


def test_manifest_byte_identical_for_byte_identical_input(tmp_path: Path) -> None:
    """If two builds produce identical config + images + parquet, their
    artifact hashes must agree. The timestamp differs (it is the
    ``timestamp_utc`` field), so we compare the hash subset."""
    cfg = parse_dataset_config(_minimal_dict())
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.png").write_bytes(b"PNGSTUB")
    parquet = tmp_path / "labels.parquet"
    parquet.write_bytes(b"PARQUETSTUB")

    args = dict(
        dataset_name=cfg.name,
        config_dict=cfg.to_dict(),
        config_sha256=cfg.config_sha256(),
        rng_seed=cfg.seed,
        backend_url="http://localhost:8080",
        backend_health=None,
        per_split_counts={"train": {"mandelbrot": 1}},
        failed=0,
        parquet_path=parquet,
        images_root=images,
    )
    write_manifest(tmp_path, **args)
    a = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    write_manifest(tmp_path, **args)
    b = json.loads((tmp_path / MANIFEST_FILENAME).read_text())

    assert a["config_sha256"] == b["config_sha256"]
    assert a["artifacts"] == b["artifacts"]
    assert a["counts"] == b["counts"]
