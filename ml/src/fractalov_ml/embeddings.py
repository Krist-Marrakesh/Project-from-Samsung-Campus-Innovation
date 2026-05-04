"""Embedding bank build + retrieval — the public API for Slice 4.

Two operations:

* :func:`build_embedding_bank` — runs the trained model over every example
  in chosen splits, writes a single ``embeddings.npz`` containing
  ``ids`` (str array), ``families`` (str array), ``splits`` (str array),
  and ``embeddings`` (float32, shape ``(N, D)``). The ``.npz`` plus the
  dataset's ``labels.parquet`` is everything a retrieval consumer needs.

* :func:`retrieve_topk` — given a unit-norm query embedding, returns the
  top-K nearest entries by cosine similarity. Cosine equals dot product on
  unit-norm vectors, which is exactly the geometry the SupCon loss
  produces, so the metric used for retrieval matches the metric used for
  training.

We use NumPy for storage + retrieval rather than FAISS / sklearn-NN. At
~10⁴ images × 128-D embeddings the bank fits in ~5 MB; the all-pairs dot
product in NumPy is microseconds. Adding FAISS would be premature: it
brings a build-time dep on a C++ library, OS-dependent wheels, and zero
benefit at this scale. Swap in if/when the bank crosses 10⁶ rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich.console import Console
from torch.utils.data import DataLoader

from .dataset import FamilyEncoder
from .dataset_v2 import FractalDatasetV2, collate
from .models.cnn_v2 import FractalCNNv2
from .training.config import pick_device


BANK_FILENAME = "embeddings.npz"


@dataclass(frozen=True)
class EmbeddingBank:
    """In-memory representation of the embedding bank.

    All arrays are aligned: ``embeddings[i]``, ``ids[i]``, ``families[i]``,
    ``splits[i]`` describe the same example.
    """
    ids: np.ndarray              # (N,) str
    families: np.ndarray         # (N,) str
    splits: np.ndarray           # (N,) str
    embeddings: np.ndarray       # (N, D) float32, L2-normalised

    @property
    def n(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            ids=self.ids,
            families=self.families,
            splits=self.splits,
            embeddings=self.embeddings.astype(np.float32, copy=False),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "EmbeddingBank":
        data = np.load(path, allow_pickle=False)
        return cls(
            ids=data["ids"].astype(str),
            families=data["families"].astype(str),
            splits=data["splits"].astype(str),
            embeddings=data["embeddings"].astype(np.float32, copy=False),
        )


def build_embedding_bank(
    *,
    data_root: Path,
    checkpoint: Path,
    splits: Iterable[str] = ("train", "val", "test"),
    out_path: Optional[Path] = None,
    base_channels: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    batch_size: int = 64,
    device_override: Optional[str] = None,
    console: Optional[Console] = None,
) -> EmbeddingBank:
    """Run the model over every example in chosen splits and persist the
    resulting embedding bank to ``out_path`` (default
    ``data_root/embeddings.npz``)."""
    console = console or Console()
    device = pick_device(device_override)
    out_path = out_path or (data_root / BANK_FILENAME)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    bc = base_channels or int(cfg.get("base_channels", 32))
    ed = embedding_dim or 128
    model = FractalCNNv2(base_channels=bc, embedding_dim=ed).to(device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    all_ids: list[str] = []
    all_families: list[str] = []
    all_splits: list[str] = []
    chunks: list[np.ndarray] = []

    for split_name in splits:
        try:
            ds = FractalDatasetV2(data_root, split=split_name)
        except (FileNotFoundError, ValueError) as ex:
            console.log(f"[yellow]skip split {split_name}: {ex}")
            continue
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
        )
        # The Dataset doesn't carry the parquet ``id`` column — it lives in
        # labels.parquet via ``image_path`` -> id stem. Take the slice of
        # rows the dataset references and join in-order.
        ids_for_split = ds.labels["id"].astype(str).tolist()
        idx = 0
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                emb = model(images).embedding   # already unit-norm
                chunks.append(emb.detach().cpu().numpy())
                bs = images.size(0)
                all_ids.extend(ids_for_split[idx: idx + bs])
                all_families.extend(batch["family_strs"])
                all_splits.extend([split_name] * bs)
                idx += bs
        console.log(f"[dim]embedded split={split_name} n={len(ds)}[/dim]")

    if not chunks:
        raise RuntimeError("no embeddings produced — were any splits available?")

    embeddings = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    bank = EmbeddingBank(
        ids=np.asarray(all_ids, dtype=object).astype(str),
        families=np.asarray(all_families, dtype=object).astype(str),
        splits=np.asarray(all_splits, dtype=object).astype(str),
        embeddings=embeddings,
    )
    bank.save(out_path)
    console.print(f"[green]wrote bank n={bank.n} dim={bank.dim} → {out_path}")
    return bank


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    id: str
    family: str
    split: str
    similarity: float


def retrieve_topk(
    bank: EmbeddingBank,
    query: np.ndarray,
    *,
    k: int = 10,
    exclude_id: Optional[str] = None,
    family_filter: Optional[str] = None,
) -> list[RetrievalHit]:
    """Return the top-K bank entries by cosine similarity.

    ``query`` may be 1-D ``(D,)`` or 2-D ``(1, D)``. It is L2-normalised in
    place if not already; this is cheap and lets callers pass raw model
    outputs. ``exclude_id`` drops a single bank row by id (useful when the
    query *is* one of the bank entries — eg ablation studies). ``family_filter``
    restricts the candidate pool to one family.
    """
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(q)
    if norm > 0.0:
        q = q / norm

    sims = bank.embeddings @ q   # (N,)

    mask = np.ones(bank.n, dtype=bool)
    if exclude_id is not None:
        mask &= bank.ids != exclude_id
    if family_filter is not None:
        mask &= bank.families == family_filter

    if not bool(mask.any()):
        return []

    candidate_idx = np.flatnonzero(mask)
    candidate_sims = sims[candidate_idx]
    k_eff = min(k, candidate_idx.size)
    # argpartition picks the top-K unsorted; argsort within only the K hits.
    top_part = np.argpartition(-candidate_sims, k_eff - 1)[:k_eff]
    top_sorted = top_part[np.argsort(-candidate_sims[top_part])]
    return [
        RetrievalHit(
            rank=rank,
            id=str(bank.ids[candidate_idx[i]]),
            family=str(bank.families[candidate_idx[i]]),
            split=str(bank.splits[candidate_idx[i]]),
            similarity=float(candidate_sims[i]),
        )
        for rank, i in enumerate(top_sorted)
    ]


def encode_image(
    image_path: Path,
    *,
    checkpoint: Path,
    base_channels: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    device_override: Optional[str] = None,
) -> np.ndarray:
    """Load model + image and return a single L2-normalised embedding
    vector. Used by the standalone retrieve CLI for off-bank queries."""
    device = pick_device(device_override)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    bc = base_channels or int(cfg.get("base_channels", 32))
    ed = embedding_dim or 128
    model = FractalCNNv2(base_channels=bc, embedding_dim=ed).to(device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).embedding
    return emb.detach().cpu().numpy().reshape(-1)


def lookup_recipe_rows(
    data_root: Path,
    ids: list[str],
) -> pd.DataFrame:
    """Join hits back to their full recipe rows from labels.parquet.

    Returned rows preserve ``ids`` order and carry every column of the
    original parquet — useful for inspecting ``c_re``/``c_im``/``exponent``
    + viewport of the retrieved neighbours, not just their ids.
    """
    df = pd.read_parquet(data_root / "labels.parquet")
    df = df[df["id"].isin(ids)].set_index("id")
    return df.reindex(ids).reset_index()
