"""Offline tests for the Slice 4 embedding pipeline.

Coverage:
* Supervised contrastive loss: zero on degenerate (single-class) batches,
  finite + differentiable on mixed batches, and quantitatively pulls
  same-class pairs together.
* Embedding head produces unit-norm vectors.
* Embedding bank round-trips through .npz.
* Retrieval ranking is correct on synthetic embeddings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fractalov_ml.embeddings import (
    EmbeddingBank,
    RetrievalHit,
    retrieve_topk,
)
from fractalov_ml.models.cnn_v2 import FractalCNNv2
from fractalov_ml.training.contrastive import supervised_contrastive_loss


# ---------------- contrastive ----------------


def test_supcon_zero_when_no_positives() -> None:
    # Every label distinct → no positives → loss == 0 by definition.
    z = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    labels = torch.tensor([0, 1, 2, 3])
    loss = supervised_contrastive_loss(z, labels)
    assert loss.item() == 0.0


def test_supcon_zero_on_singleton_batch() -> None:
    z = torch.nn.functional.normalize(torch.randn(1, 8), dim=1)
    labels = torch.tensor([0])
    loss = supervised_contrastive_loss(z, labels)
    assert loss.item() == 0.0


def test_supcon_finite_and_differentiable_on_mixed_batch() -> None:
    torch.manual_seed(0)
    raw = torch.randn(8, 16, requires_grad=True)
    z = torch.nn.functional.normalize(raw, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supervised_contrastive_loss(z, labels, temperature=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_supcon_pulls_same_class_pairs_together() -> None:
    """If we step the embeddings against the SupCon gradient a few times,
    same-label pairs should grow more similar (higher cosine) and
    different-label pairs less similar."""
    torch.manual_seed(0)
    raw = torch.randn(8, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    def same_class_sim(emb: torch.Tensor) -> float:
        e = torch.nn.functional.normalize(emb.detach(), dim=1)
        sim = e @ e.t()
        same_mask = labels.view(-1, 1) == labels.view(1, -1)
        diag_mask = ~torch.eye(emb.size(0), dtype=torch.bool)
        return float(sim[same_mask & diag_mask].mean().item())

    before = same_class_sim(raw)

    optim = torch.optim.SGD([raw], lr=0.5)
    for _ in range(20):
        optim.zero_grad()
        # Re-normalise per step so the loss sees inputs on the unit sphere.
        z_norm = torch.nn.functional.normalize(raw, dim=1)
        loss = supervised_contrastive_loss(z_norm, labels, temperature=0.1)
        loss.backward()
        optim.step()

    after = same_class_sim(raw)
    assert after > before, f"SupCon should raise within-class similarity: {before} → {after}"


def test_supcon_rejects_misshaped_inputs() -> None:
    with pytest.raises(ValueError):
        supervised_contrastive_loss(torch.zeros(4), torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError):
        supervised_contrastive_loss(torch.zeros(4, 8), torch.zeros(3, dtype=torch.long))


# ---------------- model embedding head ----------------


def test_model_embedding_is_unit_norm() -> None:
    torch.manual_seed(0)
    model = FractalCNNv2(base_channels=8, embedding_dim=32)
    x = torch.randn(5, 3, 32, 32)
    out = model(x)
    norms = out.embedding.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert out.embedding.shape == (5, 32)


def test_model_embedding_dim_property() -> None:
    model = FractalCNNv2(base_channels=8, embedding_dim=64)
    assert model.embedding_dim == 64


def test_embed_helper_runs_in_eval_mode() -> None:
    """``model.embed()`` should produce identical outputs regardless of the
    train/eval flag at call time, because it forces eval inside."""
    torch.manual_seed(0)
    model = FractalCNNv2(base_channels=8, embedding_dim=16)
    x = torch.randn(3, 3, 32, 32)

    model.train()
    a = model.embed(x)
    model.eval()
    b = model.embed(x)
    assert torch.allclose(a, b, atol=1e-6)


# ---------------- embedding bank round-trip ----------------


def test_bank_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n, d = 7, 16
    raw = rng.standard_normal(size=(n, d)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    bank = EmbeddingBank(
        ids=np.array([f"id_{i}" for i in range(n)], dtype=str),
        families=np.array(["mandelbrot", "julia"] * 3 + ["multibrot"], dtype=str),
        splits=np.array(["train"] * 5 + ["test"] * 2, dtype=str),
        embeddings=raw,
    )
    path = tmp_path / "bank.npz"
    bank.save(path)
    loaded = EmbeddingBank.load(path)
    assert loaded.n == bank.n
    assert loaded.dim == bank.dim
    assert (loaded.ids == bank.ids).all()
    assert (loaded.families == bank.families).all()
    assert (loaded.splits == bank.splits).all()
    assert np.allclose(loaded.embeddings, bank.embeddings)


# ---------------- retrieval ----------------


def _make_bank(embeddings: np.ndarray, families: list[str]) -> EmbeddingBank:
    embeddings = embeddings.astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return EmbeddingBank(
        ids=np.array([f"id_{i}" for i in range(len(embeddings))], dtype=str),
        families=np.array(families, dtype=str),
        splits=np.array(["train"] * len(embeddings), dtype=str),
        embeddings=embeddings,
    )


def test_retrieve_topk_orders_by_similarity() -> None:
    # Two clusters: anchor near (1, 0); a "near" point and a "far" one.
    bank = _make_bank(
        np.array([[1.0, 0.0], [0.99, 0.01], [-0.9, 0.1], [0.5, 0.5]]),
        ["mandelbrot"] * 4,
    )
    query = np.array([1.0, 0.0])
    hits = retrieve_topk(bank, query, k=3)
    assert len(hits) == 3
    assert [h.id for h in hits] == ["id_0", "id_1", "id_3"]
    # Sims are monotonically non-increasing.
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)


def test_retrieve_excludes_self() -> None:
    bank = _make_bank(np.eye(3), ["mandelbrot", "julia", "burning_ship"])
    hits = retrieve_topk(bank, bank.embeddings[0], k=3, exclude_id="id_0")
    assert all(h.id != "id_0" for h in hits)
    assert len(hits) == 2


def test_retrieve_family_filter() -> None:
    bank = _make_bank(
        np.array([[1.0, 0.0], [0.95, 0.05], [-1.0, 0.0]]),
        ["mandelbrot", "julia", "julia"],
    )
    query = np.array([1.0, 0.0])
    julia_hits = retrieve_topk(bank, query, k=10, family_filter="julia")
    assert all(h.family == "julia" for h in julia_hits)
    # The closer Julia (id_1) outranks the far one (id_2).
    assert julia_hits[0].id == "id_1"


def test_retrieve_handles_empty_bank() -> None:
    bank = _make_bank(np.array([[1.0, 0.0]]), ["mandelbrot"])
    hits = retrieve_topk(bank, np.array([1.0, 0.0]), k=5, exclude_id="id_0")
    assert hits == []


def test_retrieval_hit_dataclass() -> None:
    h = RetrievalHit(rank=0, id="x", family="julia", split="train", similarity=0.9)
    # Should be a frozen dataclass.
    with pytest.raises(Exception):
        h.rank = 1   # type: ignore[misc]
