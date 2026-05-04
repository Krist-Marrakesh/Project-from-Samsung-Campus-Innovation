"""Supervised contrastive (SupCon) loss for embedding-space training.

Reference: Khosla et al., "Supervised Contrastive Learning" (2020).

Given a batch of L2-normalised embeddings ``z_i`` and labels ``y_i``, the
SupCon loss for anchor ``i`` is::

    L_i = -1/|P(i)| * Σ_{p ∈ P(i)} log(  exp(z_i · z_p / τ)  /  Σ_{a ≠ i} exp(z_i · z_a / τ)  )

where ``P(i)`` is the set of indices ``p ≠ i`` with ``y_p == y_i`` (positives)
and the denominator runs over every index ``a ≠ i`` (positives + negatives).

In our setup the supervisory label is ``family``. That means the embedding
space is structured so that "two images of the same fractal family land
near each other"; *within* a family it stays uniform on the unit sphere,
which is exactly what we want for retrieval — two julia images at very
different ``c`` should still be near in family-space without collapsing
onto each other.

Why SupCon and not InfoNCE / triplet:

* InfoNCE wants self-augmented positives, which requires a two-view data
  pipeline. We have a single view per image and supervisory family labels
  — SupCon is the natural fit.
* Triplet loss requires explicit hard-negative mining. SupCon dissolves
  that into the log-sum-exp denominator, which is gentler on small batches.

Numerics: log-sum-exp is computed via the ``logits.max()``-subtraction
trick, identical to ``torch.logsumexp``. The implementation keeps a
self-mask on the diagonal so ``z_i · z_i`` doesn't dominate the
denominator at low temperature.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute SupCon over a single mini-batch.

    Parameters
    ----------
    embeddings:
        ``(B, D)`` tensor on the unit sphere. Pre-normalised: the model
        produces L2-normalised outputs in :func:`FractalCNNv2.forward`,
        so callers don't need to normalise again.
    labels:
        ``(B,)`` integer class labels.
    temperature:
        Softmax temperature τ. Lower τ → sharper distinction, higher τ →
        softer. ``0.1`` is the SupCon paper default and works well across
        our family count.

    Returns
    -------
    Scalar loss tensor. When the batch has zero positives for every anchor
    (e.g. a singleton class on every row, possible under heavy class
    imbalance), returns 0 so the joint loss does not error.
    """
    if embeddings.dim() != 2:
        raise ValueError(f"embeddings must be (B, D); got {tuple(embeddings.shape)}")
    if labels.dim() != 1 or labels.size(0) != embeddings.size(0):
        raise ValueError("labels must be (B,) and align with embeddings")

    batch_size = embeddings.size(0)
    if batch_size < 2:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    # Cosine similarity reduces to dot product because embeddings are
    # already unit-norm. ``mm`` is faster than ``einsum`` here.
    logits = embeddings @ embeddings.t() / float(temperature)

    # Self-mask: subtract a large constant on the diagonal so exp(self) is
    # effectively zero. Done *before* logsumexp for numerical stability.
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    logits = logits.masked_fill(self_mask, float("-inf"))

    # Positive mask: y_i == y_j AND i != j.
    eq = labels.view(-1, 1).eq(labels.view(1, -1))
    pos_mask = eq & ~self_mask  # (B, B), bool

    # Anchors with no positives in the batch contribute zero to the loss.
    has_pos = pos_mask.any(dim=1)
    if not bool(has_pos.any()):
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    # log(softmax) over rows, with -inf on diagonal contributing 0 to denom.
    log_prob = F.log_softmax(logits, dim=1)

    # Mean over positives per anchor. We use ``where`` to gate non-positive
    # entries to 0; the obvious ``pos_mask.float() * log_prob`` formulation
    # blows up because the self-cell carries log_prob = -inf and
    # ``0 * -inf = NaN``. ``where`` short-circuits the false branch and
    # picks the constant 0 instead.
    log_prob_pos = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    pos_count = pos_mask.float().sum(dim=1).clamp(min=1.0)
    mean_log_prob_pos = log_prob_pos.sum(dim=1) / pos_count
    loss_per_anchor = -mean_log_prob_pos
    return loss_per_anchor[has_pos].mean()
