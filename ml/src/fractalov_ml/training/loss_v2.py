"""Multi-task loss with Gaussian-NLL uncertainty heads.

Components:

* ``cls_loss`` — cross-entropy on the family head. No mask; every example
  belongs to exactly one family.
* ``c_nll`` — Gaussian negative log-likelihood for Julia c. Index-selected
  on ``has_julia`` BEFORE the loss is computed; multiplying by a 0/1 mask
  after the fact propagates ``NaN`` when the regression head produces a
  non-finite ``log σ²``, which happens often during the first few epochs.
* ``exp_nll`` — Gaussian NLL for Multibrot exponent. Same masking model.
* ``vp_nll`` — Gaussian NLL for the four viewport bounds. Always-on; every
  recipe carries a viewport so there is no mask. Powers reconstruction for
  Mandelbrot and Burning Ship which have no recipe-internal continuous
  parameter to refine.
* ``contrastive`` — supervised-contrastive on the embedding head. Off by
  default (``contrastive_weight=0``); see :mod:`.contrastive`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# log σ² clamp range. log σ² < -7 corresponds to σ < ~3e-2 (overconfident);
# log σ² > 7 to σ > ~30 (effectively no signal). Beyond these the NLL is
# numerically unstable on float32.
LOG_VAR_CLAMP_MIN = -7.0
LOG_VAR_CLAMP_MAX = 7.0


@dataclass
class LossPieces:
    total: torch.Tensor
    cls: torch.Tensor
    c_nll: torch.Tensor          # 0 if no Julia rows in batch
    exp_nll: torch.Tensor        # 0 if no Multibrot rows in batch
    vp_nll: torch.Tensor         # always present (every example has a viewport)
    contrastive: torch.Tensor    # 0 when contrastive_weight == 0
    n_julia: int
    n_multibrot: int


def gaussian_nll(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Numerically-stable Gaussian negative log-likelihood.

    NLL = 0.5 * ( (target - mean)² / σ² + log σ² ) + const

    The ``log σ²`` clamp keeps gradients finite when the head temporarily
    blows up early in training; in practice the network learns to operate
    within the clamp range within a few hundred steps.
    """
    log_var = log_var.clamp(min=LOG_VAR_CLAMP_MIN, max=LOG_VAR_CLAMP_MAX)
    inv_var = torch.exp(-log_var)
    sq_err = (target - mean) ** 2
    raw = 0.5 * (sq_err * inv_var + log_var)
    if reduction == "mean":
        return raw.mean()
    if reduction == "sum":
        return raw.sum()
    if reduction == "none":
        return raw
    raise ValueError(f"unknown reduction: {reduction}")


def compute_loss(
    output,
    *,
    family_idx: torch.Tensor,
    c_target: torch.Tensor,         # (B, 2)
    exp_target: torch.Tensor,       # (B,)
    vp_target: torch.Tensor,        # (B, 4) (xMin, xMax, yMin, yMax)
    has_julia: torch.Tensor,        # (B,) bool
    has_multibrot: torch.Tensor,    # (B,) bool
    cls_weight: float = 1.0,
    c_weight: float = 0.5,
    exp_weight: float = 0.5,
    vp_weight: float = 0.25,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.1,
) -> LossPieces:
    """Multi-task loss. ``output`` is a :class:`fractalov_ml.models.cnn_v2.ModelOutputV2`."""
    cls_loss = F.cross_entropy(output.family_logits, family_idx)

    julia_idx = torch.nonzero(has_julia, as_tuple=False).squeeze(1)
    if julia_idx.numel() > 0:
        c_nll = gaussian_nll(
            output.c_mean.index_select(0, julia_idx),
            output.c_log_var.index_select(0, julia_idx),
            c_target.index_select(0, julia_idx),
        )
    else:
        c_nll = torch.zeros((), device=output.family_logits.device)

    mb_idx = torch.nonzero(has_multibrot, as_tuple=False).squeeze(1)
    if mb_idx.numel() > 0:
        exp_nll = gaussian_nll(
            output.exp_mean.index_select(0, mb_idx),
            output.exp_log_var.index_select(0, mb_idx),
            exp_target.index_select(0, mb_idx),
        )
    else:
        exp_nll = torch.zeros((), device=output.family_logits.device)

    # Viewport MVE — unmasked, every row contributes.
    vp_nll = gaussian_nll(output.viewport_mean, output.viewport_log_var, vp_target)

    if contrastive_weight > 0.0:
        # Local import keeps the loss module light when SupCon is unused.
        from .contrastive import supervised_contrastive_loss
        con_loss = supervised_contrastive_loss(
            output.embedding, family_idx, temperature=contrastive_temperature,
        )
    else:
        con_loss = torch.zeros((), device=output.family_logits.device)

    total = (
        cls_weight * cls_loss
        + c_weight * c_nll
        + exp_weight * exp_nll
        + vp_weight * vp_nll
        + contrastive_weight * con_loss
    )

    return LossPieces(
        total=total,
        cls=cls_loss.detach(),
        c_nll=c_nll.detach(),
        exp_nll=exp_nll.detach(),
        vp_nll=vp_nll.detach(),
        contrastive=con_loss.detach(),
        n_julia=int(julia_idx.numel()),
        n_multibrot=int(mb_idx.numel()),
    )
